#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "thread.h"

namespace dxvk {

  // ==========================================================================
  // THE JOB GRAPH -- ARCHITECTURE_OVERHAUL.md sec 4.2, 4.2.1, 4.2.2, slice 6.
  //
  // WHAT IS MISSING TODAY. WorkerThreadPool::Schedule takes a closure and
  // returns a bare Future. That is the entire vocabulary: there is no
  // dependency counter, no continuation, no way for a job to make another job
  // runnable. So the frame is a sequence of phases separated by GLOBAL
  // BARRIERS -- flushGeometryBatch schedules N chunks, joins them in order in a
  // busy spin, then drains the CS thread with SynchronizeCsThread(SynchronizeAll).
  // Every phase waits for every other. That is invariant I5 being violated by
  // the primitive rather than by the code using it: "ordering is a dependency,
  // not a barrier", and a pool that can only express barriers cannot express
  // ordering any other way.
  //
  // ------------------------------------------------------------------------
  // WHY THE CONTRACT IS GRAPH-OWNED AND NOT POOL-OWNED. sec 4.2.1, and this is
  // the part that decides the whole shape of this file.
  //
  // The counters are right; Future CANNOT CARRY THEM. The existing pool is a
  // bounded task ring with recycled ids and a Future is a raw pointer into it:
  //
  //   util_threadpool.h   TaskId taskId = m_taskId++ & (m_taskCount - 1);
  //   util_threadpool.h   mutable Task* task = nullptr;
  //   util_threadpool.h   capture() calls result.reset(), clearing hasResult
  //                       under any holder still sitting in get()
  //
  // A retained Future is therefore invariant I2 being violated INSIDE the
  // primitive a graph would be built on. It is safe today only by accident:
  // flushGeometryBatch schedules ~worker-count chunks and joins them
  // immediately, so m_taskId never advances a full lap between capture and get.
  // A graph holds completion tokens ACROSS phases -- which is the entire point
  // of having one -- and removes that accident. (There is now a detector for
  // it: dxvk::g_taskRingReuseCount, sec 9 item 8.)
  //
  // Two more, and both are silent DEADLOCKS rather than corruption:
  //
  //   cancellation never completes   cancel() sets isDisposed, and the capture
  //                                  thunk is `if (!result.disposed()) set(...)`.
  //                                  A cancelled task never calls set(). A
  //                                  counter decremented inside set() never
  //                                  fires, and every job waiting on it is
  //                                  unreachable. The pool's own destructor
  //                                  uses this path, so shutdown with a live
  //                                  graph hangs.
  //   a full queue never completes   Schedule returns a default Future when
  //                                  isFull(). flushGeometryBatch survives only
  //                                  because the CALL SITE checks f.valid() and
  //                                  runs the range inline.
  //
  // So: Task and Future are demoted to "a way to get a worker to run this
  // node's body". NOTHING outside the pool holds a Future across a phase
  // boundary, and enqueue-or-run-inline moves INTO the graph instead of being
  // re-implemented at each call site.
  //
  // This class therefore does not know what a thread pool is. It takes a
  // Dispatch callable that either runs the body somewhere or reports that it
  // could not, and the graph handles the could-not. That is also what makes the
  // sec 4.2.1 gate runnable: an undersized queue is just a Dispatch that
  // returns false a lot.
  //
  // ------------------------------------------------------------------------
  // THE NODE CONTRACT, sec 4.2.2, in full:
  //
  //   node exists
  //     -> node has N unresolved dependencies
  //     -> node becomes runnable at N == 0
  //     -> node executes ANYWHERE (worker, or inline on the enqueuer)
  //     -> node signals completion EXACTLY ONCE, on every exit path
  //     -> each dependent decrements; the 0 transition enqueues it
  //     -> repeat
  //
  // A NODE MEANS "A NAMED PIECE OF STATE IS NOW VALID", not "a function ran".
  // Transform(P) and Material(P) name disjoint state, carry no edge, and run at
  // the same time. Today they are separated by a phase boundary for no reason
  // other than that a phase boundary is the only thing the pool can express.
  //
  // CANCELLATION IS COMPLETION, NOT A STATE. From a dependent's point of view a
  // cancelled node means "this node will produce no useful work, AND this node
  // is done". Any design where the dependent can observe the difference has
  // reintroduced the deadlock above. So there is no cancelled-ness to query.
  //
  // ZERO CHILDREN IS THE SAME EVENT. parallelFor(0, ...) must mean
  // completeImmediately(), never return. This tree has roughly twenty zero-work
  // paths -- no bones, no fanout, no instances, a replay hit, a filtered draw --
  // and each one that returns early instead of completing is a permanently
  // unreachable dependent. They are also why SubmitDrawDeferred is 9,922 lines:
  // in a phase system every one of them is a special case, and in a graph they
  // are all the same case.
  //
  // ------------------------------------------------------------------------
  // WHAT THIS DELIBERATELY IS NOT. sec 4.4: counters, a parallel-for that can
  // spawn, and an empty-completion job. THREE PRIMITIVES. Not a generic
  // task-graph framework, no priorities, no work-stealing of its own (the pool
  // already does that), no cross-frame pipelining -- sec 4.2.4 explains why
  // that last one is a trap rather than the next step.
  //
  // NOT THREAD-SAFE FOR GRAPH CONSTRUCTION. createNode/addEdge/submit are
  // called from one thread, the one building the frame's graph. Node BODIES run
  // anywhere and signalling is atomic. This is the same split ShardedInstancePhase
  // already relies on and it is what keeps the counters lock-free.
  // ==========================================================================

  class JobGraph {
  public:
    // Into the graph's own node array -- never into the pool's task ring. The
    // generation is what sec 4.2.1's gate is about: "the node array's
    // generation must catch any handle that outlived its node".
    struct JobHandle {
      static constexpr uint32_t kInvalidIndex = ~0u;

      uint32_t index = kInvalidIndex;
      uint32_t generation = 0u;

      bool valid() const { return index != kInvalidIndex; }
      bool operator==(const JobHandle& o) const {
        return index == o.index && generation == o.generation;
      }
      bool operator!=(const JobHandle& o) const { return !(*this == o); }
    };

    // Runs the body somewhere and returns true, or returns false because it
    // could not (queue full, pool shutting down, no pool at all). FALSE IS NOT
    // AN ERROR: the graph runs the body inline and the node completes normally,
    // which is the whole reason this returns bool rather than void.
    using Dispatch = std::function<bool(std::function<void()>)>;

    explicit JobGraph(Dispatch dispatch);
    ~JobGraph();

    JobGraph(const JobGraph&) = delete;
    JobGraph& operator=(const JobGraph&) = delete;

    // ----------------------------------------------------------------------
    // Construction. Single-threaded; see the class comment.

    // A node starts with one "unsubmitted" hold on its counter so that edges
    // can be added before it can possibly run. submit() releases that hold.
    // Without it a node with no dependencies would become runnable the instant
    // it was created and a subsequent addEdge would be a lost dependency --
    // silent, and the worst kind.
    //
    // `body` may be empty. An empty node is a pure join point, which is how a
    // fan-in is expressed without inventing a second concept.
    JobHandle createNode(const char* name, std::function<void()> body);

    // `after` becomes runnable only once `before` has signalled.
    //
    // Asserts if `before` has already signalled: adding an edge to a completed
    // node is a lost dependency, and it must be loud. This is why submit() is a
    // separate call -- it makes "still safe to add edges" a state the graph can
    // check rather than a property of when you happened to call things.
    void addEdge(JobHandle before, JobHandle after);

    // Arm the node. No further edges may target it. If its counter is already
    // at zero it becomes runnable now.
    void submit(JobHandle node);

    // ----------------------------------------------------------------------
    // THE FAN-OUT, and the counter is the composable unit rather than the node.
    //
    // A consumer attaches to the COUNTER, so it can be wired before the child
    // count is known:
    //
    //     GeometryGather(P) -> Counter G -> c0..c383 -> Instance(P)
    //
    // Instance(P).dependsOn(G) is written once. Wiring 384 individual edges
    // would mean the scheduler had to know the count at graph-build time, which
    // is precisely what a gather cannot supply.
    //
    // `join` must be a created but NOT YET SUBMITTED node whose dependents are
    // already wired. This call adds `count` holds to it, spawns `count`
    // invocations of body(i) each of which releases one, and then submits it --
    // so `join` runs once every child has finished, and its dependents follow.
    //
    // count == 0 SUBMITS `join` ANYWAY, which is the whole point: zero children
    // is the same event as N children, not an early return. See the class
    // comment on the twenty zero-work paths this is for.
    //
    // Safe to call from inside another node's body, which is what makes the
    // fan-out composable -- a gather that does not know it will produce 384
    // shards until it has run can still wire its consumer up front.
    void parallelFor(JobHandle join, uint32_t count, std::function<void(uint32_t)> body);

    // ----------------------------------------------------------------------
    // Block until every node has signalled. Runs runnable nodes on the calling
    // thread while it waits rather than spinning, so a graph whose dispatch
    // never succeeds still completes -- the undersized-queue half of sec
    // 4.2.1's gate.
    //
    // EVERY CREATED NODE MUST BE SUBMITTED (directly, or by parallelFor) BEFORE
    // CALLING THIS. A node that is never submitted still holds its creation
    // hold, never becomes runnable, never signals, and this waits for it
    // forever. That is deliberate rather than defended against: the alternative
    // is to quietly complete nodes the caller forgot about, and a dependent
    // that runs because its producer was FORGOTTEN is the same silent-wrong
    // class as a lost edge. A hang names the bug; a silent completion hides it.
    // The destructor calls cancelAll for the abnormal path.
    //
    // Also: all createNode/addEdge calls must happen before waitAll. Node
    // BODIES may call parallelFor -- that is what makes the fan-out composable
    // -- but they may not create nodes, because m_nodes is only single-thread
    // safe during construction. The counter, not the node, is the composable
    // unit; see sec 4.2.2.
    void waitAll();

    // Signal completion for every node without running its body. Every
    // dependent still becomes runnable, because cancellation is completion.
    void cancelAll();

    // ----------------------------------------------------------------------
    // A SNAPSHOT, not a reference into live state. Every one of these is
    // incremented from node bodies, which run on worker threads, so the
    // counters themselves are atomic and this is the value copied out of them.
    // Handing back a const& to a struct being written by four threads would be
    // a data race in the diagnostic that is supposed to prove the scheduler is
    // sound.
    struct Stats {
      uint32_t nodes = 0;
      uint32_t dispatched = 0;    // handed to the Dispatch and accepted
      uint32_t ranInline = 0;     // Dispatch refused, or waitAll drained it
      uint32_t cancelled = 0;
      uint32_t threw = 0;
      // Handles that named a node that had already been recycled. MUST be 0.
      // This is sec 4.2.1's gate: "the node array's generation must catch any
      // handle that outlived its node."
      uint32_t staleHandles = 0;
    };
    Stats stats() const;

    void reset();

    // ----------------------------------------------------------------------
    // SLICE 6'S ACCEPTANCE GATE, AS CODE. sec 4.2.1 states it exactly:
    //
    //   "run the graph with a deliberately undersized queue (NumTasksPerThread
    //    forced low) and with cancellation injected at random nodes. Every
    //    dependent must still become runnable, and the node array's generation
    //    must catch any handle that outlived its node. If either needs a
    //    call-site check to hold, the contract is not in the graph yet."
    //
    // Written as a self-test rather than as a paragraph because a gate nobody
    // can run is a gate nobody runs. It needs no thread pool: the undersized
    // queue is modelled by a Dispatch that refuses, which is the same condition
    // Schedule's isFull() produces and the one flushGeometryBatch currently
    // survives only by checking f.valid() at the call site.
    //
    // Returns true if every property held. Logs nothing on success.
    static bool selfTest();

  private:
    struct Node {
      const char* name = "";
      std::function<void()> body;
      // Unresolved dependencies plus one for "not yet submitted".
      std::atomic<uint32_t> pending { 0u };
      // EXACTLY ONCE, on every exit path. This is the flag that makes that
      // true rather than intended.
      std::atomic<bool> signalled { false };
      std::vector<JobHandle> dependents;
      bool submitted = false;
    };

    Node* get(JobHandle h);
    // Release ONE hold on a node's counter. At zero the node becomes runnable.
    void releaseHold(JobHandle h);
    // Signal the node itself complete -- exactly once, whatever the exit path.
    void signalComplete(JobHandle h);
    void makeRunnable(JobHandle h);
    void runNode(JobHandle h);
    bool popReady(JobHandle& out);

    Dispatch m_dispatch;

    // INDIRECTED ON PURPOSE. Node holds std::atomic members, so it is neither
    // copyable nor movable, and a std::vector<Node> could not grow. Boxing also
    // gives every node a stable address for the whole graph's life, which
    // matters because a body running on a worker can add nodes on another
    // thread's behalf during a fan-out.
    std::vector<std::unique_ptr<Node>> m_nodes;
    std::vector<uint32_t> m_generations;

    // Nodes that are runnable but have not been dispatched yet. Drained by
    // waitAll on the calling thread, which is what makes progress independent
    // of whether Dispatch ever succeeds.
    std::vector<JobHandle> m_readyQueue;
    dxvk::mutex m_readyMutex;

    std::atomic<uint32_t> m_outstanding { 0u };

    struct AtomicStats {
      std::atomic<uint32_t> nodes { 0u };
      std::atomic<uint32_t> dispatched { 0u };
      std::atomic<uint32_t> ranInline { 0u };
      std::atomic<uint32_t> cancelled { 0u };
      std::atomic<uint32_t> threw { 0u };
      std::atomic<uint32_t> staleHandles { 0u };
      void reset() {
        nodes = 0u; dispatched = 0u; ranInline = 0u;
        cancelled = 0u; threw = 0u; staleHandles = 0u;
      }
    };
    // Mutable so the const accessors that detect a stale handle can still count
    // it. A staleHandle found from a const path is exactly as important as one
    // found from a non-const path, and losing it would defeat the gate.
    mutable AtomicStats m_stats;
  };

}
