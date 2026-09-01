#include "util_job_graph.h"

#include <cassert>
#include <thread>

// selfTest only -- the graph itself deliberately has no log dependency, so that
// this header stays cheap for the hot paths that will include it.
#include "log/log.h"
#include "util_string.h"

namespace dxvk {

  JobGraph::JobGraph(Dispatch dispatch)
  : m_dispatch(std::move(dispatch)) {
  }

  JobGraph::~JobGraph() {
    // A graph destroyed with work still outstanding would leave node bodies
    // holding `this`. Cancellation is completion, so this drains rather than
    // abandons -- and because cancelAll signals every node, nothing can be left
    // waiting on a counter that will never fire.
    if (m_outstanding.load(std::memory_order_acquire) != 0u) {
      cancelAll();
    }
  }

  JobGraph::Node* JobGraph::get(JobHandle h) {
    if (!h.valid() || h.index >= m_nodes.size()) {
      return nullptr;
    }
    if (m_generations[h.index] != h.generation) {
      // sec 4.2.1's gate. A handle that outlived its node must be caught by the
      // generation, not followed.
      m_stats.staleHandles.fetch_add(1u, std::memory_order_relaxed);
      return nullptr;
    }
    return m_nodes[h.index].get();
  }

  JobGraph::JobHandle JobGraph::createNode(const char* name, std::function<void()> body) {
    const uint32_t idx = static_cast<uint32_t>(m_nodes.size());
    m_nodes.emplace_back(new Node());
    // m_generations is index-parallel but never shrinks -- see reset(). A slot
    // reused after a reset keeps the generation that made the old handle stale.
    if (idx >= m_generations.size()) {
      m_generations.push_back(0u);
    }

    Node& n = *m_nodes[idx];
    n.name = (name != nullptr) ? name : "";
    n.body = std::move(body);
    // ONE HOLD FOR "NOT YET SUBMITTED". Without it a node with no dependencies
    // would be runnable the instant it was created, and a later addEdge would
    // be a silently lost dependency. submit() releases this hold, which is what
    // makes "still safe to add edges" a state rather than a timing convention.
    n.pending.store(1u, std::memory_order_relaxed);

    m_stats.nodes.fetch_add(1u, std::memory_order_relaxed);
    m_outstanding.fetch_add(1u, std::memory_order_acq_rel);

    JobHandle h;
    h.index = idx;
    h.generation = m_generations[idx];
    return h;
  }

  void JobGraph::addEdge(JobHandle before, JobHandle after) {
    Node* b = get(before);
    Node* a = get(after);
    if (b == nullptr || a == nullptr) {
      return;
    }

    // LOUD, NOT LENIENT. An edge added to a node that has already signalled is
    // a dependency that will never be honoured, and the dependent would run
    // early rather than fail -- which is a data race that reproduces once a
    // week. There is no safe recovery here, so it asserts.
    assert(!b->signalled.load(std::memory_order_acquire) &&
           "JobGraph::addEdge: `before` has already completed -- this dependency would be lost");
    assert(!a->submitted &&
           "JobGraph::addEdge: `after` is already submitted -- no further edges may target it");

    b->dependents.push_back(after);
    a->pending.fetch_add(1u, std::memory_order_acq_rel);
  }

  void JobGraph::submit(JobHandle node) {
    Node* n = get(node);
    if (n == nullptr || n->submitted) {
      return;
    }
    n->submitted = true;
    // Release the creation hold. If nothing else is outstanding this makes the
    // node runnable right here.
    releaseHold(node);
  }

  void JobGraph::releaseHold(JobHandle h) {
    Node* n = get(h);
    if (n == nullptr) {
      return;
    }
    const uint32_t before = n->pending.fetch_sub(1u, std::memory_order_acq_rel);
    assert(before != 0u && "JobGraph: counter underflow -- a hold was released twice");
    if (before == 1u) {
      makeRunnable(h);
    }
  }

  void JobGraph::makeRunnable(JobHandle h) {
    // ENQUEUE-OR-RUN-INLINE LIVES HERE, not at the call site. sec 4.2.1: a
    // graph that assumes Schedule yields a token that will complete deadlocks
    // the first time a worker queue fills, and the existing tree survives that
    // only because flushGeometryBatch checks f.valid() by hand.
    if (m_dispatch) {
      const bool taken = m_dispatch([this, h]() { runNode(h); });
      if (taken) {
        m_stats.dispatched.fetch_add(1u, std::memory_order_relaxed);
        return;
      }
    }

    // Refused, or there is no dispatcher at all. The node is still runnable and
    // still owes its dependents a completion, so it goes on the ready queue for
    // waitAll to drain on the calling thread. Running it right here instead
    // would recurse through an arbitrarily deep chain on whichever thread
    // happened to release the last hold.
    {
      std::lock_guard<dxvk::mutex> lock(m_readyMutex);
      m_readyQueue.push_back(h);
    }
  }

  bool JobGraph::popReady(JobHandle& out) {
    std::lock_guard<dxvk::mutex> lock(m_readyMutex);
    if (m_readyQueue.empty()) {
      return false;
    }
    out = m_readyQueue.back();
    m_readyQueue.pop_back();
    return true;
  }

  void JobGraph::runNode(JobHandle h) {
    Node* n = get(h);
    if (n == nullptr) {
      return;
    }

    // EXACTLY ONCE, ON EVERY EXIT PATH -- ran to completion, threw, or was
    // cancelled underneath us. The completion is unconditional and happens
    // after the body whatever the body did, which is the sec 4.2.2 contract
    // stated as code rather than as a rule.
    struct Scope {
      JobGraph* g;
      JobHandle h;
      ~Scope() { g->signalComplete(h); }
    } scope { this, h };

    if (!n->body) {
      // A pure join point. Not a special case -- an empty body is how a fan-in
      // is expressed, and it completes exactly like any other node.
      return;
    }

    try {
      n->body();
    } catch (...) {
      // A thrown body is still a completed node. Swallowing here is deliberate:
      // letting it escape would unwind through a worker thread and skip every
      // dependent's counter, which is the deadlock this whole file exists to
      // make impossible. Counted so it cannot be silent.
      m_stats.threw.fetch_add(1u, std::memory_order_relaxed);
    }
  }

  void JobGraph::signalComplete(JobHandle h) {
    Node* n = get(h);
    if (n == nullptr) {
      return;
    }

    // IDEMPOTENT BY EXCHANGE. This is what makes "exactly once" true rather
    // than intended: cancelAll and a normal completion can race, and a double
    // signal would decrement every dependent twice and run them early.
    if (n->signalled.exchange(true, std::memory_order_acq_rel)) {
      return;
    }

    // Copy before releasing: a dependent that becomes runnable may run, finish,
    // and (through a fan-out) append to some other node's list while we are
    // still iterating this one. The vector itself is not touched after this
    // node signalled, but taking the copy makes that independent of future
    // edits rather than dependent on them.
    const std::vector<JobHandle> deps = n->dependents;

    m_outstanding.fetch_sub(1u, std::memory_order_acq_rel);

    for (const JobHandle& d : deps) {
      releaseHold(d);
    }
  }

  void JobGraph::parallelFor(JobHandle join, uint32_t count, std::function<void(uint32_t)> body) {
    Node* n = get(join);
    if (n == nullptr) {
      return;
    }

    // The creation hold must still be held, or the two releases below are one
    // too many and the counter underflows. Loud rather than lenient for the
    // same reason addEdge is: a join that fires with its fan-out still running
    // is a data race, not a glitch.
    assert(!n->submitted &&
           "JobGraph::parallelFor: `join` is already submitted -- pass a created but unsubmitted node");
    if (n->submitted) {
      return;
    }

    // ZERO CHILDREN IS THE SAME EVENT. Submitting with no holds added runs the
    // join immediately and releases its dependents -- which is exactly what N
    // children would eventually have done. An early return here is the bug sec
    // 4.2.2 names: a permanently unreachable dependent, once per zero-work
    // path, and this tree has about twenty of those.
    if (count == 0u) {
      submit(join);
      return;
    }

    // Take all the holds BEFORE dispatching any child. Otherwise child 0 could
    // finish and drive the counter to zero before child 1 has been counted, and
    // the join would fire with the fan-out still running.
    n->pending.fetch_add(count, std::memory_order_acq_rel);

    // Release the creation hold now that the real ones are in place. The join
    // cannot fire early because `count` holds outrank it.
    n->submitted = true;
    releaseHold(join);

    auto shared = std::make_shared<std::function<void(uint32_t)>>(std::move(body));

    for (uint32_t i = 0; i < count; ++i) {
      auto child = [this, join, shared, i]() {
        try {
          (*shared)(i);
        } catch (...) {
          m_stats.threw.fetch_add(1u, std::memory_order_relaxed);
        }
        // The child's ONLY obligation, and it is unconditional for the same
        // reason runNode's scope guard is.
        releaseHold(join);
      };

      if (m_dispatch && m_dispatch(child)) {
        m_stats.dispatched.fetch_add(1u, std::memory_order_relaxed);
      } else {
        // Inline, immediately. A child is a leaf -- it has no dependents of its
        // own -- so running it here cannot recurse, and doing so is strictly
        // better than queueing work that only waitAll would drain.
        m_stats.ranInline.fetch_add(1u, std::memory_order_relaxed);
        child();
      }
    }
  }

  void JobGraph::waitAll() {
    // DRAIN, DO NOT SPIN. Running ready nodes on the calling thread is what
    // makes progress independent of whether Dispatch ever succeeds -- which is
    // half of sec 4.2.1's gate ("run the graph with a deliberately undersized
    // queue ... every dependent must still become runnable").
    while (m_outstanding.load(std::memory_order_acquire) != 0u) {
      JobHandle h;
      if (popReady(h)) {
        m_stats.ranInline.fetch_add(1u, std::memory_order_relaxed);
        runNode(h);
        continue;
      }
      // Nothing runnable here and work still outstanding: it is on a worker.
      // Yield rather than burn the core the workers may need.
      std::this_thread::yield();
    }
  }

  void JobGraph::cancelAll() {
    // CANCELLATION IS COMPLETION. Every node signals, so every dependent still
    // becomes runnable and nothing can be left waiting on a counter that will
    // never fire. This is the exact failure sec 4.2.1 found in the pool's own
    // cancel path, where a cancelled task never calls set() and its waiters are
    // unreachable.
    //
    // Iterated by index because signalling a node releases holds that can make
    // others runnable, and those are signalled by this same loop when it
    // reaches them -- forward progress is guaranteed because signalled is a
    // one-way latch.
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_nodes.size()); ++i) {
      JobHandle h;
      h.index = i;
      h.generation = m_generations[i];
      Node* n = get(h);
      if (n == nullptr || n->signalled.load(std::memory_order_acquire)) {
        continue;
      }
      m_stats.cancelled.fetch_add(1u, std::memory_order_relaxed);
      signalComplete(h);
    }

    // Anything queued but never run is now moot -- its node has signalled.
    std::lock_guard<dxvk::mutex> lock(m_readyMutex);
    m_readyQueue.clear();
  }

  void JobGraph::reset() {
    // Only legal on an idle graph. Bumping every generation is what makes any
    // handle a caller kept from the previous frame stale rather than a pointer
    // into somebody else's node.
    assert(m_outstanding.load(std::memory_order_acquire) == 0u &&
           "JobGraph::reset on a graph with work outstanding");

    // BUMP, AND DO NOT CLEAR. m_generations outlives m_nodes on purpose: if it
    // were cleared too, the next frame's first node would be index 0
    // generation 0 -- exactly what a handle kept from the previous frame says,
    // and get() would hand that caller somebody else's node. Keeping the
    // generations is what makes a retained handle stale instead of wrong, which
    // is the entire reason the handle carries one.
    for (uint32_t& g : m_generations) {
      ++g;
    }
    m_nodes.clear();
    {
      std::lock_guard<dxvk::mutex> lock(m_readyMutex);
      m_readyQueue.clear();
    }
    m_outstanding.store(0u, std::memory_order_release);
    m_stats.reset();
  }

  bool JobGraph::selfTest() {
    bool ok = true;
    const auto check = [&ok](bool cond, const char* what) {
      if (!cond) {
        ok = false;
        Logger::err(str::format("[JobGraph] SELFTEST FAILED: ", what));
      }
    };

    // ---- 1. A diamond under a queue that refuses EVERY dispatch. -----------
    //
    // This is the undersized queue taken to its limit. If the contract is in
    // the graph, waitAll drains the whole thing on the calling thread and every
    // dependent still runs; if it is at the call site, this hangs or drops
    // nodes.
    {
      JobGraph g([](std::function<void()>) { return false; });

      std::atomic<uint32_t> order { 0u };
      uint32_t aAt = ~0u, bAt = ~0u, cAt = ~0u, dAt = ~0u;

      const JobHandle a = g.createNode("A", [&] { aAt = order.fetch_add(1u); });
      const JobHandle b = g.createNode("B", [&] { bAt = order.fetch_add(1u); });
      const JobHandle c = g.createNode("C", [&] { cAt = order.fetch_add(1u); });
      const JobHandle d = g.createNode("D", [&] { dAt = order.fetch_add(1u); });

      g.addEdge(a, b);
      g.addEdge(a, c);
      g.addEdge(b, d);
      g.addEdge(c, d);

      g.submit(a); g.submit(b); g.submit(c); g.submit(d);
      g.waitAll();

      check(aAt != ~0u && bAt != ~0u && cAt != ~0u && dAt != ~0u,
            "a node never ran with a dispatch that always refuses");
      check(aAt < bAt && aAt < cAt, "A did not precede its dependents");
      check(bAt < dAt && cAt < dAt, "D ran before its dependencies");
      check(g.stats().staleHandles == 0u, "stale handle observed in the diamond");
    }

    // ---- 2. Zero-child parallelFor still completes its dependent. ----------
    //
    // sec 4.2.2's twenty zero-work paths in one assertion. If parallelFor(0)
    // returns instead of submitting, `after` is unreachable and waitAll hangs.
    {
      JobGraph g([](std::function<void()> f) { f(); return true; });

      bool joinRan = false, afterRan = false;
      const JobHandle join  = g.createNode("join",  [&] { joinRan = true; });
      const JobHandle after = g.createNode("after", [&] { afterRan = true; });
      g.addEdge(join, after);
      g.submit(after);

      g.parallelFor(join, 0u, [](uint32_t) { });
      g.waitAll();

      check(joinRan,  "zero-child parallelFor did not run its join");
      check(afterRan, "zero-child parallelFor left its dependent unreachable");
    }

    // ---- 3. Fan-out: the consumer waits on the COUNTER, not the children. --
    {
      JobGraph g([](std::function<void()> f) { f(); return true; });

      constexpr uint32_t kChildren = 384u;   // sec 4.2.2's own example
      std::atomic<uint32_t> ran { 0u };
      uint32_t seenAtJoin = 0u;
      bool afterRan = false;

      const JobHandle join  = g.createNode("gather", [&] { seenAtJoin = ran.load(); });
      const JobHandle after = g.createNode("after",  [&] { afterRan = true; });
      g.addEdge(join, after);
      g.submit(after);

      g.parallelFor(join, kChildren, [&](uint32_t) { ran.fetch_add(1u); });
      g.waitAll();

      check(ran.load() == kChildren, "not every fan-out child ran");
      check(seenAtJoin == kChildren, "the join fired before its fan-out finished");
      check(afterRan, "the fan-out's consumer never became runnable");
    }

    // ---- 4. Cancellation is completion. -----------------------------------
    //
    // The pool's own cancel path is the failure being guarded against: a
    // cancelled task never calls set(), so a counter decremented inside set()
    // never fires and every waiter is unreachable. Here the producer is
    // cancelled and the dependent must STILL become runnable.
    {
      JobGraph g([](std::function<void()>) { return false; });

      bool producerRan = false, consumerRan = false;
      const JobHandle producer = g.createNode("producer", [&] { producerRan = true; });
      const JobHandle consumer = g.createNode("consumer", [&] { consumerRan = true; });
      g.addEdge(producer, consumer);
      g.submit(producer);
      g.submit(consumer);

      g.cancelAll();

      check(!producerRan, "cancelAll ran a node body");
      check(!consumerRan, "cancelAll ran a node body");
      check(g.stats().cancelled >= 2u, "cancelAll did not signal every node");
      // The real property: nothing is left outstanding, so a waitAll here
      // returns rather than hanging. A dependent that could still observe its
      // producer as "not done" is the deadlock this is testing for.
      g.waitAll();
    }

    // ---- 5. The generation catches a handle that outlived its node. --------
    {
      JobGraph g([](std::function<void()> f) { f(); return true; });

      const JobHandle stale = g.createNode("stale", nullptr);
      g.submit(stale);
      g.waitAll();
      g.reset();

      // Same index, previous generation. After a reset this index is free to be
      // handed out again, and following the stale handle would give the caller
      // somebody else's node.
      const JobHandle fresh = g.createNode("fresh", nullptr);
      check(fresh.index == stale.index, "selftest assumption: reset should reuse index 0");
      check(fresh.generation != stale.generation, "reset did not bump the generation");

      g.submit(fresh);
      g.waitAll();

      // Touching the stale handle must be caught, not followed.
      g.addEdge(stale, fresh);
      check(g.stats().staleHandles > 0u,
            "a handle that outlived its node was NOT caught by the generation");
    }

    if (ok) {
      Logger::info("[JobGraph] selfTest passed: undersized queue, zero-child fan-out, "
                   "cancellation-as-completion, and generation staleness all hold");
    }
    return ok;
  }

  JobGraph::Stats JobGraph::stats() const {
    Stats out;
    out.nodes        = m_stats.nodes.load(std::memory_order_relaxed);
    out.dispatched   = m_stats.dispatched.load(std::memory_order_relaxed);
    out.ranInline    = m_stats.ranInline.load(std::memory_order_relaxed);
    out.cancelled    = m_stats.cancelled.load(std::memory_order_relaxed);
    out.threw        = m_stats.threw.load(std::memory_order_relaxed);
    out.staleHandles = m_stats.staleHandles.load(std::memory_order_relaxed);
    return out;
  }

}
