/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <type_traits>
#include <future>
#include <assert.h>
#include "util_atomic_queue.h"
#include "util_env.h"
#include "util_math.h"
#include "util_fastops.h"
#include "util_bit.h"
#include "sync/sync_spinlock.h"

namespace dxvk {
  // NV-DXVK [TaskRing] 2026-08-31: ARCHITECTURE_OVERHAUL sec 9 item 8 asks
  // whether the task ring is ALREADY being overrun, and says the honest answer
  // is that sec 4.2.1 gives an argument rather than a measurement. This is the
  // measurement, and it is the shape that section asked for: a monotonic
  // per-slot capture counter, sampled into the Future at capture and re-checked
  // when the Future is used.
  //
  // WHAT IT CATCHES. m_tasks is a ring of m_taskCount slots indexed by
  // `m_taskId++ & (m_taskCount - 1)` (see Schedule), and a Future is a bare
  // Task* into it with no generation. If m_taskId advances a full lap between
  // capture() and get(), the slot has been re-captured under the holder --
  // capture() placement-news a new lambda and calls result.reset(), which clears
  // hasResult while a holder may be sitting in get(). Task::valid() cannot see
  // that, because there is nothing on the Task to see it with.
  //
  // WHY IT SHOULD READ ZERO TODAY. flushGeometryBatch schedules ~worker-count
  // chunks and joins them immediately, so m_taskId cannot advance a full lap in
  // between. That is a property of the ONE current call site, not of the
  // primitive -- which is exactly why sec 4.2.1 refuses to build a dependency
  // graph on it. A graph holds completion tokens across phases and removes the
  // accident.
  //
  // Silent when clean, per the [StaleTape] model: a counter and a debug assert,
  // no logging in this header (it is included by d3d11_rtx.h and rtx_types.h,
  // and a log dependency here is not worth a probe).
  inline std::atomic<uint64_t> g_taskRingReuseCount { 0 };

  const size_t kLambdaStorageCapacity = 256;
  // Note: use up to 64 bytes for state
  const size_t kResultStorageCapacity = 256 - 64;

  template<size_t Capacity = kResultStorageCapacity, bool UseWait = false>
  struct Result {
    struct Nop { };
    using OnSetCondition = std::conditional_t<UseWait, dxvk::condition_variable, Nop>;
    using ResultMutex = std::conditional_t<UseWait, dxvk::mutex, Nop>;

    template<typename T>
    void set(T&& t) {
      static_assert(sizeof(T) <= Capacity,
          "Result object storage space overrun!");

      new(storage.data()) T(std::forward<T>(t));

      set();
    }

    void set() {
      if constexpr (UseWait) {
        std::unique_lock<dxvk::mutex> lock(mtx);
        hasResult = true;
        cond.notify_one();
      } else {
        hasResult = true;
      }
    }

    void get() {
#ifdef _DEBUG
      if (isDisposed) {
        throw DxvkError("Refusing to get a disposed result!");
      }
#endif

      if constexpr (UseWait) {
        if (!hasResult) {
          std::unique_lock<dxvk::mutex> lock(mtx);
          cond.wait(lock, [this] {
            return hasResult;
          });
        }
      } else {
        while (!hasResult) {
          std::this_thread::yield();
        }
      }

      hasResult = false;
      isDisposed = true;
    }

    template<typename T>
    T get() {
      get();
      return std::move(*reinterpret_cast<T*>(storage.data()));
    }

    void reset() {
      hasResult = false;
      isDisposed = false;
    }

    void cancel() {
      hasResult = false;
      isDisposed = true;
    }

    bool disposed() const {
      return isDisposed;
    }

  private:
    std::array<uint8_t, Capacity> storage;
    std::atomic_bool hasResult = false;
    std::atomic_bool isDisposed = false;

    OnSetCondition cond;
    mutable ResultMutex mtx;
  };

  using TaskId = uint32_t;
  template<typename ResultType> struct Future;

  struct Task {
    using LambdaStorage = std::array<uint8_t, kLambdaStorageCapacity>;
    using ThunkType = void(void*);
    using ThunkStorage = std::array<uint8_t, sizeof(uintptr_t)>;

    template<typename LambdaType, typename ResultType>
    Future<ResultType> capture(LambdaType&& lambda) {
      if constexpr (sizeof(LambdaType) > sizeof(lambdaStorage)) {
        char(*__type_size)[sizeof(LambdaType)] = 1;
        static_assert(false, "Task object storage space overrun!");
      }

      // Create lambda in-place
      new (lambdaStorage.data()) LambdaType(std::forward<LambdaType>(lambda));

      // We need to use a thunk to capture the actual lambda type
      captureThunk([this]() {
        auto& lambda = *reinterpret_cast<LambdaType*>(lambdaStorage.data());

        if (!result.disposed()) {
          if constexpr (!std::is_void_v<ResultType>) {
            result.set(lambda());
          } else {
            lambda();
            result.set();
          }
        }

        lambda.~LambdaType();
      });

      result.reset();

      // Bump AFTER reset(), so the sequence a holder samples names the lambda
      // that is in the slot now rather than the one it displaced.
      const uint32_t seq = m_captureSeq.fetch_add(1u, std::memory_order_acq_rel) + 1u;

      return Future<ResultType>(*this, seq);
    }

    void operator() () {
      dispatchThunk();
    }

    template<typename ResultType>
    ResultType getResult() {
      return result.get<ResultType>();
    }

    void getResult() {
      result.get();
    }

    void cancel() {
      result.cancel();
    }

    bool valid() const {
      return !result.disposed();
    }

    // NV-DXVK [TaskRing]: which capture currently owns this slot. See
    // g_taskRingReuseCount.
    uint32_t captureSeq() const {
      return m_captureSeq.load(std::memory_order_acquire);
    }

  private:
    template<typename InvocableType>
    static inline void Thunk(void* thunkLambda) {
      (*static_cast<InvocableType*>(thunkLambda))();
    }

    template<typename TunkLambdaType>
    void captureThunk(TunkLambdaType&& thunkLambda) {
      new (thunkStorage.data()) TunkLambdaType(std::forward<TunkLambdaType>(thunkLambda));
      thunk = &Thunk<typename std::decay_t<TunkLambdaType>>;
    }

    void dispatchThunk() {
#ifdef _DEBUG
      if (!thunk) {
        throw DxvkError("Task thunk was not initialized!");
      }
#endif
      thunk(thunkStorage.data());
      thunk = nullptr;
    }

    alignas(64) LambdaStorage lambdaStorage;
    alignas(64) Result<kResultStorageCapacity> result;
    alignas(64) ThunkStorage thunkStorage;
    ThunkType* thunk = nullptr;
    // Monotonic, never reset. Wrapping at 2^32 captures of ONE slot would need
    // 2^32 * m_taskCount schedules, and a false negative there costs a missed
    // report rather than a wrong result.
    std::atomic<uint32_t> m_captureSeq { 0 };
  };

  template<typename ResultType>
  struct Future {
    Future() = default;
    explicit Future(Task& task, uint32_t captureSeq)
    : task { &task }
    , seq { captureSeq }
    { }

    ResultType get() const {
      // Checked on BOTH sides of the wait. Before catches a slot that was
      // already re-issued; after catches one re-issued while we blocked, which
      // is the case the ring makes possible and Task::valid() cannot see.
      noteIfReused();
      ResultType r = task->getResult<ResultType>();
      noteIfReused();
      task = nullptr;
      return r;
    }

    bool valid() const {
      return task != nullptr && task->valid() && !ringReused();
    }

    void cancel() const {
      task->cancel();
      task = nullptr;
    }

    // NV-DXVK [TaskRing]: true if this slot has been re-captured since this
    // Future was handed out, i.e. the Task* is now naming somebody else's work.
    // Counting rather than throwing: by the time this is observable the result
    // is already gone, so the only useful thing left is to say so exactly once
    // per occurrence and let the debug build stop on it.
    bool ringReused() const {
      return task != nullptr && task->captureSeq() != seq;
    }

  private:
    void noteIfReused() const {
      if (ringReused()) {
        g_taskRingReuseCount.fetch_add(1u, std::memory_order_relaxed);
        assert(false && "Task ring slot was re-captured while a Future still held it "
                        "-- see g_taskRingReuseCount in util_threadpool.h");
      }
    }
    mutable Task* task = nullptr;
    uint32_t seq = 0;
  };

  template<>
  struct Future<void> {
    Future() = default;
    explicit Future(Task& task, uint32_t captureSeq)
    : task { &task }
    , seq { captureSeq } { }

    void get() const {
      noteIfReused();
      task->getResult();
      noteIfReused();
      task = nullptr;
    }

    bool valid() const {
      return task != nullptr && task->valid() && !ringReused();
    }

    void cancel() const {
      task->cancel();
      task = nullptr;
    }

    // NV-DXVK [TaskRing]: true if this slot has been re-captured since this
    // Future was handed out, i.e. the Task* is now naming somebody else's work.
    // Counting rather than throwing: by the time this is observable the result
    // is already gone, so the only useful thing left is to say so exactly once
    // per occurrence and let the debug build stop on it.
    bool ringReused() const {
      return task != nullptr && task->captureSeq() != seq;
    }

  private:
    void noteIfReused() const {
      if (ringReused()) {
        g_taskRingReuseCount.fetch_add(1u, std::memory_order_relaxed);
        assert(false && "Task ring slot was re-captured while a Future still held it "
                        "-- see g_taskRingReuseCount in util_threadpool.h");
      }
    }
    mutable Task* task = nullptr;
    uint32_t seq = 0;
  };

  /**
    * \brief Implements a async task scheduler, optimized
    *        for tasks of varying execution time using a
    *        work stealing algorithm.
    *
    *  NumThreads: How many threads to spawn (up to 255)
    *  NumTasksPerThread: Size of the task queue ring buffer
    *  WorkStealing: Enables the work stealing features of the scheduler
    *  LowLatency: Enables the low-latency mode where workers will spin instead of
    *              waiting for tasks on a conditional variable
    *  (ctor)workerName: Name given to threads with the pattern: workerName(N)
    * 
    *  Example usage:
    *   // Creates 1 thread, and uses it to return PI via a future
    *   WorkerThreadPool threadPool(1, "thread-pool-name");
    *   Future<float> result = threadPool.Schedule([]{ return 3.14159265359f; });
    *   float pi = result.get();
    */
  template<size_t NumTasksPerThread, bool WorkStealing = true, bool LowLatency = true>
  class WorkerThreadPool {
    using Queue = AtomicQueue<TaskId, NumTasksPerThread>;
    using QueuePtr = std::unique_ptr<Queue>;

    struct Nop { };
    using OnAddCondition = std::conditional_t<LowLatency, Nop, dxvk::condition_variable>;
    using TaskMutex = std::conditional_t<LowLatency, Nop, dxvk::mutex>;

  public:
    WorkerThreadPool(uint8_t numThreads, const char* workerName = "Nameless Worker Thread") 
    : m_numThread(std::clamp(numThreads, (uint8_t)1u, (uint8_t)dxvk::thread::hardware_concurrency())) {
      // Note: round up to a closest power-of-two so we can use mask as modulo
      m_taskCount = 1 << (32 - bit::lzcnt(static_cast<uint32_t>(NumTasksPerThread * m_numThread) - 1));
      m_tasks.reset(new Task[m_taskCount]);
      m_workerTasks.resize(m_numThread);
      m_workerThreads.resize(m_numThread);
      // Create the work queues first!  We need to create
      // then all since work stealing may access the other
      // queues.
      for (int i = 0; i < m_numThread; i++) {
        m_workerTasks[i] = std::make_unique<Queue>();
      }
      // Must exist before any worker starts: processWork() -> executeTask()
      // locks m_queueMutex[i] on its very first iteration.
      m_queueMutex.reset(new PaddedSpinlock[m_numThread]);

      // Start the worker threads
      for (int i = 0; i < m_numThread; i++) {
        m_workerThreads[i] = std::thread([this, i, workerName] {
          env::setThreadName(str::format(workerName, "(", i, ")"));
          processWork(i);
        });
      }
    }

    ~WorkerThreadPool() {
      // Stop all the worker threads
      m_stopWork = true;

      if constexpr (!LowLatency) {
        std::unique_lock<TaskMutex> lock(m_taskMutex);
        m_condOnAdd.notify_all();
      }

      for (auto& worker : m_workerThreads) {
        worker.join();
      }

      if (m_numTasks > 0) {
        for (auto& workerTasks : m_workerTasks) {
          TaskId taskId;
          while (workerTasks->pop(taskId)) {
            // Cancel the actual task job
            m_tasks[taskId].cancel();
            // Execute the task to dispatch the destructor
            m_tasks[taskId]();
            --m_numTasks;
          }
        }
      }

      assert(m_numTasks == 0 && "Tasks left in thread pool queue after destruction!");
    }

    // Schedule a task to be executed by the thread pool
    template <uint8_t Affinity = 0xFF, typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
    Future<R> Schedule(F&& f) {
      // Is the affinity mask valid?
      const uint8_t affinityMask = std::min(popcnt_uint8(Affinity), m_numThread);

      // Schedule work on the appropriate thread
      const uint32_t thread = fast::findNthBit(Affinity, (uint8_t) (m_schedulerIndex++ % affinityMask));
      assert(thread < m_numThread);

      // Atomic queue is SPSC, so we don't need to take a lock here
      // since we know this will always be called from a single thread.

      Future<R> future;
      if (!m_workerTasks[thread]->isFull()) {
        // Get next task id
        TaskId taskId = m_taskId++ & (m_taskCount - 1);

        // Capture task lambda
        future = m_tasks[taskId].capture<F, R>(std::forward<F>(f));

        // Place task into queue
        m_workerTasks[thread]->push(std::move(taskId));

        // NV-DXVK: increment BEFORE the notify. The worker's CV predicate is
        // (m_numTasks > 0); if the increment happened after the notify (as it did
        // originally), a worker woken by notify_one could re-check the predicate,
        // still see 0, and go back to sleep before the increment landed — a lost
        // wakeup that stalls the task until the next Schedule notifies. The task is
        // already pushed above, so a worker that observes this increment will find
        // it. Harmless for the LowLatency (spin) path (no CV, no notify).
        ++m_numTasks;

        if constexpr (!LowLatency) {
          // NV-DXVK [perf]: notify with the mutex RELEASED.
          //
          // This used to notify while still holding m_taskMutex, which is the
          // classic hurry-up-and-wait: the woken worker leaves the condvar and
          // immediately blocks again on the very mutex its notifier is still
          // holding, so every wakeup costs an extra sleep/wake round trip. With
          // a burst of per-draw Schedule() calls and 30 workers on one mutex,
          // that round trip is what showed up as Schedule() sitting in the
          // kernel while the workers slept.
          //
          // The empty acquire/release below is load-bearing, not leftover - it
          // is what makes the wakeup safe to move out. A worker checks the
          // predicate while holding the mutex and wait() then releases it
          // atomically, so the producer cannot slip between those two steps
          // without also acquiring. Touching the mutex once after ++m_numTasks
          // therefore orders us either fully before the check (worker sees the
          // task and never sleeps) or fully after the block (the notify lands).
          // Removing this scope reintroduces the lost wakeup the comment above
          // describes.
          { std::unique_lock<TaskMutex> lock(m_taskMutex); }

          if constexpr (WorkStealing) {
            // Notify only one worker when workers can steal from the others
            m_condOnAdd.notify_one();
          } else {
            // Notify all workers when they cannot steal
            m_condOnAdd.notify_all();
          }
        }
      }

      return future;
    }

    // NV-DXVK [BatchSubmitDraw]: number of worker threads actually spawned (after
    // the clamp in the ctor). The frame-end batch parallel-for chunks the arena
    // into this many contiguous ranges so scheduling is O(threads), not O(draws).
    uint8_t numThreads() const { return m_numThread; }

  private:
    void processWork(const uint32_t workerId) {
      while (true) {
        // Using a conditional wait in high-latency mode
        if constexpr (!LowLatency) {
          std::unique_lock<TaskMutex> lock(m_taskMutex);
          m_condOnAdd.wait(lock, [this] {
            return m_numTasks > 0 || m_stopWork.load();
          });
        }

        // Master halt
        if (m_stopWork) {
          return;
        }

        // Try executing a task from our queue
        if (executeTask(workerId))
          continue;

        if (WorkStealing) {
          // There's no work to do!
          // Steal work from other queues
          bool workStolen = false;
          for (uint32_t i = 1; i < m_numThread; i++) {
            const uint32_t victim = (workerId + i) % m_numThread;
            // NV-DXVK [perf]: unlocked hint first. Without it a thief takes
            // (and contends) a lock per victim just to discover the queue is
            // empty - m_numThread-1 lock acquisitions per pass, on every pass,
            // by every idle worker. The check can be stale in both directions
            // and both are harmless; see AtomicQueue::isEmpty.
            if (m_workerTasks[victim]->isEmpty()) {
              continue;
            }
            if (executeTask(victim)) {
              workStolen = true;
              break;
            }
          }

          // NV-DXVK [perf]: yield in BOTH modes, not just LowLatency.
          //
          // The condvar predicate is (m_numTasks > 0), so while ANY queue holds
          // work the wait at the top of this loop returns immediately. A worker
          // that cannot get at that work - because it lives in another worker's
          // queue and lost the race for it - therefore did not sleep here: it
          // span the entire loop at full speed, re-taking m_taskMutex every
          // iteration and re-scanning every victim. That is the failure mode
          // that made adding threads make this pool SLOWER, since each extra
          // worker adds another full-speed spinner competing for the same lock
          // with the workers actually making progress.
          //
          // Yielding costs a scheduler round trip on a worker that by
          // definition has nothing to do, and hands its slice to the ones that
          // do. The LowLatency (spin) path keeps the same behaviour it had.
          if (!workStolen) {
            std::this_thread::yield();
          }
        }
      }
    }

    // True if front pop, False if back pop
    bool executeTask(const uint32_t workerId) {
      TaskId taskId;
      {
        // Since we're using an SPSC queue, we must take a lock when
        // popping, since we may be stealing (or be stolen from) by
        // another thread. NV-DXVK: that lock is now per-QUEUE (see
        // m_queueMutex) rather than one for the whole pool, so two workers
        // draining two different queues no longer serialise against each
        // other. Only contenders for THIS queue are excluded, which is the
        // entire correctness requirement.
        std::unique_lock<sync::Spinlock> lock(m_queueMutex[workerId].lock);

        if (!m_workerTasks[workerId]->pop(taskId)) {
          return false;
        }

        --m_numTasks;
      }

      // Execute the task
      m_tasks[taskId]();

      return true;
    }

    std::unique_ptr<Task[]> m_tasks;
    std::atomic<TaskId> m_taskId = 0;
    uint32_t m_taskCount;

    // Add the task to the queue and notify a worker thread
    //  just distribute evenly to all threads for some mask denoted by Affinity.
    size_t m_schedulerIndex = 0;

    uint8_t m_numThread;

    std::atomic<bool> m_stopWork = false;

    // Used conditionally to wait for tasks in high-latency mode
    TaskMutex m_taskMutex;
    OnAddCondition m_condOnAdd;

    // NV-DXVK [perf]: ONE LOCK PER QUEUE, not one for the whole pool.
    //
    // This was a single pool-wide `sync::Spinlock m_threadMutex`, taken by
    // executeTask() on EVERY pop. Because processWork()'s work-stealing loop
    // calls executeTask() once per victim, a worker that wakes to an empty
    // queue takes that one global lock up to m_numThread-1 times before it
    // finds anything - and every other worker is doing the same thing on the
    // same lock at the same time. Cost therefore grows as O(threads^2) in lock
    // traffic while the useful work stays constant, which is why ADDING
    // threads made this pool slower rather than faster.
    //
    // Measured (nsys 2026-07-28, 32-core box, auto = 30 workers, 208 items per
    // frame): the geometry pool sat inside a ~25 ms/frame window in which the
    // GPU had nothing to run, with Schedule() blocked in the kernel and the
    // workers asleep in processWork()'s condvar wait.
    //
    // A per-queue lock is the same mutual-exclusion property at the right
    // granularity: AtomicQueue is SPSC, Schedule() is the single producer, and
    // the only race is between the owning worker and any thief popping THAT
    // queue. Serialising thieves of different queues against each other was
    // never required for correctness.
    //
    // alignas(64) matters as much as the split does - a plain array of
    // spinlocks would put ~8 of them per cache line, so 30 workers polling 30
    // "different" locks would still ping-pong the same lines and reintroduce
    // the contention this removes.
    struct alignas(64) PaddedSpinlock { sync::Spinlock lock; };
    std::unique_ptr<PaddedSpinlock[]> m_queueMutex;

    std::vector<std::thread> m_workerThreads;

    // We expect high volume of potentially small tasks via "Schedule" per-
    //  frame, and require extremely low overhead to hit the 100's of FPS.
    // Use a lock-free circular queue here for two reasons (profiled):
    //  1. Non-circular queue incurs allocation overhead thats unacceptable
    //  2. Use of mutex, and CVs, incur overhead thats unacceptable
    std::vector<QueuePtr> m_workerTasks;
    std::atomic_uint32_t m_numTasks;
  };

}
