/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_TFRT_EXPERIMENTAL_RUN_HANDLER_THREAD_POOLL_RUN_HANDLER_H_
#define TENSORFLOW_CORE_TFRT_EXPERIMENTAL_RUN_HANDLER_THREAD_POOLL_RUN_HANDLER_H_
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handlerDTh {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handlerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handlerDTh() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};


#include <cstddef>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/task_function.h"  // from @tf_runtime
namespace Eigen {
struct ThreadPoolDevice;
}

namespace tfrt {
namespace tf {

class RunHandler;

// Options for RunHanler.
struct RunHandlerOptions {
  RunHandlerOptions() : priority(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handlerDTh mht_0(mht_0_v, 209, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h", "RunHandlerOptions");
}

  // Request priority.
  int priority;
};

// RunHandlerPool is a fixed size pool of pre-allocated RunHandlers
// that can be used for tracking op work for a given inference request.
// RunHandler(s) in the pool are initially 'inactive'. A RunHandler becomes
// 'active' when its unique_ptr is returned by Get() and is being used by a
// client. It becomes 'inactive' once more when its unique_ptr gets destroyed.
//
// Expected usage:
//
// * Create a single RunHandlerPool (say run_handler_pool_).
//
// * When an inference request is invoked, obtain a handler by:
// auto handler = run_handler_pool_->Get();
//
// * Use handler for scheduling all inter-op work by:
// handler->ScheduleInterOpClosure(closure);
//
// This class is thread safe.
class RunHandlerPool {
 public:
  struct Options {
    // The number of main threads.
    int num_inter_op_threads = 1;

    // The number of complimentary threads.
    int num_intra_op_threads = 1;

    // The number of max concurrent handlers.
    int max_concurrent_handler = 128;

    // The number of sub thread pool configed.
    int num_sub_thread_pool = 1;

    // The number of threads in each sub thread pool. The length of the vector
    // should equal to num_sub_thread_pool.
    std::vector<int> num_threads_in_sub_thread_pool = {1};

    // The percentage of requests the first N sub thread pool handles. The
    // length of the vector should equal to num_sub_thread_pool. For example,
    // {0.5, 1} means the first sub thread pool will handle the first 50%
    // requests based on priority and the second thread pool will handle the
    // second 50% requests based on priority.
    std::vector<double> sub_thread_request_percentage = {1.0};

    // Sleep time for non blocking threads if there is no pending task.
    int non_blocking_threads_sleep_time_micro_sec = 1000;

    // Max sleep time for blocking threads if there is no pending task and no
    // new task wakes up the thread.
    int blocking_threads_max_sleep_time_micro_sec = 1000;

    // If true, use adaptive waiting time.
    bool use_adaptive_waiting_time = true;

    // If true, threads won't wake itself up if there is no active requests.
    bool wait_if_no_active_request = true;

    // If true, threads will be waken up by new tasks.
    bool enable_wake_up = true;
  };
  explicit RunHandlerPool(Options options);
  ~RunHandlerPool();

  // Returns an inactive RunHandler from the pool.
  //
  // RunHandlers in RunHandlerPool are initially 'inactive'.
  // A RunHandler becomes 'active' when its unique_ptr its returned by Get()
  // and is being used by a client.  It becomes 'inactive' once more when the
  // unique_ptr is destroyed.
  //
  // Will block unless there is an inactive handler.
  std::unique_ptr<RunHandler> Get(
      int64_t step_id = 0, int64_t timeout_in_ms = 0,
      const RunHandlerOptions& options = RunHandlerOptions());

  // Get the priorities for active handlers. The return result is with the same
  // order of the active handler list.
  std::vector<int64_t> GetActiveHandlerPrioritiesForTesting() const;

  // Block until the system is quiescent (no pending work and no inflight work).
  void Quiesce() const;

 private:
  class Impl;
  friend class RunHandler;

  std::unique_ptr<Impl> impl_;
};

// RunHandler can be used to schedule inter/intra-op closures to run on a global
// pool shared across all Session::Run(s). The closures are enqueued to a
// handler specific queue, from which the work is stolen in a priority order
// (time of the Get() call).
//
// It can only be created via RunHandlerPool::Get().
//
// This class can be used instead of directly scheduling closures on a global
// pool since it maintains a global view across all sessions and optimizes pool
// scheduling to improve (median and tail) latency.
//
// This class is thread safe.
class RunHandler {
 public:
  void ScheduleInterOpClosure(TaskFunction fn);
  void ScheduleIntraOpClosure(TaskFunction fn);

  tensorflow::thread::ThreadPoolInterface* AsIntraThreadPoolInterface() const;

  int NumThreads() const;

  int64_t step_id() const;

  ~RunHandler();

 private:
  class Impl;
  friend class RunHandlerPool::Impl;

  explicit RunHandler(Impl* impl);

  Impl* impl_;  // NOT OWNED.
};

namespace internal {

// TODO(azaks): Refactor with thread:ThreadPool
class RunHandlerEnvironment {
 public:
  typedef tensorflow::Thread EnvThread;
  struct TaskImpl {
    TaskFunction f;
    tensorflow::Context context;
    uint64_t trace_id;
  };
  tensorflow::Env* const env_;
  const tensorflow::ThreadOptions thread_options_;
  const std::string name_;

 public:
  struct Task {
    std::unique_ptr<TaskImpl> f;
  };

  RunHandlerEnvironment(tensorflow::Env* env,
                        const tensorflow::ThreadOptions& thread_options,
                        const std::string& name);

  EnvThread* CreateThread(std::function<void()> f);

  Task CreateTask(TaskFunction f);

  void ExecuteTask(const Task& t);
};

typedef typename RunHandlerEnvironment::Task Task;
typedef Eigen::RunQueue<Task, 1024> Queue;

// To reduce cache misses, we use a doubly-linked list of Waiter structs and
// queue them in LIFO order rather than the FIFO order used by a single
// condition variable.
struct Waiter {
  Waiter() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handlerDTh mht_1(mht_1_v, 378, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h", "Waiter");

    next = this;
    prev = this;
  }
  tensorflow::condition_variable cv;
  int num_waiting_threads = 0;
  tensorflow::mutex mu;
  Waiter* next;
  Waiter* prev;
};

class ThreadWorkSource {
 public:
  ThreadWorkSource();

  ~ThreadWorkSource();

  Task EnqueueTask(Task t, bool is_blocking, bool enable_wake_up);

  Task PopBlockingTask();

  Task PopNonBlockingTask(int start_index, bool search_from_all_queue);

  int TaskQueueSize(bool is_blocking);

  int64_t GetTracemeId();

  void SetTracemeId(int64_t value);

  void SetWaiter(uint64_t version, Waiter* waiter, tensorflow::mutex* mutex);

  int64_t GetInflightTaskCount(bool is_blocking);

  void IncrementInflightTaskCount(bool is_blocking);

  void DecrementInflightTaskCount(bool is_blocking);

  int64_t GetPendingTaskCount();

  void IncrementPendingTaskCount();

  void DecrementPendingTaskCount();

  unsigned NonBlockingWorkShardingFactor();

  std::string ToString();

 private:
  struct NonBlockingQueue {
    tensorflow::mutex queue_op_mu;
    char pad[128];
    Queue queue;
  };

  int32_t non_blocking_work_sharding_factor_;
  Eigen::MaxSizeVector<NonBlockingQueue*> non_blocking_work_queues_;

  // The number of tasks that are executing now.
  std::atomic<int64_t> blocking_inflight_;
  std::atomic<int64_t> non_blocking_inflight_;

  // The number of tasks that are enqueued and not finished.
  std::atomic<int64_t> pending_tasks_;

  Queue blocking_work_queue_;
  tensorflow::mutex blocking_queue_op_mu_;
  char pad_[128];
  tensorflow::mutex waiters_mu_;
  Waiter queue_waiters_ TF_GUARDED_BY(waiters_mu_);
  std::atomic<int64_t> traceme_id_;

  tensorflow::mutex run_handler_waiter_mu_;
  uint64_t version_ TF_GUARDED_BY(run_handler_waiter_mu_);
  tensorflow::mutex* sub_thread_pool_waiter_mu_
      TF_GUARDED_BY(run_handler_waiter_mu_);
  Waiter* sub_thread_pool_waiter_ TF_GUARDED_BY(run_handler_waiter_mu_);
};

class RunHandlerThreadPool {
 public:
  struct Options {
    int num_blocking_threads;
    int num_non_blocking_threads;
    bool wait_if_no_active_request;
    int non_blocking_threads_sleep_time_micro_sec;
    int blocking_threads_max_sleep_time_micro_sec;
    bool use_adaptive_waiting_time;
    bool enable_wake_up;
    int max_concurrent_handler;
    std::vector<int> num_threads_in_sub_thread_pool;
    std::vector<double> sub_thread_request_percentage;
    Options(int num_blocking_threads, int num_non_blocking_threads,
            bool wait_if_no_active_request,
            int non_blocking_threads_sleep_time_micro_sec,
            int blocking_threads_max_sleep_time_micro_sec,
            bool use_adaptive_waiting_time, bool enable_wake_up,
            int max_concurrent_handler,
            const std::vector<int>& num_threads_in_sub_thread_pool,
            const std::vector<double>& sub_thread_request_percentage)
        : num_blocking_threads(num_blocking_threads),
          num_non_blocking_threads(num_non_blocking_threads),
          wait_if_no_active_request(wait_if_no_active_request),
          non_blocking_threads_sleep_time_micro_sec(
              non_blocking_threads_sleep_time_micro_sec),
          blocking_threads_max_sleep_time_micro_sec(
              blocking_threads_max_sleep_time_micro_sec),
          use_adaptive_waiting_time(use_adaptive_waiting_time),
          enable_wake_up(enable_wake_up),
          max_concurrent_handler(max_concurrent_handler),
          num_threads_in_sub_thread_pool(num_threads_in_sub_thread_pool),
          sub_thread_request_percentage(sub_thread_request_percentage) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handlerDTh mht_2(mht_2_v, 491, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h", "Options");
}
  };
  struct PerThread {
    constexpr PerThread() : pool(nullptr), thread_id(-1) {}
    RunHandlerThreadPool* pool;  // Parent pool, or null for normal threads.
    int thread_id;               // Worker thread index in pool.
  };

  RunHandlerThreadPool(Options options, tensorflow::Env* env,
                       const tensorflow::ThreadOptions& thread_options,
                       const std::string& name,
                       Eigen::MaxSizeVector<tensorflow::mutex>* waiters_mu,
                       Eigen::MaxSizeVector<Waiter>* queue_waiters);

  ~RunHandlerThreadPool();

  void Start();

  void StartOneThreadForTesting();

  void AddWorkToQueue(ThreadWorkSource* tws, bool is_blocking, TaskFunction fn);

  // Set work queues from which the thread 'tid' can steal its work.
  void SetThreadWorkSources(
      int tid, uint64_t version,
      const Eigen::MaxSizeVector<ThreadWorkSource*>& thread_work_sources);

  PerThread* GetPerThread();

  int CurrentThreadId() const;

  int NumThreads() const;

  int NumBlockingThreads() const;

  int NumNonBlockingThreads() const;

  void WorkerLoop(int thread_id, bool may_steal_blocking_work);

  // Search tasks from Requets range searching_range_start to
  // searching_range_end. If there is no tasks in the search range and
  // may_steal_blocking_work is true, then search from all requests.
  Task FindTask(
      int searching_range_start, int searching_range_end, int thread_id,
      int sub_thread_pool_id, int max_blocking_inflight,
      bool may_steal_blocking_work,
      const Eigen::MaxSizeVector<ThreadWorkSource*>& thread_work_sources,
      bool* task_from_blocking_queue, ThreadWorkSource** tws);

  void WaitForWorkInSubThreadPool(int thread_id, bool is_blocking,
                                  int sub_thread_pool_id);

 private:
  struct ThreadData {
    ThreadData();
    tensorflow::mutex mu;
    uint64_t new_version;
    tensorflow::condition_variable sources_not_empty;
    std::unique_ptr<tensorflow::Thread> thread;
    int current_index;
    std::unique_ptr<Eigen::MaxSizeVector<ThreadWorkSource*>>
        new_thread_work_sources TF_GUARDED_BY(mu);

    uint64_t current_version;
    // Should only be accessed by one thread.
    std::unique_ptr<Eigen::MaxSizeVector<ThreadWorkSource*>>
        current_thread_work_sources;

    int sub_thread_pool_id;
  };

  const int num_threads_;
  const int num_blocking_threads_;
  const int num_non_blocking_threads_;
  const bool adaptive_sleep_time_;
  const bool wait_if_no_active_request_;
  const int non_blocking_thread_sleep_time_;
  const int blocking_thread_max_waiting_time_;
  const bool enable_wake_up_;
  Eigen::MaxSizeVector<ThreadData> thread_data_;
  internal::RunHandlerEnvironment env_;
  std::atomic<bool> cancelled_;
  std::string name_;
  Eigen::MaxSizeVector<tensorflow::mutex>* waiters_mu_;
  Eigen::MaxSizeVector<Waiter>* queue_waiters_;

  std::vector<int> num_threads_in_sub_thread_pool_;

  // Threads in each sub thread pool will search tasks from
  // the end_request_percentage of previous sub thread pool to its own
  // end_request_percentage in a round robin fashion.
  std::vector<double> sub_thread_pool_end_request_percentage_;
};

}  // namespace internal

class RunHandlerWorkQueue : public tensorflow::tfrt_stub::WorkQueueInterface {
 public:
  explicit RunHandlerWorkQueue(std::unique_ptr<RunHandler> run_handler)
      : run_handler_(std::move(run_handler)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handlerDTh mht_3(mht_3_v, 593, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h", "RunHandlerWorkQueue");

    DCHECK(run_handler_);
  }
  ~RunHandlerWorkQueue() override = default;

  std::string name() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handlerDTh mht_4(mht_4_v, 601, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h", "name");
 return "run_handler"; }

  int GetParallelismLevel() const override;

  void AddTask(TaskFunction work) override;

  Optional<TaskFunction> AddBlockingTask(TaskFunction work,
                                         bool allow_queuing) override;

  void Await(
      llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> values) override;

  bool IsInWorkerThread() const override;

  void Quiesce() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handlerDTh mht_5(mht_5_v, 618, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h", "Quiesce");

    LOG(FATAL) << "RunHandlerWorkQueue::Quiesce() is not "  // Crash OK
                  "implemented, and supposed to be removed.";
  }

 private:
  std::unique_ptr<RunHandler> run_handler_;
};

}  // end namespace tf
}  // end namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EXPERIMENTAL_RUN_HANDLER_THREAD_POOLL_RUN_HANDLER_H_
