/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_H_
#define TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSrun_handlerDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSrun_handlerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSrun_handlerDTh() {
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


#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace Eigen {
struct ThreadPoolDevice;
}

namespace tensorflow {

class RunHandler;

// RunHandlerPool is a fixed size pool of pre-allocated RunHandlers
// that can be used for tracking inter-op work for a given Session::Run().
// RunHandler(s) in the pool are initially 'inactive'. A RunHandler becomes
// 'active' when its unique_ptr is returned by Get() and is being used by a
// client. It becomes 'inactive' once more when its unique_ptr gets destroyed.
//
// Expected usage:
//
// * Create a single RunHandlerPool (say run_handler_pool_).
//
// * When a Session::Run() is invoked, obtain a handler by:
// auto handler = run_handler_pool_->Get();
//
// * Use handler for scheduling all inter-op work by:
// handler->ScheduleInterOpClosure(closure);
//
// This class is thread safe.
class RunHandlerPool {
 public:
  explicit RunHandlerPool(int num_inter_op_threads);

  RunHandlerPool(int num_inter_op_threads, int num_intra_op_threads);
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
      const RunOptions::Experimental::RunHandlerPoolOptions& options =
          RunOptions::Experimental::RunHandlerPoolOptions());

  // Get the priorities for active handlers. The return result is with the same
  // order of the active handler list.
  std::vector<int64_t> GetActiveHandlerPrioritiesForTesting() const;

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
  void ScheduleInterOpClosure(std::function<void()> fn);
  thread::ThreadPoolInterface* AsIntraThreadPoolInterface();

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
  typedef Thread EnvThread;
  struct TaskImpl {
    std::function<void()> f;
    Context context;
    uint64 trace_id;
  };
  Env* const env_;
  const ThreadOptions thread_options_;
  const string name_;

 public:
  struct Task {
    std::unique_ptr<TaskImpl> f;
  };

  RunHandlerEnvironment(Env* env, const ThreadOptions& thread_options,
                        const string& name);

  EnvThread* CreateThread(std::function<void()> f,
                          const std::string& thread_name);

  Task CreateTask(std::function<void()> f);

  void ExecuteTask(const Task& t);
};

typedef typename RunHandlerEnvironment::Task Task;
typedef Eigen::RunQueue<Task, 1024> Queue;

// To reduce cache misses, we use a doubly-linked list of Waiter structs and
// queue them in LIFO order rather than the FIFO order used by a single
// condition variable.
struct Waiter {
  Waiter() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrun_handlerDTh mht_0(mht_0_v, 316, "", "./tensorflow/core/framework/run_handler.h", "Waiter");

    next = this;
    prev = this;
  }
  condition_variable cv;
  mutex mu;
  Waiter* next;
  Waiter* prev;
};

class ThreadWorkSource {
 public:
  ThreadWorkSource();

  ~ThreadWorkSource();

  Task EnqueueTask(Task t, bool is_blocking);

  Task PopBlockingTask();

  Task PopNonBlockingTask(int start_index, bool search_from_all_queue);

  void WaitForWork(int max_sleep_micros);

  int TaskQueueSize(bool is_blocking);

  int64_t GetTracemeId();

  void SetTracemeId(int64_t value);

  void SetWaiter(uint64 version, Waiter* waiter, mutex* mutex);

  int64_t GetInflightTaskCount(bool is_blocking);

  void IncrementInflightTaskCount(bool is_blocking);

  void DecrementInflightTaskCount(bool is_blocking);

  unsigned NonBlockingWorkShardingFactor();

  std::string ToString();

 private:
  struct NonBlockingQueue {
    mutex queue_op_mu;
    char pad[128];
    Queue queue;
  };

  int32 non_blocking_work_sharding_factor_;
  Eigen::MaxSizeVector<NonBlockingQueue*> non_blocking_work_queues_;

  std::atomic<int64_t> blocking_inflight_;
  std::atomic<int64_t> non_blocking_inflight_;

  Queue blocking_work_queue_;
  mutex blocking_queue_op_mu_;
  char pad_[128];
  mutex waiters_mu_;
  Waiter queue_waiters_ TF_GUARDED_BY(waiters_mu_);
  std::atomic<int64_t> traceme_id_;

  mutex run_handler_waiter_mu_;
  uint64 version_ TF_GUARDED_BY(run_handler_waiter_mu_);
  mutex* sub_thread_pool_waiter_mu_ TF_GUARDED_BY(run_handler_waiter_mu_);
  Waiter* sub_thread_pool_waiter_ TF_GUARDED_BY(run_handler_waiter_mu_);
};

class RunHandlerThreadPool {
 public:
  struct PerThread {
    constexpr PerThread() : pool(nullptr), thread_id(-1) {}
    RunHandlerThreadPool* pool;  // Parent pool, or null for normal threads.
    int thread_id;               // Worker thread index in pool.
  };

  RunHandlerThreadPool(int num_blocking_threads, int num_non_blocking_threads,
                       Env* env, const ThreadOptions& thread_options,
                       const string& name,
                       Eigen::MaxSizeVector<mutex>* waiters_mu,
                       Eigen::MaxSizeVector<Waiter>* queue_waiters);

  ~RunHandlerThreadPool();

  void Start();

  void StartOneThreadForTesting();

  void AddWorkToQueue(ThreadWorkSource* tws, bool is_blocking,
                      std::function<void()> fn);

  // Set work queues from which the thread 'tid' can steal its work.
  // The request with start_request_idx will be attempted first. Other requests
  // will be attempted in FIFO order based on their arrival time.
  void SetThreadWorkSources(
      int tid, int start_request_idx, uint64 version,
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

  void WaitForWork(bool is_blocking, int thread_id,
                   int32_t max_blocking_inflight);

  void WaitForWorkInSubThreadPool(bool is_blocking, int sub_thread_pool_id);

 private:
  struct ThreadData {
    ThreadData();
    mutex mu;
    uint64 new_version;
    condition_variable sources_not_empty;
    std::unique_ptr<Thread> thread;
    int current_index;
    std::unique_ptr<Eigen::MaxSizeVector<ThreadWorkSource*>>
        new_thread_work_sources TF_GUARDED_BY(mu);

    uint64 current_version;
    // Should only be accessed by one thread.
    std::unique_ptr<Eigen::MaxSizeVector<ThreadWorkSource*>>
        current_thread_work_sources;

    int sub_thread_pool_id;
  };

  const int num_threads_;
  const int num_blocking_threads_;
  const int num_non_blocking_threads_;
  Eigen::MaxSizeVector<ThreadData> thread_data_;
  internal::RunHandlerEnvironment env_;
  std::atomic<bool> cancelled_;
  string name_;
  Eigen::MaxSizeVector<mutex>* waiters_mu_;
  Eigen::MaxSizeVector<Waiter>* queue_waiters_;

  bool use_sub_thread_pool_;
  std::vector<int> num_threads_in_sub_thread_pool_;

  // Threads in each sub thread pool will search tasks from the given
  // start_request_percentage to end_request_percentage in a round robin
  // fashion.
  std::vector<double> sub_thread_pool_start_request_percentage_;
  std::vector<double> sub_thread_pool_end_request_percentage_;
};

}  // namespace internal

}  // end namespace tensorflow.

#endif  // TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_H_
