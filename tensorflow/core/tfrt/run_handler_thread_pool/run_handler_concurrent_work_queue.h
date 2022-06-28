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
#ifndef TENSORFLOW_CORE_TFRT_EXPERIMENTAL_RUN_HANDLER_THREAD_POOL_RUN_HANDLER_CONCURRENT_WORK_QUEUE_H_
#define TENSORFLOW_CORE_TFRT_EXPERIMENTAL_RUN_HANDLER_THREAD_POOL_RUN_HANDLER_CONCURRENT_WORK_QUEUE_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTh() {
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


#include <memory>

#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/thread_environment.h"  // from @tf_runtime
#include "third_party/concurrent_work_queue/lib/blocking_work_queue.h"
#include "third_party/concurrent_work_queue/lib/non_blocking_work_queue.h"

namespace tfrt {
namespace tf {

// Concurrent Work Queue based on Run Handler thread Pool. All tasks are queued
// based on requests.
class RunHandlerThreadWorkQueue
    : public tensorflow::tfrt_stub::WorkQueueInterface {
 public:
  struct Options {
    // The number of threads used for the main thread pool.
    int num_main_threads;

    // The number of threads used for complementary thread pool.
    int num_complementary_threads;

    // Timeout for InitRequest().
    // The timeout may trigger as the work queue limits the number of concurrent
    // in-flight requests for better latency.
    int64_t init_timeout_ms;

    // The number of max concurrent handlers.
    int max_concurrent_handler = 128;

    // The number of sub thread pool configed.
    int num_sub_thread_pool = 1;

    // The number of threads in each sub thread pool. The length of the vector
    // should equal to num_sub_thread_pool.
    std::vector<int> num_threads_in_sub_thread_pool = {1};

    // The percentage of requests the first N sub thread pool handles. The
    // length of the vector should equal to num_sub_thread_pool.
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

  explicit RunHandlerThreadWorkQueue(const Options& options);
  ~RunHandlerThreadWorkQueue() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTh mht_0(mht_0_v, 249, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.h", "~RunHandlerThreadWorkQueue");
}

  std::string name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTh mht_1(mht_1_v, 254, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.h", "name");

    return tensorflow::strings::StrCat(
        "RunHandlerThreadWorkQueue C++ work queue (", options_.num_main_threads,
        " main threads, ", options_.num_complementary_threads,
        " complementary threads)");
  }

  tensorflow::StatusOr<
      std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface>>
  InitializeRequest(tfrt::RequestContextBuilder* request_context_builder,
                    tensorflow::thread::ThreadPoolInterface**
                        intra_op_threadpool) const override;

  int GetParallelismLevel() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTh mht_2(mht_2_v, 270, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.h", "GetParallelismLevel");

    return options_.num_main_threads + options_.num_complementary_threads;
  }

  void AddTask(TaskFunction work) override;

  Optional<TaskFunction> AddBlockingTask(TaskFunction work,
                                         bool allow_queuing) override;

  void Quiesce() override;

  void Await(ArrayRef<RCReference<AsyncValue>> values) override;

  bool IsInWorkerThread() const override;

 private:
  Options options_;

  // Handler Pool.
  // Each request will require a handler from the pool, and release the handler
  // back to the pool once it is done.
  std::unique_ptr<RunHandlerPool> handler_pool_;

  // An id assigned to each request for tracing purpose.
  static std::atomic_int_fast64_t step_id_counter_;

  // QuiescingState for non_blocking_work_queue_ and blocking_work_queue_.
  std::unique_ptr<::tfrt::internal::QuiescingState> quiescing_state_;

  // Nonblocking queue used for cases without execution context.
  ::tfrt::internal::NonBlockingWorkQueue<ThreadingEnvironment>
      non_blocking_work_queue_;

  // Blocking queue used for cases without execution context.
  ::tfrt::internal::BlockingWorkQueue<ThreadingEnvironment>
      blocking_work_queue_;
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EXPERIMENTAL_RUN_HANDLER_THREAD_POOL_RUN_HANDLER_CONCURRENT_WORK_QUEUE_H_
