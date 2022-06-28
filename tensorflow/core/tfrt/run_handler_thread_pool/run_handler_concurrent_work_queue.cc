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
class MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc() {
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
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.h"

#include <memory>

#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

RunHandlerThreadWorkQueue::RunHandlerThreadWorkQueue(const Options& options)
    : options_(options),
      quiescing_state_(std::make_unique<::tfrt::internal::QuiescingState>()),
      non_blocking_work_queue_(quiescing_state_.get(),
                               /*num_threads=*/1),
      blocking_work_queue_(quiescing_state_.get(),
                           /*num_threads=*/1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.cc", "RunHandlerThreadWorkQueue::RunHandlerThreadWorkQueue");

  CHECK(options.num_threads_in_sub_thread_pool.size() ==  // Crash OK.
        options.num_sub_thread_pool);
  CHECK(options.sub_thread_request_percentage.size() ==  // Crash OK.
        options.num_sub_thread_pool);

  RunHandlerPool::Options pool_options;
  pool_options.num_inter_op_threads = options.num_main_threads;
  pool_options.num_intra_op_threads = options.num_complementary_threads;
  pool_options.max_concurrent_handler = options.max_concurrent_handler;
  pool_options.blocking_threads_max_sleep_time_micro_sec =
      options.blocking_threads_max_sleep_time_micro_sec;
  pool_options.non_blocking_threads_sleep_time_micro_sec =
      options.non_blocking_threads_sleep_time_micro_sec;
  pool_options.num_sub_thread_pool = options.num_sub_thread_pool;
  pool_options.num_threads_in_sub_thread_pool =
      options.num_threads_in_sub_thread_pool;
  pool_options.sub_thread_request_percentage =
      options.sub_thread_request_percentage;
  pool_options.enable_wake_up = options.enable_wake_up;
  pool_options.wait_if_no_active_request = options.wait_if_no_active_request;
  pool_options.use_adaptive_waiting_time = options.use_adaptive_waiting_time;
  handler_pool_ = absl::make_unique<RunHandlerPool>(pool_options);
}

tensorflow::StatusOr<std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface>>
RunHandlerThreadWorkQueue::InitializeRequest(
    tfrt::RequestContextBuilder* request_context_builder,
    tensorflow::thread::ThreadPoolInterface** intra_op_threadpool) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.cc", "RunHandlerThreadWorkQueue::InitializeRequest");

  DCHECK(intra_op_threadpool);
  RunHandlerOptions options;
  options.priority = request_context_builder->request_options().priority;
  std::unique_ptr<RunHandler> handler = handler_pool_->Get(
      request_context_builder->id(), options_.init_timeout_ms, options);
  if (!handler) {
    return tensorflow::errors::Internal(absl::StrCat(
        "Could not obtain RunHandler for request after waiting for ",
        options_.init_timeout_ms, " ms."));
  }

  *intra_op_threadpool = handler->AsIntraThreadPoolInterface();

  return {std::make_unique<RunHandlerWorkQueue>(std::move(handler))};
}

void RunHandlerThreadWorkQueue::AddTask(TaskFunction work) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc mht_2(mht_2_v, 254, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.cc", "RunHandlerThreadWorkQueue::AddTask");

  non_blocking_work_queue_.AddTask(std::move(work));
}

Optional<TaskFunction> RunHandlerThreadWorkQueue::AddBlockingTask(
    TaskFunction work, bool allow_queuing) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc mht_3(mht_3_v, 262, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.cc", "RunHandlerThreadWorkQueue::AddBlockingTask");

  if (allow_queuing) {
    return blocking_work_queue_.EnqueueBlockingTask(std::move(work));
  } else {
    return blocking_work_queue_.RunBlockingTask(std::move(work));
  }
  return llvm::None;
}

void RunHandlerThreadWorkQueue::Quiesce() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.cc", "RunHandlerThreadWorkQueue::Quiesce");

  handler_pool_->Quiesce();
  non_blocking_work_queue_.Quiesce();
  blocking_work_queue_.Quiesce();
}

void RunHandlerThreadWorkQueue::Await(
    ArrayRef<RCReference<AsyncValue>> values) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc mht_5(mht_5_v, 284, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.cc", "RunHandlerThreadWorkQueue::Await");

  tfrt::Await(values);
}

bool RunHandlerThreadWorkQueue::IsInWorkerThread() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queueDTcc mht_6(mht_6_v, 291, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.cc", "RunHandlerThreadWorkQueue::IsInWorkerThread");

  // TODO(b/192247530): Check if we have cases it is not true.
  return true;
}

}  // namespace tf
}  // namespace tfrt
