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
class MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc() {
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
#include "tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.h"

#include <utility>

#include "llvm/ADT/None.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/task_function.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/latch.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

using ::tensorflow::thread::ThreadPoolInterface;

StatusOr<std::unique_ptr<WorkQueueInterface>>
TfThreadPoolWorkQueue::InitializeRequest(
    ::tfrt::RequestContextBuilder* request_context_builder,
    ThreadPoolInterface** intra_op_threadpool) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.cc", "TfThreadPoolWorkQueue::InitializeRequest");

  DCHECK(intra_op_threadpool);
  *intra_op_threadpool = intra_op_threadpool_;

  return {nullptr};
}

void TfThreadPoolWorkQueue::AddTask(tfrt::TaskFunction work) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.cc", "TfThreadPoolWorkQueue::AddTask");

  auto* copy = new tfrt::TaskFunction(std::move(work));
  inter_op_threadpool_->Schedule([copy] {
    (*copy)();
    delete copy;
  });
}

void TfThreadPoolWorkQueue::AddTask(const tfrt::ExecutionContext& exec_ctx,
                                    tfrt::TaskFunction work) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.cc", "TfThreadPoolWorkQueue::AddTask");

  int64_t id = 0;
  if (auto* request_context = exec_ctx.request_ctx()) {
    id = request_context->id();
  }
  AddTask(tensorflow::tfrt_stub::WrapWork(id, "inter", std::move(work)));
}

llvm::Optional<tfrt::TaskFunction> TfThreadPoolWorkQueue::AddBlockingTask(
    tfrt::TaskFunction work, bool allow_queuing) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.cc", "TfThreadPoolWorkQueue::AddBlockingTask");

  AddTask(std::move(work));
  return llvm::None;
}

void TfThreadPoolWorkQueue::Quiesce() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc mht_4(mht_4_v, 248, "", "./tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.cc", "TfThreadPoolWorkQueue::Quiesce");

  // TODO(b/186668821): implement this
  CHECK(false);  // Crash OK
}

// From
// third_party/tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.cc
void TfThreadPoolWorkQueue::Await(
    tfrt::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> values) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc mht_5(mht_5_v, 259, "", "./tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.cc", "TfThreadPoolWorkQueue::Await");

  // We are done when values_remaining drops to zero.
  tfrt::latch values_remaining(values.size());

  // As each value becomes available, we decrement the count.
  for (auto& value : values) {
    value->AndThen([&values_remaining]() { values_remaining.count_down(); });
  }

  // Wait until all values are resolved.
  values_remaining.wait();
}

bool TfThreadPoolWorkQueue::IsInWorkerThread() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePStf_threadpool_concurrent_work_queueDTcc mht_6(mht_6_v, 275, "", "./tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.cc", "TfThreadPoolWorkQueue::IsInWorkerThread");

  // TODO(b/192247530): Check if we have cases it is not true.
  return true;
}

}  // namespace tfrt_stub
}  // namespace tensorflow
