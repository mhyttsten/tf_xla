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
#ifndef TENSORFLOW_CORE_TFRT_RUNTIME_WORK_QUEUE_INTERFACE_H_
#define TENSORFLOW_CORE_TFRT_RUNTIME_WORK_QUEUE_INTERFACE_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSruntimePSwork_queue_interfaceDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSwork_queue_interfaceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSruntimePSwork_queue_interfaceDTh() {
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


#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// This is an intermediate interface in tensorflow for injecting thread pool
// implementation into TFRT. We can add savedmodel/tensorflow specific
// methods (eg. create an intra op thread pool) without changing TFRT core.
class WorkQueueInterface : public tfrt::ConcurrentWorkQueue {
 public:
  ~WorkQueueInterface() override = 0;

  // Returns per-request work queue if possible. A nullptr should be returned if
  // the implementation does not implement the per-request work queue.
  //
  // TODO(b/198671794): Remove per-request concepts from the work queue
  // interface so that the interface is more composable. Per-request logic
  // should be handled separately.
  ABSL_DEPRECATED("Create the instance directly instead.")
  virtual StatusOr<std::unique_ptr<WorkQueueInterface>> InitializeRequest(
      tfrt::RequestContextBuilder* request_context_builder,
      thread::ThreadPoolInterface** intra_op_threadpool) const {
    *intra_op_threadpool = nullptr;
    return {nullptr};
  }
};

inline WorkQueueInterface::~WorkQueueInterface() = default;

// Creates a WorkQueueInterface from a ConcurrentWorkQueue. The returned
// WorkQueueInterface simply delegates all its public methods to the specified
// ConcurrentWorkQueue.
std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue);

// Creates a WorkQueueInterface from a ConcurrentWorkQueue. The returned
// WorkQueueInterface simply delegates all its public methods to the specified
// ConcurrentWorkQueue. The `intra_thread_pool` is stored and will be passed out
// when `InitializeRequest()` is called.
std::unique_ptr<WorkQueueInterface> WrapDefaultWorkQueue(
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue,
    thread::ThreadPoolInterface* intra_thread_pool);

// A helper function that wraps tasks with traceme events.
template <typename Callable>
tfrt::TaskFunction WrapWork(int64_t id, absl::string_view name,
                            Callable&& work) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSwork_queue_interfaceDTh mht_0(mht_0_v, 240, "", "./tensorflow/core/tfrt/runtime/work_queue_interface.h", "WrapWork");

  tensorflow::Context context(tensorflow::ContextKind::kThread);
  return tfrt::TaskFunction([id, name = std::string(name),
                             context = std::move(context),
                             work = std::forward<Callable>(work)]() mutable {
    // From TraceMeProducer in the function that launches graph execution, eg.
    // SavedModelImpl::Run().
    tensorflow::profiler::TraceMeConsumer activity(
        [&]() {
          return tensorflow::profiler::TraceMeEncode(name, {{"id", id}});
        },
        tensorflow::profiler::ContextType::kTfrtExecutor, id,
        tensorflow::profiler::TraceMeLevel::kInfo);
    tensorflow::WithContext wc(context);
    std::forward<Callable>(work)();
  });
}

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_RUNTIME_WORK_QUEUE_INTERFACE_H_
