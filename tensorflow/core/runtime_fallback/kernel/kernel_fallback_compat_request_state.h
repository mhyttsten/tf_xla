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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_COMPAT_REQUEST_STATE_H__
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_COMPAT_REQUEST_STATE_H__
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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh() {
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


#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/support/pointer_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

// FallbackResourceArray holds the tensors that are computed only once during
// initialization and read-only afterwards.
class FallbackResourceArray {
 public:
  // Set `tensor` in the array at `index`. `index` should be dense and duplicate
  // indices are not allowed.
  void SetResource(int index, tensorflow::tfrt_stub::ImmutableTensor tensor);

  // Get the resource tensor wrapped in AsyncValue value at `index`.
  tfrt::UnRefCountedAsyncValue<tensorflow::tfrt_stub::FallbackTensor>*
  GetResource(int index) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_0(mht_0_v, 215, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "GetResource");

    return resource_async_values_.at(index).get();
  }

 private:
  // `resources_` holds the ownership of all the resource tensors. Note that it
  // may not be a one-to-one mapping between `resources_` and
  // `resource_async_values_`.
  std::vector<std::unique_ptr<tensorflow::tfrt_stub::ImmutableTensor>>
      resources_;
  // `resource_async_values_` holds the UnRefCountedAsyncValue of the fallback
  // tensors that can be directly used by fallback kernels in the graph.
  std::vector<std::unique_ptr<
      tfrt::UnRefCountedAsyncValue<tensorflow::tfrt_stub::FallbackTensor>>>
      resource_async_values_;
};

// Per-request state in kernel falllback compat mode.
class KernelFallbackCompatRequestState {
 public:
  // NOTE: This is the constructor for training.
  KernelFallbackCompatRequestState(
      std::function<void(std::function<void()>)>* runner,
      const tensorflow::DeviceMgr* device_manager, int64_t step_id,
      tfrt::OwnedOrUnownedPtr<ScopedStepContainer> step_container,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      core::RefCountPtr<Rendezvous> rendezvous,
      tfrt_stub::OpKernelRunnerTable* runner_table,
      FallbackResourceArray* resource_array,
      tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
      const absl::optional<SessionMetadata>& model_metadata,
      const tensorflow::ProcessFunctionLibraryRuntime* pflr);

  // NOTE: This is the constructor for inference.
  KernelFallbackCompatRequestState(
      std::function<void(std::function<void()>)>* runner,
      const tensorflow::DeviceMgr* device_manager, int64_t step_id,
      tfrt_stub::OpKernelRunnerTable* runner_table,
      FallbackResourceArray* resource_array,
      tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
      const absl::optional<SessionMetadata>& model_metadata,
      const tensorflow::ProcessFunctionLibraryRuntime* pflr);

  // Returns the user-specified custom device for this request. It is currently
  // only used for configure per-request intra op threadpool.
  tensorflow::Device* custom_device() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_1(mht_1_v, 263, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "custom_device");
 return custom_device_.get(); }

  ScopedStepContainer* step_container() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_2(mht_2_v, 268, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "step_container");
 return step_container_.get(); }

  const tensorflow::DeviceMgr& device_manager() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_3(mht_3_v, 273, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "device_manager");

    return *device_manager_;
  }

  const tensorflow::ProcessFunctionLibraryRuntime&
  process_function_library_runtime() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_4(mht_4_v, 281, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "process_function_library_runtime");

    return *pflr_;
  }

  CollectiveExecutor* collective_executor() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_5(mht_5_v, 288, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "collective_executor");

    return collective_executor_;
  }

  tfrt_stub::OpKernelRunnerTable* runner_table() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_6(mht_6_v, 295, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "runner_table");
 return runner_table_; }

  FallbackResourceArray* resource_array() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_7(mht_7_v, 300, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "resource_array");
 return resource_array_; }

  std::function<void(std::function<void()>)>* runner() const { return runner_; }

  CancellationManager* cancellation_manager() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_8(mht_8_v, 307, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "cancellation_manager");

    return default_cancellation_manager_;
  }

  RendezvousInterface* rendezvous() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_9(mht_9_v, 314, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "rendezvous");
 return rendezvous_.get(); }

  void set_log_device_placement(bool log) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_10(mht_10_v, 319, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "set_log_device_placement");
 log_device_placement_ = log; }
  bool log_device_placement() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_11(mht_11_v, 323, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "log_device_placement");
 return log_device_placement_; }

  tensorflow::thread::ThreadPoolInterface* intra_op_threadpool() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_12(mht_12_v, 328, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "intra_op_threadpool");

    return intra_op_threadpool_;
  }

  const SessionMetadata& session_metadata() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTh mht_13(mht_13_v, 335, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h", "session_metadata");
 return session_metadata_; }

 private:
  // Below are resources needed by current tensorflow.
  std::function<void(std::function<void()>)>* runner_ = nullptr;
  ::tfrt::OwnedOrUnownedPtr<ScopedStepContainer> step_container_;
  std::unique_ptr<tensorflow::Device> custom_device_;
  std::unique_ptr<CollectiveExecutor::Handle> collective_executor_handle_;
  CollectiveExecutor* collective_executor_ = nullptr;
  core::RefCountPtr<Rendezvous> rendezvous_;
  CancellationManager* default_cancellation_manager_ = nullptr;

  const tensorflow::DeviceMgr* device_manager_ = nullptr;

  // `runner_table` holds the prepopulated tensorflow::OpKernel instances for
  // kernel fallback compat mode.
  tfrt_stub::OpKernelRunnerTable* runner_table_ = nullptr;

  // Resource array is used for keeping static values in the runtime. It is
  // accessed through tfrt_fallback_async.set_resource and
  // tfrt_fallback_async.get_resource kernels.
  FallbackResourceArray* resource_array_ = nullptr;

  tensorflow::thread::ThreadPoolInterface* intra_op_threadpool_ = nullptr;

  // Model metadata used for monitoring and tracing purpose.
  SessionMetadata session_metadata_;

  const tensorflow::ProcessFunctionLibraryRuntime* pflr_ = nullptr;

  bool log_device_placement_ = false;
};

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_COMPAT_REQUEST_STATE_H__
