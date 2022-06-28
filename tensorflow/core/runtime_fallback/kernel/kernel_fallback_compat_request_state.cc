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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTcc() {
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
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/support/pointer_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

using ::tensorflow::tfrt_stub::OpKernelRunnerTable;

void FallbackResourceArray::SetResource(
    int index, tensorflow::tfrt_stub::ImmutableTensor tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.cc", "FallbackResourceArray::SetResource");

  if (resource_async_values_.size() <= index) {
    resource_async_values_.resize(index + 1);
  }

  DCHECK(!resource_async_values_[index]);

  resources_.push_back(std::make_unique<tensorflow::tfrt_stub::ImmutableTensor>(
      std::move(tensor)));

  resource_async_values_[index] = std::make_unique<
      tfrt::UnRefCountedAsyncValue<tensorflow::tfrt_stub::FallbackTensor>>(
      resources_.back().get());
}

static CancellationManager* GetDefaultCancellationManager() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.cc", "GetDefaultCancellationManager");

  // TODO(b/167630926): Support cancellation by hooking up with TFRT's
  // mechanism.
  static auto* const default_cancellation_manager = new CancellationManager;
  return default_cancellation_manager;
}

KernelFallbackCompatRequestState::KernelFallbackCompatRequestState(
    std::function<void(std::function<void()>)>* runner,
    const tensorflow::DeviceMgr* device_manager, int64_t step_id,
    tfrt::OwnedOrUnownedPtr<ScopedStepContainer> step_container,
    std::unique_ptr<CollectiveExecutor::Handle> collective_executor_handle,
    core::RefCountPtr<Rendezvous> rendezvous, OpKernelRunnerTable* runner_table,
    FallbackResourceArray* resource_array,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
    const absl::optional<SessionMetadata>& model_metadata,
    const tensorflow::ProcessFunctionLibraryRuntime* pflr)
    : runner_(runner),
      step_container_(std::move(step_container)),
      collective_executor_handle_(std::move(collective_executor_handle)),
      collective_executor_(collective_executor_handle_
                               ? collective_executor_handle_->get()
                               : nullptr),
      rendezvous_(std::move(rendezvous)),
      default_cancellation_manager_(GetDefaultCancellationManager()),
      device_manager_(device_manager),
      runner_table_(runner_table),
      resource_array_(resource_array),
      intra_op_threadpool_(user_intra_op_threadpool),
      pflr_(pflr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTcc mht_2(mht_2_v, 258, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.cc", "KernelFallbackCompatRequestState::KernelFallbackCompatRequestState");

  DCHECK(runner_);
  DCHECK(device_manager_);
  DCHECK(runner_table_);
  DCHECK(resource_array_);
  DCHECK(rendezvous_);

  // TODO(tfrt-devs): Support customizing non-CPU devices.
  auto* device = device_manager_->HostCPU();
  if (user_intra_op_threadpool != nullptr) {
    custom_device_ = tensorflow::RenamedDevice::NewRenamedDevice(
        device->name(), device, /*owns_underlying=*/false,
        /*isolate_session_state=*/false, user_intra_op_threadpool);
  }
  if (model_metadata.has_value()) {
    session_metadata_ = *model_metadata;
  }
}

KernelFallbackCompatRequestState::KernelFallbackCompatRequestState(
    std::function<void(std::function<void()>)>* runner,
    const tensorflow::DeviceMgr* device_manager, int64_t step_id,
    OpKernelRunnerTable* runner_table, FallbackResourceArray* resource_array,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
    const absl::optional<SessionMetadata>& model_metadata,
    const tensorflow::ProcessFunctionLibraryRuntime* pflr)
    : KernelFallbackCompatRequestState(
          runner, device_manager, step_id,
          // The following code is copied from
          // third_party/tensorflow/core/common_runtime/direct_session.cc
          tfrt::OwnedOrUnownedPtr<ScopedStepContainer>{
              std::make_unique<ScopedStepContainer>(
                  step_id,
                  [step_id, device_manager](const std::string& name) {
                    for (tensorflow::Device* device :
                         device_manager->ListDevices()) {
                      auto status = device->resource_manager()->Cleanup(name);
                      (void)status;
                      tensorflow::ScopedAllocatorMgr* sam =
                          device->GetScopedAllocatorMgr();
                      if (sam) sam->Cleanup(step_id);
                    }
                  })},
          /*collective_executor=*/nullptr,
          /*rendezvous=*/
          core::RefCountPtr<RefCountedIntraProcessRendezvous>(
              new RefCountedIntraProcessRendezvous(device_manager)),
          runner_table, resource_array, user_intra_op_threadpool,
          model_metadata, pflr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_compat_request_stateDTcc mht_3(mht_3_v, 309, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.cc", "KernelFallbackCompatRequestState::KernelFallbackCompatRequestState");
}

}  // namespace tfd
}  // namespace tensorflow
