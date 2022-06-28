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

// This file declares kernel utils.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_UTILS_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh() {
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

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

template <typename T>
struct AutoReleaser {
  void operator()(T* p) const { p->Release(); }
};
template <typename T>
using AutoReleasePtr = std::unique_ptr<T, AutoReleaser<T>>;

using OwnedEagerContext = AutoReleasePtr<EagerContext>;
using OwnedEagerOperation = AutoReleasePtr<EagerOperation>;
using OwnedTensorHandle = AutoReleasePtr<TensorHandle>;
using OwnedAbstractTensorInterface = AutoReleasePtr<AbstractTensorInterface>;

// Check if a TensorHandle physically resides on GPU.
inline bool IsGpuTensorHandle(const tensorflow::TensorHandle& handle) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh mht_0(mht_0_v, 227, "", "./tensorflow/core/runtime_fallback/runtime/kernel_utils.h", "IsGpuTensorHandle");

  tensorflow::Status dummy_status;
  // BackingDeviceName is where the tensor is physically located, not where the
  // op that produces the tensor is.
  // Note that dummy_status is never set in TensorHandle::BackingDeviceName.
  absl::string_view device_name = handle.BackingDeviceName(&dummy_status);
  return absl::StrContains(device_name, "GPU");
}

// TODO(zhangqiaorjc): Allowlist more dtypes as tfrt GPU supports more.
// RuntimeFallbackTensor of supported dtypes below will be eagerly converted to
// tfrt::DenseGpuTensor after each RuntimeFallbackOpHandler::Execute.
inline bool IsSupportedByTFRTGpu(DataType dtype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh mht_1(mht_1_v, 242, "", "./tensorflow/core/runtime_fallback/runtime/kernel_utils.h", "IsSupportedByTFRTGpu");

  switch (dtype) {
    default:
      return false;
    case DataType::DT_FLOAT:
    case DataType::DT_DOUBLE:
    case DataType::DT_INT32:
      return true;
  }
}

// TODO(b/165872892): Remove this method.
// This method is needed because we use different device name in TF-TFRT
// integration and mlir test. In TF-TFRT integration, we reuse the device full
// name (e.g. /job:localhost/replica:0/task:0/device:GPU:0) from TF. But in mlir
// test, we use simplified device name "GPU:0". And lot of things in fallback
// need to be used in both cases. As a result, we need to look up the device
// with both device names.
inline const char* ConvertTfDeviceNameToTfrtDefault(const char* device_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh mht_2(mht_2_v, 264, "", "./tensorflow/core/runtime_fallback/runtime/kernel_utils.h", "ConvertTfDeviceNameToTfrtDefault");

  assert(strlen(device_name) >= 5);
  return &device_name[strlen(device_name) - 5];
}

// Create and initialize EagerContext.
tfrt::Expected<OwnedEagerContext> InitEagerContext();

tfrt::Expected<OwnedEagerContext> InitEagerContext(
    DynamicDeviceMgr* device_mgr, const SessionOptions& session_opts,
    ContextDevicePlacementPolicy default_device_placement_policy,
    bool is_async);

// Obtain EagerContext from ExecutionContext.
tfrt::Expected<EagerContext*> GetEagerContext(tfrt::ExecutionContext exec_ctx);

// Return the CoreRuntimeOp for `op_name` using fallback op_handler.
llvm::Expected<tfrt::CoreRuntimeOp> GetFallbackOp(tfrt::string_view op_name,
                                                  tfrt::HostContext* host);

constexpr char kEagerContextResourceName[] = "EagerContextResourceName";

class EagerContextResource {
 public:
  explicit EagerContextResource()
      : device_mgr_(std::make_unique<DynamicDeviceMgr>()),
        ctx_{InitEagerContext(
            device_mgr_.get(), tensorflow::SessionOptions(),
            tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
            /*is_async=*/false)} {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh mht_3(mht_3_v, 296, "", "./tensorflow/core/runtime_fallback/runtime/kernel_utils.h", "EagerContextResource");
}
  explicit EagerContextResource(
      const SessionOptions& session_opts,
      ContextDevicePlacementPolicy default_device_placement_policy,
      bool is_async)
      : device_mgr_(std::make_unique<DynamicDeviceMgr>()),
        ctx_{InitEagerContext(device_mgr_.get(), session_opts,
                              default_device_placement_policy, is_async)} {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh mht_4(mht_4_v, 306, "", "./tensorflow/core/runtime_fallback/runtime/kernel_utils.h", "EagerContextResource");
}

  tfrt::Expected<EagerContext*> GetTFEagerContext() {
    if (!ctx_) return ctx_.takeError();
    return ctx_.get().get();
  }

  DynamicDeviceMgr* GetDeviceMgr() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh mht_5(mht_5_v, 316, "", "./tensorflow/core/runtime_fallback/runtime/kernel_utils.h", "GetDeviceMgr");
 return device_mgr_.get(); }

  llvm::Error AddDevices(std::vector<std::unique_ptr<Device>> devices) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSkernel_utilsDTh mht_6(mht_6_v, 321, "", "./tensorflow/core/runtime_fallback/runtime/kernel_utils.h", "AddDevices");

    if (!ctx_) return ctx_.takeError();
    Status s = dynamic_cast<tensorflow::DynamicDeviceMgr*>(
                   ctx_.get()->local_device_mgr())
                   ->AddDevices(std::move(devices));
    if (!s.ok()) return tfrt::MakeStringError(s.error_message());
    ctx_.get()->InitPrioritizedDeviceTypeList();
    ctx_.get()->pflr()->InitializeDeviceAndFlr();
    return llvm::Error::success();
  }

 private:
  // EagerContext uses this device_mgs as local_device_mgr. We manage the
  // device_mgr_ here to allow TFRT adding new devices after EagerContext
  // initialization.
  // Today, TFRT only adds TPU devices after EagerContext initialization.
  std::unique_ptr<DynamicDeviceMgr> device_mgr_;

  tfrt::Expected<OwnedEagerContext> ctx_;
};

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_UTILS_H_
