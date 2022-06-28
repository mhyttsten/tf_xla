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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSconversion_functionDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSconversion_functionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSconversion_functionDTcc() {
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

// This file implements conversion function between TFRuntimeFallback and Host
// tensors.

#include "tensorflow/core/runtime_fallback/runtime/conversion_function.h"

#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_kernels.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_utils.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

tfrt::Expected<tfrt::DenseHostTensor>
ConvertRuntimeFallbackTensorToDenseHostTensor(
    const RuntimeFallbackTensor &tensor, const tfrt::CpuDevice &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  tensorflow::Status status;
  // Resolve ensures Tensor is on host CPU.
  OwnedAbstractTensorInterface tensor_interface{
      tensor.GetTensorHandle()->Resolve(&status)};
  if (!status.ok())
    return tfrt::MakeStringError("error resolving TensorHandle: ",
                                 status.error_message());

  void* data = tensor_interface->Data();
  size_t size = tensor_interface->ByteSize();
  // `tensor_interface` holds a reference on underlying Tensorflow buffer and is
  // held alive by HostBuffer deallocator lambda capture (a
  // llvm::unique_function), and it gets released when HostBuffer deallocator is
  // called and destroyed.
  auto host_buffer = tfrt::HostBuffer::CreateFromExternal(
      data, size,
      [tensor_interface = std::move(tensor_interface)](void*, size_t) {});
  // Assume HostBuffer::CreateFromExternal never fails.
  return tfrt::DenseHostTensor(tensor.metadata(), std::move(host_buffer));
}

static tfrt::AsyncValueRef<tfrt::StringHostTensor>
ConvertRuntimeFallbackTensorToStringHostTensor(
    const RuntimeFallbackTensor &tensor, const tfrt::Device &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  auto *host_ctx = exec_ctx.host();
  tensorflow::Status status;
  // Resolve ensures Tensor is on host CPU.
  OwnedAbstractTensorInterface tensor_interface{
      tensor.GetTensorHandle()->Resolve(&status)};
  if (!status.ok())
    return tfrt::MakeErrorAsyncValueRef(
        host_ctx,
        tfrt::StrCat("error resolving TensorHandle: ", status.error_message()));

  assert(tensor_interface->Type() == DT_STRING);

  // TODO(tfrt-devs): Consider a more efficient way to pass string
  // tensors between TFRT and TF.
  auto string_host_tensor =
      CopyTfStringTensorToStringHostTensor(tensor_interface.get(), host_ctx);
  if (!string_host_tensor)
    return tfrt::MakeErrorAsyncValueRef(
        host_ctx,
        tfrt::StrCat(
            "error converting TF string tensor to tfrt::StringHostTensor: ",
            string_host_tensor.takeError()));
  return tfrt::MakeAvailableAsyncValueRef<tfrt::StringHostTensor>(
      host_ctx, std::move(*string_host_tensor));
}

static tfrt::AsyncValueRef<RuntimeFallbackTensor>
ConvertScalarHostTensorToRuntimeFallbackTensor(
    const tfrt::AnyScalarHostTensor &tensor, const tfrt::CpuDevice &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  auto *host = exec_ctx.host();

  // The memory copy here is necessary since current TensorFlow doesn't support
  // packed TFRT representations like ScalarHostTensor.
  auto optional_dht =
      tfrt::CopyScalarHostTensorToDenseHostTensor(tensor, exec_ctx);
  if (!optional_dht)
    return MakeErrorAsyncValueRef(
        host, "error copying ScalarHostTensor to DenseHostTensor");

  return tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
      host, CopyRefDHTToRuntimeFallbackTensor(optional_dht.getValue(), host));
}

static tfrt::AsyncValueRef<RuntimeFallbackTensor>
ConvertDenseHostTensorToRuntimeFallbackTensor(
    const tfrt::DenseHostTensor &tensor, const tfrt::CpuDevice &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  auto *host = exec_ctx.host();

  // CopyRef and transfer one HostBuffer reference to RuntimeFallbackTensor.
  return tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
      host, CopyRefDHTToRuntimeFallbackTensor(tensor, host));
}

static tfrt::AsyncValueRef<RuntimeFallbackTensor>
ConvertStringHostTensorToRuntimeFallbackTensor(
    const tfrt::StringHostTensor &tensor, const tfrt::CpuDevice &src,
    const tfrt::CpuDevice &dst, const tfrt::ExecutionContext &exec_ctx) {
  auto *host = exec_ctx.host();

  return tfrt::MakeAvailableAsyncValueRef<RuntimeFallbackTensor>(
      host, CopySHTToRuntimeFallbackTensor(tensor, host));
}

static tfrt::Expected<RuntimeFallbackTensor>
TransferRuntimeFallbackToAnotherDevice(const RuntimeFallbackTensor &tensor,
                                       const tfrt::Device &src,
                                       const tfrt::Device &dst,
                                       const tfrt::ExecutionContext &exec_ctx) {
  auto eager_context_resource =
      exec_ctx.resource_context()
          ->GetResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName);
  if (!eager_context_resource.hasValue())
    return tfrt::MakeStringError(
        "Cannot get EagerContext from ExecutionContext.");
  auto expected_eager_context =
      eager_context_resource.getValue()->GetTFEagerContext();
  if (!expected_eager_context) return expected_eager_context.takeError();
  auto *eager_context = expected_eager_context.get();

  auto *th = tensor.GetTensorHandle();
  Device *tf_device;
  Status s = eager_context->FindDeviceFromName(dst.name().data(), &tf_device);
  if (!s.ok()) return tfrt::MakeStringError(s.error_message());

  auto *host = exec_ctx.host();

  TensorHandle *result_th;

  s = EagerCopyToDevice(th, eager_context, &eager_context->Executor(),
                        tf_device,
                        /*mirror=*/false, &result_th);
  if (!s.ok()) return tfrt::MakeStringError(s.error_message());
  return CreateRuntimeFallbackTensorFromTfTensorHandle(
      OwnedTensorHandle(result_th), host);
}

void RegisterTFRuntimeFallbackTensorToHostConversionFn(
    tfrt::TensorConversionFnRegistry *registry) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSconversion_functionDTcc mht_0(mht_0_v, 337, "", "./tensorflow/core/runtime_fallback/runtime/conversion_function.cc", "RegisterTFRuntimeFallbackTensorToHostConversionFn");

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertRuntimeFallbackTensorToDenseHostTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertRuntimeFallbackTensorToStringHostTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertScalarHostTensorToRuntimeFallbackTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertDenseHostTensorToRuntimeFallbackTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertStringHostTensorToRuntimeFallbackTensor));

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(TransferRuntimeFallbackToAnotherDevice));
}

}  // namespace tfd
}  // namespace tensorflow
