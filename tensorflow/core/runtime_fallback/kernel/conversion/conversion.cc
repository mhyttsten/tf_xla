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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSconversionPSconversionDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSconversionPSconversionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSconversionPSconversionDTcc() {
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

#include "tensorflow/core/runtime_fallback/kernel/conversion/conversion.h"

#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>

#include <utility>

#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/tensor_util.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_utils.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {
using tfrt::DenseHostTensor;

static tfrt::AsyncValueRef<tfrt::StringHostTensor>
ConvertKernelFallbackTensorToStringHostTensor(
    const KernelFallbackTensor& tensor, const tfrt::Device& src,
    const tfrt::CpuDevice& dst, const tfrt::ExecutionContext& exec_ctx) {
  auto dst_knfb_tensor = TransferTensorToDevice(exec_ctx, tensor, src, dst);
  auto dst_knfb_tensor_ptr = dst_knfb_tensor.AsPtr();
  auto result = tfrt::MakeUnconstructedAsyncValueRef<tfrt::StringHostTensor>();
  dst_knfb_tensor_ptr.AndThen([dst_knfb_tensor = std::move(dst_knfb_tensor),
                               result = result.CopyRef(), exec_ctx]() {
    if (dst_knfb_tensor.IsError()) {
      result.SetError(dst_knfb_tensor.GetError());
      return;
    }
    auto* host = exec_ctx.host();
    assert(!IsUnsupported(dst_knfb_tensor->metadata().dtype) &&
           "Unsupported dtype");
    const auto* tf_tensor = dst_knfb_tensor->GetTensor();
    assert(tf_tensor->dtype() == DT_STRING && "dtype is not DT_STRING");

    auto sht = tfrt::StringHostTensor::CreateUninitialized(
        tfd::GetTensorMetadata(*tf_tensor), host);
    const int64_t num_elems = tf_tensor->NumElements();
    const tensorflow::tstring* tstrings =
        reinterpret_cast<const tensorflow::tstring*>(tf_tensor->data());

    auto strings = sht->strings();
    for (int i = 0; i < num_elems; ++i) {
      strings[i] = tstrings[i];
    }

    result.emplace(std::move(*sht));
  });
  return result;
}

static tfrt::AsyncValueRef<KernelFallbackTensor>
ConvertStringHostTensorToKernelFallbackTensor(
    const tfrt::StringHostTensor& tensor, const tfrt::CpuDevice& src,
    const tfrt::Device& dst, const tfrt::ExecutionContext& exec_ctx) {
  auto tf_tensor = CopyShtToTfTensor(tensor);
  auto src_knfb_tensor =
      KernelFallbackTensor(tensor.shape(), tensor.dtype(), tf_tensor);
  return TransferTensorToDevice(exec_ctx, src_knfb_tensor, src, dst);
}

static tfrt::AsyncValueRef<tfrt::DenseHostTensor>
ConvertKernelFallbackTensorToDenseHostTensor(
    const KernelFallbackTensor& tensor, const tfrt::Device& src,
    const tfrt::CpuDevice& dst, const tfrt::ExecutionContext& exec_ctx) {
  auto dst_knfb_tensor = TransferTensorToDevice(exec_ctx, tensor, src, dst);
  auto dst_knfb_tensor_ptr = dst_knfb_tensor.AsPtr();
  auto result = tfrt::MakeUnconstructedAsyncValueRef<tfrt::DenseHostTensor>();

  dst_knfb_tensor_ptr.AndThen([dst_knfb_tensor = std::move(dst_knfb_tensor),
                               result = result.CopyRef(), exec_ctx]() {
    if (dst_knfb_tensor.IsError()) {
      result.SetError(dst_knfb_tensor.GetError());
      return;
    }
    const auto* tf_tensor = dst_knfb_tensor->GetTensor();
    void* data = tf_tensor->data();
    size_t size = tf_tensor->AllocatedBytes();
    tfrt::RCReference<tfrt::HostBuffer> host_buffer =
        tfrt::HostBuffer::CreateFromExternal(
            data, size, [tensor = std::move(*tf_tensor)](void*, size_t) {});
    // Assume HostBuffer::CreateFromExternal never fails.
    result.emplace(dst_knfb_tensor->metadata(), std::move(host_buffer));
  });
  return result;
}

static tfrt::AsyncValueRef<KernelFallbackTensor>
ConvertDenseHostTensorToKernelFallbackTensor(
    const tfrt::DenseHostTensor& tensor, const tfrt::CpuDevice& src,
    const tfrt::Device& dst, const tfrt::ExecutionContext& exec_ctx) {
  auto tf_tensor =
      MoveHostBufferToTfTensor(tensor.buffer(), tensor.dtype(), tensor.shape());
  KernelFallbackTensor src_knfb_tensor(tensor.shape(), tensor.dtype(),
                                       tf_tensor);
  return TransferTensorToDevice(exec_ctx, src_knfb_tensor, src, dst);
}

static tfrt::AsyncValueRef<KernelFallbackTensor> TransferKernelFallback(
    const KernelFallbackTensor& tensor, const tfrt::Device& src,
    const tfrt::Device& dst, const tfrt::ExecutionContext& exec_ctx) {
  return TransferTensorToDevice(exec_ctx, tensor, src, dst);
}

void RegisterKernelFallbackTensorConversionFn(
    tfrt::TensorConversionFnRegistry* registry) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSconversionPSconversionDTcc mht_0(mht_0_v, 305, "", "./tensorflow/core/runtime_fallback/kernel/conversion/conversion.cc", "RegisterKernelFallbackTensorConversionFn");

  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertKernelFallbackTensorToDenseHostTensor));
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertStringHostTensorToKernelFallbackTensor));
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertKernelFallbackTensorToStringHostTensor));
  registry->AddTensorConversionFn(
      TFRT_CONVERSION(ConvertDenseHostTensorToKernelFallbackTensor));
  registry->AddTensorConversionFn(TFRT_CONVERSION(TransferKernelFallback));
}

}  // namespace tfd
}  // namespace tensorflow
