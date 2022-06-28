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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc() {
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

// This file implements RuntimeFallbackOpHandler, responsible for running TFRT
// ops on Tensorflow.

#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_kernels.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/dispatch_utils.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_invocation.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime
// TODO(b/160798174): Avoid CUDA/ROCM macro.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tfrt/gpu/device/device.h"  // from @tf_runtime
#include "tfrt/gpu/device/device_util.h"  // from @tf_runtime
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"  // from @tf_runtime
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {
namespace tfd {
// TODO(tfrt-devs): Rename it.
class RuntimeFallbackOpHandler : public tfrt::OpHandler {
 public:
  ~RuntimeFallbackOpHandler() override;

  llvm::Expected<tfrt::CoreRuntimeOp> MakeOp(
      tfrt::string_view op_name) override;

  tfrt::string_view DeviceName() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc mht_0(mht_0_v, 234, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.cc", "DeviceName");
 return device_->name(); }

  const std::string& TfDeviceName() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.cc", "TfDeviceName");
 return tf_device_name_; }

  tfrt::RCReference<tfrt::Device> GetDeviceRef() { return device_; }

 private:
  explicit RuntimeFallbackOpHandler(tfrt::CoreRuntime* runtime,
                                    tfrt::RCReference<tfrt::Device> device,
                                    const std::string& tf_device_name);

  llvm::Error Initialize();

  friend llvm::Expected<tfrt::OpHandler*> CreateRuntimeFallbackOpHandler(
      tfrt::CoreRuntime* runtime, tfrt::string_view tf_device_name);

  tfrt::RCReference<tfrt::Device> device_;
  // Tensorflow device name, e.g., /device:CPU:0.
  std::string tf_device_name_;
};

namespace {

using tfrt::AsyncValue;
using tfrt::AsyncValueRef;
using tfrt::Chain;
using tfrt::CoreRuntime;
using tfrt::CoreRuntimeOp;
using tfrt::DenseHostTensor;
using tfrt::ExecutionContext;
using tfrt::Expected;
using tfrt::OpAttrsRef;
using tfrt::OpHandler;
using tfrt::OpInvocation;
using tfrt::OpMetadataFn;
using tfrt::raw_ostream;
using tfrt::RCReference;
using tfrt::string_view;
using tfrt::Tensor;
using tfrt::TensorMetadata;

using RuntimeFallbackDispatchFn = AsyncValueRef<Chain> (*)(
    const ExecutionContext& exec_ctx, const char* op_name,
    const char* device_name, llvm::ArrayRef<Tensor*> arguments,
    const OpAttrsRef& attrs,
    llvm::MutableArrayRef<RCReference<AsyncValue>> results);

struct RuntimeFallbackOpEntry {
  std::string op_name;
  OpMetadataFn metadata_fn = nullptr;
  // All ops use the same dispatch function.
  RuntimeFallbackDispatchFn dispatch_fn = &RuntimeFallbackExecute;
};

static Expected<tfrt::RCReference<tfrt::Device>> GetDeviceFromFallbackTensor(
    const RuntimeFallbackTensor& result_tensor,
    const ExecutionContext& exec_ctx) {
  tensorflow::Status status;
  // Obtain the device. Please note that this device is probably not
  // the device that the TensorHandle is located on. E.g. for a GPU resource
  // its device is GPU but it is physicially located on CPU.
  // We use this device because upper layer (e.g. distributed strategy) may
  // use it for colocation. On the other hand, the actual device is not widely
  // used in upper layers.
  // In the future, if we need BackingDevice in higher layer as well, we can
  // update c_api_tfrt layer to get it directly from tensorflow::TensorHandle.
  const char* tf_device_name =
      result_tensor.GetTensorHandle()->DeviceName(&status);
  if (!status.ok()) {
    return tfrt::MakeStringError(status.error_message());
  }

  // TODO(b/165872892): Unify device name for tests.
  auto device = exec_ctx.host()->GetDeviceManager()->GetDeviceRef<tfrt::Device>(
      tf_device_name);
  if (!device) {
    // Convert device name to the short form, e.g. "GPU:0".
    const char* tfrt_device_name =
        ConvertTfDeviceNameToTfrtDefault(tf_device_name);
    device = exec_ctx.host()->GetDeviceManager()->GetDeviceRef<tfrt::Device>(
        tfrt_device_name);
  }
  assert(device);
  return std::move(device);
}

struct RuntimeFallbackOpHandlerTraits {
  using InputTensorTy = Tensor;
  using OpEntryTy = RuntimeFallbackOpEntry;
  using OpHandlerInfoTy = RuntimeFallbackOpHandler*;

  static void Dispatch(const RuntimeFallbackOpEntry& op_entry,
                       RuntimeFallbackOpHandler* tf_op_handler,
                       llvm::ArrayRef<Tensor*> inputs, const OpAttrsRef& attrs,
                       llvm::ArrayRef<TensorMetadata> result_mds,
                       llvm::MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc mht_2(mht_2_v, 337, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.cc", "Dispatch");

    // Call RuntimeFallbackExecute.
    auto ch = op_entry.dispatch_fn(exec_ctx, op_entry.op_name.c_str(),
                                   tf_op_handler->TfDeviceName().c_str(),
                                   inputs, attrs, results);

    if (chain) *chain = std::move(ch);
  }

  // TODO(fishx): Remove this method.
  static tfrt::Variant<tfrt::RCReference<tfrt::Device>,
                       tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>>
  GetResultDevice(RuntimeFallbackOpHandler* tf_op_handler,
                  const tfrt::AsyncValueRef<tfrt::Tensor>& result_tensor_av,
                  const ExecutionContext& exec_ctx) {
    if (result_tensor_av.IsAvailable()) {
      if (result_tensor_av.IsError()) {
        return tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>(
            result_tensor_av.CopyRCRef());
      }
      auto expected_device = GetDeviceFromFallbackTensor(
          result_tensor_av.get<RuntimeFallbackTensor>(), exec_ctx);
      if (!expected_device) {
        return tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>(
            tfrt::MakeErrorAsyncValueRef(
                exec_ctx.host(), tfrt::StrCat(expected_device.takeError())));
      }
      return std::move(expected_device.get());
    }

    auto result_device =
        tfrt::MakeUnconstructedAsyncValueRef<tfrt::RCReference<tfrt::Device>>(
            exec_ctx.host());

    result_tensor_av.AndThen([result_tensor_av_ref = result_tensor_av.CopyRef(),
                              result_device = result_device.CopyRef(),
                              exec_ctx] {
      assert(result_tensor_av_ref.IsAvailable());
      if (result_tensor_av_ref.IsError()) {
        result_device.SetError(result_tensor_av_ref.GetError());
      }
      auto expected_device = GetDeviceFromFallbackTensor(
          result_tensor_av_ref.get<RuntimeFallbackTensor>(), exec_ctx);
      result_device.emplace(GetDeviceFromFallbackTensor(
          result_tensor_av_ref.get<RuntimeFallbackTensor>(), exec_ctx));
    });
    return std::move(result_device);
  }

  static tfrt::Variant<tfrt::RCReference<tfrt::Device>,
                       tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>>
  GetResultDevice(const RuntimeFallbackOpEntry& op_entry,
                  RuntimeFallbackOpHandler* tf_op_handler,
                  const tfrt::AsyncValueRef<tfrt::Tensor>& result_tensor_av,
                  int index, const ExecutionContext& exec_ctx) {
    return GetResultDevice(tf_op_handler, result_tensor_av, exec_ctx);
  }
};

}  // namespace

Expected<CoreRuntimeOp> RuntimeFallbackOpHandler::MakeOp(string_view op_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc mht_3(mht_3_v, 402, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.cc", "RuntimeFallbackOpHandler::MakeOp");

  // NOTE(fishx): Copying string here will cost extra overhead in graph
  // execution. Because in current implementation, we needs to prepare the op
  // before each executions.
  // TODO(fishx): Avoid this heap allocation by getting op registration
  // information from current TF.
  RuntimeFallbackOpEntry op_entry;
  if (!op_name.consume_front("tf."))
    return tfrt::MakeStringError(op_name, " does not start with 'tf.'");
  op_entry.op_name.assign(op_name.begin(), op_name.end());
  return CoreRuntimeOp(
      [op_entry = std::move(op_entry), this](const OpInvocation& invocation) {
        // If the op does not have outputs, then it is expected to output an
        // out chain.
        bool update_chain = invocation.results.empty();

        // Convert the argument tensors to RuntimeFallbackTensors.
        for (auto& argument : invocation.arguments) {
          argument = argument.TransferToSameDevice(
              invocation.exec_ctx, RuntimeFallbackTensor::kTensorType);
        }

        tfrt::ExecuteOnOpHandler<RuntimeFallbackOpHandlerTraits>(
            update_chain, invocation, std::move(op_entry), this);

// TODO(b/160798174): Avoid CUDA/ROCM macro.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
        // If the RuntimeFallbackTensor contains a tensorflow::TensorHandle
        // that holds a GPU tensor, convert it to tfrt::DenseGpuTensor, and
        // populate the correct device name to the result tfrt::TensorHandle.
        //
        // Note that if the GPU tensor contains a DataType that is not natively
        // supported by TFRT, e.g. Resource DataType, we skip the conversion.
        //
        // If the RuntimeFallbackTensor's tensorflow::TensorHandle holds a CPU
        // tensor, do not convert it to DenseHostTensor (it will be lazily
        // converted) for performance reason.
        for (auto& result : invocation.results) {
          auto* host_ctx = invocation.exec_ctx.host();
          auto* result_tensor_av = result.GetAsyncTensor();

          if (!result_tensor_av->IsAvailable())
            host_ctx->Await(FormRef(result_tensor_av));

          if (result_tensor_av->IsError()) continue;

          auto result_tensor_tf_th =
              result_tensor_av->get<RuntimeFallbackTensor>().GetTensorHandle();

          // Check if we need to convert the RuntimeFallbackTensor.
          if (!(IsGpuTensorHandle(*result_tensor_tf_th) &&
                IsSupportedByTFRTGpu(result_tensor_tf_th->DataType())))
            continue;

          result = result.TransferToSameDevice(
              invocation.exec_ctx, tfrt::gpu::DenseGpuTensor::kTensorType);
        }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      },
      // device and arg_tensor_type are not used in runtime fallback ops.
      /*is_fallback=*/true, /*device=*/device_);
}

llvm::Expected<tfrt::OpHandler*> CreateRuntimeFallbackOpHandler(
    tfrt::CoreRuntime* runtime, tfrt::string_view tf_device_name) {
  // TODO(fishx): Remove the device field from fallback op handler.
  std::unique_ptr<RuntimeFallbackOpHandler> op_handler(
      new RuntimeFallbackOpHandler(
          runtime, runtime->GetHostContext()->GetHostDeviceRef(),
          tf_device_name.str()));
  if (auto error = op_handler->Initialize()) {
    return std::move(error);
  }
  auto op_handler_ptr = op_handler.get();
  runtime->TakeOpHandler(std::move(op_handler));
  return op_handler_ptr;
}

RuntimeFallbackOpHandler::RuntimeFallbackOpHandler(
    CoreRuntime* runtime, tfrt::RCReference<tfrt::Device> device,
    const std::string& tf_device_name)
    : OpHandler("tf", runtime, nullptr),
      device_(std::move(device)),
      tf_device_name_(tf_device_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tf_device_name: \"" + tf_device_name + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc mht_4(mht_4_v, 489, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.cc", "RuntimeFallbackOpHandler::RuntimeFallbackOpHandler");
}

RuntimeFallbackOpHandler::~RuntimeFallbackOpHandler() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc mht_5(mht_5_v, 494, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.cc", "RuntimeFallbackOpHandler::~RuntimeFallbackOpHandler");
}

llvm::Error RuntimeFallbackOpHandler::Initialize() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_op_handlerDTcc mht_6(mht_6_v, 499, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.cc", "RuntimeFallbackOpHandler::Initialize");

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  Status status = InjectTfGpuResources();
  if (!status.ok()) {
    return tfrt::MakeStringError(tfrt::StrCat("error injecting GPU resources: ",
                                              status.error_message()));
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  return llvm::Error::success();
}

}  // namespace tfd
}  // namespace tensorflow
