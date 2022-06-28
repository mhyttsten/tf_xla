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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc() {
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

// TFRT kernels for calling directly into current TF kernels, bypassing the
// current TF runtime.

#include "llvm/Support/raw_ostream.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/attr_util.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tfrt/cpu/core_runtime/cpu_op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_frame.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime

namespace tensorflow {

// Directly invoke TFRTOpKernel::Compute on a kernel specified by
// 'op_name'. Pass a TFRTOpKernelContext that forwards to the provided
// AsyncKernelFrame.
//
// Directly invoked kernels must be registered with the
// REGISTER_KERNEL_FALLBACK_KERNEL macro and must work with the
// TFRTOpKernel{,Construction,Context} objects instead of the usual
// OpKernel objects.
static void TFDForwardKernel(tfrt::RemainingArguments arguments,
                             tfrt::RemainingResults results,
                             tfrt::StringAttribute op_name,
                             tfrt::RemainingAttributes attributes,
                             tfrt::AsyncKernelFrame* frame,
                             const tfrt::ExecutionContext& exec_ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_kernels.cc", "TFDForwardKernel");

  for (int i = 0; i < results.size(); ++i) {
    results.AllocateAt<tensorflow::Tensor>(i);
  }
  std::string op_name_str = op_name.str();
  tfrt::OpAttrs opattrs;
  Status s = FillOpAttrs(attributes, &opattrs);
  if (!s.ok()) {
    frame->ReportError("TFDForwardKernel: Error while parsing attributes: ",
                       s.error_message());
  }

  tfrt::OpAttrsRef opattrsref(opattrs);
  bool work_enqueued = tfd::KernelFallbackExecute(
      exec_ctx, op_name_str, arguments.values(), results.values(), opattrsref,
      tfd::KernelFallbackOutputType::TENSOR);
  if (!work_enqueued) {
    frame->ReportError("TFDForwardKernel: couldn't EnqueueBlockingWork");
  }
}

// Return an initialized scalar Tensor with the specified value.
static void TFDConstantTensor(tfrt::Argument<int32_t> value,
                              tfrt::Result<Tensor> tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc mht_1(mht_1_v, 245, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_kernels.cc", "TFDConstantTensor");

  // TODO(annarev): tensor.Emplace(value.get()) would be simpler but
  // it causes a missing typeinfo error when using -fno-rtti. Investigate
  // if we can make it work with no-rtti.
  Tensor out(DT_INT32, TensorShape({}));
  out.flat<int32>()(0) = value.get();
  tensor.Emplace(out);
}

// Print a Tensor.
static void TFDPrintTensor(tfrt::Argument<Tensor> tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc mht_2(mht_2_v, 258, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_kernels.cc", "TFDPrintTensor");

  llvm::outs() << "tensor=" << tensor.get().DebugString() << "\n";
  llvm::outs().flush();
}

// Log a Tensor.
static void TFDLogTensor(tfrt::Argument<Tensor> tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_kernels.cc", "TFDLogTensor");

  LOG(INFO) << "tensor=" << tensor.get().DebugString() << "\n";
}

void CreateKernelFallbackOpHandlerKernel(
    tfrt::Result<tfrt::OpHandler*> op_handler,
    const tfrt::ExecutionContext& exec_ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc mht_4(mht_4_v, 276, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_kernels.cc", "CreateKernelFallbackOpHandlerKernel");

  auto* runtime = tfrt::CoreRuntime::GetFromHostContext(exec_ctx.host());
  assert(runtime);
  auto op_handler_ptr = tensorflow::tfd::CreateKernelFallbackOpHandler(
      runtime, exec_ctx.host()->GetHostDeviceRef());
  assert(op_handler_ptr);
  op_handler.Emplace(op_handler_ptr.get());
}

tfrt::Chain AddKernelFallbackImplicitConversionKernel(
    tfrt::Argument<tfrt::OpHandler*> op_handler,
    const tfrt::ExecutionContext& exec_ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc mht_5(mht_5_v, 290, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_kernels.cc", "AddKernelFallbackImplicitConversionKernel");

  assert(op_handler.get()->GetName() == tfrt::CpuOpHandler::kName);
  tfrt::CpuOpHandler* cpu_op_handler =
      static_cast<tfrt::CpuOpHandler*>(op_handler.get());
  cpu_op_handler->AddImplicitConversion(KernelFallbackTensor::kTensorType,
                                        tfrt::DenseHostTensor::kTensorType);
  cpu_op_handler->AddImplicitConversion(KernelFallbackTensor::kTensorType,
                                        tfrt::AnyScalarHostTensor::kTensorType);
  cpu_op_handler->AddImplicitConversion(KernelFallbackTensor::kTensorType,
                                        tfrt::StringHostTensor::kTensorType);
  return {};
}

void RegisterKernelFallbackKernels(tfrt::KernelRegistry* registry) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_kernelsDTcc mht_6(mht_6_v, 306, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_kernels.cc", "RegisterKernelFallbackKernels");

  registry->AddKernel("tfd.forward_kernel", TFRT_KERNEL(TFDForwardKernel));
  registry->AddKernel("tfd.constant_tensor", TFRT_KERNEL(TFDConstantTensor));
  registry->AddKernel("tfd.print_tensor", TFRT_KERNEL(TFDPrintTensor));
  registry->AddKernel("tfd.log_tensor", TFRT_KERNEL(TFDLogTensor));
  registry->AddKernel("corert.create_kernel_fallback_op_handler",
                      TFRT_KERNEL(CreateKernelFallbackOpHandlerKernel));
  registry->AddKernel("corert.add_kernel_fallback_implicit_conversions",
                      TFRT_KERNEL(AddKernelFallbackImplicitConversionKernel));
}
}  // namespace tensorflow
