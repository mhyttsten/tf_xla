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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_executeDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_executeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_executeDTcc() {
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
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute.h"

#include <assert.h>

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h"
#include "tfrt/common/compat/eigen/thread_pool_device.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

namespace {
using tensorflow::KernelFallbackTensor;
using tfrt::AsyncValue;
using tfrt::RCReference;
}  // namespace

void SetError(const tfrt::ExecutionContext& exec_ctx,
              llvm::SmallVector<RCReference<AsyncValue>, 4>* results,
              tfrt::string_view message) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("message: \"" + std::string(message.data(), message.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_executeDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute.cc", "SetError");

  RCReference<tfrt::ErrorAsyncValue> error = EmitErrorAsync(exec_ctx, message);
  for (auto& result : *results) {
    result->SetError(error->GetError());
  }
}

bool KernelFallbackExecute(
    const tfrt::ExecutionContext& exec_ctx, tfrt::string_view op_name,
    tfrt::ArrayRef<AsyncValue*> arguments,
    tfrt::MutableArrayRef<RCReference<AsyncValue>> results,
    const tfrt::OpAttrsRef& attrs, KernelFallbackOutputType output_type) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPSkernel_fallback_executeDTcc mht_1(mht_1_v, 231, "", "./tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute.cc", "KernelFallbackExecute");

  // Remove tf. prefix.
  op_name.consume_front("tf.");
  std::string op_name_str = op_name.str();

  llvm::SmallVector<RCReference<AsyncValue>, 4> inputs;
  inputs.reserve(arguments.size());
  for (AsyncValue* input : arguments) {
    inputs.push_back(FormRef(input));
  }
  llvm::SmallVector<RCReference<AsyncValue>, 4> outputs(results.begin(),
                                                        results.end());

  // Always run TFRTOpKernel::Compute on the blocking thread pool to
  // avoid deadlock. Many TF kernels block until their intra-op closures
  // complete.
  bool work_enqueued = EnqueueBlockingWork(
      exec_ctx.host(),
      [exec_ctx, inputs = std::move(inputs), outputs = std::move(outputs),
       op_name_str = std::move(op_name_str), attrs = attrs.freeze(),
       output_type = output_type]() mutable {
        TFRTOpKernelConstruction op_kernel_construction(attrs);
        std::unique_ptr<TFRTOpKernel> op =
            tfrt_forwarding_kernel_factories->CreateKernel(
                op_name_str, &op_kernel_construction);

        // Forward kernel construction error.
        if (op_kernel_construction.error().hasValue()) {
          SetError(exec_ctx, &outputs,
                   op_kernel_construction.error().getValue());
          return;
        }

        const TFRTOpMeta* op_meta =
            tfrt_forwarding_op_meta_map->GetOpMeta(op_name_str);
        if (op_meta == nullptr) {
          SetError(exec_ctx, &outputs,
                   tfrt::StrCat("No TFRTOpMeta for op_name ", op_name_str));
          return;
        }

        TFRTOpKernelContext op_kernel_context(inputs, outputs.size(), op_meta,
                                              exec_ctx.host());
        op->Compute(&op_kernel_context);

        // Forward the context's error or outputs to raii_frame.
        if (op_kernel_context.error().hasValue()) {
          SetError(exec_ctx, &outputs, op_kernel_context.error().getValue());
          return;
        } else {
          for (int i = 0, e = outputs.size(); i != e; ++i) {
            // Expected result could be either a tensorflow::Tensor
            // (in case we call kernel directly), or KernelFallbackTensor
            // (if called from OpHandler).
            if (output_type == KernelFallbackOutputType::TENSOR) {
              outputs[i]->emplace<tensorflow::Tensor>(
                  op_kernel_context.output(i));
            } else {
              assert(output_type ==
                     KernelFallbackOutputType::KERNEL_FALLBACK_TENSOR);
              outputs[i]->emplace<KernelFallbackTensor>(
                  KernelFallbackTensor::Create(op_kernel_context.output(i)));
            }
          }
        }
      });

  return work_enqueued;
}
}  // namespace tfd
}  // namespace tensorflow
