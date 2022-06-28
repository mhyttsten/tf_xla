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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_xla_op_kernelDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_xla_op_kernelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_xla_op_kernelDTcc() {
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

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"

namespace tensorflow {

Status MlirXlaOpKernel::ContextToXlaArgs(
    XlaOpKernelContext* ctx, std::vector<XlaCompiler::Argument>& xla_args) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_xla_op_kernelDTcc mht_0(mht_0_v, 194, "", "./tensorflow/compiler/tf2xla/mlir_xla_op_kernel.cc", "MlirXlaOpKernel::ContextToXlaArgs");

  // Collect arguments that are registered as CompileTimeConstantInput.
  std::vector<int> registered_consts_vec;
  TF_RETURN_IF_ERROR(tensorflow::XlaOpRegistry::CompileTimeConstantInputs(
      *this, &registered_consts_vec));
  llvm::SmallDenseSet<int, 4> registered_consts;
  registered_consts.insert(registered_consts_vec.begin(),
                           registered_consts_vec.end());

  int num_inputs = ctx->num_inputs();
  xla_args.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    // TODO(b/180448774): Handle kResource and kTensorList.
    XlaExpression::Kind ctx_kind_i = ctx->InputExpression(i).kind();
    if (ctx_kind_i != XlaExpression::Kind::kXlaOp &&
        ctx_kind_i != XlaExpression::Kind::kConstant)
      return tensorflow::errors::InvalidArgument(
          absl::StrCat("Input ", i, " to an MlirXlaOpKernel is invalid: ",
                       ctx->InputExpression(i).HumanString()));
    XlaCompiler::Argument arg;
    arg.type = ctx->input_type(i);
    arg.shape = ctx->InputXlaShape(i).ValueOrDie();
    arg.name = absl::StrCat("_arg", i);
    if (registered_consts.count(i)) {
      arg.kind = XlaCompiler::Argument::kConstant;
      TF_ASSIGN_OR_RETURN(arg.constant_value, ctx->ConstantInputTensor(i));
    } else {
      arg.kind = XlaCompiler::Argument::kParameter;
    }
    xla_args.push_back(arg);
  }
  return Status::OK();
}

MlirXlaOpKernel::MlirXlaOpKernel(OpKernelConstruction* ctx)
    : XlaOpKernel(ctx),
      // Since this kernel implements lowering for a single TF operation, we
      // disable MLIR threading for efficiency purpose (avoid starting a large
      // number of threads eagerly).
      mlir_ctx_(mlir::MLIRContext::Threading::DISABLED) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_xla_op_kernelDTcc mht_1(mht_1_v, 236, "", "./tensorflow/compiler/tf2xla/mlir_xla_op_kernel.cc", "MlirXlaOpKernel::MlirXlaOpKernel");
}

Status MlirXlaOpKernel::ConstructXlaOp(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_xla_op_kernelDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/tf2xla/mlir_xla_op_kernel.cc", "MlirXlaOpKernel::ConstructXlaOp");

  // Create input XlaArguments.
  std::vector<XlaCompiler::Argument> xla_args;
  TF_RETURN_IF_ERROR(ContextToXlaArgs(ctx, xla_args));

  // Create input XlaOps.
  llvm::SmallVector<xla::XlaOp, 4> xla_params(ctx->num_inputs());
  for (int i = 0, end = ctx->num_inputs(); i < end; ++i) {
    xla_params[i] = ctx->Input(i);
  }

  // Create outputs.
  std::vector<DataType> result_dtypes(ctx->num_outputs());
  for (int i = 0, end = result_dtypes.size(); i < end; ++i) {
    result_dtypes[i] = ctx->expected_output_dtype(i);
  }

  // When there are no data-flow outputs from the node, the node is used as a
  // control output by the graph to TensorflowDialect importer.
  std::vector<std::string> control_rets;
  if (result_dtypes.empty()) {
    control_rets.push_back(def().name());
  }

  // Get the context's device.
  auto device = dynamic_cast<Device*>(ctx->op_kernel_context()->device());
  if (!device) {
    return tensorflow::errors::InvalidArgument(
        "Expected the XlaOpKernelContext argument's device to have type "
        "Device.");
  }

  // Create a graph that wraps the kernel.
  TF_ASSIGN_OR_RETURN(auto graph, CreateGraph(def(), xla_args, result_dtypes));

  // Compile the graph to HLO.
  GraphDebugInfo debug_info;
  std::vector<xla::XlaOp> returns(1);
  TF_RETURN_IF_ERROR(BuildHloFromGraph(
      *graph, *ctx->builder(), mlir_ctx_, xla_params, returns,
      mlir::SpanToArrayRef<XlaCompiler::Argument>(xla_args), control_rets,
      device->device_type(),
      *ctx->function_library()->GetFunctionLibraryDefinition(), debug_info,
      {}));

  // Set context outputs.
  for (int i = 0, end = returns.size(); i < end; ++i) {
    ctx->SetOutput(i, returns[i]);
  }

  return Status::OK();
}

void MlirXlaOpKernel::Compile(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSmlir_xla_op_kernelDTcc mht_3(mht_3_v, 297, "", "./tensorflow/compiler/tf2xla/mlir_xla_op_kernel.cc", "MlirXlaOpKernel::Compile");

  auto status = ConstructXlaOp(ctx);
  if (!status.ok()) {
    errors::AppendToMessage(&status, "Failure to legalize ", def().name(),
                            " using MlirXlaOpKernel in the tf2xla bridge.");
  }
  OP_REQUIRES_OK(ctx, status);
}

}  // namespace tensorflow
