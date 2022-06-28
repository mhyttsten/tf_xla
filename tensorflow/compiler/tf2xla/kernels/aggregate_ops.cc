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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSaggregate_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSaggregate_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSaggregate_opsDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

class AddNOp : public XlaOpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSaggregate_opsDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/tf2xla/kernels/aggregate_ops.cc", "AddNOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSaggregate_opsDTcc mht_1(mht_1_v, 201, "", "./tensorflow/compiler/tf2xla/kernels/aggregate_ops.cc", "Compile");

    if (!ctx->ValidateInputsAreSameShape(this)) return;

    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("AddN requires at least one argument"));

    XlaExpression::Kind kind = ctx->InputExpression(0).kind();
    xla::XlaOp sum;
    switch (kind) {
      case XlaExpression::Kind::kTensorList: {
        // Check that all TensorLists are initialized.
        for (int i = 1; i < ctx->num_inputs(); ++i) {
          xla::XlaOp list = ctx->Input(i);
          bool is_initialized;
          OP_REQUIRES_OK(ctx, IsTensorListInitialized(list, &is_initialized));
          OP_REQUIRES(
              ctx, is_initialized,
              errors::InvalidArgument("TensorList input #", i,
                                      " for AddN op is an uninitialized list"));
        }
        // Nested TensorList is not supported.
        bool is_nested_list;
        OP_REQUIRES_OK(ctx, IsNestedTensorList(ctx->Input(0), &is_nested_list));
        OP_REQUIRES(ctx, !is_nested_list,
                    errors::Unimplemented(
                        "Nested TensorList is not supported for AddN op"));

        OP_REQUIRES_OK(ctx, GetTensorListBuffer(ctx->Input(0), &sum));
        xla::Shape sum_shape;
        OP_REQUIRES_OK(ctx,
                       GetTensorListBufferShape(ctx->Input(0), &sum_shape));
        for (int i = 1; i < ctx->num_inputs(); ++i) {
          xla::XlaOp operand;
          OP_REQUIRES_OK(ctx, GetTensorListBuffer(ctx->Input(i), &operand));
          // Check that the shapes match.
          xla::Shape operand_shape;
          OP_REQUIRES_OK(
              ctx, GetTensorListBufferShape(ctx->Input(i), &operand_shape));
          OP_REQUIRES(
              ctx, sum_shape.dimensions() == operand_shape.dimensions(),
              errors::InvalidArgument(
                  "TensorList arguments to AddN must all have the same ",
                  "shape.\n", "Expected: ", sum_shape.DebugString(), "\n",
                  "Found: ", operand_shape.DebugString()));
          sum = xla::Add(sum, operand);
        }
        xla::XlaOp push_index;
        OP_REQUIRES_OK(ctx, GetTensorListPushIndex(ctx->Input(0), &push_index));
        OP_REQUIRES_OK(ctx, BuildNonNestedTensorList(sum, push_index, &sum));
        ctx->SetTensorListOutput(0, sum);
        break;
      }
      default:
        sum = ctx->Input(0);
        for (int i = 1; i < ctx->num_inputs(); ++i) {
          sum = xla::Add(sum, ctx->Input(i));
        }
        ctx->SetOutput(0, sum);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AddNOp);
};

REGISTER_XLA_OP(Name("AddN").AllowVariantTypes(), AddNOp);

}  // namespace
}  // namespace tensorflow
