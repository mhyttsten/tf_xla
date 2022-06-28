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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlower_upper_bound_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlower_upper_bound_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlower_upper_bound_opsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

// Builds a LowerBound or UpperBound op, the distinction lying in
// comparison_direction: GT => LowerBoundOp, GE => UpperBoundOp.
// Note that this is an O(MN) algorithm: all entries in each sorted_inputs row
// are considered, and their sorted nature is not fully exploited.
void BuildLowerUpperBoundOp(XlaOpKernelContext* ctx, DataType out_dtype,
                            xla::ComparisonDirection comparison_direction) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlower_upper_bound_opsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/tf2xla/kernels/lower_upper_bound_ops.cc", "BuildLowerUpperBoundOp");

  const TensorShape sorted_inputs_shape = ctx->InputShape("sorted_inputs");
  const TensorShape values_shape = ctx->InputShape("values");
  const xla::XlaOp sorted_inputs = ctx->Input("sorted_inputs");
  const xla::XlaOp values = ctx->Input("values");

  // We are assuming both inputs are 2D, which they will be given the current
  // implementation of tf.searchsorted.
  OP_REQUIRES(ctx, sorted_inputs_shape.dims() == 2,
              errors::FailedPrecondition("sorted_inputs must be 2D"));
  OP_REQUIRES(ctx, values_shape.dims() == 2,
              errors::FailedPrecondition("values must be 2D"));

  // Add a new inner dimension to values, to allow broadcasting along the inner
  // dimension of sorted_sequence.
  auto new_values_shape = values_shape;
  new_values_shape.InsertDim(/* d */ 2, /* size */ 1);
  auto values_reshaped = xla::Reshape(values, new_values_shape.dim_sizes());

  // Add a new penultimate dimension to sorted_inputs, to allow broadcasting of
  // sorted_sequence entries for each value.
  auto new_sorted_inputs_shape = sorted_inputs_shape;
  new_sorted_inputs_shape.InsertDim(/* d */ 1, /* size */ 1);
  auto sorted_inputs_reshaped =
      xla::Reshape(sorted_inputs, new_sorted_inputs_shape.dim_sizes());

  // We are relying on broadcasting to compare each value against each entry in
  // the associated sorted_inputs row.
  // The reshapes above leave the tensors with equal rank of 3, so broadcast
  // dimensions are not explicitly specified.
  auto comparison = xla::Compare(values_reshaped, sorted_inputs_reshaped, {},
                                 comparison_direction);

  const DataType accumulation_type = XlaHelpers::SumAccumulationType(out_dtype);

  // Convert boolean comparison results to integers so we can sum them.
  auto comparison_int =
      XlaHelpers::ConvertElementType(comparison, accumulation_type);

  // Sum the comparison results over the inner dimension to find the index for
  // each value.
  xla::XlaBuilder* builder = ctx->builder();
  auto reduced =
      xla::Reduce(comparison_int, XlaHelpers::Zero(builder, accumulation_type),
                  *ctx->GetOrCreateAdd(accumulation_type), {2});

  ctx->SetOutput(0, reduced);
}

class LowerBoundOp : public XlaOpKernel {
 public:
  explicit LowerBoundOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlower_upper_bound_opsDTcc mht_1(mht_1_v, 257, "", "./tensorflow/compiler/tf2xla/kernels/lower_upper_bound_ops.cc", "LowerBoundOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlower_upper_bound_opsDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/tf2xla/kernels/lower_upper_bound_ops.cc", "Compile");

    BuildLowerUpperBoundOp(ctx, out_dtype_, xla::ComparisonDirection::kGt);
  }

 private:
  DataType out_dtype_;
};

REGISTER_XLA_OP(Name("LowerBound"), LowerBoundOp);

class UpperBoundOp : public XlaOpKernel {
 public:
  explicit UpperBoundOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlower_upper_bound_opsDTcc mht_3(mht_3_v, 279, "", "./tensorflow/compiler/tf2xla/kernels/lower_upper_bound_ops.cc", "UpperBoundOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlower_upper_bound_opsDTcc mht_4(mht_4_v, 286, "", "./tensorflow/compiler/tf2xla/kernels/lower_upper_bound_ops.cc", "Compile");

    BuildLowerUpperBoundOp(ctx, out_dtype_, xla::ComparisonDirection::kGe);
  }

 private:
  DataType out_dtype_;
};

REGISTER_XLA_OP(Name("UpperBound"), UpperBoundOp);

}  // namespace
}  // namespace tensorflow
