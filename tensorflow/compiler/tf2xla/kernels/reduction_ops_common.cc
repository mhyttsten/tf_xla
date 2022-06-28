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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_ops_commonDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_ops_commonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_ops_commonDTcc() {
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

// XLA-specific reduction Ops.

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/kernels/reduction_ops.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {

XlaReductionOp::XlaReductionOp(OpKernelConstruction* ctx,
                               DataType reduction_type)
    : XlaOpKernel(ctx), reduction_type_(reduction_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_ops_commonDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops_common.cc", "XlaReductionOp::XlaReductionOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  OP_REQUIRES_OK(
      ctx, DataTypeToPrimitiveType(reduction_type_, &xla_reduction_type_));
}

// The default finalizer converts the results back into the input type. This can
// be overridden.
xla::XlaOp XlaReductionOp::BuildFinalizer(
    xla::XlaBuilder* /*builder*/, const xla::XlaOp& /*input*/,
    const xla::XlaOp& reduce_output,
    const std::vector<int64_t>& /*dimensions_to_reduce*/) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_ops_commonDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops_common.cc", "XlaReductionOp::BuildFinalizer");

  return XlaHelpers::ConvertElementType(reduce_output, input_type(0));
}

void XlaReductionOp::Compile(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_ops_commonDTcc mht_2(mht_2_v, 222, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops_common.cc", "XlaReductionOp::Compile");

  const TensorShape data_shape = ctx->InputShape(0);
  const TensorShape axes_tensor_shape = ctx->InputShape(1);
  VLOG(1) << "ReductionOp: " << ctx->op_kernel().name();

  if (axes_tensor_shape.num_elements() == 0) {
    // The reduction axes is an empty vector, which means there are no
    // axes to reduce so just pass the input directly through to the
    // output.
    ctx->SetOutput(0, ctx->Input(0));
    return;
  }

  OP_REQUIRES(ctx, axes_tensor_shape.dims() <= 1,
              errors::InvalidArgument(
                  "Expected scalar or vector as index argument, got ",
                  axes_tensor_shape.DebugString()));

  // Evaluate the constant, reshaping to a 1-vector if it is a scalar.
  std::vector<int64_t> axes;
  xla::Literal axes_literal;
  OP_REQUIRES_OK(ctx, ctx->ConstantInputReshapedToIntVector(1, &axes));

  VLOG(1) << "data shape: " << data_shape.DebugString();
  VLOG(1) << "axes      : " << absl::StrJoin(axes, ",");

  absl::InlinedVector<bool, 4> bitmap(data_shape.dims(), false);
  std::vector<int64_t> xla_axes;
  auto num_elements = axes_tensor_shape.num_elements();
  xla_axes.reserve(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    int64_t index = axes[i];
    OP_REQUIRES(ctx,
                !(index < -data_shape.dims() || index >= data_shape.dims()),
                errors::InvalidArgument("Invalid reduction dimension (", index,
                                        " for input with ", data_shape.dims(),
                                        " dimension(s)"));
    index = (index + data_shape.dims()) % data_shape.dims();
    OP_REQUIRES(
        ctx, !bitmap[index],
        errors::InvalidArgument(
            "Invalid reduction arguments: Axes contains duplicate dimension: ",
            index));
    bitmap[index] = true;
    xla_axes.push_back(index);
  }

  std::vector<int64_t> final_shape;
  for (int i = 0; i < data_shape.dims(); ++i) {
    if (!bitmap[i]) {
      // If we are not reducing along dimension i.
      int64_t dim = data_shape.dim_size(i);
      final_shape.push_back(dim);
    } else if (keep_dims_) {
      // We are reducing along dimension i, but we want to keep the
      // same number of dimensions, so we set the dimension of i to
      // '1'.
      final_shape.push_back(1);
    }
  }

  string desc = ctx->op_kernel().name();

  xla::XlaBuilder* const b = ctx->builder();
  // Construct the builder for the reduction lambda.
  xla::XlaBuilder r(absl::StrCat(desc, "-reduction"));
  xla::PrimitiveType type;
  TF_CHECK_OK(DataTypeToPrimitiveType(reduction_type_, &type));

  auto data = xla::ConvertElementType(ctx->Input(0), type);
  // Call virtual method to get the initial value.
  auto initial = xla::ConvertElementType(InitialValue(b), type);
  // Make two scalar parameters of the desired type for the lambda.
  auto rx = xla::Parameter(&r, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  auto ry = xla::Parameter(&r, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  // Call virtual method to build the reduction lambda.
  BuildReducer(&r, rx, ry);
  xla::XlaComputation reduction_computation = r.Build().ConsumeValueOrDie();

  auto reduce = xla::Reduce(data, initial, reduction_computation, xla_axes);
  auto finalized = BuildFinalizer(b, data, reduce, xla_axes);
  auto result = keep_dims_ ? xla::Reshape(finalized, final_shape) : finalized;
  ctx->SetOutput(0, result);
}

}  // namespace tensorflow
