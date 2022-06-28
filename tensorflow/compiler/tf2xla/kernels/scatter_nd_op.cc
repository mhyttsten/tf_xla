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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

// Check whether updates.shape = indices.shape[:batch_dim] +
// buffer_shape[num_index_dims:]
Status ValidateUpdateShape(const TensorShape& buffer_shape,
                           const TensorShape& indices_shape,
                           const TensorShape& updates_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "ValidateUpdateShape");

  if (indices_shape.dims() < 1) {
    return errors::InvalidArgument(
        "indices shape must have >= 1 dimension; got ",
        indices_shape.DebugString());
  }

  const int64_t num_index_dims =
      indices_shape.dim_size(indices_shape.dims() - 1);
  const int64_t batch_dim = indices_shape.dims() - 1;

  auto shape_err = [&]() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "lambda");

    return errors::InvalidArgument(
        "Must have updates.shape = indices.shape[:batch_dim] + ",
        "buffer_shape[num_index_dims:], got updates.shape: ",
        updates_shape.DebugString(),
        ", indices.shape: ", indices_shape.DebugString(),
        ", buffer_shape: ", buffer_shape.DebugString(),
        ", num_index_dims: ", num_index_dims, ", and batch_dim: ", batch_dim);
  };

  if (updates_shape.dims() < batch_dim) return shape_err();
  if (buffer_shape.dims() <
      num_index_dims + (updates_shape.dims() - batch_dim)) {
    return shape_err();
  }
  if (updates_shape.dims() !=
      batch_dim + buffer_shape.dims() - num_index_dims) {
    return shape_err();
  }
  for (int d = 0; d < batch_dim; ++d) {
    if (updates_shape.dim_size(d) != indices_shape.dim_size(d)) {
      return shape_err();
    }
  }
  for (int d = 0; d < updates_shape.dims() - batch_dim; ++d) {
    if (updates_shape.dim_size(d + batch_dim) !=
        buffer_shape.dim_size(d + num_index_dims)) {
      return shape_err();
    }
  }
  return Status::OK();
}

class ScatterNdOp : public XlaOpKernel {
 public:
  explicit ScatterNdOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_2(mht_2_v, 257, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "ScatterNdOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_3(mht_3_v, 262, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "Compile");

    DataType dtype = context->input_type(1);

    TensorShape indices_shape = context->InputShape(0);
    TensorShape updates_shape = context->InputShape(1);

    TensorShape buffer_shape;
    OP_REQUIRES_OK(context, context->ConstantInputAsShape(2, &buffer_shape));

    OP_REQUIRES(
        context, TensorShapeUtils::IsVectorOrHigher(buffer_shape),
        errors::InvalidArgument("Output must be at least 1-D, ",
                                "got shape: ", buffer_shape.DebugString()));

    OP_REQUIRES(
        context,
        buffer_shape.num_elements() > 0 || (indices_shape.num_elements() == 0 &&
                                            updates_shape.num_elements() == 0),
        errors::InvalidArgument(
            "Indices and updates specified for empty output. indices shape: ",
            indices_shape.DebugString()));

    OP_REQUIRES_OK(context, ValidateUpdateShape(buffer_shape, indices_shape,
                                                updates_shape));

    xla::XlaBuilder* builder = context->builder();
    auto buffer = xla::Broadcast(XlaHelpers::Zero(builder, dtype),
                                 buffer_shape.dim_sizes());
    auto indices = context->Input(0);
    auto updates = context->Input(1);
    auto combine =
        context->input_xla_type(1) == xla::PRED ? CombineBool : CombineNum;
    auto result =
        XlaScatter(buffer, updates, indices,
                   /*indices_are_vectors=*/true, /*combiner=*/combine, builder);
    OP_REQUIRES_OK(context, result.status());
    context->SetOutput(0, result.ValueOrDie());
  }

 private:
  static xla::XlaOp CombineNum(const xla::XlaOp x, const xla::XlaOp y,
                               xla::XlaBuilder* builder) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_4(mht_4_v, 306, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "CombineNum");

    (void)builder;
    return xla::Add(x, y);
  }
  static xla::XlaOp CombineBool(const xla::XlaOp x, const xla::XlaOp y,
                                xla::XlaBuilder* builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_5(mht_5_v, 314, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "CombineBool");

    (void)builder;
    return xla::Or(x, y);
  }
};

REGISTER_XLA_OP(Name("ScatterNd").CompileTimeConstantInput("shape"),
                ScatterNdOp);

void CompileTensorScatter(
    XlaOpKernelContext* context,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp, xla::XlaBuilder*)>&
        combiner) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_6(mht_6_v, 329, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "CompileTensorScatter");

  TensorShape buffer_shape = context->InputShape(0);
  TensorShape indices_shape = context->InputShape(1);
  TensorShape updates_shape = context->InputShape(2);

  OP_REQUIRES(
      context, TensorShapeUtils::IsVectorOrHigher(buffer_shape),
      errors::InvalidArgument("Output must be at least 1-D, ",
                              "got shape: ", buffer_shape.DebugString()));

  OP_REQUIRES(
      context,
      buffer_shape.num_elements() > 0 || (indices_shape.num_elements() == 0 &&
                                          updates_shape.num_elements() == 0),
      errors::InvalidArgument(
          "Indices and updates specified for empty output. indices shape: ",
          indices_shape.DebugString()));

  OP_REQUIRES_OK(
      context, ValidateUpdateShape(buffer_shape, indices_shape, updates_shape));

  xla::XlaBuilder* builder = context->builder();
  auto buffer = context->Input(0);
  auto indices = context->Input(1);
  auto updates = context->Input(2);
  auto result = XlaScatter(buffer, updates, indices,
                           /*indices_are_vectors=*/true, combiner, builder);
  OP_REQUIRES_OK(context, result.status());
  context->SetOutput(0, result.ValueOrDie());
}

class TensorScatterAddOp : public XlaOpKernel {
 public:
  explicit TensorScatterAddOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_7(mht_7_v, 366, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "TensorScatterAddOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_8(mht_8_v, 371, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "Compile");

    CompileTensorScatter(context,
                         [](xla::XlaOp x, xla::XlaOp y, xla::XlaBuilder*) {
                           return xla::Add(x, y);
                         });
  }
};

class TensorScatterMaxOp : public XlaOpKernel {
 public:
  explicit TensorScatterMaxOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_9(mht_9_v, 385, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "TensorScatterMaxOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_10(mht_10_v, 390, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "Compile");

    CompileTensorScatter(context,
                         [](xla::XlaOp x, xla::XlaOp y, xla::XlaBuilder*) {
                           return xla::Max(x, y);
                         });
  }
};

class TensorScatterMinOp : public XlaOpKernel {
 public:
  explicit TensorScatterMinOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_11(mht_11_v, 404, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "TensorScatterMinOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_12(mht_12_v, 409, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "Compile");

    CompileTensorScatter(context,
                         [](xla::XlaOp x, xla::XlaOp y, xla::XlaBuilder*) {
                           return xla::Min(x, y);
                         });
  }
};

class TensorScatterSubOp : public XlaOpKernel {
 public:
  explicit TensorScatterSubOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_13(mht_13_v, 423, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "TensorScatterSubOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_14(mht_14_v, 428, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "Compile");

    CompileTensorScatter(context,
                         [](xla::XlaOp x, xla::XlaOp y, xla::XlaBuilder*) {
                           return xla::Sub(x, y);
                         });
  }
};

class TensorScatterUpdateOp : public XlaOpKernel {
 public:
  explicit TensorScatterUpdateOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_15(mht_15_v, 442, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "TensorScatterUpdateOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSscatter_nd_opDTcc mht_16(mht_16_v, 447, "", "./tensorflow/compiler/tf2xla/kernels/scatter_nd_op.cc", "Compile");

    CompileTensorScatter(
        context, [](xla::XlaOp, xla::XlaOp y, xla::XlaBuilder*) { return y; });
  }
};

REGISTER_XLA_OP(Name("TensorScatterAdd"), TensorScatterAddOp);
REGISTER_XLA_OP(Name("TensorScatterMax"), TensorScatterMaxOp);
REGISTER_XLA_OP(Name("TensorScatterMin"), TensorScatterMinOp);
REGISTER_XLA_OP(Name("TensorScatterSub"), TensorScatterSubOp);
REGISTER_XLA_OP(Name("TensorScatterUpdate"), TensorScatterUpdateOp);

}  // namespace
}  // namespace tensorflow
