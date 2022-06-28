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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc() {
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

#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/tf2xla/kernels/shape_util.h"
#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/resource_variable_util.h"
#include "tensorflow/core/kernels/scatter_nd_util.h"

namespace tensorflow {
namespace {

Status ValidateAssignUpdateVariableOpShapes(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ValidateAssignUpdateVariableOpShapes");

  DataType variable_dtype;
  TensorShape variable_shape;
  TensorShape value_shape = ctx->InputShape(1);
  TF_RETURN_IF_ERROR(
      ctx->GetVariableTypeAndShape(0, &variable_dtype, &variable_shape));
  TF_RETURN_IF_ERROR(
      ValidateAssignUpdateVariableOpShapes(variable_shape, value_shape));
  return Status::OK();
}

class VarIsInitializedOp : public XlaOpKernel {
 public:
  explicit VarIsInitializedOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "VarIsInitializedOp");
}
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_2(mht_2_v, 222, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Compile");

    XlaResource* variable;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(0, &variable));
    ctx->SetOutput(
        0, xla::ConstantR0<bool>(ctx->builder(), variable->initialized()));
  }
};
REGISTER_XLA_OP(Name("VarIsInitializedOp"), VarIsInitializedOp);

class VariableShapeOp : public XlaOpKernel {
 public:
  explicit VariableShapeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "VariableShapeOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_4(mht_4_v, 243, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Compile");

    DataType variable_dtype;
    TensorShape shape;
    OP_REQUIRES_OK(ctx,
                   ctx->GetVariableTypeAndShape(0, &variable_dtype, &shape));
    Tensor shape_constant(out_dtype_, TensorShape({shape.dims()}));
    OP_REQUIRES_OK(ctx, TensorShapeToConstant(shape, &shape_constant));
    ctx->SetConstantOutput(0, shape_constant);
  }

 private:
  DataType out_dtype_;
};
REGISTER_XLA_OP(Name("VariableShape").CompilationOnly().IsMetadataOp(),
                VariableShapeOp);

class ReadVariableOp : public XlaOpKernel {
 public:
  explicit ReadVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_5(mht_5_v, 264, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ReadVariableOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_6(mht_6_v, 271, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Compile");

    xla::XlaOp handle;
    OP_REQUIRES_OK(
        ctx, ctx->ReadVariableInput(0, dtype_, /*shape=*/nullptr, &handle));
    ctx->SetOutput(0, handle);
  }

 private:
  DataType dtype_;
};
REGISTER_XLA_OP(Name("ReadVariableOp").CompilationOnly(), ReadVariableOp);

class AssignVariableOp : public XlaOpKernel {
 public:
  explicit AssignVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_7(mht_7_v, 288, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "AssignVariableOp");
}
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_8(mht_8_v, 292, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Compile");

    OP_REQUIRES_OK(ctx,
                   ctx->AssignVariable(0, ctx->input_type(1), ctx->Input(1)));
  }
};
REGISTER_XLA_OP(Name("AssignVariableOp").CompilationOnly(), AssignVariableOp);

class AssignAddVariableOp : public XlaOpKernel {
 public:
  explicit AssignAddVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_9(mht_9_v, 304, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "AssignAddVariableOp");
}
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_10(mht_10_v, 308, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Compile");

    DataType type = ctx->input_type(1);
    xla::XlaOp handle;
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput(0, type, /*shape=*/nullptr, &handle));
    OP_REQUIRES_OK(ctx, ValidateAssignUpdateVariableOpShapes(ctx));
    handle = xla::Add(handle, ctx->Input(1));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, handle));
  }
};
REGISTER_XLA_OP(
    Name("AssignAddVariableOp").TypeConstraint("dtype", kNumericTypes),
    AssignAddVariableOp);

class AssignSubVariableOp : public XlaOpKernel {
 public:
  explicit AssignSubVariableOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_11(mht_11_v, 327, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "AssignSubVariableOp");
}
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_12(mht_12_v, 331, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Compile");

    DataType type = ctx->input_type(1);
    xla::XlaOp handle;
    OP_REQUIRES_OK(ctx,
                   ctx->ReadVariableInput(0, type, /*shape=*/nullptr, &handle));
    OP_REQUIRES_OK(ctx, ValidateAssignUpdateVariableOpShapes(ctx));
    handle = xla::Sub(handle, ctx->Input(1));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, type, handle));
  }
};
REGISTER_XLA_OP(
    Name("AssignSubVariableOp").TypeConstraint("dtype", kNumericTypes),
    AssignSubVariableOp);

class ResourceGatherOp : public XlaOpKernel {
 public:
  explicit ResourceGatherOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_13(mht_13_v, 350, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceGatherOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_dims", &batch_dims_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_14(mht_14_v, 356, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Compile");

    DataType type = ctx->expected_output_dtype(0);

    TensorShape input_shape;
    xla::XlaOp input;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, type, &input_shape, &input));

    xla::XlaOp gather;
    OP_REQUIRES_OK(ctx, XlaGatherWithBatchDimsOpImpl(ctx, input, input_shape,
                                                     batch_dims_, &gather));
    ctx->SetOutput(0, gather);
  }

 private:
  int32 batch_dims_;
};
REGISTER_XLA_OP(Name("ResourceGather"), ResourceGatherOp);

class ResourceScatterOp : public XlaOpKernel {
 public:
  explicit ResourceScatterOp(
      OpKernelConstruction* context, bool indices_are_vectors,
      std::function<xla::XlaOp(const xla::XlaOp&, const xla::XlaOp&,
                               xla::XlaBuilder*)>
          combiner)
      : XlaOpKernel(context),
        indices_are_vectors_(indices_are_vectors),
        combiner_(std::move(combiner)) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_15(mht_15_v, 386, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_16(mht_16_v, 391, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Compile");

    xla::XlaBuilder* builder = context->builder();

    DataType dtype = context->input_type(2);
    TensorShape var_shape;
    xla::XlaOp var_value;
    OP_REQUIRES_OK(
        context, context->ReadVariableInput(0, dtype, &var_shape, &var_value));
    // This check is only required for ScatterNdOps.
    if (indices_are_vectors_) {
      OP_REQUIRES_OK(context, ValidateScatterNdUpdateShape(
                                  var_shape, context->InputShape(1),
                                  context->InputShape(2)));
    }

    const xla::XlaOp indices = context->Input(1);
    const xla::XlaOp updates = context->Input(2);

    auto result = XlaScatter(var_value, updates, indices, indices_are_vectors_,
                             combiner_, builder);
    OP_REQUIRES_OK(context, result.status());
    OP_REQUIRES_OK(context,
                   context->AssignVariable(0, dtype, result.ValueOrDie()));
  }

 private:
  const bool indices_are_vectors_;
  const std::function<xla::XlaOp(const xla::XlaOp&, const xla::XlaOp&,
                                 xla::XlaBuilder*)>
      combiner_;
};

class ResourceScatterAddOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterAddOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_17(mht_17_v, 429, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterAddOp");
}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_18(mht_18_v, 436, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Combine");

    return xla::Add(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterAdd"), ResourceScatterAddOp);

class ResourceScatterSubOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterSubOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_19(mht_19_v, 448, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterSubOp");
}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_20(mht_20_v, 455, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Combine");

    return xla::Sub(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterSub"), ResourceScatterSubOp);

class ResourceScatterMulOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterMulOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_21(mht_21_v, 467, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterMulOp");
}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_22(mht_22_v, 474, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Combine");

    return xla::Mul(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterMul"), ResourceScatterMulOp);

class ResourceScatterDivOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterDivOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_23(mht_23_v, 486, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterDivOp");
}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_24(mht_24_v, 493, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Combine");

    return xla::Div(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterDiv"), ResourceScatterDivOp);

class ResourceScatterMinOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterMinOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_25(mht_25_v, 505, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterMinOp");
}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_26(mht_26_v, 512, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Combine");

    return xla::Min(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterMin"), ResourceScatterMinOp);

class ResourceScatterMaxOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterMaxOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false, Combine) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_27(mht_27_v, 524, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterMaxOp");
}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_28(mht_28_v, 531, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Combine");

    return xla::Max(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterMax"), ResourceScatterMaxOp);

class ResourceScatterUpdateOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterUpdateOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/false,
                          /*combiner=*/{}) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_29(mht_29_v, 544, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterUpdateOp");
}
};
REGISTER_XLA_OP(Name("ResourceScatterUpdate"), ResourceScatterUpdateOp);

class ResourceScatterNdUpdateOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterNdUpdateOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/true,
                          /*combiner=*/{}) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_30(mht_30_v, 555, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterNdUpdateOp");
}
};
REGISTER_XLA_OP(Name("ResourceScatterNdUpdate"), ResourceScatterNdUpdateOp);

class ResourceScatterNdAddOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterNdAddOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/true,
                          /*combiner=*/Combine) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_31(mht_31_v, 566, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterNdAddOp");
}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_32(mht_32_v, 573, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Combine");

    return xla::Add(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterNdAdd"), ResourceScatterNdAddOp);

class ResourceScatterNdSubOp : public ResourceScatterOp {
 public:
  explicit ResourceScatterNdSubOp(OpKernelConstruction* context)
      : ResourceScatterOp(context, /*indices_are_vectors=*/true,
                          /*combiner=*/Combine) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_33(mht_33_v, 586, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "ResourceScatterNdSubOp");
}

 private:
  static xla::XlaOp Combine(const xla::XlaOp& x, const xla::XlaOp& y,
                            xla::XlaBuilder* builder) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSvariable_opsDTcc mht_34(mht_34_v, 593, "", "./tensorflow/compiler/tf2xla/kernels/variable_ops.cc", "Combine");

    return xla::Sub(x, y);
  }
};
REGISTER_XLA_OP(Name("ResourceScatterNdSub"), ResourceScatterNdSubOp);

}  // namespace
}  // namespace tensorflow
