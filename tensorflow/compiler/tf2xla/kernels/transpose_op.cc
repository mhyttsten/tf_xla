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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStranspose_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStranspose_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStranspose_opDTcc() {
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

// XLA-specific Transpose Op. This is very different to the Eigen
// version in third_party/tensorflow because XLA's reshape neatly
// handles all transposes, while Eigen needs a restricted DoTranspose
// helper.

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace {

class TransposeOp : public XlaOpKernel {
 public:
  explicit TransposeOp(OpKernelConstruction* ctx, bool conjugate = false)
      : XlaOpKernel(ctx), conjugate_(conjugate) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStranspose_opDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/tf2xla/kernels/transpose_op.cc", "TransposeOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStranspose_opDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/tf2xla/kernels/transpose_op.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape("x");
    const TensorShape perm_tensor_shape = ctx->InputShape("perm");

    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm_tensor_shape),
                errors::InvalidArgument("perm must be a vector, not ",
                                        perm_tensor_shape.DebugString()));

    const int dims = input_shape.dims();
    OP_REQUIRES(ctx, dims == perm_tensor_shape.num_elements(),
                errors::InvalidArgument("transpose expects a vector of size ",
                                        input_shape.dims(),
                                        ". But input(1) is a vector of size ",
                                        perm_tensor_shape.num_elements()));

    std::vector<int64_t> perm;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector("perm", &perm));

    std::vector<int64_t> transposed_order;
    // Check whether permutation is a permutation of integers of [0 .. dims).
    absl::InlinedVector<bool, 8> bits(dims);
    bool is_identity = true;
    for (int i = 0; i < dims; ++i) {
      const int64_t d = perm[i];
      OP_REQUIRES(
          ctx, 0 <= d && d < dims,
          errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
      bits[d] = true;
      transposed_order.push_back(d);
      if (d != i) {
        is_identity = false;
      }
    }
    for (int i = 0; i < dims; ++i) {
      OP_REQUIRES(
          ctx, bits[i],
          errors::InvalidArgument(i, " is missing from 'perm' argument."));
    }

    xla::XlaOp transposed;
    // 0-D, 1-D, and identity transposes do nothing.
    if (dims <= 1 || is_identity) {
      transposed = ctx->Input("x");
    } else {
      transposed = xla::Transpose(ctx->Input("x"), transposed_order);
    }

    // Conjugate the transposed result if this is ConjugateTransposeOp.
    if (conjugate_) {
      ctx->SetOutput(0, xla::Conj(transposed));
    } else {
      ctx->SetOutput(0, transposed);
    }
  }

 private:
  const bool conjugate_;
};

class ConjugateTransposeOp : public TransposeOp {
 public:
  explicit ConjugateTransposeOp(OpKernelConstruction* ctx)
      : TransposeOp(ctx, /*conjugate=*/true) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStranspose_opDTcc mht_2(mht_2_v, 278, "", "./tensorflow/compiler/tf2xla/kernels/transpose_op.cc", "ConjugateTransposeOp");
}
};

REGISTER_XLA_OP(Name("Transpose").CompileTimeConstantInput("perm"),
                TransposeOp);

REGISTER_XLA_OP(Name("ConjugateTranspose").CompileTimeConstantInput("perm"),
                ConjugateTransposeOp);

// InvertPermutation frequently forms part of the gradient of Transpose.
//
// inv = InvertPermutationOp(p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32 or int64.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.

class InvertPermutationOp : public XlaOpKernel {
 public:
  explicit InvertPermutationOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStranspose_opDTcc mht_3(mht_3_v, 301, "", "./tensorflow/compiler/tf2xla/kernels/transpose_op.cc", "InvertPermutationOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStranspose_opDTcc mht_4(mht_4_v, 306, "", "./tensorflow/compiler/tf2xla/kernels/transpose_op.cc", "Compile");

    DataType dtype = ctx->expected_output_dtype(0);
    Status status;
    switch (dtype) {
      case DT_INT32:
        InvertPermutation<int32>(ctx);
        break;
      case DT_INT64:
        InvertPermutation<int64_t>(ctx);
        break;
      default:
        // This should never happen since we restrict this kernel to only match
        // inputs with supported Tensor datatype.
        OP_REQUIRES_OK(ctx, errors::InvalidArgument(
                                "InvertPermutation expects x as either ",
                                "int32 or int64, not ", DataTypeString(dtype)));
    }
  }

  template <typename T>
  void InvertPermutation(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStranspose_opDTcc mht_5(mht_5_v, 329, "", "./tensorflow/compiler/tf2xla/kernels/transpose_op.cc", "InvertPermutation");

    OP_REQUIRES(ctx,
                FastBoundsCheck(ctx->InputShape(0).num_elements(),
                                std::numeric_limits<T>::max()),
                errors::InvalidArgument(
                    "permutation of nonnegative integers must have <= ",
                    std::numeric_limits<T>::max(), " elements"));

    auto e = ctx->InputExpression(0);
    auto* client = ctx->compiler() ? ctx->compiler()->client() : nullptr;
    auto tensor_or_status = e.ResolveConstant(client);
    OP_REQUIRES_OK(ctx, tensor_or_status.status());
    // If the input is a constant, we also want the output to be a constant.
    // Some models rely on the result of InvertPermutation being a constant.
    // TODO(b/32495713): Remove this when we can check whether Scatter is
    // constant. Right now, we always assume it is non-constant because we don't
    // check the embedded computation.
    if (tensor_or_status.ValueOrDie().has_value()) {
      std::vector<int64_t> perm;
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(0, &perm));

      int size = perm.size();

      std::vector<T> output(size);
      std::fill_n(output.data(), size, -1);
      for (int i = 0; i < size; ++i) {
        const int64_t d = perm[i];
        OP_REQUIRES(ctx, FastBoundsCheck(d, size),
                    errors::InvalidArgument(d, " is not between 0 and ", size));
        OP_REQUIRES(ctx, output[d] == -1,
                    errors::InvalidArgument(d, " is duplicated in the input."));
        output[d] = i;
      }

      ctx->SetOutput(0, xla::ConstantR1<T>(ctx->builder(), output));
    } else {
      auto indices = ctx->Input(0);
      T size = ctx->InputShape(0).num_elements();
      auto iota =
          xla::Iota(ctx->builder(),
                    xla::primitive_util::NativeToPrimitiveType<T>(), size);
      auto result = XlaScatter(iota, iota, indices,
                               /*indices_are_vectors=*/false, /*combiner=*/{},
                               ctx->builder());
      OP_REQUIRES_OK(ctx, result.status());
      ctx->SetOutput(0, result.ValueOrDie());
    }
  }
};

REGISTER_XLA_OP(
    Name("InvertPermutation").TypeConstraint("T", {DT_INT32, DT_INT64}),
    InvertPermutationOp);

}  // namespace
}  // namespace tensorflow
