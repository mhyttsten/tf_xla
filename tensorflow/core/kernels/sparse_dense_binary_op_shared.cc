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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_sharedDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_sharedDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_sharedDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// SparseDenseBinaryOpShared is the shared code for binary coefficient-wise
// (cwise) operations of the following form:
//
//   sparse_t <binary cwise op> dense_t -> new sparse_t
//
// where:
//
//   (1) "binary cwise op" can be, for example, cdiv, cmul, cfloordiv, etc.
//   (2) LIMITATION: we only support broadcasting the dense side to the sparse
//       side.  In other words, NumDims(sparse_t) >= NumDims(dense_t), and if
//       they are equal, each dim size of sparse_t >= that of dense_t.
//   (3) Note that the result is a new sparse tensor, which means the implicitly
//       zero elements of sparse_t do not participate.  (Hence, this should not
//       be used for, say, cadd.)
//
// The only output is a vector of flat values with shape [nnz], since this op
// does not change neither the indices nor the shape of the sparse operand.
//
// See docs of all registered ops in ../ops/sparse_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/util/bcast.h"

using Eigen::TensorRef;
using tensorflow::gtl::ArraySlice;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, typename Functor>
class SparseDenseBinaryOpShared : public OpKernel {
 public:
  explicit SparseDenseBinaryOpShared(OpKernelConstruction *ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_sharedDTcc mht_0(mht_0_v, 228, "", "./tensorflow/core/kernels/sparse_dense_binary_op_shared.cc", "SparseDenseBinaryOpShared");
}

  void Compute(OpKernelContext *ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_sharedDTcc mht_1(mht_1_v, 233, "", "./tensorflow/core/kernels/sparse_dense_binary_op_shared.cc", "Compute");

    const Tensor *indices_t, *values_t, *shape_t, *dense_t;
    OP_REQUIRES_OK(ctx, ctx->input("sp_indices", &indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &values_t));
    OP_REQUIRES_OK(ctx, ctx->input("sp_shape", &shape_t));
    OP_REQUIRES_OK(ctx, ctx->input("dense", &dense_t));

    // Validations.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(indices_t->shape()),
                errors::InvalidArgument(
                    "Input sp_indices should be a matrix but received shape: ",
                    indices_t->shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(values_t->shape()) &&
                    TensorShapeUtils::IsVector(shape_t->shape()),
                errors::InvalidArgument(
                    "Inputs sp_values and sp_shape should be vectors "
                    "but received shapes: ",
                    values_t->shape().DebugString(), " and ",
                    shape_t->shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape_t->shape()),
        errors::InvalidArgument("Input sp_shape must be a vector. Got: ",
                                shape_t->shape().DebugString()));
    OP_REQUIRES(
        ctx, values_t->dim_size(0) == indices_t->dim_size(0),
        errors::InvalidArgument(
            "The first dimension of values and indices should match. (",
            values_t->dim_size(0), " vs. ", indices_t->dim_size(0), ")"));
    OP_REQUIRES(
        ctx, shape_t->shape().dim_size(0) == indices_t->shape().dim_size(1),
        errors::InvalidArgument(
            "Number of dimensions must match second dimension of indices. ",
            "Got ", shape_t->shape().dim_size(0),
            " dimensions, indices shape: ", indices_t->shape().DebugString()));
    OP_REQUIRES(ctx, shape_t->NumElements() > 0,
                errors::InvalidArgument(
                    "The shape argument requires at least one element."));

    const auto indices_mat = indices_t->matrix<int64_t>();
    const auto shape_vec = shape_t->vec<int64_t>();
    TensorShape lhs_shape;
    OP_REQUIRES_OK(ctx, TensorShape::BuildTensorShape(shape_vec, &lhs_shape));
    const auto lhs_dims = BCast::FromShape(lhs_shape);
    const auto rhs_dims = BCast::FromShape(dense_t->shape());
    BCast b(lhs_dims, rhs_dims, false);  // false for keeping the same num dims.

    // True iff (size(lhs) >= size(rhs)) and all dims in lhs is greater or equal
    // to dims in rhs (from right to left).
    auto VecGreaterEq = [](ArraySlice<int64_t> lhs, ArraySlice<int64_t> rhs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_dense_binary_op_sharedDTcc mht_2(mht_2_v, 285, "", "./tensorflow/core/kernels/sparse_dense_binary_op_shared.cc", "lambda");

      if (lhs.size() < rhs.size()) return false;
      for (size_t i = 0; i < rhs.size(); ++i) {
        if (lhs[lhs.size() - 1 - i] < rhs[rhs.size() - 1 - i]) return false;
      }
      return true;
    };
    OP_REQUIRES(ctx, VecGreaterEq(lhs_dims, rhs_dims) && b.IsValid(),
                errors::InvalidArgument(
                    "SparseDenseBinaryOpShared broadcasts dense to sparse "
                    "only; got incompatible shapes: [",
                    absl::StrJoin(lhs_dims, ","), "] vs. [",
                    absl::StrJoin(rhs_dims, ","), "]"));

    Tensor *output_values = nullptr;
    Tensor dense_gathered;
    const int64_t nnz = indices_t->dim_size(0);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({nnz}), &output_values));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, TensorShape({nnz}),
                                &dense_gathered));
    bool op_is_div = false;
    if (absl::StrContains(ctx->op_kernel().type_string_view(), "Div")) {
      op_is_div = true;
    }
    // Pulls relevant entries from the dense side, with reshape and broadcasting
    // *of the dense side* taken into account.  Use a TensorRef to avoid blowing
    // up memory.
    //
    // We can directly use the sparse indices to look up dense side, because
    // "b.y_reshape()" and "b.y_bcast()" are guaranteed to have rank "ndims".
    auto dense_gathered_flat = dense_gathered.flat<T>();
    const int ndims = lhs_dims.size();
    switch (ndims) {
#define CASE(NDIM)                                                             \
  case NDIM: {                                                                 \
    TensorRef<Eigen::Tensor<const T, NDIM, Eigen::RowMajor>> rhs_ref =         \
        dense_t->shaped<T, NDIM>(b.y_reshape())                                \
            .broadcast(BCast::ToIndexArray<NDIM>(b.y_bcast()));                \
    Eigen::array<Eigen::DenseIndex, NDIM> idx;                                 \
    bool indices_valid = true;                                                 \
    for (int i = 0; i < nnz; ++i) {                                            \
      for (int d = 0; d < NDIM; ++d) {                                         \
        idx[d] = internal::SubtleMustCopy(indices_mat(i, d));                  \
        if (!FastBoundsCheck(idx[d], rhs_ref.dimension(d))) {                  \
          indices_valid = false;                                               \
        }                                                                      \
      }                                                                        \
      OP_REQUIRES(                                                             \
          ctx, indices_valid,                                                  \
          errors::InvalidArgument("Provided indices are out-of-bounds w.r.t. " \
                                  "dense side with broadcasted shape"));       \
      dense_gathered_flat(i) = rhs_ref.coeff(idx);                             \
      if (op_is_div) {                                                         \
        OP_REQUIRES(ctx, dense_gathered_flat(i) != 0,                          \
                    errors::InvalidArgument(                                   \
                        "SparseDenseCwiseDiv cannot divide by zero,"           \
                        "but input dense tensor contains zero "));             \
      }                                                                        \
    }                                                                          \
    break;                                                                     \
  }

      CASE(1);
      CASE(2);
      CASE(3);
      CASE(4);
      CASE(5);
      default:
        OP_REQUIRES(
            ctx, false,
            errors::InvalidArgument("Only tensors with ranks between 1 and 5 "
                                    "are currently supported.  Tensor rank: ",
                                    ndims));
#undef CASE
    }

    output_values->flat<T>().device(ctx->eigen_device<Device>()) =
        values_t->flat<T>().binaryExpr(dense_gathered_flat,
                                       typename Functor::func());
  }
};

// NOTE(aselle): If Div is extended to non-reals, make sure to use the same
// separation of operator semantics as done for dense cwise ops. I.e. you
// should make SparseDenseCwiseRealDiv, SparseDenseCwiseTruncateDiv,
// SparseDenseCwiseFloorDiv, and then deprecate, SparseDenseCwiseDiv.
// TODO(zongheng): extend to other eligible cwise operations as requested.
#define REGISTER_KERNELS(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseDenseCwiseMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseDenseBinaryOpShared<CPUDevice, T, functor::mul<T>>)              \
                                                                             \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseDenseCwiseDiv").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseDenseBinaryOpShared<CPUDevice, T, functor::div<T>>)              \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseDenseCwiseAdd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseDenseBinaryOpShared<CPUDevice, T, functor::add<T>>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow
