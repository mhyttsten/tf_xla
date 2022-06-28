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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_add_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_add_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_add_opDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_tensor_dense_add_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
// NOTE: does not support GPU yet.

namespace {

template <typename Index>
Status ValidateInputs(const Tensor *a_indices, const Tensor *a_values,
                      const Tensor *a_shape, const Tensor *b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_add_opDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/kernels/sparse_tensor_dense_add_op.cc", "ValidateInputs");

  if (!TensorShapeUtils::IsMatrix(a_indices->shape())) {
    return errors::InvalidArgument(
        "Input a_indices should be a matrix but received shape: ",
        a_indices->shape().DebugString());
  }
  if (!TensorShapeUtils::IsVector(a_values->shape()) ||
      !TensorShapeUtils::IsVector(a_shape->shape())) {
    return errors::InvalidArgument(
        "Inputs a_values and a_shape should be vectors "
        "but received shapes: ",
        a_values->shape().DebugString(), " and ",
        a_shape->shape().DebugString());
  }
  int64_t nnz = a_indices->dim_size(0);
  int64_t ndims = a_indices->dim_size(1);
  if (a_values->dim_size(0) != nnz) {
    return errors::InvalidArgument("Dimensions ", nnz, " and ",
                                   a_values->dim_size(0),
                                   " are not compatible");
  }
  if (a_shape->dim_size(0) != ndims) {
    return errors::InvalidArgument("Dimensions ", ndims, " and ",
                                   a_shape->dim_size(0), " are not compatible");
  }
  if (a_shape->NumElements() != b->dims()) {
    return errors::InvalidArgument(
        "Two operands have different ranks; received: ", a_shape->NumElements(),
        " and ", b->dims());
  }
  const auto a_shape_flat = a_shape->flat<Index>();
  for (int i = 0; i < b->dims(); ++i) {
    if (a_shape_flat(i) != b->dim_size(i)) {
      return errors::InvalidArgument(
          "Dimension ", i,
          " does not equal (no broadcasting is supported): sparse side ",
          a_shape_flat(i), " vs dense side ", b->dim_size(i));
    }
  }

  // Check for invalid indices.
  const auto a_indices_mat = a_indices->flat_inner_dims<Index>();

  for (int64_t zidx = 0; zidx < nnz; ++zidx) {
    for (int64_t didx = 0; didx < ndims; ++didx) {
      const Index idx = a_indices_mat(zidx, didx);
      if (idx < 0 || idx >= a_shape_flat(didx)) {
        return errors::InvalidArgument(
            "Sparse tensor has an invalid index on dimension ", didx,
            ": "
            "a_indices(",
            zidx, ",", didx, ") = ", idx,
            ", dense tensor shape: ", a_shape_flat);
      }
    }
  }

  return Status::OK();
}

}  // namespace

template <typename Device, typename T, typename Index>
class SparseTensorDenseAddOp : public OpKernel {
 public:
  explicit SparseTensorDenseAddOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_add_opDTcc mht_1(mht_1_v, 274, "", "./tensorflow/core/kernels/sparse_tensor_dense_add_op.cc", "SparseTensorDenseAddOp");
}

  void Compute(OpKernelContext *ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_tensor_dense_add_opDTcc mht_2(mht_2_v, 279, "", "./tensorflow/core/kernels/sparse_tensor_dense_add_op.cc", "Compute");

    const Tensor *a_indices_t, *a_values_t, *a_shape_t, *b;
    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("a_values", &a_values_t));
    OP_REQUIRES_OK(ctx, ctx->input("a_shape", &a_shape_t));
    OP_REQUIRES_OK(ctx, ctx->input("b", &b));
    OP_REQUIRES_OK(
        ctx, ValidateInputs<Index>(a_indices_t, a_values_t, a_shape_t, b));

    Tensor *out_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, b->shape(), &out_t));

    const int ndims = static_cast<int>(a_indices_t->dim_size(1));
    const auto a_indices_mat = a_indices_t->flat_inner_dims<Index>();
    const auto a_values_flat = a_values_t->flat<T>();

    switch (ndims) {
#define NDIMS_CASE(N)                                                     \
  case N: {                                                               \
    auto out_tensor = out_t->tensor<T, N>();                              \
    out_tensor.device(ctx->eigen_device<Device>()) = b->tensor<T, N>();   \
    const Index result =                                                  \
        functor::ScatterNdFunctor<Device, T, Index, N,                    \
                                  scatter_op::UpdateOp::ADD>()(           \
            ctx->eigen_device<Device>(), a_indices_mat, a_values_flat,    \
            out_tensor);                                                  \
    OP_REQUIRES(                                                          \
        ctx, result == -1,                                                \
        errors::InvalidArgument(                                          \
            "Sparse tensor has some invalid index on dimension ", result, \
            "; dense tensor shape: ", b->shape().DebugString()));         \
  } break;

      NDIMS_CASE(1);
      NDIMS_CASE(2);
      NDIMS_CASE(3);
      NDIMS_CASE(4);
      NDIMS_CASE(5);
      default:
        OP_REQUIRES(
            ctx, false,
            errors::InvalidArgument("Only tensors with ranks between 1 and 5 "
                                    "are currently supported.  Tensor rank: ",
                                    ndims));
#undef NDIMS_CASE
    }
  }
};

namespace functor {
template <typename T, typename Index, int NDIMS>
struct ScatterNdFunctor<CPUDevice, T, Index, NDIMS, scatter_op::UpdateOp::ADD> {
  Index operator()(const CPUDevice &d,
                   typename TTypes<Index>::ConstMatrix indices,
                   typename TTypes<T>::ConstFlat updates,
                   typename TTypes<T, NDIMS>::Tensor out) {
    Eigen::array<Eigen::DenseIndex, NDIMS> idx;
    const int num_nnz = static_cast<int>(indices.dimension(0));
    for (int i = 0; i < num_nnz; ++i) {
      for (int d = 0; d < NDIMS; ++d) {
        idx[d] = internal::SubtleMustCopy(indices(i, d));
        if (!FastBoundsCheck(idx[d], out.dimension(d))) {
          return d;  // on failure: d nonnegative
        }
      }
      out(idx) += updates(i);
    }
    return -1;  // on success
  }
};
}  // namespace functor

#define REGISTER_KERNELS_CPU(TypeT, TypeIndex)                        \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorDenseAdd")                \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<TypeT>("T")             \
                              .TypeConstraint<TypeIndex>("Tindices"), \
                          SparseTensorDenseAddOp<CPUDevice, TypeT, TypeIndex>)

#define REGISTER_KERNELS(T)         \
  REGISTER_KERNELS_CPU(T, int64_t); \
  REGISTER_KERNELS_CPU(T, int32)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
#undef REGISTER_KERNELS_CPU
}  // namespace tensorflow
