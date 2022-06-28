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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_sparse_binary_op_sharedDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_sparse_binary_op_sharedDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_sparse_binary_op_sharedDTcc() {
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

// SparseSparseBinaryOpShared is the shared code for binary coefficient-wise
// (cwise) operations of the following form:
//
//   sparse_t <binary cwise op> sparse_t -> new sparse_t
//
// The output SparseTensor may store up to "a_nnz + b_nnz" elements.

// IMPLEMENTATION DETAILS (not part of the interface specification).
//
// This kernel implements the "union" semantics on the non-zeros: namely, any
// non-zero from either side participate in the calculations, and any resultant
// zeros will NOT be excluded from the output storage.
//
// (In the future, we could always add a pruning op the prunes away the zeros,
// if desirable.)

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
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {
// Unions the sparse indices and outputs corresponding values: namely, if a
// non-zero appear in one side, it will participate in the calculation, where
// the counterpart on the other side is either a value or an implicit zero.
//
// On exit, outputs the augmented values in "{a,b}_augmented_values", and fills
// "entries_to_copy" with "(from_a?, index)" pairs.  All three vectors have the
// same size.
//
// The input and output sparse tensors are assumed ordered in the canonical
// row-major order.
template <typename T>
void UnionSparseIndicesAndValues(
    typename TTypes<int64_t>::ConstMatrix a_indices_mat,
    typename TTypes<T>::ConstFlat a_values, int64_t a_nnz,
    typename TTypes<int64_t>::ConstMatrix b_indices_mat,
    typename TTypes<T>::ConstFlat b_values, int64_t b_nnz, int num_dims,
    std::vector<T> *a_augmented_values, std::vector<T> *b_augmented_values,
    std::vector<std::pair<bool, int64>> *entries_to_copy) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_sparse_binary_op_sharedDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/kernels/sparse_sparse_binary_op_shared.cc", "UnionSparseIndicesAndValues");

  entries_to_copy->reserve(a_nnz + b_nnz);
  a_augmented_values->reserve(a_nnz);
  b_augmented_values->reserve(b_nnz);

  int64_t i = 0, j = 0;
  const T kZero = T(0);
  while (i < a_nnz && j < b_nnz) {
    switch (sparse::DimComparator::cmp(a_indices_mat, b_indices_mat, i, j,
                                       num_dims)) {
      case -1:
        entries_to_copy->emplace_back(true, i);
        a_augmented_values->push_back(a_values(i));
        b_augmented_values->push_back(kZero);
        ++i;
        break;
      case 0:
        entries_to_copy->emplace_back(true, i);
        a_augmented_values->push_back(a_values(i));
        b_augmented_values->push_back(b_values(j));
        ++i;
        ++j;
        break;
      case 1:
        entries_to_copy->emplace_back(false, j);
        a_augmented_values->push_back(kZero);
        b_augmented_values->push_back(b_values(j));
        ++j;
        break;
    }
  }
  // Handles leftovers; at most one loop runs.
  while (i < a_nnz) {
    entries_to_copy->emplace_back(/* is_a */ true, i);
    a_augmented_values->push_back(a_values(i++));
    b_augmented_values->push_back(kZero);
  }
  while (j < b_nnz) {
    entries_to_copy->emplace_back(/* is_a */ false, j);
    a_augmented_values->push_back(kZero);
    b_augmented_values->push_back(b_values(j++));
  }
}
}  // anonymous namespace

// Device: CPUDevice.  GPU kernel is not supported currently.
// T: dtype of the SparseTensor's.
// Functor: binary cwise operation to perform on the corresponding operand
// values.  See cwise_ops.h for a list of possible functors to register with.
template <typename Device, typename T, typename Functor>
class SparseSparseBinaryOpShared : public OpKernel {
 public:
  explicit SparseSparseBinaryOpShared(OpKernelConstruction *ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_sparse_binary_op_sharedDTcc mht_1(mht_1_v, 293, "", "./tensorflow/core/kernels/sparse_sparse_binary_op_shared.cc", "SparseSparseBinaryOpShared");
}

  void Compute(OpKernelContext *ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_sparse_binary_op_sharedDTcc mht_2(mht_2_v, 298, "", "./tensorflow/core/kernels/sparse_sparse_binary_op_shared.cc", "Compute");

    const Tensor *a_indices_t, *a_values_t, *a_shape_t, *b_indices_t,
        *b_values_t, *b_shape_t;
    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("a_values", &a_values_t));
    OP_REQUIRES_OK(ctx, ctx->input("a_shape", &a_shape_t));
    OP_REQUIRES_OK(ctx, ctx->input("b_indices", &b_indices_t));
    OP_REQUIRES_OK(ctx, ctx->input("b_values", &b_values_t));
    OP_REQUIRES_OK(ctx, ctx->input("b_shape", &b_shape_t));

    // Validations.
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsMatrix(a_indices_t->shape()) &&
            TensorShapeUtils::IsMatrix(b_indices_t->shape()),
        errors::InvalidArgument("Inputs a_indices and b_indices should be "
                                "matrices but received shapes: ",
                                a_indices_t->shape().DebugString(), ", ",
                                b_indices_t->shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(a_values_t->shape()) &&
                    TensorShapeUtils::IsVector(b_values_t->shape()),
                errors::InvalidArgument(
                    "Inputs a_values and b_values should be vectors "
                    "but received shapes: ",
                    a_values_t->shape().DebugString(), " and ",
                    b_values_t->shape().DebugString()));

    const int64_t a_nnz = a_indices_t->dim_size(0);
    const int64_t b_nnz = b_indices_t->dim_size(0);

    const auto a_values = a_values_t->vec<T>();
    const auto b_values = b_values_t->vec<T>();

    OP_REQUIRES(
        ctx, a_values.size() == a_nnz && b_values.size() == b_nnz,
        errors::InvalidArgument("Expected ", a_nnz, " and ", b_nnz,
                                " non-empty input values, got ",
                                a_values.size(), " and ", b_values.size()));

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(a_shape_t->shape()) &&
                    TensorShapeUtils::IsVector(b_shape_t->shape()),
                errors::InvalidArgument(
                    "Input shapes should be a vector but received shapes ",
                    a_shape_t->shape().DebugString(), " and ",
                    b_shape_t->shape().DebugString()));
    const int num_dims = a_indices_t->dim_size(1);
    OP_REQUIRES(
        ctx, a_shape_t->NumElements() == num_dims,
        errors::InvalidArgument("Second dimension of a_indices and length of "
                                "a_shape must match, got ",
                                num_dims, " and ", a_shape_t->NumElements()));
    OP_REQUIRES(ctx, num_dims > 0,
                errors::InvalidArgument("Tensors must not be empty"));
    OP_REQUIRES(ctx, a_shape_t->IsSameSize(*b_shape_t),
                errors::InvalidArgument(
                    "Operands do not have the same ranks; got shapes: ",
                    a_shape_t->SummarizeValue(10), " and ",
                    b_shape_t->SummarizeValue(10)));
    const auto a_shape = a_shape_t->flat<int64_t>();
    const auto b_shape = b_shape_t->flat<int64_t>();
    for (int i = 0; i < a_shape_t->NumElements(); ++i) {
      OP_REQUIRES(ctx, a_shape(i) == b_shape(i),
                  errors::InvalidArgument("Operands' shapes do not match: got ",
                                          a_shape(i), " and ", b_shape(i),
                                          " for dimension ", i));
    }

    const auto a_indices_mat = a_indices_t->matrix<int64_t>();
    const auto b_indices_mat = b_indices_t->matrix<int64_t>();
    std::vector<T> a_augmented_values, b_augmented_values;
    std::vector<std::pair<bool, int64>> entries_to_copy;  // from_a?, idx
    UnionSparseIndicesAndValues(a_indices_mat, a_values, a_nnz, b_indices_mat,
                                b_values, b_nnz, num_dims, &a_augmented_values,
                                &b_augmented_values, &entries_to_copy);

    // Allocates and fills output tensors.
    const int64_t sum_nnz = a_augmented_values.size();
    Tensor *output_indices_t, *output_values_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({sum_nnz, num_dims}),
                                        &output_indices_t));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({sum_nnz}), &output_values_t));
    auto output_indices_mat = output_indices_t->matrix<int64_t>();

    for (int64_t i = 0; i < sum_nnz; ++i) {
      const bool from_a = entries_to_copy[i].first;
      const int64_t idx = entries_to_copy[i].second;
      output_indices_mat.chip<0>(i) =
          from_a ? a_indices_mat.chip<0>(idx) : b_indices_mat.chip<0>(idx);
    }

    // Performs the functor operation using Eigen.
    //
    // Note that the two stack-allocated std::vector's may not be aligned. Using
    // allocate_temp() would've given us aligned storage, but we do not know
    // their sizes in advance, so we couldn't use allocate_temp() anyway.
    //
    // TODO(zongheng): measure if it's worthwhile to somehow force alignment.
    using UnalignedTensorMap =
        Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                         Eigen::Unaligned>;
    auto a_augmented_values_t =
        UnalignedTensorMap(a_augmented_values.data(), sum_nnz);
    auto b_augmented_values_t =
        UnalignedTensorMap(b_augmented_values.data(), sum_nnz);
    output_values_t->flat<T>().device(ctx->eigen_device<Device>()) =
        a_augmented_values_t.binaryExpr(b_augmented_values_t,
                                        typename Functor::func());
  }
};

#define REGISTER_KERNELS(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseSparseMinimum").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseSparseBinaryOpShared<CPUDevice, T, functor::minimum<T>>)         \
                                                                             \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseSparseMaximum").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseSparseBinaryOpShared<CPUDevice, T, functor::maximum<T>>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow
