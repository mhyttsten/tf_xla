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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_opDTcc() {
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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

template <typename T, typename Treal>
class SparseAddOp : public OpKernel {
 public:
  explicit SparseAddOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_opDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/kernels/sparse_add_op.cc", "SparseAddOp");
}

  void Compute(OpKernelContext *ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_opDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/kernels/sparse_add_op.cc", "Compute");

    // (0) validations
    const Tensor *a_indices, *b_indices, *a_values_t, *b_values_t, *a_shape,
        *b_shape, *thresh_t;

    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices));
    OP_REQUIRES_OK(ctx, ctx->input("b_indices", &b_indices));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsMatrix(a_indices->shape()) &&
                    TensorShapeUtils::IsMatrix(b_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be matrices but received shapes: ",
                    a_indices->shape().DebugString(), " and ",
                    b_indices->shape().DebugString()));
    const int64_t a_nnz = a_indices->dim_size(0);
    const int64_t b_nnz = b_indices->dim_size(0);
    const int num_dims = a_indices->dim_size(1);
    OP_REQUIRES(ctx, b_indices->dim_size(1) == num_dims,
                errors::InvalidArgument(
                    "Input indices must have the same dimension, got ",
                    num_dims, " and ", b_indices->dim_size(1)));

    OP_REQUIRES_OK(ctx, ctx->input("a_values", &a_values_t));
    OP_REQUIRES_OK(ctx, ctx->input("b_values", &b_values_t));

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(a_values_t->shape()) &&
                    TensorShapeUtils::IsVector(b_values_t->shape()),
                errors::InvalidArgument(
                    "Input values should be vectors but received shapes: ",
                    a_values_t->shape().DebugString(), " and ",
                    b_values_t->shape().DebugString()));
    auto a_values = ctx->input(1).vec<T>();
    auto b_values = ctx->input(4).vec<T>();
    OP_REQUIRES(
        ctx, a_values.size() == a_nnz && b_values.size() == b_nnz,
        errors::InvalidArgument("Expected ", a_nnz, " and ", b_nnz,
                                " non-empty input values, got ",
                                a_values.size(), " and ", b_values.size()));

    OP_REQUIRES_OK(ctx, ctx->input("a_shape", &a_shape));
    OP_REQUIRES_OK(ctx, ctx->input("b_shape", &b_shape));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(a_shape->shape()) &&
                    TensorShapeUtils::IsVector(b_shape->shape()),
                errors::InvalidArgument(
                    "Input shapes should be a vector but received shapes ",
                    a_shape->shape().DebugString(), " and ",
                    b_shape->shape().DebugString()));
    OP_REQUIRES(
        ctx, a_shape->NumElements() == num_dims,
        errors::InvalidArgument("Second dimension of a_indices and length of "
                                "a_shape must match, got ",
                                num_dims, " and ", a_shape->NumElements()));
    OP_REQUIRES(ctx, num_dims > 0,
                errors::InvalidArgument("Tesors must not be empty"));
    OP_REQUIRES(
        ctx, a_shape->IsSameSize(*b_shape),
        errors::InvalidArgument(
            "Operands do not have the same ranks; got shapes: ",
            a_shape->SummarizeValue(10), " and ", b_shape->SummarizeValue(10)));
    const auto a_shape_flat = a_shape->flat<int64_t>();
    const auto b_shape_flat = b_shape->flat<int64_t>();
    for (int i = 0; i < a_shape->NumElements(); ++i) {
      OP_REQUIRES(ctx, a_shape_flat(i) == b_shape_flat(i),
                  errors::InvalidArgument(
                      "Operands' shapes do not match: got ", a_shape_flat(i),
                      " and ", b_shape_flat(i), " for dimension ", i));
    }

    OP_REQUIRES_OK(ctx, ctx->input("thresh", &thresh_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(thresh_t->shape()),
                errors::InvalidArgument(
                    "The magnitude threshold must be a scalar: got shape ",
                    thresh_t->shape().DebugString()));
    // std::abs() so that it works for complex{64,128} values as well
    const Treal thresh = thresh_t->scalar<Treal>()();

    // (1) do a pass over inputs, and append values and indices to vectors
    auto a_indices_mat = a_indices->matrix<int64_t>();
    auto b_indices_mat = b_indices->matrix<int64_t>();
    std::vector<std::pair<bool, int64>> entries_to_copy;  // from_a?, idx
    entries_to_copy.reserve(a_nnz + b_nnz);
    std::vector<T> out_values;

    // The input and output sparse tensors are assumed to be ordered along
    // increasing dimension number.
    int64_t i = 0, j = 0;
    T s;
    while (i < a_nnz && j < b_nnz) {
      switch (sparse::DimComparator::cmp(a_indices_mat, b_indices_mat, i, j,
                                         num_dims)) {
        case -1:
          entries_to_copy.emplace_back(true, i);
          out_values.push_back(a_values(i));
          ++i;
          break;
        case 0:
          s = a_values(i) + b_values(j);
          if (thresh <= std::abs(s)) {
            entries_to_copy.emplace_back(true, i);
            out_values.push_back(s);
          }
          ++i;
          ++j;
          break;
        case 1:
          entries_to_copy.emplace_back(false, j);
          out_values.push_back(b_values(j));
          ++j;
          break;
      }
    }

#define HANDLE_LEFTOVERS(A_OR_B, IDX, IS_A)     \
  while (IDX < A_OR_B##_nnz) {                  \
    entries_to_copy.emplace_back(IS_A, IDX);    \
    out_values.push_back(A_OR_B##_values(IDX)); \
    ++IDX;                                      \
  }

    // at most one of these calls appends new values
    HANDLE_LEFTOVERS(a, i, true);
    HANDLE_LEFTOVERS(b, j, false);
#undef HANDLE_LEFTOVERS

    // (2) allocate and fill output tensors
    const int64_t sum_nnz = out_values.size();
    Tensor *out_indices_t, *out_values_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({sum_nnz, num_dims}),
                                        &out_indices_t));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({sum_nnz}), &out_values_t));
    auto out_indices_mat = out_indices_t->matrix<int64_t>();
    auto out_values_flat = out_values_t->vec<T>();

    for (i = 0; i < sum_nnz; ++i) {
      const bool from_a = entries_to_copy[i].first;
      const int64_t idx = entries_to_copy[i].second;
      out_indices_mat.chip<0>(i) =
          from_a ? a_indices_mat.chip<0>(idx) : b_indices_mat.chip<0>(idx);
    }
    if (sum_nnz > 0) {
      std::copy_n(out_values.begin(), sum_nnz, &out_values_flat(0));
    }
    ctx->set_output(2, *a_shape);
  }
};

#define REGISTER_KERNELS(type, thresh_type)                           \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseAddOp<type, thresh_type>)

// The list below is equivalent to TF_CALL_REAL_NUMBER_TYPES, minus uint8.  This
// is because std::abs() on uint8 does not compile.
REGISTER_KERNELS(float, float);
REGISTER_KERNELS(double, double);
REGISTER_KERNELS(int64_t, int64);
REGISTER_KERNELS(int32, int32);
REGISTER_KERNELS(int16, int16);
REGISTER_KERNELS(int8, int8);
REGISTER_KERNELS(complex64, float);
REGISTER_KERNELS(complex128, double);
#undef REGISTER_KERNELS
}  // namespace tensorflow
