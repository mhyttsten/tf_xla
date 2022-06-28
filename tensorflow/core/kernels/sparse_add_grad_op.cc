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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_grad_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_grad_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_grad_opDTcc() {
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
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

template <typename T>
class SparseAddGradOp : public OpKernel {
 public:
  explicit SparseAddGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_grad_opDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/sparse_add_grad_op.cc", "SparseAddGradOp");
}

  void Compute(OpKernelContext *ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_add_grad_opDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/kernels/sparse_add_grad_op.cc", "Compute");

    // Gradient for op: SparseAdd(a, b) == sum.
    const Tensor *backprop_val_grad, *a_indices, *b_indices, *sum_indices;
    OP_REQUIRES_OK(ctx, ctx->input("backprop_val_grad", &backprop_val_grad));
    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices));
    OP_REQUIRES_OK(ctx, ctx->input("b_indices", &b_indices));
    OP_REQUIRES_OK(ctx, ctx->input("sum_indices", &sum_indices));

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsMatrix(a_indices->shape()) &&
                    TensorShapeUtils::IsMatrix(b_indices->shape()) &&
                    TensorShapeUtils::IsMatrix(sum_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be matrices but received shapes: ",
                    a_indices->shape().DebugString(), " and ",
                    b_indices->shape().DebugString(), " and ",
                    sum_indices->shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(backprop_val_grad->shape()),
        errors::InvalidArgument(
            "Input backprop_val_grad should be a vector but received shape: ",
            backprop_val_grad->shape().DebugString()));
    OP_REQUIRES(
        ctx,
        a_indices->dim_size(1) == b_indices->dim_size(1) &&
            b_indices->dim_size(1) == sum_indices->dim_size(1),
        errors::InvalidArgument("The densified operands should have the same "
                                "ndims; for A, B, sum got: ",
                                a_indices->dim_size(1), b_indices->dim_size(1),
                                sum_indices->dim_size(1)));
    OP_REQUIRES(
        ctx, backprop_val_grad->NumElements() == sum_indices->dim_size(0),
        errors::InvalidArgument("# elements of backprop_val_grad and # rows of "
                                "sum_indices should match (#nnz of sum): got ",
                                backprop_val_grad->NumElements(), " and ",
                                sum_indices->dim_size(0)));

    const int num_dims = a_indices->dim_size(1);
    const int64_t a_nnz = a_indices->dim_size(0);
    const int64_t b_nnz = b_indices->dim_size(0);
    const int64_t sum_nnz = backprop_val_grad->NumElements();

    const auto a_indices_mat = a_indices->matrix<int64_t>();
    const auto b_indices_mat = b_indices->matrix<int64_t>();
    const auto sum_indices_mat = sum_indices->matrix<int64_t>();

    Tensor *a_val_grad, *b_val_grad;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({a_nnz}), &a_val_grad));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({b_nnz}), &b_val_grad));

    T *a_val_grad_flat = a_val_grad->flat<T>().data();
    T *b_val_grad_flat = b_val_grad->flat<T>().data();
    const T *backprop_val_grad_flat = backprop_val_grad->flat<T>().data();
    memset(a_val_grad_flat, 0, sizeof(T) * a_nnz);
    memset(b_val_grad_flat, 0, sizeof(T) * b_nnz);

#define COMPARE(a_or_b, idx)                                                \
  switch (sparse::DimComparator::cmp(a_or_b##_indices_mat, sum_indices_mat, \
                                     idx, k, num_dims)) {                   \
    case 0:                                                                 \
      a_or_b##_val_grad_flat[idx] = backprop_val_grad_flat[k];              \
      ++idx;                                                                \
      break;                                                                \
    case -1:                                                                \
      ++idx;                                                                \
      a_or_b##_idx_geq = false;                                             \
      break;                                                                \
    case 1:                                                                 \
      break;                                                                \
  }

    // Set-intersect the indices; fill in grads for positions in the
    // intersection.
    int64_t i = 0, j = 0, k = 0;
    bool a_idx_geq, b_idx_geq;
    while (i < a_nnz && j < b_nnz && k < sum_nnz) {
      a_idx_geq = b_idx_geq = true;
      COMPARE(a, i);
      COMPARE(b, j);
      // increment pointer into sum_indices iff both the current A, B indices >=
      // the current sum index.
      if (a_idx_geq && b_idx_geq) ++k;
    }

    // at most one loop below will run
    while (i < a_nnz && k < sum_nnz) {
      a_idx_geq = true;
      COMPARE(a, i);
      if (a_idx_geq) ++k;
    }
    while (j < b_nnz && k < sum_nnz) {
      b_idx_geq = true;
      COMPARE(b, j);
      if (b_idx_geq) ++k;
    }
#undef COMPARE
  }
};

#define REGISTER_KERNELS(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseAddGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseAddGradOp<type>)

// This op should work for any T that SparseAdd is registered with.
REGISTER_KERNELS(float);
REGISTER_KERNELS(double);
REGISTER_KERNELS(int64_t);
REGISTER_KERNELS(int32);
REGISTER_KERNELS(int16);
REGISTER_KERNELS(int8);
REGISTER_KERNELS(complex64);
REGISTER_KERNELS(complex128);
#undef REGISTER_KERNELS
}  // namespace tensorflow
