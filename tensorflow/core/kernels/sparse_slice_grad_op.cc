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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_grad_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_grad_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_grad_opDTcc() {
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

#include "tensorflow/core/kernels/sparse_slice_grad_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename T>
struct SparseSliceGradFunctor<CPUDevice, T> {
  void operator()(OpKernelContext *ctx,
                  typename TTypes<T>::ConstFlat backprop_val_grad,
                  typename TTypes<int64_t>::ConstMatrix input_indices_mat,
                  typename TTypes<int64_t>::ConstFlat input_start_flat,
                  typename TTypes<int64_t>::ConstMatrix output_indices_mat,
                  typename TTypes<T>::Flat val_grad) const {
    const int64_t input_nnz = input_indices_mat.dimension(0);
    const int num_dims = input_indices_mat.dimension(1);

    T *val_grad_flat = val_grad.data();
    const T *backprop_val_grad_flat = backprop_val_grad.data();
    memset(val_grad_flat, 0, sizeof(T) * input_nnz);

    // Fill gradients for position where indices of input and output are same.
    int64_t j = 0;
    for (int64_t i = 0; i < input_nnz && j < backprop_val_grad.dimension(0);
         ++i) {
      bool is_same = true;
      for (int d = 0; d < num_dims; ++d) {
        const int64_t a = input_indices_mat(i, d);
        const int64_t b = output_indices_mat(j, d);
        const int64_t offset = input_start_flat(d);
        if (a != b + offset) {
          is_same = false;
          break;
        }
      }
      if (is_same) {
        val_grad_flat[i] = backprop_val_grad_flat[j];
        ++j;
      }
    }
    OP_REQUIRES(
        ctx, backprop_val_grad.dimension(0) == j,
        errors::Internal("Elements of backprop_val_grad aren't all propagated. "
                         "Num elements:",
                         backprop_val_grad.dimension(0), ", used: ", j));
  }
};

}  // namespace functor

template <typename Device, typename T>
class SparseSliceGradOp : public OpKernel {
 public:
  explicit SparseSliceGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_grad_opDTcc mht_0(mht_0_v, 246, "", "./tensorflow/core/kernels/sparse_slice_grad_op.cc", "SparseSliceGradOp");
}

  void Compute(OpKernelContext *ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_grad_opDTcc mht_1(mht_1_v, 251, "", "./tensorflow/core/kernels/sparse_slice_grad_op.cc", "Compute");

    const Tensor *backprop_val_grad, *input_indices, *output_indices, *input_start;
    OP_REQUIRES_OK(ctx, ctx->input("backprop_val_grad", &backprop_val_grad));
    OP_REQUIRES_OK(ctx, ctx->input("input_indices", &input_indices));
    OP_REQUIRES_OK(ctx, ctx->input("input_start", &input_start));
    OP_REQUIRES_OK(ctx, ctx->input("output_indices", &output_indices));

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsMatrix(input_indices->shape()) &&
                    TensorShapeUtils::IsMatrix(output_indices->shape()),
                errors::InvalidArgument(
                    "Input and output indices should be matrices "
                    "but received shapes: ",
                    input_indices->shape().DebugString(), " and ",
                    output_indices->shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(backprop_val_grad->shape()),
        errors::InvalidArgument(
            "Input backprop_val_grad should be a vector but received shape: ",
            backprop_val_grad->shape().DebugString()));
    OP_REQUIRES(
        ctx,
        input_indices->dim_size(1) == output_indices->dim_size(1),
        errors::InvalidArgument("The input and output should have the same "
                                "ndims: got: ", input_indices->dim_size(1), " and ",
                                output_indices->dim_size(1)));
    OP_REQUIRES(
        ctx, output_indices->dim_size(0) <= input_indices->dim_size(0),
        errors::InvalidArgument("# rows of output_indices should be not greater "
                                "than of input_indices, got ",
                                output_indices->dim_size(0), " and ",
                                input_indices->dim_size(0)));
    OP_REQUIRES(
        ctx, backprop_val_grad->NumElements() == output_indices->dim_size(0),
        errors::InvalidArgument("# elements of backprop_val_grad and # rows of "
                                "output_indices should match (#nnz of sum): got ",
                                backprop_val_grad->NumElements(), " and ",
                                output_indices->dim_size(0)));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_start->shape()),
                errors::InvalidArgument(
                    "The input_start should be a vector but received shape ",
                    input_start->shape().DebugString()));

    const int num_dims = input_indices->dim_size(1);
    OP_REQUIRES(ctx, num_dims == input_start->NumElements(),
                errors::InvalidArgument(
                    "Expected input_start to be a vector of length ", num_dims,
                    " but got length ", input_start->NumElements()));

    const int64_t input_nnz = input_indices->dim_size(0);

    Tensor *val_grad;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({input_nnz}), &val_grad));

    if (input_nnz == 0) return;

    functor::SparseSliceGradFunctor<Device, T>()(
        ctx, backprop_val_grad->flat<T>(), input_indices->matrix<int64_t>(),
        input_start->flat<int64_t>(), output_indices->matrix<int64_t>(),
        val_grad->flat<T>());
  }
};

#define REGISTER_KERNELS(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseSliceGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseSliceGradOp<CPUDevice, type>)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
