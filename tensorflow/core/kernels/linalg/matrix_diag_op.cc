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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/linalg/matrix_diag_op.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class MatrixDiagPartOp : public OpKernel {
 public:
  explicit MatrixDiagPartOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/kernels/linalg/matrix_diag_op.cc", "MatrixDiagPartOp");

    // MatrixDiagPartV3-specific.
    if (context->HasAttr("align")) {
      functor::ReadAlignment(context, &left_align_superdiagonal_,
                             &left_align_subdiagonal_);
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/kernels/linalg/matrix_diag_op.cc", "Compute");

    const Tensor& input = context->input(0);

    // MatrixDiagPart and MatrixDiagPartV2 both use this OpKernel.
    // MatrixDiagPart only has one input, so we have to check the number of
    // inputs before reading additional parameters in MatrixDiagV2.
    int32_t lower_diag_index = 0;
    int32_t upper_diag_index = 0;
    T padding_value(0);

    // MatrixDiagPartV2-specific.
    if (context->num_inputs() > kNumV1Inputs) {
      auto& diag_index = context->input(1);
      OP_REQUIRES(context,
                  TensorShapeUtils::IsScalar(diag_index.shape()) ||
                      TensorShapeUtils::IsVector(diag_index.shape()),
                  errors::InvalidArgument(
                      "diag_index must be a scalar or vector, received shape: ",
                      diag_index.shape().DebugString()));
      OP_REQUIRES(context, diag_index.NumElements() > 0,
                  errors::InvalidArgument(
                      "Expected diag_index to have at least 1 element"));
      lower_diag_index = diag_index.flat<int32>()(0);
      upper_diag_index = lower_diag_index;
      if (TensorShapeUtils::IsVector(diag_index.shape())) {
        auto diag_index_size = diag_index.dim_size(0);
        OP_REQUIRES(
            context, 0 < diag_index_size && diag_index_size <= 2,
            errors::InvalidArgument(
                "diag_index must have only one or two elements, received ",
                diag_index_size, " elements."));
        if (diag_index_size > 1) {
          upper_diag_index = diag_index.flat<int32>()(1);
        }
      }
      const Tensor& padding_in = context->input(2);
      OP_REQUIRES(context, padding_in.NumElements() == 1,
                  errors::InvalidArgument("Padding must be scalar."));
      padding_value = padding_in.flat<T>()(0);
    }
    const TensorShape& input_shape = input.shape();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));

    // Make sure lower_diag_index and upper_diag_index is valid.
    const int rank = input_shape.dims();
    const Eigen::Index num_rows = input_shape.dim_size(rank - 2);
    const Eigen::Index num_cols = input_shape.dim_size(rank - 1);
    OP_REQUIRES(  // Checks lower_diag_index == 0 for when matrix shape = 0.
        context,
        (-num_rows < lower_diag_index && lower_diag_index < num_cols) ||
            lower_diag_index == 0,
        errors::InvalidArgument(
            "lower_diag_index is out of bound: ", lower_diag_index,
            ". It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(context,
                (-num_rows < upper_diag_index && upper_diag_index < num_cols) ||
                    upper_diag_index == 0,
                errors::InvalidArgument(
                    "upper_diag_index is out of bound: ", upper_diag_index,
                    " It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(
        context, lower_diag_index <= upper_diag_index,
        errors::InvalidArgument(
            "lower_diag_index must not be larger than upper_diag_index: ",
            lower_diag_index, " > ", upper_diag_index));

    TensorShape output_shape;
    for (int i = 0; i < rank - 2; ++i) {
      output_shape.AddDim(input_shape.dim_size(i));
    }
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    if (num_diags > 1) output_shape.AddDim(num_diags);
    const int32_t max_diag_len =
        std::min(num_rows + std::min(upper_diag_index, 0),
                 num_cols - std::max(lower_diag_index, 0));
    output_shape.AddDim(max_diag_len);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_reshaped = output->flat<T>();
    auto input_reshaped = input.flat_inner_dims<T, 3>();
    functor::MatrixDiagPart<Device, T>::Compute(
        context, context->eigen_device<Device>(), input_reshaped,
        output_reshaped, lower_diag_index, upper_diag_index, max_diag_len,
        padding_value, left_align_superdiagonal_, left_align_subdiagonal_);
  }

 private:
  bool left_align_superdiagonal_ = true;
  bool left_align_subdiagonal_ = true;
  static constexpr int kNumV1Inputs = 1;
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixDiagPartOp);
};

template <typename Device, typename T>
class MatrixDiagOp : public OpKernel {
 public:
  explicit MatrixDiagOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc mht_2(mht_2_v, 334, "", "./tensorflow/core/kernels/linalg/matrix_diag_op.cc", "MatrixDiagOp");

    // MatrixDiagV3-specific.
    if (context->HasAttr("align")) {
      functor::ReadAlignment(context, &left_align_superdiagonal_,
                             &left_align_subdiagonal_);
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc mht_3(mht_3_v, 345, "", "./tensorflow/core/kernels/linalg/matrix_diag_op.cc", "Compute");

    const Tensor& diagonal = context->input(0);

    // MatrixDiag and MatrixDiagV2 both use this OpKernel. MatrixDiag only has
    // one input, so we have to check the number of inputs before reading
    // additional parameters in MatrixDiagV2.
    int32_t lower_diag_index = 0;
    int32_t upper_diag_index = 0;
    int32_t num_rows = -1;
    int32_t num_cols = -1;
    T padding_value(0);

    // MatrixDiagOpV2-specific.
    if (context->num_inputs() > kNumV1Inputs) {
      auto& diag_index = context->input(1);
      OP_REQUIRES(context,
                  TensorShapeUtils::IsScalar(diag_index.shape()) ||
                      TensorShapeUtils::IsVector(diag_index.shape()),
                  errors::InvalidArgument(
                      "diag_index must be a scalar or vector, received shape: ",
                      diag_index.shape().DebugString()));
      OP_REQUIRES(context, diag_index.NumElements() > 0,
                  errors::InvalidArgument(
                      "Expected diag_index to have at least 1 element"));
      lower_diag_index = diag_index.flat<int32>()(0);
      upper_diag_index = lower_diag_index;
      if (TensorShapeUtils::IsVector(diag_index.shape())) {
        auto diag_index_size = diag_index.dim_size(0);
        OP_REQUIRES(
            context, 0 < diag_index_size && diag_index_size <= 2,
            errors::InvalidArgument(
                "diag_index must have only one or two elements, received ",
                diag_index_size, " elements."));
        if (diag_index_size > 1) {
          upper_diag_index = diag_index.flat<int32>()(1);
        }
      }

      auto& num_rows_tensor = context->input(2);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_rows_tensor.shape()),
                  errors::InvalidArgument("num_rows must be a scalar"));
      num_rows = num_rows_tensor.flat<int32>()(0);

      auto& num_cols_tensor = context->input(3);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_cols_tensor.shape()),
                  errors::InvalidArgument("num_cols must be a scalar"));
      num_cols = num_cols_tensor.flat<int32>()(0);

      auto& padding_value_tensor = context->input(4);
      OP_REQUIRES(context,
                  TensorShapeUtils::IsScalar(padding_value_tensor.shape()),
                  errors::InvalidArgument("padding_value must be a scalar"));
      padding_value = padding_value_tensor.flat<T>()(0);
    }

    // Size validations.
    const TensorShape& diagonal_shape = diagonal.shape();
    const int diag_rank = diagonal_shape.dims();
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(diagonal_shape),
                errors::InvalidArgument(
                    "diagonal must be at least 1-dim, received shape: ",
                    diagonal.shape().DebugString()));
    OP_REQUIRES(
        context, lower_diag_index <= upper_diag_index,
        errors::InvalidArgument(
            "lower_diag_index must not be larger than upper_diag_index: ",
            lower_diag_index, " > ", upper_diag_index));
    OP_REQUIRES(context,
                lower_diag_index == upper_diag_index ||
                    diagonal_shape.dim_size(diag_rank - 2) == num_diags,
                errors::InvalidArgument(
                    "The number of diagonals provided in the input does not "
                    "match the lower_diag_index and upper_diag_index range."));

    const Eigen::Index max_diag_len = diagonal_shape.dim_size(diag_rank - 1);
    const int32_t min_num_rows = max_diag_len - std::min(upper_diag_index, 0);
    const int32_t min_num_cols = max_diag_len + std::max(lower_diag_index, 0);
    OP_REQUIRES(context, num_rows == -1 || num_rows >= min_num_rows,
                errors::InvalidArgument("The number of rows is too small."));
    OP_REQUIRES(context, num_cols == -1 || num_cols >= min_num_cols,
                errors::InvalidArgument("The number of columns is too small."));

    // If both num_rows and num_cols are unknown, assume that output is square.
    // Otherwise, use smallest possible values.
    if (num_rows == -1 && num_cols == -1) {
      num_rows = std::max(min_num_rows, min_num_cols);
      num_cols = num_rows;
    } else if (num_rows == -1) {
      num_rows = min_num_rows;
    } else if (num_cols == -1) {
      num_cols = min_num_cols;
    }
    OP_REQUIRES(context, num_rows == min_num_rows || num_cols == min_num_cols,
                errors::InvalidArgument(
                    "The number of rows or columns is not consistent with "
                    "the specified d_lower, d_upper, and diagonal."));

    TensorShape output_shape = diagonal_shape;
    if (num_diags == 1) {  // Output has rank `rank+1`.
      output_shape.set_dim(diag_rank - 1, num_rows);
      output_shape.AddDim(num_cols);
    } else {  // Output has rank `rank`.
      output_shape.set_dim(diag_rank - 2, num_rows);
      output_shape.set_dim(diag_rank - 1, num_cols);
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_reshaped = output->flat_inner_dims<T, 3>();
    auto diag_reshaped = diagonal.flat<T>();
    functor::MatrixDiag<Device, T>::Compute(
        context, context->eigen_device<Device>(), diag_reshaped,
        output_reshaped, lower_diag_index, upper_diag_index, max_diag_len,
        padding_value, left_align_superdiagonal_, left_align_subdiagonal_);
  }

 private:
  bool left_align_superdiagonal_ = true;
  bool left_align_subdiagonal_ = true;
  static constexpr int kNumV1Inputs = 1;
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixDiagOp);
};

#define REGISTER_MATRIX_DIAG(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MatrixDiag").Device(DEVICE_CPU).TypeConstraint<type>("T"),       \
      MatrixDiagOp<CPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MatrixDiagV2").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      MatrixDiagOp<CPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MatrixDiagV3").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      MatrixDiagOp<CPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MatrixDiagPart").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      MatrixDiagPartOp<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MatrixDiagPartV2").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixDiagPartOp<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MatrixDiagPartV3").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixDiagPartOp<CPUDevice, type>);

TF_CALL_POD_TYPES(REGISTER_MATRIX_DIAG);
#undef REGISTER_MATRIX_DIAG

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_DIAG(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("BatchMatrixDiag").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixDiagOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixDiagPart")                       \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<type>("T"),                   \
                          MatrixDiagPartOp<CPUDevice, type>);
TF_CALL_POD_TYPES(REGISTER_BATCH_MATRIX_DIAG);
#undef REGISTER_BATCH_MATRIX_DIAG

// Implementation of the functor specialization for CPU.
namespace functor {

void ReadAlignment(OpKernelConstruction* context,
                   bool* left_align_superdiagonal,
                   bool* left_align_subdiagonal) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc mht_4(mht_4_v, 513, "", "./tensorflow/core/kernels/linalg/matrix_diag_op.cc", "ReadAlignment");

  string align;
  OP_REQUIRES_OK(context, context->GetAttr("align", &align));

  *left_align_superdiagonal = align == "LEFT_LEFT" || align == "LEFT_RIGHT";
  *left_align_subdiagonal = align == "LEFT_LEFT" || align == "RIGHT_LEFT";
}

std::pair<int, int> ComputeDiagLenAndContentOffset(
    int diag_index, int max_diag_len, int num_rows, int num_cols,
    bool left_align_superdiagonal, bool left_align_subdiagonal) {
  const bool left_align = (diag_index >= 0 && left_align_superdiagonal) ||
                          (diag_index <= 0 && left_align_subdiagonal);
  const int diag_len = std::min(num_rows + std::min(0, diag_index),
                                num_cols - std::max(0, diag_index));
  const int content_offset = (left_align) ? 0 : (max_diag_len - diag_len);
  return {diag_len, content_offset};
}

template <typename T>
struct MatrixDiag<CPUDevice, T> {
  static void Compute(OpKernelContext* context, const CPUDevice& device,
                      typename TTypes<T>::ConstTensor& diag,
                      typename TTypes<T, 3>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len, const T padding_value,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc mht_5(mht_5_v, 544, "", "./tensorflow/core/kernels/linalg/matrix_diag_op.cc", "Compute");

    // 10 in cost_per_batch is from existing heuristic.
    // TODO(penporn): Tune for the best constant in cost_per_batch.
    const Eigen::Index num_batches = output.dimension(0);
    const Eigen::Index num_rows = output.dimension(1);
    const Eigen::Index num_cols = output.dimension(2);
    const Eigen::Index cost_per_batch = 10 * num_rows * num_cols;

    auto compute_shard = [&output, &num_rows, &num_cols, &diag,
                          &lower_diag_index, &upper_diag_index, &max_diag_len,
                          &padding_value, &left_align_superdiagonal,
                          &left_align_subdiagonal](Eigen::Index begin,
                                                   Eigen::Index end) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc mht_6(mht_6_v, 559, "", "./tensorflow/core/kernels/linalg/matrix_diag_op.cc", "lambda");

      const int num_diags = upper_diag_index - lower_diag_index + 1;
      const int diag_elements_in_batch = num_diags * max_diag_len;
      Eigen::Index diag_batch_base_index = begin * diag_elements_in_batch;
      for (Eigen::Index batch = begin; batch < end; ++batch) {
        for (Eigen::Index i = 0; i < output.dimension(1); ++i) {
          for (Eigen::Index j = 0; j < output.dimension(2); ++j) {
            const int diag_index = j - i;
            const int diag_index_in_input = upper_diag_index - diag_index;
            int diag_len, content_offset;
            std::tie(diag_len, content_offset) = ComputeDiagLenAndContentOffset(
                diag_index, max_diag_len, num_rows, num_cols,
                left_align_superdiagonal, left_align_subdiagonal);
            const int index_in_the_diagonal =
                j - std::max<Eigen::Index>(diag_index, 0) + content_offset;
            if (lower_diag_index <= diag_index &&
                diag_index <= upper_diag_index) {
              output(batch, i, j) = diag(diag_batch_base_index +
                                         diag_index_in_input * max_diag_len +
                                         index_in_the_diagonal);
            } else {
              output(batch, i, j) = padding_value;
            }
          }
        }
        diag_batch_base_index += diag_elements_in_batch;
      }
    };
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(num_batches, cost_per_batch,
                             std::move(compute_shard));
  }
};

template <typename T>
struct MatrixDiagPart<CPUDevice, T> {
  static void Compute(OpKernelContext* context, const CPUDevice& device,
                      typename TTypes<T, 3>::ConstTensor& input,
                      typename TTypes<T>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len, const T padding_value,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc mht_7(mht_7_v, 606, "", "./tensorflow/core/kernels/linalg/matrix_diag_op.cc", "Compute");

    // 10 in cost_per_batch is from existing heuristic.
    // TODO(penporn): Tune for the best constant in cost_per_batch.
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    const Eigen::Index output_elements_in_batch = num_diags * max_diag_len;
    const Eigen::Index cost_per_batch = 10 * output_elements_in_batch;
    const Eigen::Index num_batches = input.dimension(0);
    const Eigen::Index num_rows = input.dimension(1);
    const Eigen::Index num_cols = input.dimension(2);

    auto compute_shard = [&output, &input, &num_rows, &num_cols,
                          &upper_diag_index, &max_diag_len, &num_diags,
                          &output_elements_in_batch, &padding_value,
                          &left_align_superdiagonal, &left_align_subdiagonal](
                             Eigen::Index begin, Eigen::Index end) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_diag_opDTcc mht_8(mht_8_v, 623, "", "./tensorflow/core/kernels/linalg/matrix_diag_op.cc", "lambda");

      Eigen::Index output_base_index = begin * output_elements_in_batch;
      for (Eigen::Index batch = begin; batch < end; ++batch) {
        for (Eigen::Index m = 0; m < num_diags; ++m) {
          const Eigen::Index diag_index = upper_diag_index - m;
          Eigen::Index y_offset = std::max<Eigen::Index>(0, -diag_index);
          Eigen::Index x_offset = std::max<Eigen::Index>(0, diag_index);
          int diag_len, content_offset;
          std::tie(diag_len, content_offset) = ComputeDiagLenAndContentOffset(
              diag_index, max_diag_len, num_rows, num_cols,
              left_align_superdiagonal, left_align_subdiagonal);

          // Fills the diagonal.
          for (Eigen::Index n = 0; n < diag_len; ++n) {
            output(output_base_index + content_offset + n) =
                input(batch, n + y_offset, n + x_offset);
          }

          // Padding.
          const bool left_align = (content_offset == 0);
          const Eigen::Index padding_start = (left_align) ? diag_len : 0;
          const Eigen::Index padding_end =
              (left_align) ? max_diag_len : content_offset;
          for (Eigen::Index n = padding_start; n < padding_end; ++n) {
            output(output_base_index + n) = padding_value;
          }
          output_base_index += max_diag_len;
        }
      }
    };
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(num_batches, cost_per_batch,
                             std::move(compute_shard));
  }
};

}  // namespace functor

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void MatrixDiag<GPUDevice, T>::Compute(                                      \
      OpKernelContext* context, const GPUDevice& device,                       \
      typename TTypes<T>::ConstTensor& diag,                                   \
      typename TTypes<T, 3>::Tensor& output,                                   \
      const Eigen::Index lower_diag_index,                                     \
      const Eigen::Index upper_diag_index, const Eigen::Index max_diag_len,    \
      const T padding_value, const bool left_align_superdiagonal,              \
      const bool left_align_subdiagonal);                                      \
  extern template struct MatrixDiag<GPUDevice, T>;                             \
  template <>                                                                  \
  void MatrixDiagPart<GPUDevice, T>::Compute(                                  \
      OpKernelContext* context, const GPUDevice& device,                       \
      typename TTypes<T, 3>::ConstTensor& input,                               \
      typename TTypes<T>::Tensor& output, const Eigen::Index lower_diag_index, \
      const Eigen::Index upper_diag_index, const Eigen::Index max_diag_len,    \
      const T padding_value, const bool left_align_superdiagonal,              \
      const bool left_align_subdiagonal);                                      \
  extern template struct MatrixDiagPart<GPUDevice, T>;

TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_SPEC);

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_MATRIX_DIAG_GPU(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixDiag").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      MatrixDiagOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(Name("MatrixDiagV2")                             \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .HostMemory("k")                             \
                              .HostMemory("num_rows")                      \
                              .HostMemory("num_cols")                      \
                              .HostMemory("padding_value"),                \
                          MatrixDiagOp<GPUDevice, type>);                  \
  REGISTER_KERNEL_BUILDER(Name("MatrixDiagV3")                             \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .HostMemory("k")                             \
                              .HostMemory("num_rows")                      \
                              .HostMemory("num_cols")                      \
                              .HostMemory("padding_value"),                \
                          MatrixDiagOp<GPUDevice, type>);                  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixDiagPart").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      MatrixDiagPartOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(Name("MatrixDiagPartV2")                         \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .HostMemory("k")                             \
                              .HostMemory("padding_value"),                \
                          MatrixDiagPartOp<GPUDevice, type>);              \
  REGISTER_KERNEL_BUILDER(Name("MatrixDiagPartV3")                         \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<type>("T")                   \
                              .HostMemory("k")                             \
                              .HostMemory("padding_value"),                \
                          MatrixDiagPartOp<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_MATRIX_DIAG_GPU);
#undef REGISTER_MATRIX_DIAG_GPU

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_DIAG_GPU(type)                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("BatchMatrixDiag").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      MatrixDiagOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixDiagPart")                       \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<type>("T"),                   \
                          MatrixDiagPartOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_BATCH_MATRIX_DIAG_GPU);
#undef REGISTER_BATCH_MATRIX_DIAG_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
