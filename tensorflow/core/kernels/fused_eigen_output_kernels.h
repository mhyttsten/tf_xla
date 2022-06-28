/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Output kernels for fusing computation into Eigen Tensor contractions:
//   (1) FusedConv2DOp
//   (2) FusedMatMulOp
//
// Supported fused computations:
//   (1) {Conv2D/MatMul} + BiasAdd + <Activation>
//   (2) {Conv2D/MatMul} + FusedBatchNorm + <Activation>
//
// Activation: Relu, Relu6, Elu, etc...

#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EIGEN_OUTPUT_KERNELS_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EIGEN_OUTPUT_KERNELS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh() {
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


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

enum class FusedComputationType {
  kUndefined,
  kBiasAdd,
  kBiasAddWithRelu,
  kBiasAddWithRelu6,
  kBiasAddWithElu,
  kBiasAddWithLeakyRelu,
  kFusedBatchNorm,
  kFusedBatchNormWithRelu,
  kFusedBatchNormWithRelu6,
  kFusedBatchNormWithElu,
  kFusedBatchNormWithLeakyRelu
};

// We have to pass around additional arguments for all possible fusion types.
struct FusedComputationArgs {
  float epsilon = 0.0;          // Used by `FusedBatchNorm` fusion only
  float leakyrelu_alpha = 0.0;  // Used by `LeakyRelu` fusion only
};

struct FusedComputationPattern {
  FusedComputationType fused_computation;
  std::vector<string> fused_ops;
};

// Parse attributes from the kernel construction context, and verifies that they
// specify valid fused computation pattern.
Status InitializeFusedComputation(
    OpKernelConstruction* context, const string& kernel_name,
    const std::vector<FusedComputationPattern>& patterns,
    FusedComputationType* fused_computation,
    FusedComputationArgs* fused_computation_args);

// Type alias for the tensor contraction output mapper.
template <typename Scalar, typename StorageIndex>
using ContractionOutputMapper =
    Eigen::internal::blas_data_mapper<Scalar, StorageIndex, Eigen::ColMajor>;

// Returns input expression without any transformations.
struct Identity {
  template <typename XprType>
  static auto apply(XprType expr) -> XprType {
    return expr;
  };
};

// Applies `Relu` to the passed input expression.
struct Relu {
  template <typename XprType>
  static auto apply(XprType expr)
      -> decltype(expr.cwiseMax(std::declval<typename XprType::Scalar>())) {
    return expr.cwiseMax(static_cast<typename XprType::Scalar>(0));
  };
};

// Applies `Relu6` to the passed input expression.
struct Relu6 {
  template <typename XprType>
  static auto apply(XprType expr)
      -> decltype(expr.cwiseMax(std::declval<typename XprType::Scalar>())
                      .cwiseMin(std::declval<typename XprType::Scalar>())) {
    return expr.cwiseMax(static_cast<typename XprType::Scalar>(0))
        .cwiseMin(static_cast<typename XprType::Scalar>(6));
  };
};

// Applies `Elu` to the passed input expression.
struct Elu {
  template <typename XprType>
  static auto apply(XprType expr) -> decltype(
      (expr < std::declval<typename XprType::Scalar>())
          .select(expr.exp() -
                      expr.constant(std::declval<typename XprType::Scalar>()),
                  expr)) {
    return (expr < static_cast<typename XprType::Scalar>(0))
        .select(expr.exp() -
                    expr.constant(static_cast<typename XprType::Scalar>(1)),
                expr);
  };
};

// Applies `LeakyRelu` to the passed input expression.
struct LeakyRelu {
  template <typename XprType>
  static auto apply(XprType expr, const float leakyrelu_alpha) -> decltype(
      (expr < std::declval<typename XprType::Scalar>())
          .select(expr *
                      expr.constant(std::declval<typename XprType::Scalar>()),
                  expr)) {
    return (expr < static_cast<typename XprType::Scalar>(0))
        .select(expr * expr.constant(static_cast<typename XprType::Scalar>(
                           leakyrelu_alpha)),
                expr);
  };
};

template <typename T>
struct BiasAddArgs {
  const T* bias_add_data = nullptr;
  float leakyrelu_alpha;

  static bool IsSupported(FusedComputationType fusion) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh mht_0(mht_0_v, 306, "", "./tensorflow/core/kernels/fused_eigen_output_kernels.h", "IsSupported");

    return fusion == FusedComputationType::kBiasAdd ||
           fusion == FusedComputationType::kBiasAddWithRelu ||
           fusion == FusedComputationType::kBiasAddWithRelu6 ||
           fusion == FusedComputationType::kBiasAddWithElu ||
           fusion == FusedComputationType::kBiasAddWithLeakyRelu;
  }
};

template <typename T>
struct FusedBatchNormArgs {
  const T* scale_data = nullptr;
  const T* offset_data = nullptr;
  const T* estimated_mean_data = nullptr;
  const T* estimated_variance_data = nullptr;

  // Precomputed expression:
  //   scaling_factor = (estimated_variance + epsilon).rsqrt() * scale
  Eigen::Tensor<T, 1, Eigen::RowMajor> scaling_factor;

  float leakyrelu_alpha;

  static bool IsSupported(FusedComputationType fusion) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh mht_1(mht_1_v, 331, "", "./tensorflow/core/kernels/fused_eigen_output_kernels.h", "IsSupported");

    return fusion == FusedComputationType::kFusedBatchNorm ||
           fusion == FusedComputationType::kFusedBatchNormWithRelu ||
           fusion == FusedComputationType::kFusedBatchNormWithRelu6 ||
           fusion == FusedComputationType::kFusedBatchNormWithElu ||
           fusion == FusedComputationType::kFusedBatchNormWithLeakyRelu;
  }
};

// TensorContraction swaps lhs with rhs, and changes layout from RowMajor
// (default in Tensorflow) to ColMajor (preferred in Eigen), and computes matmul
// using these tensors.
//
// (1) Spatial Convolution (see eigen_spatial_convolutions.h):
//
//   TensorContraction output matrix (before reshape) has a ColMajor layout, and
//   has dimensions:
//   - rows: output_channels
//   - cols: all other dimensions
//
//   First element in every column is:
//     [batch ??, height ??, width ??, out_channel = i]
//
//   We do not know what are the values of the 'batch', 'height', and 'width'
//   here (if we know original dimensions, they can be computed from 'j').
//
//   Each column of an output block is a continuous slice along the output
//   channel dimension, so we can use it to efficiently compute any
//   transformation that depends only on a channel value (e.g. add channel
//   bias).
//
// (2) Matrix Multiplication (see matmul_op.cc):
//
//   For the `MxK * KxN` matrix multiplication, output matrix has a `MxN`
//   dimensions. Each column in output block is a slice of the innermost
//   dimension of the output matrix starting at offset 'i'.
//
//   Example: In Tensorflow MatMul [8x32] * [32x64], each output block column
//   will correspond to MatMul output row of size 64 (because Tensorflow uses
//   row major storage order).

// Output kernel that fuses BiasAdd operation into the output of tensor
// contraction + activation function defined by Activation.
template <typename T, typename Activation = Identity>
struct BiasAddOutputKernel {
  explicit BiasAddOutputKernel(const BiasAddArgs<T>& args)
      : bias_data(args.bias_add_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh mht_2(mht_2_v, 380, "", "./tensorflow/core/kernels/fused_eigen_output_kernels.h", "BiasAddOutputKernel");
}

  template <typename StorageIndex, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, StorageIndex>& output_mapper,
      const Eigen::TensorContractionParams& params, StorageIndex i,
      StorageIndex j, StorageIndex num_rows, StorageIndex num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* bias_base = bias_data + i;
    typename TTypes<T>::UnalignedConstTensor bias(bias_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename TTypes<T>::UnalignedTensor output(output_base, num_rows);
      const auto expr = output + bias;
      output = Activation::template apply<decltype(expr)>(expr);
    }
  }

 private:
  const T* bias_data;
};

template <typename T>
struct BiasAddOutputKernel<T, LeakyRelu> {
  explicit BiasAddOutputKernel(const BiasAddArgs<T>& args)
      : bias_data(args.bias_add_data), leakyrelu_alpha(args.leakyrelu_alpha) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh mht_3(mht_3_v, 410, "", "./tensorflow/core/kernels/fused_eigen_output_kernels.h", "BiasAddOutputKernel");
}

  template <typename StorageIndex, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, StorageIndex>& output_mapper,
      const Eigen::TensorContractionParams& params, StorageIndex i,
      StorageIndex j, StorageIndex num_rows, StorageIndex num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* bias_base = bias_data + i;
    typename TTypes<T>::UnalignedConstTensor bias(bias_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename TTypes<T>::UnalignedTensor output(output_base, num_rows);
      const auto expr = output + bias;
      output = LeakyRelu::template apply<decltype(expr)>(expr, leakyrelu_alpha);
    }
  }

 private:
  const T* bias_data;
  float leakyrelu_alpha;
};

// Output kernel that fuses FusedBatchNorm operation into the output of tensor
// contraction + activation function defined by Activation.
template <typename T, typename Activation = Identity>
struct FusedBatchNormOutputKernel {
  FusedBatchNormOutputKernel(T epsilon, const FusedBatchNormArgs<T>& args)
      : epsilon(epsilon),
        scaling_factor_data(args.scaling_factor.data()),
        offset_data(args.offset_data),
        estimated_mean_data(args.estimated_mean_data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh mht_4(mht_4_v, 446, "", "./tensorflow/core/kernels/fused_eigen_output_kernels.h", "FusedBatchNormOutputKernel");
}

  template <typename StorageIndex, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, StorageIndex>& output_mapper,
      const Eigen::TensorContractionParams& params, StorageIndex i,
      StorageIndex j, StorageIndex num_rows, StorageIndex num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* scaling_factor_base = scaling_factor_data + i;
    const T* offset_base = offset_data + i;
    const T* mean_base = estimated_mean_data + i;

    typename TTypes<T>::UnalignedConstTensor scaling_factor(scaling_factor_base,
                                                            num_rows);
    typename TTypes<T>::UnalignedConstTensor offset(offset_base, num_rows);
    typename TTypes<T>::UnalignedConstTensor mean(mean_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename TTypes<T>::UnalignedTensor output(output_base, num_rows);

      auto scaled = (output - mean) * scaling_factor;
      auto shifted = scaled + offset;

      output = Activation::template apply<decltype(shifted)>(shifted);
    }
  }

 private:
  T epsilon;
  const T* scaling_factor_data;
  const T* offset_data;
  const T* estimated_mean_data;
};

template <typename T>
struct FusedBatchNormOutputKernel<T, LeakyRelu> {
  FusedBatchNormOutputKernel(T epsilon, const FusedBatchNormArgs<T>& args)
      : epsilon(epsilon),
        scaling_factor_data(args.scaling_factor.data()),
        offset_data(args.offset_data),
        estimated_mean_data(args.estimated_mean_data),
        leakyrelu_alpha(args.leakyrelu_alpha) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh mht_5(mht_5_v, 492, "", "./tensorflow/core/kernels/fused_eigen_output_kernels.h", "FusedBatchNormOutputKernel");
}

  template <typename StorageIndex, typename Scalar>
  EIGEN_ALWAYS_INLINE void operator()(
      const ContractionOutputMapper<Scalar, StorageIndex>& output_mapper,
      const Eigen::TensorContractionParams& params, StorageIndex i,
      StorageIndex j, StorageIndex num_rows, StorageIndex num_cols) const {
    DCHECK(params.swapped_arguments);

    const T* scaling_factor_base = scaling_factor_data + i;
    const T* offset_base = offset_data + i;
    const T* mean_base = estimated_mean_data + i;

    typename TTypes<T>::UnalignedConstTensor scaling_factor(scaling_factor_base,
                                                            num_rows);
    typename TTypes<T>::UnalignedConstTensor offset(offset_base, num_rows);
    typename TTypes<T>::UnalignedConstTensor mean(mean_base, num_rows);

    for (int col = 0; col < num_cols; ++col) {
      T* output_base = &output_mapper(0, col);
      typename TTypes<T>::UnalignedTensor output(output_base, num_rows);

      auto scaled = (output - mean) * scaling_factor;
      auto shifted = scaled + offset;

      output = LeakyRelu::template apply<decltype(shifted)>(shifted,
                                                            leakyrelu_alpha);
    }
  }

 private:
  T epsilon;
  const T* scaling_factor_data;
  const T* offset_data;
  const T* estimated_mean_data;
  float leakyrelu_alpha;
};

// Type aliases for the output kernels, purely for the sake of better launch
// dispatching code readability.
template <typename T>
using WithBiasAdd = BiasAddOutputKernel<T>;
template <typename T>
using WithBiasAddAndRelu = BiasAddOutputKernel<T, Relu>;
template <typename T>
using WithBiasAddAndRelu6 = BiasAddOutputKernel<T, Relu6>;
template <typename T>
using WithBiasAddAndElu = BiasAddOutputKernel<T, Elu>;
template <typename T>
using WithBiasAddAndLeakyRelu = BiasAddOutputKernel<T, LeakyRelu>;
template <typename T>
using WithFusedBatchNorm = FusedBatchNormOutputKernel<T>;
template <typename T>
using WithFusedBatchNormAndRelu = FusedBatchNormOutputKernel<T, Relu>;
template <typename T>
using WithFusedBatchNormAndRelu6 = FusedBatchNormOutputKernel<T, Relu6>;
template <typename T>
using WithFusedBatchNormAndElu = FusedBatchNormOutputKernel<T, Elu>;
template <typename T>
using WithFusedBatchNormAndLeakyRelu = FusedBatchNormOutputKernel<T, LeakyRelu>;

template <typename T>
Status InitBiasAddArgs(OpKernelContext* context, BiasAddArgs<T>* args,
                       const float* leakyrelu_alpha = nullptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh mht_6(mht_6_v, 558, "", "./tensorflow/core/kernels/fused_eigen_output_kernels.h", "InitBiasAddArgs");

  // Bias of the following dimensions: [ output_depth ]
  const Tensor& bias = context->input(2);

  if (bias.dims() != 1)
    return errors::InvalidArgument("bias must be 1-dimensional",
                                   bias.shape().DebugString());

  const auto data_ptr = [](const Tensor& tensor) -> const T* {
    return reinterpret_cast<const T*>(tensor.tensor_data().data());
  };

  args->bias_add_data = data_ptr(bias);

  if (leakyrelu_alpha) {
    args->leakyrelu_alpha = *leakyrelu_alpha;
  }

  return Status::OK();
}

template <typename T>
Status InitFusedBatchNormArgs(OpKernelContext* context, float epsilon,
                              FusedBatchNormArgs<T>* args,
                              const float* leakyrelu_alpha = nullptr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfused_eigen_output_kernelsDTh mht_7(mht_7_v, 585, "", "./tensorflow/core/kernels/fused_eigen_output_kernels.h", "InitFusedBatchNormArgs");

  const Tensor& scale = context->input(2);
  const Tensor& offset = context->input(3);
  const Tensor& estimated_mean = context->input(4);
  const Tensor& estimated_variance = context->input(5);

  if (scale.dims() != 1)
    return errors::InvalidArgument("scale must be 1-dimensional",
                                   scale.shape().DebugString());
  if (offset.dims() != 1)
    return errors::InvalidArgument("offset must be 1-dimensional",
                                   offset.shape().DebugString());
  if (estimated_mean.dims() != 1)
    return errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                   estimated_mean.shape().DebugString());
  if (estimated_variance.dims() != 1)
    return errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                   estimated_variance.shape().DebugString());

  const auto data_ptr = [](const Tensor& tensor) -> const T* {
    return reinterpret_cast<const T*>(tensor.tensor_data().data());
  };

  args->scale_data = data_ptr(scale);
  args->offset_data = data_ptr(offset);
  args->estimated_mean_data = data_ptr(estimated_mean);
  args->estimated_variance_data = data_ptr(estimated_variance);

  // Precompute scaling factor once for all output blocks (kernels).
  args->scaling_factor =
      (estimated_variance.flat<T>() + static_cast<T>(epsilon)).rsqrt() *
      scale.flat<T>();

  if (leakyrelu_alpha) {
    args->leakyrelu_alpha = *leakyrelu_alpha;
  }

  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EIGEN_OUTPUT_KERNELS_H_
