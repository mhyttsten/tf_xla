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
class MHTracer_DTPStensorflowPScorePSkernelsPSquantized_batch_norm_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_batch_norm_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSquantized_batch_norm_opDTcc() {
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

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/quantization_utils.h"

namespace tensorflow {

namespace {

// A slow but straightforward implementation of batch normalization.
template <typename T1, typename T2>
void ReferenceBatchNorm(const Tensor& input, const float input_min,
                        const float input_max, const Tensor& mean,
                        float mean_min, float mean_max, const Tensor& var,
                        float var_min, float var_max, const Tensor& beta,
                        float beta_min, float beta_max, const Tensor& gamma,
                        float gamma_min, float gamma_max,
                        float variance_epsilon, bool scale_after_normalization,
                        Tensor* output, float* output_min, float* output_max) {
  auto input_flat = input.flat<T1>();
  auto mean_flat = mean.flat<T1>();
  auto var_flat = var.flat<T1>();
  auto beta_flat = beta.flat<T1>();
  auto gamma_flat = gamma.flat<T1>();
  auto output_flat = output->flat<T2>();

  const int depth = mean.dim_size(0);
  const int row_count = input_flat.size() / depth;

  *output_min = std::numeric_limits<float>::max();
  *output_max = std::numeric_limits<float>::lowest();
  for (int pass = 0; pass < 2; ++pass) {
    const bool is_range_pass = (pass == 0);
    for (int row_index = 0; row_index < row_count; ++row_index) {
      for (int channel = 0; channel < depth; ++channel) {
        const int input_index = (row_index * depth) + channel;
        const float input_value =
            QuantizedToFloat(input_flat(input_index), input_min, input_max);
        const float mean_value =
            QuantizedToFloat(mean_flat(channel), mean_min, mean_max);
        const float var_value =
            QuantizedToFloat(var_flat(channel), var_min, var_max);
        const float beta_value =
            QuantizedToFloat(beta_flat(channel), beta_min, beta_max);
        const float gamma_value =
            QuantizedToFloat(gamma_flat(channel), gamma_min, gamma_max);
        float output_value;
        if (scale_after_normalization) {
          output_value = (((input_value - mean_value) /
                           sqrtf(var_value + variance_epsilon)) *
                          gamma_value) +
                         beta_value;
        } else {
          output_value = ((input_value - mean_value) /
                          sqrtf(var_value + variance_epsilon)) +
                         beta_value;
        }
        if (is_range_pass) {
          *output_min = std::min(output_value, *output_min);
          *output_max = std::max(output_value, *output_max);
        } else {
          output_flat(input_index) =
              FloatToQuantized<T2>(output_value, *output_min, *output_max);
        }
      }
    }
  }
}

// An implementation of batch normalization that does the main calculations
// using only fixed-point arithmetic. There's a prologue with some floating
// calculations, but assuming the weights are constant these could be hoisted to
// an offline process, or baked into the weights.
template <typename T1, typename T2>
void FixedPointBatchNorm(const Tensor& input, const float input_min,
                         const float input_max, const Tensor& mean,
                         float mean_min, float mean_max, const Tensor& var,
                         float var_min, float var_max, const Tensor& beta,
                         float beta_min, float beta_max, const Tensor& gamma,
                         float gamma_min, float gamma_max,
                         float variance_epsilon, bool scale_after_normalization,
                         Tensor* output, float* output_min, float* output_max) {
  auto input_flat = input.flat<T1>();
  auto mean_flat = mean.flat<T1>();
  auto var_flat = var.flat<T1>();
  auto beta_flat = beta.flat<T1>();
  auto gamma_flat = gamma.flat<T1>();
  auto output_flat = output->flat<T2>();

  const int depth = mean.dim_size(0);
  const int row_count = input_flat.size() / depth;

  // The range here is chosen so that typical input values fit in without any
  // overflow or loss of precision, going from +1m to -1m with 10 bits of fixed
  // point precision.
  *output_min = -(1 << 20);
  *output_max = (1 << 20);

  Tensor scale_tensor(DataTypeToEnum<T2>::v(), {depth});
  auto scale_flat = scale_tensor.flat<T2>();
  Tensor offset_tensor(DataTypeToEnum<T2>::v(), {depth});
  auto offset_flat = offset_tensor.flat<T2>();
  for (int channel = 0; channel < depth; ++channel) {
    const float mean_value =
        QuantizedToFloat(mean_flat(channel), mean_min, mean_max);
    const float var_value =
        QuantizedToFloat(var_flat(channel), var_min, var_max);
    const float beta_value =
        QuantizedToFloat(beta_flat(channel), beta_min, beta_max);
    const float gamma_value =
        QuantizedToFloat(gamma_flat(channel), gamma_min, gamma_max);
    float scale_value;
    if (scale_after_normalization) {
      scale_value = (1.0f / sqrtf(var_value + variance_epsilon)) * gamma_value;
    } else {
      scale_value = (1.0f / sqrtf(var_value + variance_epsilon));
    }
    const float offset_value = (-mean_value * scale_value) + beta_value;
    scale_flat(channel) =
        FloatToQuantized<T2>(scale_value, *output_min, *output_max);
    offset_flat(channel) =
        FloatToQuantized<T2>(offset_value, *output_min, *output_max);
  }

  const T2 one_in_output_space =
      FloatToQuantized<T2>(1.0f, *output_min, *output_max);
  for (int row_index = 0; row_index < row_count; ++row_index) {
    for (int channel = 0; channel < depth; ++channel) {
      const int input_index = (row_index * depth) + channel;
      const T2 input_value =
          RequantizeInNewRange<T1, T2>(input_flat(input_index), input_min,
                                       input_max, *output_min, *output_max);
      const T2 scale_value = scale_flat(channel);
      const T2 offset_value = offset_flat(channel);
      const T2 output_value =
          ((input_value * scale_value) / one_in_output_space) + offset_value;
      output_flat(input_index) = output_value;
    }
  }
}

}  // namespace

template <typename T1, typename T2>
class QuantizedBatchNormOp : public OpKernel {
 public:
  explicit QuantizedBatchNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_batch_norm_opDTcc mht_0(mht_0_v, 336, "", "./tensorflow/core/kernels/quantized_batch_norm_op.cc", "QuantizedBatchNormOp");

    OP_REQUIRES_OK(context,
                   context->GetAttr("variance_epsilon", &variance_epsilon_));
    OP_REQUIRES_OK(context, context->GetAttr("scale_after_normalization",
                                             &scale_after_normalization_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_batch_norm_opDTcc mht_1(mht_1_v, 346, "", "./tensorflow/core/kernels/quantized_batch_norm_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const auto& input_min_tensor = context->input(1);
    OP_REQUIRES(context, input_min_tensor.NumElements() == 1,
                errors::InvalidArgument("input_min must have 1 element"));
    const float input_min = input_min_tensor.flat<float>()(0);
    const auto& input_max_tensor = context->input(2);
    OP_REQUIRES(context, input_max_tensor.NumElements() == 1,
                errors::InvalidArgument("input_max must have 1 element"));
    const float input_max = input_max_tensor.flat<float>()(0);
    const Tensor& mean = context->input(3);
    const auto& mean_min_tensor = context->input(4);
    OP_REQUIRES(context, mean_min_tensor.NumElements() == 1,
                errors::InvalidArgument("mean_min must have 1 element"));
    const float mean_min = mean_min_tensor.flat<float>()(0);
    const auto& mean_max_tensor = context->input(5);
    OP_REQUIRES(context, mean_max_tensor.NumElements() == 1,
                errors::InvalidArgument("mean_max must have 1 element"));
    const float mean_max = mean_max_tensor.flat<float>()(0);
    const Tensor& var = context->input(6);
    const auto& var_min_tensor = context->input(7);
    OP_REQUIRES(context, var_min_tensor.NumElements() == 1,
                errors::InvalidArgument("var_min must have 1 element"));
    const float var_min = var_min_tensor.flat<float>()(0);
    const auto& var_max_tensor = context->input(8);
    OP_REQUIRES(context, var_max_tensor.NumElements() == 1,
                errors::InvalidArgument("var_max must have 1 element"));
    const float var_max = var_max_tensor.flat<float>()(0);
    const Tensor& beta = context->input(9);
    const auto& beta_min_tensor = context->input(10);
    OP_REQUIRES(context, beta_min_tensor.NumElements() == 1,
                errors::InvalidArgument("beta_min must have 1 element"));
    const float beta_min = beta_min_tensor.flat<float>()(0);
    const auto& beta_max_tensor = context->input(11);
    OP_REQUIRES(context, beta_max_tensor.NumElements() == 1,
                errors::InvalidArgument("beta_max must have 1 element"));
    const float beta_max = beta_max_tensor.flat<float>()(0);
    const Tensor& gamma = context->input(12);
    const auto& gamma_min_tensor = context->input(13);
    OP_REQUIRES(context, gamma_min_tensor.NumElements() == 1,
                errors::InvalidArgument("gamma_min must have 1 element"));
    const float gamma_min = gamma_min_tensor.flat<float>()(0);
    const auto& gamma_max_tensor = context->input(14);
    OP_REQUIRES(context, gamma_max_tensor.NumElements() == 1,
                errors::InvalidArgument("gamma_max must have 1 element"));
    const float gamma_max = gamma_max_tensor.flat<float>()(0);

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, mean.dims() == 1,
                errors::InvalidArgument("mean must be 1-dimensional",
                                        mean.shape().DebugString()));
    OP_REQUIRES(context, var.dims() == 1,
                errors::InvalidArgument("var must be 1-dimensional",
                                        var.shape().DebugString()));
    OP_REQUIRES(context, beta.dims() == 1,
                errors::InvalidArgument("beta must be 1-dimensional",
                                        beta.shape().DebugString()));
    OP_REQUIRES(context, gamma.dims() == 1,
                errors::InvalidArgument("gamma must be 1-dimensional",
                                        gamma.shape().DebugString()));
    OP_REQUIRES(context, mean.NumElements() > 1,
                errors::InvalidArgument("Must have at least a mean value",
                                        gamma.shape().DebugString()));
    OP_REQUIRES(context, mean.NumElements() > 1,
                errors::InvalidArgument("Must have at least a mean value"));
    const auto last_dim = input.shape().dims() - 1;
    OP_REQUIRES(context,
                mean.shape().dim_size(0) == input.shape().dim_size(last_dim),
                errors::InvalidArgument("Must provide as many means as the "
                                        "last dimension of the input tensor: ",
                                        mean.shape().DebugString(), " vs. ",
                                        input.shape().DebugString()));
    OP_REQUIRES(
        context, mean.shape().dim_size(0) == var.shape().dim_size(0),
        errors::InvalidArgument(
            "Mean and variance tensors must have the same shape: ",
            mean.shape().DebugString(), " vs. ", var.shape().DebugString()));
    OP_REQUIRES(
        context, mean.shape().dim_size(0) == beta.shape().dim_size(0),
        errors::InvalidArgument(
            "Mean and beta tensors must have the same shape: ",
            mean.shape().DebugString(), " vs. ", beta.shape().DebugString()));
    OP_REQUIRES(
        context, mean.shape().dim_size(0) == gamma.shape().dim_size(0),
        errors::InvalidArgument(
            "Mean and gamma tensors must have the same shape: ",
            mean.shape().DebugString(), " vs. ", gamma.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    float output_min;
    float output_max;
    FixedPointBatchNorm<T1, T2>(input, input_min, input_max, mean, mean_min,
                                mean_max, var, var_min, var_max, beta, beta_min,
                                beta_max, gamma, gamma_min, gamma_max,
                                variance_epsilon_, scale_after_normalization_,
                                output, &output_min, &output_max);

    Tensor* output_min_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {}, &output_min_tensor));
    output_min_tensor->flat<float>()(0) = output_min;

    Tensor* output_max_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, {}, &output_max_tensor));
    output_max_tensor->flat<float>()(0) = output_max;
  }

 private:
  float variance_epsilon_;
  bool scale_after_normalization_;
};

REGISTER_KERNEL_BUILDER(Name("QuantizedBatchNormWithGlobalNormalization")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint32>("out_type"),
                        QuantizedBatchNormOp<quint8, qint32>);

}  // namespace tensorflow
