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

#ifndef TENSORFLOW_CORE_KERNELS_QUANTIZE_AND_DEQUANTIZE_OP_H_
#define TENSORFLOW_CORE_KERNELS_QUANTIZE_AND_DEQUANTIZE_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh() {
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
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

enum QuantizerRoundMode {
  // Round half up: if the fraction of y is exactly 0.5, then
  // round(y) = y + 0.5
  // E.g., -5.5 gets rounded to -5, -5.4 goes to -5,
  // 5.4 goes to 5, and 5.5 goes to 6.
  ROUND_HALF_UP,
  // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
  // the nearest even integer to y.
  // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5 becomes
  // -24, and -24.5 gets rounded to 24.
  ROUND_HALF_TO_EVEN,
};

namespace functor {

// TODO(pauldonnelly): 'signed_input' should really be called 'signed_output'.

template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleFunctor {
  void operator()(const Device& d, typename TTypes<T>::ConstVec input,
                  bool signed_input, int num_bits, bool range_given,
                  Tensor* input_min_tensor, Tensor* input_max_tensor,
                  QuantizerRoundMode round_mode, bool narrow_range,
                  typename TTypes<T>::Vec output);
};

template <typename Device, typename T>
struct QuantizeAndDequantizePerChannelFunctor {
  void operator()(const Device& d, typename TTypes<T, 3>::ConstTensor input,
                  bool signed_input, int num_bits, bool range_given,
                  Tensor* input_min_tensor, Tensor* input_max_tensor,
                  QuantizerRoundMode round_mode, bool narrow_range,
                  typename TTypes<T, 3>::Tensor output);
};

template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleGradientFunctor {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat gradient,
                  typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::ConstScalar input_min,
                  typename TTypes<T>::ConstScalar input_max,
                  typename TTypes<T>::Flat input_backprop,
                  typename TTypes<T>::Scalar input_min_backprop,
                  typename TTypes<T>::Scalar input_max_backprop);
};

template <typename Device, typename T>
struct QuantizeAndDequantizePerChannelGradientFunctor {
  void operator()(const Device& d, typename TTypes<T, 3>::ConstTensor gradient,
                  typename TTypes<T, 3>::ConstTensor input,
                  const Tensor* input_min_tensor,
                  const Tensor* input_max_tensor,
                  typename TTypes<T, 3>::Tensor input_backprop,
                  typename TTypes<T>::Flat input_min_backprop,
                  typename TTypes<T>::Flat input_max_backprop);
};

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T, typename Func,
          typename Vec = typename TTypes<T>::Vec,
          typename ConstVec = typename TTypes<T>::ConstVec>
void ClampScaleAndRound(const Device& d, ConstVec input, T min_range,
                        T max_range, T scale, T inverse_scale, Func round_func,
                        Vec output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh mht_0(mht_0_v, 260, "", "./tensorflow/core/kernels/quantize_and_dequantize_op.h", "ClampScaleAndRound");

  output.device(d) = (input.cwiseMin(max_range).cwiseMax(min_range) * scale)
                         .unaryExpr(round_func) *
                     inverse_scale;
}

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T, typename Vec = typename TTypes<T>::Vec,
          typename ConstVec = typename TTypes<T>::ConstVec>
void ClampScaleAndRound(const Device& d, ConstVec input, T min_range,
                        T max_range, T scale, T inverse_scale,
                        QuantizerRoundMode round_mode, Vec output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh mht_1(mht_1_v, 274, "", "./tensorflow/core/kernels/quantize_and_dequantize_op.h", "ClampScaleAndRound");

  switch (round_mode) {
    case ROUND_HALF_TO_EVEN:
      ClampScaleAndRound(d, input, min_range, max_range, scale, inverse_scale,
                         Eigen::internal::scalar_round_half_to_even_op<T>(),
                         output);
      break;
    case ROUND_HALF_UP:
      ClampScaleAndRound(d, input, min_range, max_range, scale, inverse_scale,
                         Eigen::internal::scalar_round_up_op<T>(), output);
      break;
  }
}

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T, typename Func,
          typename Vec = typename TTypes<T>::Vec,
          typename ConstVec = typename TTypes<T>::ConstVec>
void ScaleAndRound(const Device& d, ConstVec input, T scale, T inverse_scale,
                   Func round_func, Vec output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh mht_2(mht_2_v, 296, "", "./tensorflow/core/kernels/quantize_and_dequantize_op.h", "ScaleAndRound");

  output.device(d) = (input * scale).unaryExpr(round_func) * inverse_scale;
}

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T, typename Vec = typename TTypes<T>::Vec,
          typename ConstVec = typename TTypes<T>::ConstVec>
void ScaleAndRound(const Device& d, ConstVec input, T scale, T inverse_scale,
                   QuantizerRoundMode round_mode, Vec output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh mht_3(mht_3_v, 307, "", "./tensorflow/core/kernels/quantize_and_dequantize_op.h", "ScaleAndRound");

  switch (round_mode) {
    case ROUND_HALF_TO_EVEN:
      ScaleAndRound(d, input, scale, inverse_scale,
                    Eigen::internal::scalar_round_half_to_even_op<T>(), output);
      break;
    case ROUND_HALF_UP:
      ScaleAndRound(d, input, scale, inverse_scale,
                    Eigen::internal::scalar_round_up_op<T>(), output);
      break;
  }
}

template <typename T>
void ComputeQuantizationRange(bool signed_input, int num_bits,
                              QuantizerRoundMode round_mode, bool narrow_range,
                              T* min_range, T* max_range, T* scale,
                              T* inverse_scale) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh mht_4(mht_4_v, 327, "", "./tensorflow/core/kernels/quantize_and_dequantize_op.h", "ComputeQuantizationRange");

  // Calculate the range for the simulated integer quantization:
  // e.g. [-127,127] for signed = true, narrow_range = true, num_bits = 8,
  // or [-128,127] for signed = true, narrow_range = false, num_bits = 8,
  // or [0, 255] for signed = false, num_bits = 8.
  const int64_t min_quantized =
      signed_input ? narrow_range ? -(1ULL << (num_bits - 1)) + 1
                                  : -(1ULL << (num_bits - 1))
                   : 0;
  const int64_t max_quantized =
      signed_input ? (1ULL << (num_bits - 1)) - 1 : (1ULL << num_bits) - 1;
  // Determine the maximum scaling factor that would scale
  // [min_range, max_range] to not exceed [min_quantized, max_quantized],
  // while keeping 0 unchanged.
  const T scale_from_min_side = (min_quantized * *min_range > 0)
                                    ? min_quantized / *min_range
                                    : std::numeric_limits<T>::max();
  const T scale_from_max_side = (max_quantized * *max_range > 0)
                                    ? max_quantized / *max_range
                                    : std::numeric_limits<T>::max();

  // Note: Avoids changing the side of the range that determines scale.
  if (scale_from_min_side < scale_from_max_side) {
    *scale = scale_from_min_side;
    *inverse_scale = *min_range / min_quantized;
    *max_range = max_quantized * *inverse_scale;
  } else {
    *scale = scale_from_max_side;
    *inverse_scale = *max_range / max_quantized;
    *min_range = min_quantized * *inverse_scale;
  }
}

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleImpl {
  static void Compute(const Device& d, typename TTypes<T>::ConstVec input,
                      bool signed_input, int num_bits, bool range_given,
                      Tensor* input_min_tensor, Tensor* input_max_tensor,
                      QuantizerRoundMode round_mode, bool narrow_range,
                      typename TTypes<T>::Vec output) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh mht_5(mht_5_v, 370, "", "./tensorflow/core/kernels/quantize_and_dequantize_op.h", "Compute");

    T min_range;
    T max_range;
    auto input_min = input_min_tensor->scalar<T>();
    auto input_max = input_max_tensor->scalar<T>();
    if (!range_given) {
      input_min.device(d) = input.minimum();
      input_max.device(d) = input.maximum();
      d.memcpyDeviceToHost(&min_range, input_min.data(), sizeof(T));
      d.memcpyDeviceToHost(&max_range, input_max.data(), sizeof(T));
    } else {
      // Copy the range values from their respective tensors on the host.
      min_range = input_min_tensor->scalar<T>()();
      max_range = input_max_tensor->scalar<T>()();
    }

    T scale, inverse_scale;
    ComputeQuantizationRange(signed_input, num_bits, round_mode, narrow_range,
                             &min_range, &max_range, &scale, &inverse_scale);

    if (range_given) {
      // Note: The clamping here is to avoid overflow in the quantized type.
      // The semantics of the op does not guarantee to clamp to the specified
      // min_range and max_range - because we may have changed either min_range
      // or max_range.
      ClampScaleAndRound(d, input, min_range, max_range, scale, inverse_scale,
                         round_mode, output);
    } else {
      ScaleAndRound(d, input, scale, inverse_scale, round_mode, output);
    }
  }
};

// The implementation below runs on both CPU and GPU.

template <typename Device, typename T>
struct QuantizeAndDequantizePerChannelImpl {
  static void Compute(const Device& d, typename TTypes<T, 3>::ConstTensor input,
                      bool signed_input, int num_bits, bool range_given,
                      Tensor* input_min_tensor, Tensor* input_max_tensor,
                      QuantizerRoundMode round_mode, bool narrow_range,
                      typename TTypes<T, 3>::Tensor output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh mht_6(mht_6_v, 414, "", "./tensorflow/core/kernels/quantize_and_dequantize_op.h", "Compute");

    using Index = typename tensorflow::TTypes<T>::ConstTensor::Index;
    int num_channels = input.dimension(1);
    auto input_min = input_min_tensor->vec<T>();
    auto input_max = input_max_tensor->vec<T>();
    std::vector<T> min_range(num_channels);
    std::vector<T> max_range(num_channels);

    if (!range_given) {
      Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<2> > reduce_dims;
      input_min.device(d) = input.minimum(reduce_dims);
      input_max.device(d) = input.maximum(reduce_dims);
      d.memcpyDeviceToHost(min_range.data(), input_min.data(),
                           num_channels * sizeof(T));
      d.memcpyDeviceToHost(max_range.data(), input_max.data(),
                           num_channels * sizeof(T));
    } else {
      // Copy the range values from their respective tensors on the host.
      std::memcpy(min_range.data(), input_min_tensor->vec<T>().data(),
                  num_channels * sizeof(T));
      std::memcpy(max_range.data(), input_max_tensor->vec<T>().data(),
                  num_channels * sizeof(T));
    }

    for (Index i = 0; i < num_channels; ++i) {
      const auto input_chip = input.template chip<1>(i);
      auto output_chip = output.template chip<1>(i);

      T scale, inverse_scale;
      ComputeQuantizationRange(signed_input, num_bits, round_mode, narrow_range,
                               &min_range[i], &max_range[i], &scale,
                               &inverse_scale);
      if (range_given) {
        ClampScaleAndRound(d, input_chip, min_range[i], max_range[i], scale,
                           inverse_scale, round_mode, output_chip);
      } else {
        ScaleAndRound(d, input_chip, scale, inverse_scale, round_mode,
                      output_chip);
      }
    }
  }
};

template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleGradientImpl {
  static void Compute(const Device& d, typename TTypes<T>::ConstFlat gradient,
                      typename TTypes<T>::ConstFlat input,
                      typename TTypes<T>::ConstScalar input_min,
                      typename TTypes<T>::ConstScalar input_max,
                      typename TTypes<T>::Flat input_backprop,
                      typename TTypes<T>::Scalar input_min_backprop,
                      typename TTypes<T>::Scalar input_max_backprop) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh mht_7(mht_7_v, 468, "", "./tensorflow/core/kernels/quantize_and_dequantize_op.h", "Compute");

    const T min_val = input_min();
    const T max_val = input_max();
    const auto in_range =
        (input >= min_val && input <= max_val)
            .select(input.constant(1.0f), input.constant(0.0f));
    input_backprop.device(d) = gradient * in_range;
    input_min_backprop.device(d) = input_min_backprop.constant(0.0f);
    input_max_backprop.device(d) = input_max_backprop.constant(0.0f);
  }
};

template <typename Device, typename T>
struct QuantizeAndDequantizePerChannelGradientImpl {
  static void Compute(const Device& d,
                      typename TTypes<T, 3>::ConstTensor gradient,
                      typename TTypes<T, 3>::ConstTensor input,
                      const Tensor* input_min_tensor,
                      const Tensor* input_max_tensor,
                      typename TTypes<T, 3>::Tensor input_backprop,
                      typename TTypes<T>::Flat input_min_backprop,
                      typename TTypes<T>::Flat input_max_backprop) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_and_dequantize_opDTh mht_8(mht_8_v, 492, "", "./tensorflow/core/kernels/quantize_and_dequantize_op.h", "Compute");

    using Index = typename tensorflow::TTypes<T>::ConstTensor::Index;
    auto input_min = input_min_tensor->vec<T>();
    auto input_max = input_max_tensor->vec<T>();
    int num_channels = input.dimension(1);
    for (Index i = 0; i < num_channels; ++i) {
      const auto gradient_chip = gradient.template chip<1>(i);
      const auto input_chip = input.template chip<1>(i);
      const T min_val = input_min(i);
      const T max_val = input_max(i);
      const auto in_range =
          (input_chip >= min_val && input_chip <= max_val)
              .select(input_chip.constant(1.0f), input_chip.constant(0.0f));
      input_backprop.template chip<1>(i).device(d) = gradient_chip * in_range;
    }
    input_min_backprop.device(d) = input_min_backprop.constant(0.0f);
    input_max_backprop.device(d) = input_max_backprop.constant(0.0f);
  }
};

}  // end of namespace functor
}  // end of namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_QUANTIZE_AND_DEQUANTIZE_OP_H_
