/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SOFTMAX_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SOFTMAX_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsoftmaxDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsoftmaxDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsoftmaxDTh() {
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


#include <limits>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace reference_ops {

inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const float* input_data,
                    const RuntimeShape& output_shape, float* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsoftmaxDTh mht_0(mht_0_v, 201, "", "./tensorflow/lite/kernels/internal/reference/softmax.h", "Softmax");

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c) {
      max = std::max(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      const float exp_c = std::exp((input_data[i * depth + c] - max) *
                                   static_cast<float>(params.beta));
      output_data[i * depth + c] = exp_c;
      sum += exp_c;
    }

    // Compute result.
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] = output_data[i * depth + c] / sum;
    }
  }
}

// Quantized softmax with int8_t/uint8_t input and int8_t/uint8_t/int16_t
// output.
template <typename InputT, typename OutputT>
inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const InputT* input_data,
                    const RuntimeShape& output_shape, OutputT* output_data) {
  const int32_t input_beta_multiplier = params.input_multiplier;
  const int32_t input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
  using FixedPointAccum =
      gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    InputT max_in_row = std::numeric_limits<InputT>::min();
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    int num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));

    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);

        FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
        int32_t unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(),
            num_bits_over_unit + 31 - (sizeof(OutputT) * 8));

        const int32_t shifted_output =
            unsat_output +
            static_cast<int32_t>(std::numeric_limits<OutputT>::min());

        output_data[i * depth + c] = static_cast<OutputT>(std::max(
            std::min(shifted_output,
                     static_cast<int32_t>(std::numeric_limits<OutputT>::max())),
            static_cast<int32_t>(std::numeric_limits<OutputT>::min())));
      } else {
        output_data[i * depth + c] = std::numeric_limits<OutputT>::min();
      }
    }
  }
}

// Computes exp(input - max_input)
inline int16_t SoftMaxCalculateExp(const SoftmaxParams& params,
                                   const int16_t* input_data, const int depth,
                                   int16_t max_in_row, int i, int c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsoftmaxDTh mht_1(mht_1_v, 322, "", "./tensorflow/lite/kernels/internal/reference/softmax.h", "SoftMaxCalculateExp");

  int32_t input_diff = input_data[i * depth + c] - max_in_row;
  // scale the input_diff such that [-65535, 0] correspond to [-10.0, 0.0]
  // exp lut generated with range [-10, 0], as exp(-10) is negligible.
  int32_t scaled_diff = MultiplyByQuantizedMultiplier(
      input_diff, params.input_multiplier, params.input_left_shift);
  // recenter to [-32768, 32767]
  int32_t sym_scaled_diff = scaled_diff + 32767;
  int16_t sat_sym_scaled_diff =
      std::min(std::max(sym_scaled_diff, static_cast<int32_t>(-32768)),
               static_cast<int32_t>(32767));
  // apply the exp() LUT activation function
  return lut_lookup(sat_sym_scaled_diff, params.exp_lut);
}
// Quantized softmax with int16_t input and int16_t output.
inline void SoftmaxInt16(const SoftmaxParams& params,
                         const RuntimeShape& input_shape,
                         const int16_t* input_data,
                         const RuntimeShape& output_shape,
                         int16_t* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsoftmaxDTh mht_2(mht_2_v, 344, "", "./tensorflow/lite/kernels/internal/reference/softmax.h", "SoftmaxInt16");

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    // Find the largest element
    int16_t max_in_row = std::numeric_limits<int16_t>::min();
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    // This loops computes the exp values and their sum. We will need the exp
    // values later on in the function so we cache them in the output_data
    // buffer. This is an optimization done to avoid calculating the exp values
    // twice making use of the output_data buffer as scratch memory.
    int32_t sum_of_exps = 0;  // Q16.15 fixed point format.
    int16_t* exp_results_Q015 = output_data + i * depth;
    for (int c = 0; c < depth; ++c) {
      exp_results_Q015[c] =
          SoftMaxCalculateExp(params, input_data, depth, max_in_row, i, c);
      sum_of_exps += exp_results_Q015[c];
    }

    // Compute the reciprocal 1/sum_of_exps
    uint8_t headroom_plus_one =
        CountLeadingZeros(static_cast<uint32_t>(sum_of_exps));
    int32_t shifted_sum =
        ((static_cast<int64_t>(sum_of_exps) << (headroom_plus_one - 1)) +
         (1 << 13)) >>
        14;
    // since the LUT computes 1/(1 + x) we need to first compute x = (sum - 1).
    // also, the LUT expects a symmetrical input, so we must also recenter x
    // from [0, 65535] to [-32768, 32767].
    int32_t sym_shifted_sum = shifted_sum + (-((1 << 15) + (1 << 16)));
    int16_t sat_sym_shifted_sum = static_cast<int16_t>(
        std::min(std::max(sym_shifted_sum, static_cast<int32_t>(-32768)),
                 static_cast<int32_t>(32767)));
    // apply 1/(1 + x) LUT activation function
    int16_t reciprocal_scale_Q015 =
        lut_lookup(sat_sym_shifted_sum, params.one_over_one_plus_x_lut);

    // Rescale the exp_result with reciprocal
    // range of output is [0, 32767] correspond to [0.0, 1.0]
    for (int c = 0; c < depth; ++c) {
      uint8_t right_shift = 31 - headroom_plus_one;
      int64_t round = 1 << (right_shift - 1);
      int32_t result = (static_cast<int64_t>(exp_results_Q015[c]) *
                            static_cast<int64_t>(reciprocal_scale_Q015) +
                        round) >>
                       right_shift;
      output_data[i * depth + c] = static_cast<int16_t>(
          std::min(std::max(result, static_cast<int32_t>(0)),
                   static_cast<int32_t>(32767)));
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SOFTMAX_H_
