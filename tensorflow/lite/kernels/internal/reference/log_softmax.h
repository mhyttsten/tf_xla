/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOG_SOFTMAX_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOG_SOFTMAX_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlog_softmaxDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlog_softmaxDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlog_softmaxDTh() {
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


#include <algorithm>
#include <cstddef>
#include <limits>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {

inline void LogSoftmax(const SoftmaxParams& params,
                       const RuntimeShape& input_shape, const float* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlog_softmaxDTh mht_0(mht_0_v, 199, "", "./tensorflow/lite/kernels/internal/reference/log_softmax.h", "LogSoftmax");

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // log(exp(x[i])/sum(exp(x[i]))) == log(exp(x[i]+C)/sum(exp(x[i]+C)))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c) {
      max = std::max(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      sum += std::exp(input_data[i * depth + c] - max);
    }

    // Compute result.
    const float log_sum = std::log(sum);
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] = input_data[i * depth + c] - max - log_sum;
    }
  }
}

inline void LogSoftmax(const SoftmaxParams& params,
                       const RuntimeShape& input_shape,
                       const uint8_t* input_data,
                       const RuntimeShape& output_shape, uint8_t* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlog_softmaxDTh mht_1(mht_1_v, 235, "", "./tensorflow/lite/kernels/internal/reference/log_softmax.h", "LogSoftmax");

  const int32_t input_multiplier = params.input_multiplier;
  const int32_t input_left_shift = params.input_left_shift;
  const int32_t reverse_scaling_divisor = params.reverse_scaling_divisor;
  const int32_t reverse_scaling_right_shift =
      params.reverse_scaling_right_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large
  // as -32 before multiplying by input_beta_multiplier, and therefore as
  // large as -16 afterwards.  Note that exp(-8) is definitely not
  // insignificant to accumulation, but exp(-16) definitely is.
  static constexpr int kScaledDiffIntegerBits = 5;
  static constexpr int kAccumulationIntegerBits = 12;
  static constexpr int kOutputIntegerBits = 4;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
  using FixedPointAccum =
      gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    uint8_t max_in_row = 0;
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
                input_diff, input_multiplier, input_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    const int32_t fixed_log_sum_of_exps =
        log_x_for_x_greater_than_or_equal_to_1<kScaledDiffIntegerBits>(
            sum_of_exps)
            .raw();

    // rescaled_diff_min is smallest representable in
    // Q(kScaledDiffIntegerBits).(31-kScaledDiffIntegerBits) plus the
    // log-sub-exps that will be subtracted in the loop.
    //
    // The thresholds diff_min, etc are negative.
    const int rescaled_diff_min =
        fixed_log_sum_of_exps + std::numeric_limits<int32_t>::lowest();
    const int adjusted_diff_min =
        std::max(static_cast<int32_t>(
                     diff_min - 1),  // Note use of > below instead of >= above.
                 MultiplyByQuantizedMultiplierSmallerThanOneExp(
                     rescaled_diff_min, reverse_scaling_divisor,
                     -reverse_scaling_right_shift));

    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff > adjusted_diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
        int32_t unsat_output =
            gemmlowp::RoundingDivideByPOT(
                (input_diff_rescaled - fixed_log_sum_of_exps),
                31 - kScaledDiffIntegerBits - kOutputIntegerBits) +
            255;

        output_data[i * depth + c] = static_cast<uint8_t>(
            std::max(std::min(unsat_output, static_cast<int32_t>(255)),
                     static_cast<int32_t>(0)));
      } else {
        // Set output to smallest value.
        output_data[i * depth + c] = 0;
      }
    }
  }
}

template <typename T>
inline void LogSoftmaxQuantized(const SoftmaxParams& params,
                                const size_t outer_size, const size_t depth,
                                const RuntimeShape& input_shape,
                                const T* input_data,
                                const RuntimeShape& output_shape,
                                T* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlog_softmaxDTh mht_2(mht_2_v, 334, "", "./tensorflow/lite/kernels/internal/reference/log_softmax.h", "LogSoftmaxQuantized");

  const int32_t input_multiplier = params.input_multiplier;
  const int32_t input_left_shift = params.input_left_shift;
  const int32_t reverse_scaling_divisor = params.reverse_scaling_divisor;
  const int32_t reverse_scaling_right_shift =
      params.reverse_scaling_right_shift;
  const int diff_min = params.diff_min;

  static constexpr T kMinT8 = std::numeric_limits<T>::min();
  static constexpr T kMaxT8 = std::numeric_limits<T>::max();
  static constexpr int32_t kMinInt32 = std::numeric_limits<int32_t>::min();

  // All IntegerBits must agree with Prepare function.
  // Input is chosen as Q5.26 so exp(-1 * 2^5 * 2^-1) = exp(-16) is negligible.
  static constexpr int kInputIntegerBits = 5;
  static constexpr int kAccumulationIntegerBits = 12;
  static constexpr int kOutputIntegerBits = 4;
  using F5 = gemmlowp::FixedPoint<int32_t, kInputIntegerBits>;
  using F12 = gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;

  for (size_t outer_index = 0; outer_index < outer_size; ++outer_index) {
    T max_in_row = kMinT8;
    for (size_t inner_index = 0; inner_index < depth; ++inner_index) {
      max_in_row =
          std::max(max_in_row, input_data[outer_index * depth + inner_index]);
    }

    // Accumulator "sum_of_exps_in_q12" is safe from overflowing in 2^12 steps.
    F12 sum_of_exps_in_q12 = F12::FromRaw(0);
    for (size_t inner_index = 0; inner_index < depth; ++inner_index) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[outer_index * depth + inner_index]) -
          max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_in_q5 = MultiplyByQuantizedMultiplier(
            input_diff, input_multiplier, input_left_shift);
        sum_of_exps_in_q12 =
            sum_of_exps_in_q12 +
            gemmlowp::Rescale<kAccumulationIntegerBits>(
                exp_on_negative_values(F5::FromRaw(input_diff_in_q5)));
      }
    }

    const int32_t log_sum_of_exps_in_q5 =
        log_x_for_x_greater_than_or_equal_to_1<kInputIntegerBits>(
            sum_of_exps_in_q12)
            .raw();

    // Potentially reduced the valid range. shifted_log_sum_of_exps_in_q5 is
    // smallest representable in Q5.26 plus the log_sum_of_exps.
    const int32_t shifted_log_sum_of_exps_in_q5 =
        log_sum_of_exps_in_q5 + kMinInt32;
    const int32_t adjusted_diff_min =
        std::max(static_cast<int32_t>(diff_min - 1),
                 MultiplyByQuantizedMultiplier(shifted_log_sum_of_exps_in_q5,
                                               reverse_scaling_divisor,
                                               -reverse_scaling_right_shift));

    for (size_t inner_index = 0; inner_index < depth; ++inner_index) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[outer_index * depth + inner_index]) -
          max_in_row;
      // Note use of > below instead of >= above.
      if (input_diff > adjusted_diff_min) {
        const int32_t input_diff_in_q5 = MultiplyByQuantizedMultiplier(
            input_diff, input_multiplier, input_left_shift);

        // Rescale and downcast.
        int32_t output_in_q27 =
            gemmlowp::RoundingDivideByPOT(
                (input_diff_in_q5 - log_sum_of_exps_in_q5),
                31 - kInputIntegerBits - kOutputIntegerBits) +
            kMaxT8;

        output_in_q27 =
            std::max(std::min(output_in_q27, static_cast<int32_t>(kMaxT8)),
                     static_cast<int32_t>(kMinT8));
        output_data[outer_index * depth + inner_index] =
            static_cast<T>(output_in_q27);
      } else {
        output_data[outer_index * depth + inner_index] = kMinT8;
      }
    }
  }
}

inline void LogSoftmax(const SoftmaxParams& params, const size_t outer_size,
                       const size_t depth, const RuntimeShape& input_shape,
                       const int8_t* input_data,
                       const RuntimeShape& output_shape, int8_t* output_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlog_softmaxDTh mht_3(mht_3_v, 426, "", "./tensorflow/lite/kernels/internal/reference/log_softmax.h", "LogSoftmax");

  LogSoftmaxQuantized(params, outer_size, depth, input_shape, input_data,
                      output_shape, output_data);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOG_SOFTMAX_H_
