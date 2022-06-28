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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LOG_SOFTMAX_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LOG_SOFTMAX_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSinteger_opsPSlog_softmaxDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSinteger_opsPSlog_softmaxDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSinteger_opsPSlog_softmaxDTh() {
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


#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

inline void LogSoftmax(int32_t input_multiplier, int32_t input_shift,
                       int32_t reverse_multiplier, int32_t reverse_shift,
                       int32_t diff_min, int32_t outer_size, int32_t depth,
                       const int8* input_data, int8* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSinteger_opsPSlog_softmaxDTh mht_0(mht_0_v, 195, "", "./tensorflow/lite/kernels/internal/reference/integer_ops/log_softmax.h", "LogSoftmax");

  static constexpr int8_t kMinInt8 = std::numeric_limits<int8_t>::min();
  static constexpr int8_t kMaxInt8 = std::numeric_limits<int8_t>::max();
  static constexpr int32_t kMinInt32 = std::numeric_limits<int32_t>::min();

  // [-16, 0] is mapped to [-128, 127] with 1/16 as scale and 127 as zero
  // point. This nudges the output to [-255/16, 0].
  static constexpr int32_t kOutputZeroPoint = 127;

  // All IntegerBits must agree with Prepare function.
  // Input is chosen as Q5.26 so exp(-1 * 2^5 * 2^-1) = exp(-16) is negligible.
  static constexpr int kInputIntegerBits = 5;
  static constexpr int kAccumulationIntegerBits = 12;
  static constexpr int kOutputIntegerBits = 4;
  using F5 = gemmlowp::FixedPoint<int32, kInputIntegerBits>;
  using F12 = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;

  for (int outer_index = 0; outer_index < outer_size; ++outer_index) {
    int8 max_in_row = kMinInt8;
    for (int inner_index = 0; inner_index < depth; ++inner_index) {
      max_in_row =
          std::max(max_in_row, input_data[outer_index * depth + inner_index]);
    }

    // Accumulator "sum_of_exps_in_q12" is safe from overflowing in 2^12 steps.
    F12 sum_of_exps_in_q12 = F12::FromRaw(0);
    for (int inner_index = 0; inner_index < depth; ++inner_index) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[outer_index * depth + inner_index]) -
          max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_in_q5 = MultiplyByQuantizedMultiplier(
            input_diff, input_multiplier, input_shift);
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
    const int32_t adjusted_diff_min = std::max(
        diff_min - 1,
        MultiplyByQuantizedMultiplier(shifted_log_sum_of_exps_in_q5,
                                      reverse_multiplier, -reverse_shift));

    for (int inner_index = 0; inner_index < depth; ++inner_index) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[outer_index * depth + inner_index]) -
          max_in_row;
      // Note use of > below instead of >= above.
      if (input_diff > adjusted_diff_min) {
        const int32_t input_diff_in_q5 = MultiplyByQuantizedMultiplier(
            input_diff, input_multiplier, input_shift);

        // Rescale and downcast.
        int32_t output_in_q27 =
            gemmlowp::RoundingDivideByPOT(
                (input_diff_in_q5 - log_sum_of_exps_in_q5),
                31 - kInputIntegerBits - kOutputIntegerBits) +
            kOutputZeroPoint;

        output_in_q27 =
            std::max(std::min(output_in_q27, static_cast<int32_t>(kMaxInt8)),
                     static_cast<int32_t>(kMinInt8));
        output_data[outer_index * depth + inner_index] =
            static_cast<int8_t>(output_in_q27);
      } else {
        output_data[outer_index * depth + inner_index] = kMinInt8;
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LOG_SOFTMAX_H_
