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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh() {
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


#include <type_traits>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh mht_0(mht_0_v, 200, "", "./tensorflow/lite/kernels/internal/reference/add.h", "Add");

  T activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] + input2_data[i], activation_min, activation_max);
  }
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.

// This function is used for 8-bit as well as for 16-bit, but the accumulator
// is 32-bit for both cases. The overflow does not happen due to the
// choice of the shift (20 or 15, accordingly - see add.cc for more comments).
template <typename T>
inline void AddElementwise(int size, const ArithmeticParams& params,
                           const T* input1_data, const T* input2_data,
                           T* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh mht_1(mht_1_v, 224, "", "./tensorflow/lite/kernels/internal/reference/add.h", "AddElementwise");

  TFLITE_DCHECK_GT(params.input1_offset, -std::numeric_limits<T>::max());
  TFLITE_DCHECK_GT(params.input2_offset, -std::numeric_limits<T>::max());
  TFLITE_DCHECK_LT(params.input1_offset, std::numeric_limits<T>::max());
  TFLITE_DCHECK_LT(params.input2_offset, std::numeric_limits<T>::max());

  for (int i = 0; i < size; ++i) {
    const int32_t input1_val = params.input1_offset + input1_data[i];
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

// Scalar-broadcast add that can be used for inner loop of more general
// broadcast add, so that, for example, scalar-broadcast with batch will still
// be fast.
inline void AddScalarBroadcast(int size, const ArithmeticParams& params,
                               uint8_t input1_data, const uint8_t* input2_data,
                               uint8_t* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh mht_2(mht_2_v, 261, "", "./tensorflow/lite/kernels/internal/reference/add.h", "AddScalarBroadcast");

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);

  const int32_t input1_val = params.input1_offset + input1_data;
  const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
  const int32_t scaled_input1_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input1_val, params.input1_multiplier, params.input1_shift);
  for (int i = 0; i < size; ++i) {
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8_t* input1_data,
                const RuntimeShape& input2_shape, const uint8_t* input2_data,
                const RuntimeShape& output_shape, uint8_t* output_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh mht_3(mht_3_v, 296, "", "./tensorflow/lite/kernels/internal/reference/add.h", "Add");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void AddGeneralParamScale(const ArithmeticParams& params,
                                 const RuntimeShape& input1_shape,
                                 const int16_t* input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int16_t* input2_data,
                                 const RuntimeShape& output_shape,
                                 int16_t* output_data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh mht_4(mht_4_v, 318, "", "./tensorflow/lite/kernels/internal/reference/add.h", "AddGeneralParamScale");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  int max_value = std::numeric_limits<int16_t>::max();

  TFLITE_DCHECK_GT(params.input1_offset, -max_value);
  TFLITE_DCHECK_GT(params.input2_offset, -max_value);
  TFLITE_DCHECK_LT(params.input1_offset, max_value);
  TFLITE_DCHECK_LT(params.input2_offset, max_value);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16_t* input1_data,
                const RuntimeShape& input2_shape, const int16_t* input2_data,
                const RuntimeShape& output_shape, int16_t* output_data,
                bool pot_scale = true) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh mht_5(mht_5_v, 340, "", "./tensorflow/lite/kernels/internal/reference/add.h", "Add");

  if (!pot_scale) {
    AddGeneralParamScale(params, input1_shape, input1_data, input2_shape,
                         input2_data, output_shape, output_data);
    return;
  }

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int input1_shift = params.input1_shift;
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  const int16_t output_activation_min = params.quantized_activation_min;
  const int16_t output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK(input1_shift == 0 || params.input2_shift == 0);
  TFLITE_DCHECK_LE(input1_shift, 0);
  TFLITE_DCHECK_LE(params.input2_shift, 0);
  const int16_t* not_shift_input =
      input1_shift == 0 ? input1_data : input2_data;
  const int16_t* shift_input = input1_shift == 0 ? input2_data : input1_data;
  const int input_right_shift =
      input1_shift == 0 ? -params.input2_shift : -input1_shift;

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
    F0 scaled_input = F0::FromRaw(
        gemmlowp::RoundingDivideByPOT(shift_input[i], input_right_shift));
    F0 result = gemmlowp::SaturatingAdd(scaled_input, input_ready_scaled);
    const int16_t raw_output = result.raw();
    const int16_t clamped_output = std::min(
        output_activation_max, std::max(output_activation_min, raw_output));
    output_data[i] = clamped_output;
  }
}

template <typename T>
inline typename std::enable_if<!is_small_integer<T>::value, void>::type
BroadcastAdd4DSlow(const ArithmeticParams& params,
                   const RuntimeShape& input1_shape, const T* input1_data,
                   const RuntimeShape& input2_shape, const T* input2_data,
                   const RuntimeShape& output_shape, T* output_data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh mht_6(mht_6_v, 388, "", "./tensorflow/lite/kernels/internal/reference/add.h", "BroadcastAdd4DSlow");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  T activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax<T>(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] +
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  activation_min, activation_max);
        }
      }
    }
  }
}

// This function is used for 8-bit as well as for 16-bit, but the accumulator
// is 32-bit for both cases. The overflow does not happen due to the
// choice of the shift (20 or 15, accordingly - see add.cc for more comments).
template <typename T>
inline typename std::enable_if<is_small_integer<T>::value, void>::type
BroadcastAdd4DSlow(const ArithmeticParams& params,
                   const RuntimeShape& input1_shape, const T* input1_data,
                   const RuntimeShape& input2_shape, const T* input2_data,
                   const RuntimeShape& output_shape, T* output_data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh mht_7(mht_7_v, 436, "", "./tensorflow/lite/kernels/internal/reference/add.h", "BroadcastAdd4DSlow");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          const int32_t input1_val =
              params.input1_offset +
              input1_data[SubscriptToIndex(desc1, b, y, x, c)];
          const int32_t input2_val =
              params.input2_offset +
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
          const int32_t shifted_input1_val =
              input1_val * (1 << params.left_shift);
          const int32_t shifted_input2_val =
              input2_val * (1 << params.left_shift);
          const int32_t scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input1_val, params.input1_multiplier,
                  params.input1_shift);
          const int32_t scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input2_val, params.input2_multiplier,
                  params.input2_shift);
          const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
          const int32_t raw_output =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  raw_sum, params.output_multiplier, params.output_shift) +
              params.output_offset;
          const int32_t clamped_output =
              std::min(params.quantized_activation_max,
                       std::max(params.quantized_activation_min, raw_output));
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              static_cast<T>(clamped_output);
        }
      }
    }
  }
}

inline void BroadcastAddFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const uint8_t* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const uint8_t* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 uint8_t* output_data) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSaddDTh mht_8(mht_8_v, 502, "", "./tensorflow/lite/kernels/internal/reference/add.h", "BroadcastAddFivefold");

  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input1_multiplier = unswitched_params.input2_multiplier;
  switched_params.input1_shift = unswitched_params.input2_shift;
  switched_params.input2_offset = unswitched_params.input1_offset;
  switched_params.input2_multiplier = unswitched_params.input1_multiplier;
  switched_params.input2_shift = unswitched_params.input1_shift;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const uint8_t* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const uint8_t* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise add of
  // sections of the arrays.
  uint8_t* output_data_ptr = output_data;
  const uint8_t* input1_data_ptr = input1_data;
  const uint8_t* input2_data_reset = input2_data;
  // In the fivefold pattern, y0, y2 and y4 are not broadcast, and so shared
  // between input shapes. y3 for input 1 is always broadcast, and so the
  // dimension there is 1, whereas optionally y1 might be broadcast for input 2.
  // Put another way,
  // input1.shape.FlatSize = y0 * y1 * y2 * y4,
  // input2.shape.FlatSize = y0 * y2 * y3 * y4.
  int y0 = params.broadcast_shape[0];
  int y1 = params.broadcast_shape[1];
  int y2 = params.broadcast_shape[2];
  int y3 = params.broadcast_shape[3];
  int y4 = params.broadcast_shape[4];
  if (y4 > 1) {
    // General fivefold pattern, with y4 > 1 so there is a non-broadcast inner
    // dimension.
    for (int i0 = 0; i0 < y0; ++i0) {
      const uint8_t* input2_data_ptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          for (int i3 = 0; i3 < y3; ++i3) {
            AddElementwise(y4, params, input1_data_ptr, input2_data_ptr,
                           output_data_ptr);
            input2_data_ptr += y4;
            output_data_ptr += y4;
          }
          // We have broadcast y4 of input1 data y3 times, and now move on.
          input1_data_ptr += y4;
        }
      }
      // We have broadcast y2*y3*y4 of input2 data y1 times, and now move on.
      input2_data_reset = input2_data_ptr;
    }
  } else {
    // Special case of y4 == 1, in which the innermost loop is a single element
    // and can be combined with the next (y3) as an inner broadcast.
    //
    // Note that this handles the case of pure scalar broadcast when
    // y0 == y1 == y2 == 1. With low overhead it handles cases such as scalar
    // broadcast with batch (as y2 > 1).
    //
    // NOTE The process is the same as the above general case except simplified
    // for y4 == 1 and the loop over y3 is contained within the
    // AddScalarBroadcast function.
    for (int i0 = 0; i0 < y0; ++i0) {
      const uint8_t* input2_data_ptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          AddScalarBroadcast(y3, params, *input1_data_ptr, input2_data_ptr,
                             output_data_ptr);
          input2_data_ptr += y3;
          output_data_ptr += y3;
          input1_data_ptr += 1;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_H_
