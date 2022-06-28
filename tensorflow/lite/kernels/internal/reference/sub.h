/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SUB_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SUB_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh() {
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


#include <stdint.h>

#include <algorithm>
#include <limits>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

inline void SubNonBroadcast(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const float* input1_data,
                            const RuntimeShape& input2_shape,
                            const float* input2_data,
                            const RuntimeShape& output_shape,
                            float* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "SubNonBroadcast");

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.float_activation_min,
        params.float_activation_max);
  }
}

inline void SubNonBroadcast(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const int32_t* input1_data,
                            const RuntimeShape& input2_shape,
                            const int32_t* input2_data,
                            const RuntimeShape& output_shape,
                            int32_t* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_1(mht_1_v, 226, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "SubNonBroadcast");

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.quantized_activation_min,
        params.quantized_activation_max);
  }
}

// TODO(b/151345304): We can implement BroadcastSub on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <int N = 5>
inline void BroadcastSubSlow(const ArithmeticParams& params,
                             const RuntimeShape& input1_shape,
                             const float* input1_data,
                             const RuntimeShape& input2_shape,
                             const float* input2_data,
                             const RuntimeShape& output_shape,
                             float* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/float");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_2(mht_2_v, 273, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "lambda");

    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] -
                input2_data[SubscriptToIndex(desc2, indexes)],
            params.float_activation_min, params.float_activation_max);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <int N = 5>
inline void BroadcastSubSlow(const ArithmeticParams& params,
                             const RuntimeShape& input1_shape,
                             const int32_t* input1_data,
                             const RuntimeShape& input2_shape,
                             const int32_t* input2_data,
                             const RuntimeShape& output_shape,
                             int32_t* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/int32_t");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_3(mht_3_v, 316, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "lambda");

    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] -
                input2_data[SubscriptToIndex(desc2, indexes)],
            params.quantized_activation_min, params.quantized_activation_max);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <int N = 5>
void BroadcastSubSlow(const ArithmeticParams& params,
                      const RuntimeShape& input1_shape,
                      const int64_t* input1_data,
                      const RuntimeShape& input2_shape,
                      const int64_t* input2_data,
                      const RuntimeShape& output_shape, int64_t* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/int64_t");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_4(mht_4_v, 358, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "lambda");

    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] -
                input2_data[SubscriptToIndex(desc2, indexes)],
            params.int64_activation_min, params.int64_activation_max);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <typename T, int N = 5>
void BroadcastSubSlow(const ArithmeticParams& params,
                      const RuntimeShape& input1_shape, const T* input1_data,
                      const RuntimeShape& input2_shape, const T* input2_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/templated");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_5(mht_5_v, 398, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "lambda");

    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] -
                input2_data[SubscriptToIndex(desc2, indexes)],
            params.quantized_activation_min, params.quantized_activation_max);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <int N = 5>
inline void BroadcastSub16POTSlow(const ArithmeticParams& params,
                                  const RuntimeShape& input1_shape,
                                  const int16_t* input1_data,
                                  const RuntimeShape& input2_shape,
                                  const int16_t* input2_data,
                                  const RuntimeShape& output_shape,
                                  int16_t* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSub16POTSlow/int16_t");
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_6(mht_6_v, 438, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "lambda");

    const int32_t input1_val = input1_data[SubscriptToIndex(desc1, indexes)];
    const int32_t input2_val = input2_data[SubscriptToIndex(desc2, indexes)];
    const int32_t scaled_input1_val =
        gemmlowp::RoundingDivideByPOT(input1_val, -params.input1_shift);
    const int32_t scaled_input2_val =
        gemmlowp::RoundingDivideByPOT(input2_val, -params.input2_shift);
    const int32_t raw_output = scaled_input1_val - scaled_input2_val;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[SubscriptToIndex(output_desc, indexes)] =
        static_cast<int16_t>(clamped_output);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <typename T, int N = 5>
void BroadcastQuantSubSlow(const ArithmeticParams& params,
                           const RuntimeShape& input1_shape,
                           const T* input1_data,
                           const RuntimeShape& input2_shape,
                           const T* input2_data,
                           const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastQuantSubSlow/T");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_7(mht_7_v, 487, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "lambda");

    const int32_t input1_val =
        params.input1_offset + input1_data[SubscriptToIndex(desc1, indexes)];
    const int32_t input2_val =
        params.input2_offset + input2_data[SubscriptToIndex(desc2, indexes)];
    const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sub = scaled_input1_val - scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sub, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[SubscriptToIndex(output_desc, indexes)] =
        static_cast<T>(clamped_output);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
template <typename T>
inline void SubElementwise(int size, const ArithmeticParams& params,
                           const T* input1_data, const T* input2_data,
                           T* output_data) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_8(mht_8_v, 522, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "SubElementwise");

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
    const int32_t raw_sub = scaled_input1_val - scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sub, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8_t* input1_data,
                const RuntimeShape& input2_shape, const uint8_t* input2_data,
                const RuntimeShape& output_shape, uint8_t* output_data) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_9(mht_9_v, 552, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "Sub");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_10(mht_10_v, 571, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "Sub");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GE(params.input1_offset, -128);
  TFLITE_DCHECK_GE(params.input2_offset, -128);
  // offset = -quantization_params.zero_point in PrepareGeneralSubOp().
  // So it's maximum can be 128 not 127.
  TFLITE_DCHECK_LE(params.input1_offset, 128);
  TFLITE_DCHECK_LE(params.input2_offset, 128);
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16_t* input1_data,
                const RuntimeShape& input2_shape, const int16_t* input2_data,
                const RuntimeShape& output_shape, int16_t* output_data) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_11(mht_11_v, 593, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "Sub");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_EQ(params.input1_offset, 0);
  TFLITE_DCHECK_EQ(params.input2_offset, 0);
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

template <typename T>
void Sub(const ArithmeticParams& params, const RuntimeShape& input1_shape,
         const T* input1_data, const RuntimeShape& input2_shape,
         const T* input2_data, const RuntimeShape& output_shape,
         T* output_data) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_12(mht_12_v, 612, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "Sub");

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
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              input1_data[SubscriptToIndex(desc1, b, y, x, c)] -
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
        }
      }
    }
  }
}

inline void SetActivationMinMax(const ArithmeticParams& params,
                                int32_t* activation_min,
                                int32_t* activation_max) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_13(mht_13_v, 649, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "SetActivationMinMax");

  *activation_min = params.quantized_activation_min;
  *activation_max = params.quantized_activation_max;
}

inline void SetActivationMinMax(const ArithmeticParams& params,
                                float* activation_min, float* activation_max) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_14(mht_14_v, 658, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "SetActivationMinMax");

  *activation_min = params.float_activation_min;
  *activation_max = params.float_activation_max;
}

inline void SetActivationMinMax(const ArithmeticParams& params,
                                int64_t* activation_min,
                                int64_t* activation_max) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_15(mht_15_v, 668, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "SetActivationMinMax");

  *activation_min = params.int64_activation_min;
  *activation_max = params.int64_activation_max;
}

template <typename T>
inline void SubWithActivation(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsubDTh mht_16(mht_16_v, 680, "", "./tensorflow/lite/kernels/internal/reference/sub.h", "SubWithActivation");

  ruy::profiler::ScopeLabel label("SubWithActivation");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  T activation_min, activation_max;
  SetActivationMinMax(params, &activation_min, &activation_max);

  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], activation_min, activation_max);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SUB_H_
