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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DIV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DIV_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh() {
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

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void DivCheckArithmeticParams(const ArithmeticParams& params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh mht_0(mht_0_v, 196, "", "./tensorflow/lite/kernels/internal/reference/div.h", "DivCheckArithmeticParams");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  // Input offset is negative input zero point. Activation tensors are
  // asymmetric quantized so they span the full int8 range.
  constexpr int32_t max_value =
      static_cast<int32_t>(std::numeric_limits<T>::max());
  TFLITE_DCHECK_GE(params.input1_offset, -max_value);
  TFLITE_DCHECK_LE(params.input1_offset, max_value);
  TFLITE_DCHECK_GE(params.input2_offset, -max_value);
  TFLITE_DCHECK_LE(params.input2_offset, max_value);
  TFLITE_DCHECK_GE(params.output_offset, -max_value);
  TFLITE_DCHECK_LE(params.output_offset, max_value);
}

// Element-wise div that can often be used for inner loop of broadcast Div as
// well as the non-broadcast Div.
template <typename T>
inline void DivElementwise(int size, const ArithmeticParams& params,
                           const T* input1_data, const T* input2_data,
                           T* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh mht_1(mht_1_v, 219, "", "./tensorflow/lite/kernels/internal/reference/div.h", "DivElementwise");

  DivCheckArithmeticParams<T>(params);

  for (int i = 0; i < size; ++i) {
    int32_t input1_val = params.input1_offset + input1_data[i];
    int32_t input2_val = params.input2_offset + input2_data[i];
    TFLITE_DCHECK_NE(input2_val, 0);
    if (input2_val < 0) {
      // Invert signs to avoid a negative input2_val as input2_inv needs to be
      // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
      input1_val = -input1_val;
      input2_val = -input2_val;
    }
    int recip_shift;
    const int32_t input2_inv = GetReciprocal(input2_val, 31, &recip_shift);
    const int headroom = CountLeadingSignBits(input1_val);
    const int32_t unscaled_quotient =
        MultiplyByQuantizedMultiplierGreaterThanOne(input1_val, input2_inv,
                                                    headroom);
    const int total_shift = params.output_shift - recip_shift - headroom;
    const int32_t unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            unscaled_quotient, params.output_multiplier, total_shift);
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8_t* input1_data,
                const RuntimeShape& input2_shape, const uint8_t* input2_data,
                const RuntimeShape& output_shape, uint8_t* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh mht_2(mht_2_v, 256, "", "./tensorflow/lite/kernels/internal/reference/div.h", "Div");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  DivElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh mht_3(mht_3_v, 271, "", "./tensorflow/lite/kernels/internal/reference/div.h", "Div");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  DivElementwise(flat_size, params, input1_data, input2_data, output_data);
}

template <typename T, int N = 5>
inline void BroadcastDivSlowQuantized(
    const ArithmeticParams& params, const RuntimeShape& unextended_input1_shape,
    const T* input1_data, const RuntimeShape& unextended_input2_shape,
    const T* input2_data, const RuntimeShape& unextended_output_shape,
    T* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), N);

  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_output_shape),
                 &output_desc);

  DivCheckArithmeticParams<T>(params);

  auto div_func = [&](int indexes[N]) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh mht_4(mht_4_v, 303, "", "./tensorflow/lite/kernels/internal/reference/div.h", "lambda");

    int32_t input1_val =
        params.input1_offset + input1_data[SubscriptToIndex(desc1, indexes)];
    int32_t input2_val =
        params.input2_offset + input2_data[SubscriptToIndex(desc2, indexes)];
    TFLITE_DCHECK_NE(input2_val, 0);
    if (input2_val < 0) {
      // Invert signs to avoid a negative input2_val as input2_inv needs to be
      // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
      input1_val = -input1_val;
      input2_val = -input2_val;
    }
    int recip_shift;
    const int32_t input2_inv = GetReciprocal(input2_val, 31, &recip_shift);
    const int headroom = CountLeadingSignBits(input1_val);
    const int32_t unscaled_quotient =
        MultiplyByQuantizedMultiplierGreaterThanOne(input1_val, input2_inv,
                                                    headroom);
    const int total_shift = params.output_shift - recip_shift - headroom;
    const int32_t unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            unscaled_quotient, params.output_multiplier, total_shift);
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[SubscriptToIndex(output_desc, indexes)] =
        static_cast<T>(clamped_output);
  };
  NDOpsHelper<N>(output_desc, div_func);
}

template <int N = 5>
inline void BroadcastDivSlow(const ArithmeticParams& params,
                             const RuntimeShape& unextended_input1_shape,
                             const uint8_t* input1_data,
                             const RuntimeShape& unextended_input2_shape,
                             const uint8_t* input2_data,
                             const RuntimeShape& unextended_output_shape,
                             uint8_t* output_data) {
  BroadcastDivSlowQuantized<uint8_t, N>(
      params, unextended_input1_shape, input1_data, unextended_input2_shape,
      input2_data, unextended_output_shape, output_data);
}

template <int N = 5>
inline void BroadcastDivSlow(const ArithmeticParams& params,
                             const RuntimeShape& unextended_input1_shape,
                             const int8_t* input1_data,
                             const RuntimeShape& unextended_input2_shape,
                             const int8_t* input2_data,
                             const RuntimeShape& unextended_output_shape,
                             int8_t* output_data) {
  BroadcastDivSlowQuantized<int8_t, N>(
      params, unextended_input1_shape, input1_data, unextended_input2_shape,
      input2_data, unextended_output_shape, output_data);
}

// TODO(jiawen): We can implement BroadcastDiv on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <typename T, int N = 5>
void BroadcastDivSlow(const ArithmeticParams& params,
                      const RuntimeShape& unextended_input1_shape,
                      const T* input1_data,
                      const RuntimeShape& unextended_input2_shape,
                      const T* input2_data,
                      const RuntimeShape& unextended_output_shape,
                      T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), N);

  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_output_shape),
                 &output_desc);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.

  auto div_func = [&](int indexes[N]) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh mht_5(mht_5_v, 400, "", "./tensorflow/lite/kernels/internal/reference/div.h", "lambda");

    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] /
                input2_data[SubscriptToIndex(desc2, indexes)],
            output_activation_min, output_activation_max);
  };
  NDOpsHelper<N>(output_desc, div_func);
}

template <typename T>
inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSdivDTh mht_6(mht_6_v, 417, "", "./tensorflow/lite/kernels/internal/reference/div.h", "Div");

  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] / input2_data[i], output_activation_min,
        output_activation_max);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DIV_H_
