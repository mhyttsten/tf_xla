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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_COMPARISONS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_COMPARISONS_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh() {
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


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline bool EqualFn(T lhs, T rhs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh mht_0(mht_0_v, 196, "", "./tensorflow/lite/kernels/internal/reference/comparisons.h", "EqualFn");

  return lhs == rhs;
}

template <typename T>
inline bool NotEqualFn(T lhs, T rhs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh mht_1(mht_1_v, 204, "", "./tensorflow/lite/kernels/internal/reference/comparisons.h", "NotEqualFn");

  return lhs != rhs;
}

template <typename T>
inline bool GreaterFn(T lhs, T rhs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh mht_2(mht_2_v, 212, "", "./tensorflow/lite/kernels/internal/reference/comparisons.h", "GreaterFn");

  return lhs > rhs;
}
template <typename T>
inline bool GreaterEqualFn(T lhs, T rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh mht_3(mht_3_v, 219, "", "./tensorflow/lite/kernels/internal/reference/comparisons.h", "GreaterEqualFn");

  return lhs >= rhs;
}
template <typename T>
inline bool LessFn(T lhs, T rhs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh mht_4(mht_4_v, 226, "", "./tensorflow/lite/kernels/internal/reference/comparisons.h", "LessFn");

  return lhs < rhs;
}
template <typename T>
inline bool LessEqualFn(T lhs, T rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh mht_5(mht_5_v, 233, "", "./tensorflow/lite/kernels/internal/reference/comparisons.h", "LessEqualFn");

  return lhs <= rhs;
}

template <typename T>
using ComparisonFn = bool (*)(T, T);

template <typename T, ComparisonFn<T> F>
inline void ComparisonImpl(
    const ComparisonParams& op_params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, bool* output_data) {
  const int64_t flatsize =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int64_t i = 0; i < flatsize; ++i) {
    output_data[i] = F(input1_data[i], input2_data[i]);
  }
}

template <ComparisonFn<float> F>
inline void Comparison(const ComparisonParams& op_params,
                       const RuntimeShape& input1_shape,
                       const float* input1_data,
                       const RuntimeShape& input2_shape,
                       const float* input2_data,
                       const RuntimeShape& output_shape, bool* output_data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh mht_6(mht_6_v, 261, "", "./tensorflow/lite/kernels/internal/reference/comparisons.h", "Comparison");

  ComparisonImpl<float, F>(op_params, input1_shape, input1_data, input2_shape,
                           input2_data, output_shape, output_data);
}

template <typename T, ComparisonFn<int32_t> F>
inline void ComparisonWithScaling(
    const ComparisonParams& op_params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, bool* output_data) {
  int left_shift = op_params.left_shift;
  int32_t input1_offset = op_params.input1_offset;
  int32_t input1_multiplier = op_params.input1_multiplier;
  int input1_shift = op_params.input1_shift;
  int32_t input2_offset = op_params.input2_offset;
  int32_t input2_multiplier = op_params.input2_multiplier;
  int input2_shift = op_params.input2_shift;

  const int64_t flatsize =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int64_t i = 0; i < flatsize; ++i) {
    const int32_t input1_val = input1_offset + input1_data[i];
    const int32_t input2_val = input2_offset + input2_data[i];
    const int32_t shifted_input1_val = input1_val * (1 << left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << left_shift);
    const int32_t scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, input1_multiplier, input1_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, input2_multiplier, input2_shift);
    output_data[i] = F(scaled_input1_val, scaled_input2_val);
  }
}

struct BroadcastComparison4DSlowCommon {
  const RuntimeShape output_shape;
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
};

inline BroadcastComparison4DSlowCommon BroadcastComparison4DSlowPreprocess(
    const RuntimeShape& unextended_input1_shape,
    const RuntimeShape& unextended_input2_shape,
    const RuntimeShape& unextended_output_shape) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh mht_7(mht_7_v, 308, "", "./tensorflow/lite/kernels/internal/reference/comparisons.h", "BroadcastComparison4DSlowPreprocess");

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  return {RuntimeShape::ExtendedShape(4, unextended_output_shape), desc1,
          desc2};
}

template <typename T, ComparisonFn<T> F>
inline void BroadcastComparison4DSlowImpl(
    const ComparisonParams& op_params,
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const T* input2_data,
    const RuntimeShape& unextended_output_shape, bool* output_data) {
  const BroadcastComparison4DSlowCommon dims =
      BroadcastComparison4DSlowPreprocess(unextended_input1_shape,
                                          unextended_input2_shape,
                                          unextended_output_shape);

  for (int b = 0; b < dims.output_shape.Dims(0); ++b) {
    for (int y = 0; y < dims.output_shape.Dims(1); ++y) {
      for (int x = 0; x < dims.output_shape.Dims(2); ++x) {
        for (int c = 0; c < dims.output_shape.Dims(3); ++c) {
          output_data[Offset(dims.output_shape, b, y, x, c)] =
              F(input1_data[SubscriptToIndex(dims.desc1, b, y, x, c)],
                input2_data[SubscriptToIndex(dims.desc2, b, y, x, c)]);
        }
      }
    }
  }
}

template <ComparisonFn<float> F>
inline void BroadcastComparison4DSlow(const ComparisonParams& op_params,
                                      const RuntimeShape& input1_shape,
                                      const float* input1_data,
                                      const RuntimeShape& input2_shape,
                                      const float* input2_data,
                                      const RuntimeShape& output_shape,
                                      bool* output_data) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScomparisonsDTh mht_8(mht_8_v, 354, "", "./tensorflow/lite/kernels/internal/reference/comparisons.h", "BroadcastComparison4DSlow");

  BroadcastComparison4DSlowImpl<float, F>(op_params, input1_shape, input1_data,
                                          input2_shape, input2_data,
                                          output_shape, output_data);
}

template <typename T, ComparisonFn<int32_t> F>
inline void BroadcastComparison4DSlowWithScaling(
    const ComparisonParams& op_params,
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const T* input2_data,
    const RuntimeShape& unextended_output_shape, bool* output_data) {
  const BroadcastComparison4DSlowCommon dims =
      BroadcastComparison4DSlowPreprocess(unextended_input1_shape,
                                          unextended_input2_shape,
                                          unextended_output_shape);

  int left_shift = op_params.left_shift;
  int32_t input1_offset = op_params.input1_offset;
  int32_t input1_multiplier = op_params.input1_multiplier;
  int input1_shift = op_params.input1_shift;
  int32_t input2_offset = op_params.input2_offset;
  int32_t input2_multiplier = op_params.input2_multiplier;
  int input2_shift = op_params.input2_shift;

  for (int b = 0; b < dims.output_shape.Dims(0); ++b) {
    for (int y = 0; y < dims.output_shape.Dims(1); ++y) {
      for (int x = 0; x < dims.output_shape.Dims(2); ++x) {
        for (int c = 0; c < dims.output_shape.Dims(3); ++c) {
          const int32_t input1_val =
              input1_offset +
              input1_data[SubscriptToIndex(dims.desc1, b, y, x, c)];
          const int32_t input2_val =
              input2_offset +
              input2_data[SubscriptToIndex(dims.desc2, b, y, x, c)];
          const int32_t shifted_input1_val = input1_val * (1 << left_shift);
          const int32_t shifted_input2_val = input2_val * (1 << left_shift);
          const int32_t scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input1_val, input1_multiplier, input1_shift);
          const int32_t scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input2_val, input2_multiplier, input2_shift);
          output_data[Offset(dims.output_shape, b, y, x, c)] =
              F(scaled_input1_val, scaled_input2_val);
        }
      }
    }
  }
}

#define TFLITE_COMPARISON_OP(name)                                             \
  inline void name(const ComparisonParams& op_params,                          \
                   const RuntimeShape& input1_shape, const float* input1_data, \
                   const RuntimeShape& input2_shape, const float* input2_data, \
                   const RuntimeShape& output_shape, bool* output_data) {      \
    Comparison<name##Fn>(op_params, input1_shape, input1_data, input2_shape,   \
                         input2_data, output_shape, output_data);              \
  }                                                                            \
  template <typename T>                                                        \
  inline void name##NoScaling(                                                 \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    ComparisonImpl<T, name##Fn>(op_params, input1_shape, input1_data,          \
                                input2_shape, input2_data, output_shape,       \
                                output_data);                                  \
  }                                                                            \
  template <typename T>                                                        \
  inline void name##WithScaling(                                               \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    ComparisonWithScaling<T, name##Fn>(op_params, input1_shape, input1_data,   \
                                       input2_shape, input2_data,              \
                                       output_shape, output_data);             \
  }                                                                            \
  template <typename T>                                                        \
  inline void Broadcast4DSlow##name##NoScaling(                                \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    BroadcastComparison4DSlowImpl<T, name##Fn>(                                \
        op_params, input1_shape, input1_data, input2_shape, input2_data,       \
        output_shape, output_data);                                            \
  }                                                                            \
  inline void Broadcast4DSlow##name(                                           \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const float* input1_data, const RuntimeShape& input2_shape,              \
      const float* input2_data, const RuntimeShape& output_shape,              \
      bool* output_data) {                                                     \
    BroadcastComparison4DSlow<name##Fn>(op_params, input1_shape, input1_data,  \
                                        input2_shape, input2_data,             \
                                        output_shape, output_data);            \
  }                                                                            \
  template <typename T>                                                        \
  inline void Broadcast4DSlow##name##WithScaling(                              \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    BroadcastComparison4DSlowWithScaling<T, name##Fn>(                         \
        op_params, input1_shape, input1_data, input2_shape, input2_data,       \
        output_shape, output_data);                                            \
  }
TFLITE_COMPARISON_OP(Equal);
TFLITE_COMPARISON_OP(NotEqual);
TFLITE_COMPARISON_OP(Greater);
TFLITE_COMPARISON_OP(GreaterEqual);
TFLITE_COMPARISON_OP(Less);
TFLITE_COMPARISON_OP(LessEqual);
#undef TFLITE_COMPARISON_OP

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_COMPARISONS_H_
