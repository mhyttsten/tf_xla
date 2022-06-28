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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_BILINEAR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_BILINEAR_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSresize_bilinearDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSresize_bilinearDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSresize_bilinearDTh() {
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
#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

inline void ComputeInterpolationValues(const float value, const float scale,
                                       const bool half_pixel_centers,
                                       int32_t input_size, float* scaled_value,
                                       int32_t* lower_bound,
                                       int32_t* upper_bound) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSresize_bilinearDTh mht_0(mht_0_v, 202, "", "./tensorflow/lite/kernels/internal/reference/resize_bilinear.h", "ComputeInterpolationValues");

  if (half_pixel_centers) {
    *scaled_value = (value + 0.5f) * scale - 0.5f;
  } else {
    *scaled_value = value * scale;
  }
  float scaled_value_floor = std::floor(*scaled_value);
  *lower_bound = std::max(static_cast<int32_t>(scaled_value_floor),
                          static_cast<int32_t>(0));
  *upper_bound =
      std::min(static_cast<int32_t>(std::ceil(*scaled_value)), input_size - 1);
}

template <typename T>
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const T* input_data,
                           const RuntimeShape& unextended_output_size_shape,
                           const int32_t* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSresize_bilinearDTh mht_1(mht_1_v, 225, "", "./tensorflow/lite/kernels/internal/reference/resize_bilinear.h", "ResizeBilinear");

  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_size_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32_t input_height = input_shape.Dims(1);
  int32_t input_width = input_shape.Dims(2);
  int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.Dims(0), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(1), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(2), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(3), 2);
  int32_t output_height =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 0)];
  int32_t output_width =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 1)];

  float height_scale = static_cast<float>(input_height) / output_height;
  float width_scale = static_cast<float>(input_width) / output_width;
  if (op_params.align_corners && output_height > 1) {
    height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
  }
  if (op_params.align_corners && output_width > 1) {
    width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
  }
  const float rounding_offset = std::numeric_limits<T>::is_integer ? .5f : .0f;

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y;
      int32_t y0, y1;
      ComputeInterpolationValues(y, height_scale, op_params.half_pixel_centers,
                                 input_height, &input_y, &y0, &y1);
      for (int x = 0; x < output_width; ++x) {
        float input_x;
        int32_t x0, x1;
        ComputeInterpolationValues(x, width_scale, op_params.half_pixel_centers,
                                   input_width, &input_x, &x0, &x1);
        for (int c = 0; c < depth; ++c) {
          T interpolation =
              static_cast<T>(input_data[Offset(input_shape, b, y0, x0, c)] *
                                 (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y1, x0, c)] *
                                 (input_y - y0) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y0, x1, c)] *
                                 (1 - (input_y - y0)) * (input_x - x0) +
                             input_data[Offset(input_shape, b, y1, x1, c)] *
                                 (input_y - y0) * (input_x - x0) +
                             rounding_offset);
          output_data[Offset(output_shape, b, y, x, c)] = interpolation;
        }
      }
    }
  }
}

inline void ComputeInterpolationValuesInteger(
    const int32_t value, const int32_t scale_10, const bool half_pixel_centers,
    int32_t input_size, int32_t* scaled_value, int32_t* lower_bound,
    int32_t* upper_bound) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSresize_bilinearDTh mht_2(mht_2_v, 297, "", "./tensorflow/lite/kernels/internal/reference/resize_bilinear.h", "ComputeInterpolationValuesInteger");

  if (half_pixel_centers) {
    *scaled_value = value * scale_10 + scale_10 / 2 - (1 << 9);
  } else {
    *scaled_value = value * scale_10;
  }
  constexpr int32_t zero = 0;
  *lower_bound = std::max(*scaled_value / (1 << 10), zero);
  *upper_bound =
      std::min((*scaled_value + (1 << 10) - 1) / (1 << 10), input_size - 1);
}

// Same as above but doesn't use any floating-point for the resize
template <typename T>
inline void ResizeBilinearInteger(
    const tflite::ResizeBilinearParams& op_params,
    const RuntimeShape& unextended_input_shape, const T* input_data,
    const RuntimeShape& unextended_output_size_shape,
    const int32_t* output_size_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSresize_bilinearDTh mht_3(mht_3_v, 319, "", "./tensorflow/lite/kernels/internal/reference/resize_bilinear.h", "ResizeBilinearInteger");

  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_size_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int32_t input_height = input_shape.Dims(1);
  const int32_t input_width = input_shape.Dims(2);
  const int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.Dims(0), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(1), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(2), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(3), 2);
  const int32_t output_height =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 0)];
  const int32_t output_width =
      output_size_data[Offset(output_size_shape, 0, 0, 0, 1)];

  int32_t height_scale_10 =
      ((1 << 10) * input_height + output_height / 2) / output_height;
  int32_t width_scale_10 =
      ((1 << 10) * input_width + output_width / 2) / output_width;
  if (op_params.align_corners && output_height > 1) {
    height_scale_10 =
        ((1 << 10) * (input_height - 1) + (output_height - 1) / 2) /
        (output_height - 1);
  }
  if (op_params.align_corners && output_width > 1) {
    width_scale_10 = ((1 << 10) * (input_width - 1) + (output_width - 1) / 2) /
                     (output_width - 1);
  }

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      int32_t input_y, y0, y1;
      ComputeInterpolationValuesInteger(y, height_scale_10,
                                        op_params.half_pixel_centers,
                                        input_height, &input_y, &y0, &y1);
      for (int x = 0; x < output_width; ++x) {
        int32_t input_x, x0, x1;
        ComputeInterpolationValuesInteger(x, width_scale_10,
                                          op_params.half_pixel_centers,
                                          input_width, &input_x, &x0, &x1);
        for (int c = 0; c < depth; ++c) {
          const int64_t output_20_ll =
              static_cast<int64_t>(
                  input_data[Offset(input_shape, b, y0, x0, c)]) *
              ((1 << 10) - (input_y - (1 << 10) * y0)) *
              ((1 << 10) - (input_x - (1 << 10) * x0));
          const int64_t output_20_lu =
              static_cast<int64_t>(
                  input_data[Offset(input_shape, b, y1, x0, c)]) *
              (input_y - (1 << 10) * y0) *
              ((1 << 10) - (input_x - (1 << 10) * x0));
          const int64_t output_20_rl =
              static_cast<int64_t>(
                  input_data[Offset(input_shape, b, y0, x1, c)]) *
              ((1 << 10) - (input_y - (1 << 10) * y0)) *
              (input_x - (1 << 10) * x0);
          const int64_t output_20_ru =
              static_cast<int64_t>(
                  input_data[Offset(input_shape, b, y1, x1, c)]) *
              (input_y - (1 << 10) * y0) * (input_x - (1 << 10) * x0);
          const int64_t output_20 =
              output_20_ll + output_20_lu + output_20_rl + output_20_ru;
          const int64_t round = (output_20 > 0) ? (1 << 19) : -(1 << 19);
          const T interpolation =
              static_cast<T>((output_20 + round) / (1 << 20));
          output_data[Offset(output_shape, b, y, x, c)] = interpolation;
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_RESIZE_BILINEAR_H_
