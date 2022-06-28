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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONCATENATION_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONCATENATION_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSconcatenationDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSconcatenationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSconcatenationDTh() {
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
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename Scalar>
inline void Concatenation(const ConcatenationParams& params,
                          const RuntimeShape* const* input_shapes,
                          const Scalar* const* input_data,
                          const RuntimeShape& output_shape,
                          Scalar* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSconcatenationDTh mht_0(mht_0_v, 201, "", "./tensorflow/lite/kernels/internal/reference/concatenation.h", "Concatenation");

  int axis = params.axis;
  int inputs_count = params.inputs_count;
  const int concat_dimensions = output_shape.DimensionsCount();
  TFLITE_DCHECK_LT(axis, concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    TFLITE_DCHECK_EQ(input_shapes[i]->DimensionsCount(), concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++) {
      if (j != axis) {
        MatchingDim(*input_shapes[i], j, output_shape, j);
      }
    }
    concat_size += input_shapes[i]->Dims(axis);
  }
  TFLITE_DCHECK_EQ(concat_size, output_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_shape.Dims(i);
  }
  // For all input arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i) {
    base_inner_size *= output_shape.Dims(i);
  }

  Scalar* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size = input_shapes[i]->Dims(axis) * base_inner_size;
      const Scalar* input_ptr = input_data[i] + k * copy_size;
      memcpy(output_ptr, input_ptr, copy_size * sizeof(Scalar));
      output_ptr += copy_size;
    }
  }
}

// TODO(b/174275780): The quantized implementation of concatentation isn't fully
// quantized as it takes scale as a floating point value. This should be fixed
// when optimizng this routine further.
inline void ConcatenationWithScaling(const ConcatenationParams& params,
                                     const RuntimeShape* const* input_shapes,
                                     const uint8_t* const* input_data,
                                     const RuntimeShape& output_shape,
                                     uint8_t* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSconcatenationDTh mht_1(mht_1_v, 250, "", "./tensorflow/lite/kernels/internal/reference/concatenation.h", "ConcatenationWithScaling");

  int axis = params.axis;
  const int32_t* input_zeropoint = params.input_zeropoint;
  const float* input_scale = params.input_scale;
  int inputs_count = params.inputs_count;
  const int32_t output_zeropoint = params.output_zeropoint;
  const float output_scale = params.output_scale;

  const int concat_dimensions = output_shape.DimensionsCount();
  TFLITE_DCHECK_LT(axis, concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    TFLITE_DCHECK_EQ(input_shapes[i]->DimensionsCount(), concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++) {
      if (j != axis) {
        MatchingDim(*input_shapes[i], j, output_shape, j);
      }
    }
    concat_size += input_shapes[i]->Dims(axis);
  }
  TFLITE_DCHECK_EQ(concat_size, output_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_shape.Dims(i);
  }
  // For all input arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i) {
    base_inner_size *= output_shape.Dims(i);
  }

  const float inverse_output_scale = 1.f / output_scale;
  uint8_t* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size = input_shapes[i]->Dims(axis) * base_inner_size;
      const uint8_t* input_ptr = input_data[i] + k * copy_size;
      if (input_zeropoint[i] == output_zeropoint &&
          input_scale[i] == output_scale) {
        memcpy(output_ptr, input_ptr, copy_size);
      } else {
        const float scale = input_scale[i] * inverse_output_scale;
        const float bias = -input_zeropoint[i] * scale;
        for (int j = 0; j < copy_size; ++j) {
          const int32_t value = static_cast<int32_t>(tflite::TfLiteRound(
                                    input_ptr[j] * scale + bias)) +
                                output_zeropoint;
          output_ptr[j] = static_cast<uint8_t>(
              std::max<int32_t>(std::min<int32_t>(255, value), 0));
        }
      }
      output_ptr += copy_size;
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONCATENATION_H_
