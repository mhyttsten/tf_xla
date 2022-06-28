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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CUMSUM_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CUMSUM_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScumsumDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScumsumDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScumsumDTh() {
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
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace reference_ops {

template <typename T>
inline void CumSum(const T* input_data, const RuntimeShape& shape, int32_t axis,
                   bool exclusive, bool reverse, T* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScumsumDTh mht_0(mht_0_v, 199, "", "./tensorflow/lite/kernels/internal/reference/cumsum.h", "CumSum");

  const int32_t rank = shape.DimensionsCount();
  TFLITE_DCHECK_GE(rank, 1);
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, rank);

  size_t inner = 1;
  size_t outer = 1;
  size_t depth = 1;
  for (int32_t i = 0; i < rank; i++) {
    if (i < axis)
      inner *= shape.Dims(i);
    else if (i > axis)
      outer *= shape.Dims(i);
    else
      depth = shape.Dims(i);
  }

  for (size_t outer_index = 0; outer_index < outer; outer_index++) {
    size_t outer_index_adj;
    if (reverse)
      outer_index_adj = (outer - 1) - outer_index;
    else
      outer_index_adj = outer_index;
    for (size_t inner_index = 0; inner_index < inner; inner_index++) {
      T accumulator = 0;
      size_t inner_index_adj;
      if (reverse)
        inner_index_adj = (inner - 1) - inner_index;
      else
        inner_index_adj = inner_index;
      for (size_t depth_index = 0; depth_index < depth; depth_index++) {
        size_t depth_index_adj;
        if (reverse)
          depth_index_adj = (depth - 1) - depth_index;
        else
          depth_index_adj = depth_index;

        size_t index = outer_index_adj;
        index += inner_index_adj * depth * outer;
        index += depth_index_adj * outer;

        if (exclusive) {
          output_data[index] = accumulator;
          accumulator += input_data[index];
        } else {
          accumulator += input_data[index];
          output_data[index] = accumulator;
        }
      }
    }
  }
}

//
// Quantized INT8 CUMSUM
//
inline void CumSum(const ArithmeticParams& params, const int8_t* input_data,
                   const RuntimeShape& shape, int32_t axis, bool exclusive,
                   bool reverse, int8_t* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePScumsumDTh mht_1(mht_1_v, 261, "", "./tensorflow/lite/kernels/internal/reference/cumsum.h", "CumSum");

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  // Input offset is negative input zero point. Activation tensors are
  // asymmetric quantized so they span the full int8 range.
  // All inputs should have same zero-point and scale, this is checked during
  // Prepare stage.
  TFLITE_DCHECK_GE(-params.input1_offset, std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_LE(-params.input1_offset, std::numeric_limits<int8_t>::max());

  const int32_t rank = shape.DimensionsCount();
  TFLITE_DCHECK_GE(rank, 1);
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, rank);

  size_t inner = 1;
  size_t outer = 1;
  size_t depth = 1;
  for (int32_t i = 0; i < rank; i++) {
    if (i < axis)
      inner *= shape.Dims(i);
    else if (i > axis)
      outer *= shape.Dims(i);
    else
      depth = shape.Dims(i);
  }

  for (size_t outer_index = 0; outer_index < outer; outer_index++) {
    size_t outer_index_adj;
    if (reverse)
      outer_index_adj = (outer - 1) - outer_index;
    else
      outer_index_adj = outer_index;
    for (size_t inner_index = 0; inner_index < inner; inner_index++) {
      int32_t accumulator = params.input1_offset;  // accumulator = 0
      accumulator *= (1 << params.left_shift);
      accumulator = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          accumulator, params.input1_multiplier, params.input1_shift);

      size_t inner_index_adj;
      if (reverse)
        inner_index_adj = (inner - 1) - inner_index;
      else
        inner_index_adj = inner_index;

      for (size_t depth_index = 0; depth_index < depth; depth_index++) {
        size_t depth_index_adj;
        if (reverse)
          depth_index_adj = (depth - 1) - depth_index;
        else
          depth_index_adj = depth_index;

        size_t index = outer_index_adj;
        index += inner_index_adj * depth * outer;
        index += depth_index_adj * outer;

        const int32_t y = params.input1_offset + input_data[index];
        const int32_t shifted_y = y * (1 << params.left_shift);
        const int32_t scaled_y = MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_y, params.input1_multiplier, params.input1_shift);

        int32_t scaled_output;
        if (exclusive) {
          scaled_output = accumulator;
          accumulator += scaled_y;
        } else {
          accumulator += scaled_y;
          scaled_output = accumulator;
        }

        const int32_t raw_output =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                scaled_output, params.output_multiplier, params.output_shift) +
            params.output_offset;
        const int32_t clamped_output =
            std::min(params.quantized_activation_max,
                     std::max(params.quantized_activation_min, raw_output));
        output_data[index] = static_cast<int8_t>(clamped_output);
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CUMSUM_H_
