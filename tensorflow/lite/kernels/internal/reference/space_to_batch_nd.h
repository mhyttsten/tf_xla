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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SPACE_TO_BATCH_ND_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SPACE_TO_BATCH_ND_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSspace_to_batch_ndDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSspace_to_batch_ndDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSspace_to_batch_ndDTh() {
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


#include <cmath>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

// TODO(b/135760455): Move this method anonymous namespace in a cc file.
inline RuntimeShape ExtendShapeSpaceToBatch(const RuntimeShape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSspace_to_batch_ndDTh mht_0(mht_0_v, 197, "", "./tensorflow/lite/kernels/internal/reference/space_to_batch_nd.h", "ExtendShapeSpaceToBatch");

  if (shape.DimensionsCount() == 4) {
    return shape;
  }
  RuntimeShape new_shape(4, 1);
  new_shape.SetDim(0, shape.Dims(0));
  new_shape.SetDim(1, shape.Dims(1));
  new_shape.SetDim(3, shape.Dims(2));
  return new_shape;
}

template <typename T>
inline void SpaceToBatchND(const SpaceToBatchParams& params,
                           const RuntimeShape& unextended_input1_shape,
                           const T* input1_data,
                           const RuntimeShape& unextended_input2_shape,
                           const int32_t* block_shape_data,
                           const RuntimeShape& unextended_input3_shape,
                           const int32_t* paddings_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSspace_to_batch_ndDTh mht_1(mht_1_v, 220, "", "./tensorflow/lite/kernels/internal/reference/space_to_batch_nd.h", "SpaceToBatchND");

  ruy::profiler::ScopeLabel label("SpaceToBatchND");
  TFLITE_DCHECK_GE(unextended_input1_shape.DimensionsCount(), 3);
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(unextended_input1_shape.DimensionsCount(),
                   unextended_output_shape.DimensionsCount());

  // Extends the input/output shape from 3D to 4D if needed, NHC -> NH1C.
  const RuntimeShape input1_shape =
      ExtendShapeSpaceToBatch(unextended_input1_shape);
  const RuntimeShape output_shape =
      ExtendShapeSpaceToBatch(unextended_output_shape);

  const int depth = input1_shape.Dims(3);
  const int input_width = input1_shape.Dims(2);
  const int input_height = input1_shape.Dims(1);
  const int input_batch_size = input1_shape.Dims(0);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int block_shape_height = block_shape_data[0];
  const int block_shape_width =
      unextended_input1_shape.DimensionsCount() == 4 ? block_shape_data[1] : 1;
  const int padding_top = paddings_data[0];
  const int padding_left =
      unextended_input1_shape.DimensionsCount() == 4 ? paddings_data[2] : 0;

  // For uint8 quantized, the correct padding "zero value" is the output offset.
  const int32_t pad_value = params.output_offset;
  for (int out_b = 0; out_b < output_batch_size; ++out_b) {
    int input_batch = out_b % input_batch_size;
    int shift_w = (out_b / input_batch_size) % block_shape_width;
    int shift_h = (out_b / input_batch_size) / block_shape_width;
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        T* out = output_data + Offset(output_shape, out_b, out_h, out_w, 0);
        if (out_h * block_shape_height + shift_h < padding_top ||
            out_h * block_shape_height + shift_h >=
                padding_top + input_height ||
            out_w * block_shape_width + shift_w < padding_left ||
            out_w * block_shape_width + shift_w >= padding_left + input_width) {
          // This may not execute correctly when pad_value != 0 and T != uint8.
          memset(out, pad_value, depth * sizeof(T));
        } else {
          const T* in =
              input1_data +
              Offset(input1_shape, input_batch,
                     (out_h * block_shape_height + shift_h) - padding_top,
                     (out_w * block_shape_width + shift_w) - padding_left, 0);
          memcpy(out, in, depth * sizeof(T));
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SPACE_TO_BATCH_ND_H_
