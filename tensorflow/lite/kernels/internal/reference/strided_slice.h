/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRIDED_SLICE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRIDED_SLICE_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSstrided_sliceDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSstrided_sliceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSstrided_sliceDTh() {
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


#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const RuntimeShape& unextended_output_shape,
                         SequentialTensorWriter<T>* writer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSstrided_sliceDTh mht_0(mht_0_v, 202, "", "./tensorflow/lite/kernels/internal/reference/strided_slice.h", "StridedSlice");

  using strided_slice::LoopCondition;
  using strided_slice::StartForAxis;
  using strided_slice::StopForAxis;

  ruy::profiler::ScopeLabel label("StridedSlice");

  // Note that the output_shape is not used herein.
  tflite::StridedSliceParams params_copy = op_params;

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 5);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(5, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(5, unextended_output_shape);

  // Reverse and pad to 5 dimensions because that is what the runtime code
  // requires (ie. all shapes must be 5D and are given backwards).
  strided_slice::StridedSlicePadIndices(&params_copy, 5);

  const int start_0 = StartForAxis(params_copy, input_shape, 0);
  const int stop_0 = StopForAxis(params_copy, input_shape, 0, start_0);
  const int start_1 = StartForAxis(params_copy, input_shape, 1);
  const int stop_1 = StopForAxis(params_copy, input_shape, 1, start_1);
  const int start_2 = StartForAxis(params_copy, input_shape, 2);
  const int stop_2 = StopForAxis(params_copy, input_shape, 2, start_2);
  const int start_3 = StartForAxis(params_copy, input_shape, 3);
  const int stop_3 = StopForAxis(params_copy, input_shape, 3, start_3);
  const int start_4 = StartForAxis(params_copy, input_shape, 4);
  const int stop_4 = StopForAxis(params_copy, input_shape, 4, start_4);

  for (int offset_0 = start_0 * input_shape.Dims(1),
           end_0 = stop_0 * input_shape.Dims(1),
           step_0 = params_copy.strides[0] * input_shape.Dims(1);
       !LoopCondition(offset_0, end_0, params_copy.strides[0]);
       offset_0 += step_0) {
    for (int offset_1 = (offset_0 + start_1) * input_shape.Dims(2),
             end_1 = (offset_0 + stop_1) * input_shape.Dims(2),
             step_1 = params_copy.strides[1] * input_shape.Dims(2);
         !LoopCondition(offset_1, end_1, params_copy.strides[1]);
         offset_1 += step_1) {
      for (int offset_2 = (offset_1 + start_2) * input_shape.Dims(3),
               end_2 = (offset_1 + stop_2) * input_shape.Dims(3),
               step_2 = params_copy.strides[2] * input_shape.Dims(3);
           !LoopCondition(offset_2, end_2, params_copy.strides[2]);
           offset_2 += step_2) {
        for (int offset_3 = (offset_2 + start_3) * input_shape.Dims(4),
                 end_3 = (offset_2 + stop_3) * input_shape.Dims(4),
                 step_3 = params_copy.strides[3] * input_shape.Dims(4);
             !LoopCondition(offset_3, end_3, params_copy.strides[3]);
             offset_3 += step_3) {
          for (int offset_4 = offset_3 + start_4, end_4 = offset_3 + stop_4;
               !LoopCondition(offset_4, end_4, params_copy.strides[4]);
               offset_4 += params_copy.strides[4]) {
            writer->Write(offset_4);
          }
        }
      }
    }
  }
}

template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSstrided_sliceDTh mht_1(mht_1_v, 273, "", "./tensorflow/lite/kernels/internal/reference/strided_slice.h", "StridedSlice");

  SequentialTensorWriter<T> writer(input_data, output_data);
  StridedSlice<T>(op_params, unextended_input_shape, unextended_output_shape,
                  &writer);
}

template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const TfLiteTensor* input,
                         const RuntimeShape& unextended_output_shape,
                         TfLiteTensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSstrided_sliceDTh mht_2(mht_2_v, 287, "", "./tensorflow/lite/kernels/internal/reference/strided_slice.h", "StridedSlice");

  SequentialTensorWriter<T> writer(input, output);
  StridedSlice<T>(op_params, unextended_input_shape, unextended_output_shape,
                  &writer);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRIDED_SLICE_H_
