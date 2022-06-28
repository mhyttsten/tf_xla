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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PAD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PAD_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSpadDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSpadDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSpadDTh() {
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


#include <vector>

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// TFLite Pad supports activation tensors with up to 5 dimensions.
constexpr int PadKernelMaxDimensionCount() { return 5; }

// There are two versions of pad: Pad and PadV2.  In PadV2 there is a second
// scalar input that provides the padding value.  Therefore pad_value_ptr can be
// equivalent to a simple input1_data.  For Pad, it should point to a zero
// value.
//
// Note that two typenames are required, so that T=P=int32_t is considered a
// specialization distinct from P=int32_t.
template <typename T, typename P>
inline void PadImpl(const tflite::PadParams& op_params,
                    const RuntimeShape& input_shape, const T* input_data,
                    const P* pad_value_ptr, const RuntimeShape& output_shape,
                    T* output_data) {
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(PadKernelMaxDimensionCount(), input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(PadKernelMaxDimensionCount(), output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, PadKernelMaxDimensionCount());
  TFLITE_DCHECK_LE(op_params.right_padding_count, PadKernelMaxDimensionCount());

  // Runtime calls are currently fixed at 5 dimensions. Copy inputs so we can
  // pad them to 5 dims (yes, we are "padding the padding").
  int left_padding_copy[PadKernelMaxDimensionCount()];
  for (int i = 0; i < PadKernelMaxDimensionCount(); i++) {
    left_padding_copy[i] = 0;
  }
  for (int i = 0; i < op_params.left_padding_count; ++i) {
    left_padding_copy[i + PadKernelMaxDimensionCount() -
                      op_params.left_padding_count] = op_params.left_padding[i];
  }
  int right_padding_copy[PadKernelMaxDimensionCount()];
  for (int i = 0; i < PadKernelMaxDimensionCount(); i++) {
    right_padding_copy[i] = 0;
  }
  for (int i = 0; i < op_params.right_padding_count; ++i) {
    right_padding_copy[i + PadKernelMaxDimensionCount() -
                       op_params.right_padding_count] =
        op_params.right_padding[i];
  }

  const int output_batch = ext_output_shape.Dims(0);
  const int output_plane = ext_output_shape.Dims(1);
  const int output_height = ext_output_shape.Dims(2);
  const int output_width = ext_output_shape.Dims(3);
  const int output_depth = ext_output_shape.Dims(4);

  const int left_b_padding = left_padding_copy[0];
  const int left_p_padding = left_padding_copy[1];
  const int left_h_padding = left_padding_copy[2];
  const int left_w_padding = left_padding_copy[3];
  const int left_d_padding = left_padding_copy[4];

  const int right_b_padding = right_padding_copy[0];
  const int right_p_padding = right_padding_copy[1];
  const int right_h_padding = right_padding_copy[2];
  const int right_w_padding = right_padding_copy[3];
  const int right_d_padding = right_padding_copy[4];

  const T pad_value = *pad_value_ptr;

  const T* in_ptr = input_data;
  T* out_ptr = output_data;
  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_p = 0; out_p < output_plane; ++out_p) {
      for (int out_h = 0; out_h < output_height; ++out_h) {
        for (int out_w = 0; out_w < output_width; ++out_w) {
          for (int out_d = 0; out_d < output_depth; ++out_d) {
            if (out_b < left_b_padding ||
                out_b >= output_batch - right_b_padding ||
                out_p < left_p_padding ||
                out_p >= output_plane - right_p_padding ||
                out_h < left_h_padding ||
                out_h >= output_height - right_h_padding ||
                out_w < left_w_padding ||
                out_w >= output_width - right_w_padding ||
                out_d < left_d_padding ||
                out_d >= output_depth - right_d_padding) {
              *out_ptr++ = pad_value;
            } else {
              *out_ptr++ = *in_ptr++;
            }
          }
        }
      }
    }
  }
}

template <typename T, typename P>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const P* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

// The second (pad-value) input can be int32_t when, say, the first is uint8_t.
template <typename T>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const int32_t* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSpadDTh mht_0(mht_0_v, 300, "", "./tensorflow/lite/kernels/internal/reference/pad.h", "Pad");

  const T converted_pad_value = static_cast<T>(*pad_value_ptr);
  PadImpl(op_params, input_shape, input_data, &converted_pad_value,
          output_shape, output_data);
}

// This version avoids conflicting template matching.
template <>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const int32_t* input_data,
                const int32_t* pad_value_ptr, const RuntimeShape& output_shape,
                int32_t* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSpadDTh mht_1(mht_1_v, 314, "", "./tensorflow/lite/kernels/internal/reference/pad.h", "Pad");

  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

template <typename T, typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const P* pad_value_ptr,
                          const RuntimeShape& output_shape, T* output_data) {
  Pad(op_params, input_shape, input_data, pad_value_ptr, output_shape,
      output_data);
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const float* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          float* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSpadDTh mht_2(mht_2_v, 336, "", "./tensorflow/lite/kernels/internal/reference/pad.h", "PadImageStyle");

  Pad(op_params, input_shape, input_data, pad_value_ptr, output_shape,
      output_data);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PAD_H_
