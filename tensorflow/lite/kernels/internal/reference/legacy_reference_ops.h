/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEGACY_REFERENCE_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEGACY_REFERENCE_OPS_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh() {
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
#include <sys/types.h>

#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/legacy_types.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

static constexpr int kDepthwiseReverseShift = -1;

inline void ShapeFromDims(const tflite::Dims<4>& dims, RuntimeShape* shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "ShapeFromDims");

  shape->BuildFrom(
      {dims.sizes[3], dims.sizes[2], dims.sizes[1], dims.sizes[0]});
}

inline void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          const float* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height,
                          int dilation_width_factor, int dilation_height_factor,
                          int pad_width, int pad_height, int depth_multiplier,
                          float output_activation_min,
                          float output_activation_max, float* output_data,
                          const Dims<4>& output_dims) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_1(mht_1_v, 222, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "DepthwiseConv");

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.depth_multiplier = depth_multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  DepthwiseConv(op_params, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                bias_data, DimsToShape(output_dims), output_data);
}

inline void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          const float* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, int depth_multiplier,
                          float output_activation_min,
                          float output_activation_max, float* output_data,
                          const Dims<4>& output_dims) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_2(mht_2_v, 251, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "DepthwiseConv");

  DepthwiseConv(input_data, input_dims, filter_data, filter_dims, bias_data,
                bias_dims, stride_width, stride_height, 1, 1, pad_width,
                pad_height, depth_multiplier, output_activation_min,
                output_activation_max, output_data, output_dims);
}

// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                   const float* filter_data, const Dims<4>& filter_dims,
                   const float* bias_data, const Dims<4>& bias_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int depth_multiplier, float* output_data,
                   const Dims<4>& output_dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_3(mht_3_v, 268, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "DepthwiseConv");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  DepthwiseConv(input_data, input_dims, filter_data, filter_dims, bias_data,
                bias_dims, stride_width, stride_height, pad_width, pad_height,
                depth_multiplier, output_activation_min, output_activation_max,
                output_data, output_dims);
}

// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const float* input_data, const Dims<4>& input_dims,
                   const float* filter_data, const Dims<4>& filter_dims,
                   const float* bias_data, const Dims<4>& bias_dims, int stride,
                   int pad_width, int pad_height, int depth_multiplier,
                   float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_4(mht_4_v, 286, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "DepthwiseConv");

  DepthwiseConv<Ac>(input_data, input_dims, filter_data, filter_dims, bias_data,
                    bias_dims, stride, stride, pad_width, pad_height,
                    depth_multiplier, output_data, output_dims);
}

inline void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                          int32 input_offset, const uint8* filter_data,
                          const Dims<4>& filter_dims, int32 filter_offset,
                          const int32* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height,
                          int dilation_width_factor, int dilation_height_factor,
                          int pad_width, int pad_height, int depth_multiplier,
                          int32 output_offset, int32 output_multiplier,
                          int output_shift, int32 output_activation_min,
                          int32 output_activation_max, uint8* output_data,
                          const Dims<4>& output_dims) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_5(mht_5_v, 305, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "DepthwiseConv");

  tflite::DepthwiseParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.depth_multiplier = depth_multiplier;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kDepthwiseReverseShift * output_shift;

  DepthwiseConv(op_params, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                bias_data, DimsToShape(output_dims), output_data);
}

inline void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                          int32 input_offset, const uint8* filter_data,
                          const Dims<4>& filter_dims, int32 filter_offset,
                          const int32* bias_data, const Dims<4>& bias_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, int depth_multiplier,
                          int32 output_offset, int32 output_multiplier,
                          int output_shift, int32 output_activation_min,
                          int32 output_activation_max, uint8* output_data,
                          const Dims<4>& output_dims) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_6(mht_6_v, 342, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "DepthwiseConv");

  DepthwiseConv(input_data, input_dims, input_offset, filter_data, filter_dims,
                filter_offset, bias_data, bias_dims, stride_width,
                stride_height, 1, 1, pad_width, pad_height, depth_multiplier,
                output_offset, output_multiplier, output_shift,
                output_activation_min, output_activation_max, output_data,
                output_dims);
}

// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                   int32 input_offset, const uint8* filter_data,
                   const Dims<4>& filter_dims, int32 filter_offset,
                   const int32* bias_data, const Dims<4>& bias_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int depth_multiplier, int32 output_offset,
                   int32 output_multiplier, int output_shift,
                   int32 output_activation_min, int32 output_activation_max,
                   uint8* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_7(mht_7_v, 364, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "DepthwiseConv");

  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  DepthwiseConv(input_data, input_dims, input_offset, filter_data, filter_dims,
                filter_offset, bias_data, bias_dims, stride_width,
                stride_height, pad_width, pad_height, depth_multiplier,
                output_offset, output_multiplier, output_shift,
                output_activation_min, output_activation_max, output_data,
                output_dims);
}

// Legacy, for compatibility with old checked-in code.
template <FusedActivationFunctionType Ac>
void DepthwiseConv(const uint8* input_data, const Dims<4>& input_dims,
                   int32 input_offset, const uint8* filter_data,
                   const Dims<4>& filter_dims, int32 filter_offset,
                   const int32* bias_data, const Dims<4>& bias_dims, int stride,
                   int pad_width, int pad_height, int depth_multiplier,
                   int32 output_offset, int32 output_multiplier,
                   int output_shift, int32 output_activation_min,
                   int32 output_activation_max, uint8* output_data,
                   const Dims<4>& output_dims) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_8(mht_8_v, 390, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "DepthwiseConv");

  DepthwiseConv<Ac>(input_data, input_dims, input_offset, filter_data,
                    filter_dims, filter_offset, bias_data, bias_dims, stride,
                    stride, pad_width, pad_height, depth_multiplier,
                    output_offset, output_multiplier, output_shift,
                    output_activation_min, output_activation_max, output_data,
                    output_dims);
}

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int dilation_width_factor,
                 int dilation_height_factor, int pad_width, int pad_height,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_9(mht_9_v, 409, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Conv");

  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  Conv(op_params, DimsToShape(input_dims), input_data, DimsToShape(filter_dims),
       filter_data, DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
       output_data, DimsToShape(im2col_dims), im2col_data);
}

template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int dilation_width_factor,
          int dilation_height_factor, int pad_width, int pad_height,
          float* output_data, const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_10(mht_10_v, 437, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Conv");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, dilation_width_factor,
       dilation_height_factor, pad_width, pad_height, output_activation_min,
       output_activation_max, output_data, output_dims, im2col_data,
       im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_11(mht_11_v, 457, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Conv");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, 1, 1, pad_width, pad_height,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_12(mht_12_v, 476, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Conv");

  Conv<Ac>(input_data, input_dims, filter_data, filter_dims, bias_data,
           bias_dims, stride, stride, 1, 1, pad_width, pad_height, output_data,
           output_dims, im2col_data, im2col_dims);
}

inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int dilation_width_factor,
                 int dilation_height_factor, int pad_width, int pad_height,
                 int32 output_offset, int32 output_multiplier, int output_shift,
                 int32 output_activation_min, int32 output_activation_max,
                 uint8* output_data, const Dims<4>& output_dims,
                 uint8* im2col_data, const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemmlowp_context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_13(mht_13_v, 495, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Conv");

  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;
  op_params.dilation_width_factor = dilation_width_factor;
  op_params.dilation_height_factor = dilation_height_factor;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  Conv(op_params, DimsToShape(input_dims), input_data, DimsToShape(filter_dims),
       filter_data, DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
       output_data, DimsToShape(im2col_dims), im2col_data, gemmlowp_context);
}

inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemmlowp_context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_14(mht_14_v, 532, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Conv");

  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride_width, stride_height, 1, 1,
       pad_width, pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemmlowp_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemmlowp_context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_15(mht_15_v, 555, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Conv");

  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride_width, stride_height,
       pad_width, pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemmlowp_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const uint8* input_data, const Dims<4>& input_dims,
          int32 input_offset, const uint8* filter_data,
          const Dims<4>& filter_dims, int32 filter_offset,
          const int32* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, int32 output_offset,
          int32 output_multiplier, int output_shift,
          int32 output_activation_min, int32 output_activation_max,
          uint8* output_data, const Dims<4>& output_dims, uint8* im2col_data,
          const Dims<4>& im2col_dims, gemmlowp::GemmContext* gemmlowp_context) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_16(mht_16_v, 585, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Conv");

  Conv<Ac>(input_data, input_dims, input_offset, filter_data, filter_dims,
           filter_offset, bias_data, bias_dims, stride, stride, pad_width,
           pad_height, output_offset, output_multiplier, output_shift,
           output_activation_min, output_activation_max, output_data,
           output_dims, im2col_data, im2col_dims, gemmlowp_context);
}

inline void TransposeConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, float* output_data,
                          const Dims<4>& output_dims, float* im2col_data,
                          const Dims<4>& im2col_dims) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_17(mht_17_v, 601, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "TransposeConv");

  tflite::ConvParams op_params;
  // Padding type is ignored, but still set.
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = pad_width;
  op_params.padding_values.height = pad_height;
  op_params.stride_width = stride_width;
  op_params.stride_height = stride_height;

  TransposeConv(op_params, DimsToShape(input_dims), input_data,
                DimsToShape(filter_dims), filter_data,
                /*bias_shape*/ RuntimeShape(), /*bias*/ nullptr,
                DimsToShape(output_dims), output_data, DimsToShape(im2col_dims),
                im2col_data);
}

inline void TransposeConv(
    const ConvParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& output_shape,
    float* output_data, const RuntimeShape& im2col_shape, float* im2col_data) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_18(mht_18_v, 624, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "TransposeConv");

  TransposeConv(params, input_shape, input_data, filter_shape, filter_data,
                /*bias_shape*/ RuntimeShape(), /*bias*/ nullptr, output_shape,
                output_data, im2col_shape, im2col_data);
}

inline void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                           const float* weights_data,
                           const Dims<4>& weights_dims, const float* bias_data,
                           const Dims<4>& bias_dims,
                           float output_activation_min,
                           float output_activation_max, float* output_data,
                           const Dims<4>& output_dims) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_19(mht_19_v, 639, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "FullyConnected");

  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(weights_dims), weights_data,
                 DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
                 output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                    const float* weights_data, const Dims<4>& weights_dims,
                    const float* bias_data, const Dims<4>& bias_dims,
                    float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_20(mht_20_v, 658, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "FullyConnected");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  FullyConnected(input_data, input_dims, weights_data, weights_dims, bias_data,
                 bias_dims, output_activation_min, output_activation_max,
                 output_data, output_dims);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data, gemmlowp::GemmContext*) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_21(mht_21_v, 674, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "FullyConnected");

  FullyConnected(params, input_shape, input_data, filter_shape, filter_data,
                 bias_shape, bias_data, output_shape, output_data);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    int16* output_data, gemmlowp::GemmContext*) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_22(mht_22_v, 687, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "FullyConnected");

  FullyConnected(params, input_shape, input_data, filter_shape, filter_data,
                 bias_shape, bias_data, output_shape, output_data);
}

inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, uint8* output_data,
                           const Dims<4>& output_dims,
                           gemmlowp::GemmContext* gemmlowp_context) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_23(mht_23_v, 703, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "FullyConnected");

  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                 bias_data, DimsToShape(output_dims), output_data,
                 gemmlowp_context);
}

inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, int16* output_data,
                           const Dims<4>& output_dims,
                           gemmlowp::GemmContext* gemmlowp_context) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_24(mht_24_v, 731, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "FullyConnected");

  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                 bias_data, DimsToShape(output_dims), output_data,
                 gemmlowp_context);
}

inline void ShuffledFullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& weights_shape,
    const uint8* shuffled_weights_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    int16* output_data, uint8* shuffled_input_workspace_data,
    gemmlowp::GemmContext*) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_25(mht_25_v, 757, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "ShuffledFullyConnected");

  ShuffledFullyConnected(params, input_shape, input_data, weights_shape,
                         shuffled_weights_data, bias_shape, bias_data,
                         output_shape, output_data,
                         shuffled_input_workspace_data);
}

inline void ShuffledFullyConnected(
    const uint8* input_data, const Dims<4>& input_dims,
    const uint8* shuffled_weights_data, const Dims<4>& weights_dims,
    const int32* bias_data, const Dims<4>& bias_dims, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    int16* output_data, const Dims<4>& output_dims,
    uint8* shuffled_input_workspace_data,
    gemmlowp::GemmContext* gemmlowp_context) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_26(mht_26_v, 774, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "ShuffledFullyConnected");

  tflite::FullyConnectedParams op_params;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  ShuffledFullyConnected(op_params, DimsToShape(input_dims), input_data,
                         DimsToShape(weights_dims), shuffled_weights_data,
                         DimsToShape(bias_dims), bias_data,
                         DimsToShape(output_dims), output_data,
                         shuffled_input_workspace_data, gemmlowp_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_offset, const uint8* filter_data,
                    const Dims<4>& filter_dims, int32 filter_offset,
                    const int32* bias_data, const Dims<4>& bias_dims,
                    int32 output_offset, int32 output_multiplier,
                    int output_shift, int32 output_activation_min,
                    int32 output_activation_max, uint8* output_data,
                    const Dims<4>& output_dims,
                    gemmlowp::GemmContext* gemmlowp_context) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_27(mht_27_v, 802, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "FullyConnected");

  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  FullyConnected(input_data, input_dims, input_offset, filter_data, filter_dims,
                 filter_offset, bias_data, bias_dims, output_offset,
                 output_multiplier, output_shift, output_activation_min,
                 output_activation_max, output_data, output_dims,
                 gemmlowp_context);
}

inline void LstmCell(const float* input_data, const Dims<4>& input_dims,
                     const float* prev_activ_data,
                     const Dims<4>& prev_activ_dims, const float* weights_data,
                     const Dims<4>& weights_dims, const float* bias_data,
                     const Dims<4>& bias_dims, const float* prev_state_data,
                     const Dims<4>& prev_state_dims, float* output_state_data,
                     const Dims<4>& output_state_dims, float* output_activ_data,
                     const Dims<4>& output_activ_dims, float* concat_temp_data,
                     const Dims<4>& concat_temp_dims, float* activ_temp_data,
                     const Dims<4>& activ_temp_dims) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_28(mht_28_v, 831, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "LstmCell");

  tflite::LstmCellParams op_params;
  // Float LSTM cell does not need parameters to be set: leave untouched.

  LstmCell(op_params, DimsToShape(input_dims), input_data,
           DimsToShape(prev_activ_dims), prev_activ_data,
           DimsToShape(weights_dims), weights_data, DimsToShape(bias_dims),
           bias_data, DimsToShape(prev_state_dims), prev_state_data,
           DimsToShape(output_state_dims), output_state_data,
           DimsToShape(output_activ_dims), output_activ_data,
           DimsToShape(concat_temp_dims), concat_temp_data,
           DimsToShape(activ_temp_dims), activ_temp_data);
}

template <int StateIntegerBits>
void LstmCell(const uint8* input_data_uint8, const Dims<4>& input_dims,
              const uint8* prev_activ_data_uint8,
              const Dims<4>& prev_activ_dims, const uint8* weights_data_uint8,
              const Dims<4>& weights_dims, const int32* bias_data_int32,
              const Dims<4>& bias_dims, const int16* prev_state_data_int16,
              const Dims<4>& prev_state_dims, int16* output_state_data_int16,
              const Dims<4>& output_state_dims, uint8* output_activ_data_uint8,
              const Dims<4>& output_activ_dims, uint8* concat_temp_data_uint8,
              const Dims<4>& concat_temp_dims, int16* activ_temp_data_int16,
              const Dims<4>& activ_temp_dims, int32 weights_zero_point,
              int32 accum_multiplier, int accum_shift,
              gemmlowp::GemmContext* gemmlowp_context) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_29(mht_29_v, 860, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "LstmCell");

  tflite::LstmCellParams op_params;
  op_params.weights_zero_point = weights_zero_point;
  op_params.accum_multiplier = accum_multiplier;
  op_params.accum_shift = accum_shift;

  LstmCell<StateIntegerBits>(
      op_params, DimsToShape(input_dims), input_data_uint8,
      DimsToShape(prev_activ_dims), prev_activ_data_uint8,
      DimsToShape(weights_dims), weights_data_uint8, DimsToShape(bias_dims),
      bias_data_int32, DimsToShape(prev_state_dims), prev_state_data_int16,
      DimsToShape(output_state_dims), output_state_data_int16,
      DimsToShape(output_activ_dims), output_activ_data_uint8,
      DimsToShape(concat_temp_dims), concat_temp_data_uint8,
      DimsToShape(activ_temp_dims), activ_temp_data_int16, gemmlowp_context);
}

template <typename T>
void BroadcastDiv(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_30(mht_30_v, 884, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BroadcastDiv");

  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  BroadcastDivSlow(op_params, DimsToShape(input1_dims), input1_data,
                   DimsToShape(input2_dims), input2_data,
                   DimsToShape(output_dims), output_data);
}

template <typename T>
inline void Div(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T output_activation_min, T output_activation_max,
                T* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_31(mht_31_v, 900, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Div");

  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  Div(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac, typename Scalar>
inline void Concatenation(int concat_dim, const Scalar* const* input_data,
                          const Dims<4>* const* input_dims, int inputs_count,
                          Scalar* output_data, const Dims<4>& output_dims) {
  // For now we don't have a model with a Concatenation with fused activation.
  TFLITE_DCHECK_EQ(Ac, FusedActivationFunctionType::kNone);

  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::ConcatenationParams op_params;
  op_params.axis = 3 - concat_dim;
  op_params.inputs_count = inputs_count;

  Concatenation(op_params, input_shapes_indirect.data(), input_data,
                DimsToShape(output_dims), output_data);
}

inline void Concatenation(int concat_dim, const uint8* const* input_data,
                          const Dims<4>* const* input_dims,
                          const int32* input_zeropoint,
                          const float* input_scale, int inputs_count,
                          uint8* output_data, const Dims<4>& output_dims,
                          const int32 output_zeropoint,
                          const float output_scale) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_32(mht_32_v, 939, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Concatenation");

  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::ConcatenationParams op_params;
  op_params.axis = 3 - concat_dim;
  op_params.input_zeropoint = input_zeropoint;
  op_params.input_scale = input_scale;
  op_params.inputs_count = inputs_count;
  op_params.output_zeropoint = output_zeropoint;
  op_params.output_scale = output_scale;

  ConcatenationWithScaling(op_params, input_shapes_indirect.data(), input_data,
                           DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac, typename Scalar>
void DepthConcatenation(const Scalar* const* input_data,
                        const Dims<4>* const* input_dims, int inputs_count,
                        Scalar* output_data, const Dims<4>& output_dims) {
  // For now we don't have a model with a Concatenation with fused activation.
  TFLITE_DCHECK_EQ(Ac, FusedActivationFunctionType::kNone);

  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::ConcatenationParams op_params;
  op_params.inputs_count = inputs_count;

  DepthConcatenation(op_params, input_shapes_indirect.data(), input_data,
                     DimsToShape(output_dims), output_data);
}

template <typename Scalar>
void TensorFlowSplit(const Scalar* input_data, const Dims<4>& input_dims,
                     int axis, int outputs_count, Scalar* const* output_data,
                     const Dims<4>* const* output_dims) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_33(mht_33_v, 984, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "TensorFlowSplit");

  std::vector<RuntimeShape> output_shapes(outputs_count);
  std::vector<const RuntimeShape*> output_shapes_indirect(outputs_count);
  for (int i = 0; i < outputs_count; ++i) {
    ShapeFromDims(*output_dims[i], &output_shapes[i]);
    output_shapes_indirect[i] = &output_shapes[i];
  }
  tflite::SplitParams op_params;
  op_params.axis = 3 - axis;
  op_params.num_split = outputs_count;

  Split(op_params, DimsToShape(input_dims), input_data,
        output_shapes_indirect.data(), output_data);
}

template <FusedActivationFunctionType Ac, typename Scalar>
void TensorFlowSplit(const Scalar* input_data, const Dims<4>& input_dims,
                     int outputs_count, Scalar* const* output_data,
                     const Dims<4>* const* output_dims) {
  TFLITE_DCHECK_GE(outputs_count, 1);
  for (int i = 0; i < outputs_count; i++) {
    /* batches = */ MatchingArraySize(*output_dims[i], 3, input_dims, 3);
    /* height = */ MatchingArraySize(*output_dims[i], 2, input_dims, 2);
    /* width = */ MatchingArraySize(*output_dims[i], 1, input_dims, 1);
  }
  // For now we don't have a model with a Split with fused activation.
  TFLITE_DCHECK_EQ(Ac, FusedActivationFunctionType::kNone);

  TensorFlowSplit(input_data, input_dims, /*axis=*/0, outputs_count,
                  output_data, output_dims);
}

inline void Softmax(const float* input_data, const RuntimeShape& input_shape,
                    float beta, float* output_data,
                    const RuntimeShape& output_shape) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_34(mht_34_v, 1021, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Softmax");

  SoftmaxParams params;
  params.beta = beta;
  Softmax(params, input_shape, input_data, output_shape, output_data);
}

inline void Softmax(const uint8* input_data, const RuntimeShape& input_shape,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const RuntimeShape& output_shape) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_35(mht_35_v, 1033, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Softmax");

  SoftmaxParams params;
  params.input_multiplier = input_beta_multiplier;
  params.input_left_shift = input_beta_left_shift;
  params.diff_min = diff_min;
  Softmax(params, input_shape, input_data, output_shape, output_data);
}

inline void LogSoftmax(const float* input_data, const RuntimeShape& input_shape,
                       float* output_data, const RuntimeShape& output_shape) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_36(mht_36_v, 1045, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "LogSoftmax");

  SoftmaxParams params;
  // No params currently used for float LogSoftmax.
  LogSoftmax(params, input_shape, input_data, output_shape, output_data);
}

inline void LogSoftmax(const uint8* input_data, const RuntimeShape& input_shape,
                       int32 input_multiplier, int32 input_left_shift,
                       int32 reverse_scaling_divisor,
                       int32 reverse_scaling_right_shift, int diff_min,
                       uint8* output_data, const RuntimeShape& output_shape) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_37(mht_37_v, 1058, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "LogSoftmax");

  SoftmaxParams params;
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  params.reverse_scaling_divisor = reverse_scaling_divisor;
  params.reverse_scaling_right_shift = reverse_scaling_right_shift;
  params.diff_min = diff_min;
  LogSoftmax(params, input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const LogisticParams& params,
                     const RuntimeShape& input_shape, const uint8* input_data,
                     const RuntimeShape& output_shape, uint8* output_data) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_38(mht_38_v, 1073, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Logistic");

  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int32 input_multiplier = params.input_multiplier;
  const int input_left_shift = params.input_left_shift;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const uint8 input_val_u8 = input_data[i];
    const int32 input_val_centered =
        static_cast<int32>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered <= -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered >= input_range_radius) {
      output_val = 255;
    } else {
      const int32 input_val_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_val_centered, input_multiplier, input_left_shift);
      using FixedPoint4 = gemmlowp::FixedPoint<int32, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::logistic(input_val_f4);
      // Convert from Q0.31 to Q23.8.
      using gemmlowp::RoundingDivideByPOT;
      int32 output_val_s32 = RoundingDivideByPOT(output_val_f0.raw(), 23);
      if (output_val_s32 == 256) {
        output_val_s32 = 255;
      }
      // Reinterpret as U0.8.
      TFLITE_DCHECK_GE(output_val_s32, 0);
      TFLITE_DCHECK_LE(output_val_s32, 255);
      output_val = static_cast<uint8>(output_val_s32);
    }
    output_data[i] = output_val;
  }
}

inline void Logistic(const uint8* input_data, const RuntimeShape& input_shape,
                     int32 input_zero_point, int32 input_range_radius,
                     int32 input_multiplier, int input_left_shift,
                     uint8* output_data, const RuntimeShape& output_shape) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_39(mht_39_v, 1118, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Logistic");

  LogisticParams params;
  params.input_zero_point = input_zero_point;
  params.input_range_radius = input_range_radius;
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  Logistic(params, input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const RuntimeShape& input_shape, const int16* input_data,
                     const RuntimeShape& output_shape, int16* output_data) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_40(mht_40_v, 1131, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Logistic");

  LogisticParams params;
  // No params currently needed by int16 Logistic.
  Logistic(params, input_shape, input_data, output_shape, output_data);
}

inline void Tanh(const uint8* input_data, const RuntimeShape& input_shape,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const RuntimeShape& output_shape) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_41(mht_41_v, 1143, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Tanh");

  TanhParams params;
  params.input_zero_point = input_zero_point;
  params.input_range_radius = input_range_radius;
  params.input_multiplier = input_multiplier;
  params.input_left_shift = input_left_shift;
  Tanh(params, input_shape, input_data, output_shape, output_data);
}

inline void Tanh(const int16* input_data, const RuntimeShape& input_shape,
                 int input_left_shift, int16* output_data,
                 const RuntimeShape& output_shape) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_42(mht_42_v, 1157, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Tanh");

  TanhParams params;
  params.input_left_shift = input_left_shift;
  Tanh(params, input_shape, input_data, output_shape, output_data);
}

inline void Dequantize(const uint8* input_data, const Dims<4>& input_dims,
                       int32 zero_point, double scale, float* output_data,
                       const Dims<4>& output_dims) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_43(mht_43_v, 1168, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Dequantize");

  tflite::DequantizationParams op_params;
  op_params.zero_point = zero_point;
  op_params.scale = scale;

  Dequantize(op_params, DimsToShape(input_dims), input_data,
             DimsToShape(output_dims), output_data);
}

inline void FakeQuant(const float* input_data, const Dims<4>& input_dims,
                      float rmin, float rmax, int num_bits, float* output_data,
                      const Dims<4>& output_dims) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_44(mht_44_v, 1182, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "FakeQuant");

  tflite::FakeQuantParams op_params;
  op_params.num_bits = num_bits;
  op_params.minmax.min = rmin;
  op_params.minmax.max = rmax;

  FakeQuant(op_params, DimsToShape(input_dims), input_data,
            DimsToShape(output_dims), output_data);
}

template <typename T>
inline void Gather(const T* input_data, const Dims<4>& input_dims,
                   int input_rank, const int32* coords_data,
                   const Dims<4>& coords_dims, T* output_data,
                   const Dims<4>& output_dims) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_45(mht_45_v, 1199, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Gather");

  tflite::GatherParams op_params;
  op_params.axis = 4 - input_rank;
  op_params.batch_dims = 0;

  Gather(op_params, DimsToShape(input_dims), input_data,
         DimsToShape(coords_dims), coords_data, DimsToShape(output_dims),
         output_data);
}

inline uint32 LegacyReverseBits32(uint32 n) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_46(mht_46_v, 1212, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "LegacyReverseBits32");

  n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);
  n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);
  n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);
  return (((n & 0xFF) << 24) | ((n & 0xFF00) << 8) | ((n & 0xFF0000) >> 8) |
          ((n & 0xFF000000) >> 24));
}

inline void StridedSliceReverseIndices(tflite::StridedSliceParams* p) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_47(mht_47_v, 1223, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "StridedSliceReverseIndices");

  TFLITE_CHECK_EQ(p->start_indices_count, p->stop_indices_count);
  TFLITE_CHECK_EQ(p->stop_indices_count, p->strides_count);

  std::reverse(p->start_indices, p->start_indices + p->start_indices_count);
  std::reverse(p->stop_indices, p->stop_indices + p->stop_indices_count);
  std::reverse(p->strides, p->strides + p->strides_count);

  p->begin_mask = LegacyReverseBits32(static_cast<uint32>(p->begin_mask)) >>
                  (32 - p->start_indices_count);
  p->ellipsis_mask =
      LegacyReverseBits32(static_cast<uint32>(p->ellipsis_mask)) >>
      (32 - p->start_indices_count);
  p->end_mask = LegacyReverseBits32(static_cast<uint32>(p->end_mask)) >>
                (32 - p->start_indices_count);
  p->new_axis_mask =
      LegacyReverseBits32(static_cast<uint32>(p->new_axis_mask)) >>
      (32 - p->start_indices_count);
  p->shrink_axis_mask =
      LegacyReverseBits32(static_cast<uint32>(p->shrink_axis_mask)) >>
      (32 - p->start_indices_count);
}

template <typename T>
inline void StridedSlice(const T* input_data, const Dims<4>& input_dims,
                         int begin_mask, int end_mask, int shrink_axis_mask,
                         const std::vector<int>& start_indices,
                         const std::vector<int>& stop_indices,
                         const std::vector<int>& strides, T* output_data,
                         const Dims<4>& output_dims) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_48(mht_48_v, 1255, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "StridedSlice");

  TFLITE_DCHECK_EQ(start_indices.size(), 4);
  auto op_params = strided_slice::BuildStridedSliceParams(
      begin_mask, end_mask, shrink_axis_mask, start_indices, stop_indices,
      strides);
  StridedSliceReverseIndices(&op_params);

  StridedSlice(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

template <typename T>
inline void Mean(const T* input_data, const Dims<4>& input_dims,
                 const std::vector<int>& reduction_indices, T* output_data,
                 const Dims<4>& output_dims) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_49(mht_49_v, 1272, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Mean");

  tflite::MeanParams op_params;
  op_params.axis_count = reduction_indices.size();
  for (int i = 0; i < op_params.axis_count; ++i) {
    op_params.axis[i] = reduction_indices[op_params.axis_count - 1 - i];
  }

  Mean(op_params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

template <typename T>
void Transpose(const T* input, const Dims<4>& input_dims, T* output,
               const Dims<4>& output_dims, const int* permuted_axes) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_50(mht_50_v, 1288, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Transpose");

  TransposeParams params;
  params.perm_count = 4;
  for (int i = 0; i < 4; ++i) {
    params.perm[i] = 3 - permuted_axes[3 - i];
  }
  Transpose(params, DimsToShape(input_dims), input, DimsToShape(output_dims),
            output);
}

template <typename T, ComparisonFn<T> F>
inline void Comparison(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, const Dims<4>& input2_dims,
                       bool* output_data, const Dims<4>& output_dims) {
  ComparisonParams op_params;
  // No parameters needed.
  ComparisonImpl<T, F>(op_params, DimsToShape(input1_dims), input1_data,
                       DimsToShape(input2_dims), input2_data,
                       DimsToShape(output_dims), output_data);
}

template <typename T, ComparisonFn<int32> F>
inline void Comparison(int left_shift, const T* input1_data,
                       const Dims<4>& input1_dims, int32 input1_offset,
                       int32 input1_multiplier, int input1_shift,
                       const T* input2_data, const Dims<4>& input2_dims,
                       int32 input2_offset, int32 input2_multiplier,
                       int input2_shift, bool* output_data,
                       const Dims<4>& output_dims) {
  tflite::ComparisonParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.input2_shift = kReverseShift * input2_shift;

  ComparisonWithScaling<T, F>(op_params, DimsToShape(input1_dims), input1_data,
                              DimsToShape(input2_dims), input2_data,
                              DimsToShape(output_dims), output_data);
}

template <typename T, ComparisonFn<T> F>
inline void BroadcastComparison(const T* input1_data,
                                const Dims<4>& input1_dims,
                                const T* input2_data,
                                const Dims<4>& input2_dims, bool* output_data,
                                const Dims<4>& output_dims) {
  ComparisonParams op_params;
  // No parameters needed.
  BroadcastComparison4DSlowImpl<T, F>(op_params, DimsToShape(input1_dims),
                                      input1_data, DimsToShape(input2_dims),
                                      input2_data, DimsToShape(output_dims),
                                      output_data);
}

template <typename T, ComparisonFn<int32> F>
inline void BroadcastComparison(int left_shift, const T* input1_data,
                                const Dims<4>& input1_dims, int32 input1_offset,
                                int32 input1_multiplier, int input1_shift,
                                const T* input2_data,
                                const Dims<4>& input2_dims, int32 input2_offset,
                                int32 input2_multiplier, int input2_shift,
                                bool* output_data, const Dims<4>& output_dims) {
  ComparisonParams op_params;

  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.input2_shift = kReverseShift * input2_shift;

  BroadcastComparison4DSlowWithScaling<T, F>(
      op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

#define TFLITE_LEGACY_COMPARISON_OP(name)                                     \
  template <typename T>                                                       \
  inline void name(const T* input1_data, const Dims<4>& input1_dims,          \
                   const T* input2_data, const Dims<4>& input2_dims,          \
                   bool* output_data, const Dims<4>& output_dims) {           \
    ruy::profiler::ScopeLabel label(#name);                                   \
    Comparison<T, name##Fn>(input1_data, input1_dims, input2_data,            \
                            input2_dims, output_data, output_dims);           \
  }                                                                           \
  template <typename T>                                                       \
  inline void name(                                                           \
      int left_shift, const T* input1_data, const Dims<4>& input1_dims,       \
      int32 input1_offset, int32 input1_multiplier, int input1_shift,         \
      const T* input2_data, const Dims<4>& input2_dims, int32 input2_offset,  \
      int32 input2_multiplier, int input2_shift, bool* output_data,           \
      const Dims<4>& output_dims) {                                           \
    ruy::profiler::ScopeLabel label(#name "/8bit");                           \
    Comparison<T, name##Fn>(left_shift, input1_data, input1_dims,             \
                            input1_offset, input1_multiplier, input1_shift,   \
                            input2_data, input2_dims, input2_offset,          \
                            input2_multiplier, input2_shift, output_data,     \
                            output_dims);                                     \
  }                                                                           \
  template <typename T>                                                       \
  inline void Broadcast##name(                                                \
      const T* input1_data, const Dims<4>& input1_dims, const T* input2_data, \
      const Dims<4>& input2_dims, bool* output_data,                          \
      const Dims<4>& output_dims) {                                           \
    ruy::profiler::ScopeLabel label("Broadcast" #name);                       \
    BroadcastComparison<T, name##Fn>(input1_data, input1_dims, input2_data,   \
                                     input2_dims, output_data, output_dims);  \
  }                                                                           \
  template <typename T>                                                       \
  inline void Broadcast##name(                                                \
      int left_shift, const T* input1_data, const Dims<4>& input1_dims,       \
      int32 input1_offset, int32 input1_multiplier, int input1_shift,         \
      const T* input2_data, const Dims<4>& input2_dims, int32 input2_offset,  \
      int32 input2_multiplier, int input2_shift, bool* output_data,           \
      const Dims<4>& output_dims) {                                           \
    ruy::profiler::ScopeLabel label("Broadcast" #name "/8bit");               \
    BroadcastComparison<T, name##Fn>(left_shift, input1_data, input1_dims,    \
                                     input1_offset, input1_multiplier,        \
                                     input1_shift, input2_data, input2_dims,  \
                                     input2_offset, input2_multiplier,        \
                                     input2_shift, output_data, output_dims); \
  }
TFLITE_LEGACY_COMPARISON_OP(Equal);
TFLITE_LEGACY_COMPARISON_OP(NotEqual);
TFLITE_LEGACY_COMPARISON_OP(Greater);
TFLITE_LEGACY_COMPARISON_OP(GreaterEqual);
TFLITE_LEGACY_COMPARISON_OP(Less);
TFLITE_LEGACY_COMPARISON_OP(LessEqual);
#undef TFLITE_LEGACY_COMPARISON_OP

template <typename D, typename T>
inline void Select(const D* input_condition_data,
                   const Dims<4>& input_condition_dims, const T* input_x_data,
                   const Dims<4>& input_x_dims, const T* input_y_data,
                   const Dims<4>& input_y_dims, T* output_data,
                   const Dims<4>& output_dims) {
  Select(DimsToShape(input_condition_dims), input_condition_data,
         DimsToShape(input_x_dims), input_x_data, DimsToShape(input_y_dims),
         input_y_data, DimsToShape(output_dims), output_data);
}

template <typename D, typename T>
inline void RankOneSelect(const D* input_condition_data,
                          const Dims<4>& input_condition_dims,
                          const T* input_x_data, const Dims<4>& input_x_dims,
                          const T* input_y_data, const Dims<4>& input_y_dims,
                          T* output_data, const Dims<4>& output_dims) {
  RankOneSelect(DimsToShape(input_condition_dims), input_condition_data,
                DimsToShape(input_x_dims), input_x_data,
                DimsToShape(input_y_dims), input_y_data,
                DimsToShape(output_dims), output_data);
}

template <typename T, typename TI>
inline void SparseToDense(const std::vector<std::vector<TI>>& indices,
                          const T* values, T default_value, T* output_data,
                          const Dims<4>& output_dims, bool value_is_scalar) {
  SparseToDense(indices, values, default_value, value_is_scalar,
                DimsToShape(output_dims), output_data);
}

template <typename Scalar>
void Pack(int dim, const Scalar* const* input_data,
          const Dims<4>* const* input_dims, int inputs_count,
          Scalar* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_51(mht_51_v, 1464, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Pack");

  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::PackParams op_params;
  op_params.axis = 3 - dim;
  op_params.inputs_count = inputs_count;

  Pack(op_params, input_shapes_indirect.data(), input_data,
       DimsToShape(output_dims), output_data);
}

template <typename Scalar>
void Unpack(int axis, const Scalar* input_data, const Dims<4>& input_dims,
            int dimensions, int outputs_count, Scalar* const* output_datas,
            const Dims<4>& output_dims) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_52(mht_52_v, 1485, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Unpack");

  tflite::UnpackParams op_params;
  op_params.axis = 3 - axis;
  op_params.num_split = outputs_count;

  Unpack(op_params, DimsToShape(input_dims), input_data,
         DimsToShape(output_dims), output_datas);
}

template <typename Scalar>
void Pack(int dim, const Scalar* const* input_data,
          const Dims<4>* const* input_dims, const int32* input_zeropoint,
          const float* input_scale, int inputs_count, Scalar* output_data,
          const Dims<4>& output_dims, const int32 output_zeropoint,
          const float output_scale) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_53(mht_53_v, 1502, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Pack");

  std::vector<RuntimeShape> input_shapes(inputs_count);
  std::vector<const RuntimeShape*> input_shapes_indirect(inputs_count);
  for (int i = 0; i < inputs_count; ++i) {
    ShapeFromDims(*input_dims[i], &input_shapes[i]);
    input_shapes_indirect[i] = &input_shapes[i];
  }
  tflite::PackParams op_params;
  op_params.axis = 3 - dim;
  op_params.input_zeropoint = input_zeropoint;
  op_params.input_scale = input_scale;
  op_params.inputs_count = inputs_count;
  op_params.output_zeropoint = output_zeropoint;
  op_params.output_scale = output_scale;

  PackWithScaling(op_params, input_shapes_indirect.data(), input_data,
                  DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const RuntimeShape& input_shape,
                     float* output_data, const RuntimeShape& output_shape) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_54(mht_54_v, 1526, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "L2Normalization");

  static_assert(Ac == FusedActivationFunctionType::kNone, "");
  tflite::L2NormalizationParams op_params;
  // No params need to be set for float.

  L2Normalization(op_params, input_shape, input_data, output_shape,
                  output_data);
}

inline void L2Normalization(const uint8* input_data,
                            const RuntimeShape& input_shape,
                            int32 input_zero_point, uint8* output_data,
                            const RuntimeShape& output_shape) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_55(mht_55_v, 1541, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "L2Normalization");

  tflite::L2NormalizationParams op_params;
  op_params.input_zero_point = input_zero_point;

  L2Normalization(op_params, input_shape, input_data, output_shape,
                  output_data);
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_56(mht_56_v, 1554, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "L2Normalization");

  L2Normalization<Ac>(input_data, DimsToShape(input_dims), output_data,
                      DimsToShape(output_dims));
}

inline void L2Normalization(const uint8* input_data, const Dims<4>& input_dims,
                            int32 input_zero_point, uint8* output_data,
                            const Dims<4>& output_dims) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_57(mht_57_v, 1564, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "L2Normalization");

  L2Normalization(input_data, DimsToShape(input_dims), input_zero_point,
                  output_data, DimsToShape(output_dims));
}

inline void Relu(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_58(mht_58_v, 1573, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Relu");

  Relu(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

inline void Relu1(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_59(mht_59_v, 1582, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Relu1");

  Relu1(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
        output_data);
}

inline void Relu6(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_60(mht_60_v, 1591, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Relu6");

  Relu6(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
        output_data);
}

inline void ReluX(uint8 min_value, uint8 max_value, const uint8* input_data,
                  const RuntimeShape& input_shape, uint8* output_data,
                  const RuntimeShape& output_shape) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_61(mht_61_v, 1601, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "ReluX");

  tflite::ActivationParams params;
  params.quantized_activation_max = max_value;
  params.quantized_activation_min = min_value;
  ReluX(params, input_shape, input_data, output_shape, output_data);
}

template <FusedActivationFunctionType Ac>
inline void Add(int left_shift, const uint8* input1_data,
                const Dims<4>& input1_dims, int32 input1_offset,
                int32 input1_multiplier, int input1_shift,
                const uint8* input2_data, const Dims<4>& input2_dims,
                int32 input2_offset, int32 input2_multiplier, int input2_shift,
                int32 output_offset, int32 output_multiplier, int output_shift,
                int32 output_activation_min, int32 output_activation_max,
                uint8* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_62(mht_62_v, 1619, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Add");

  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }

  tflite::ArithmeticParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Add(const int32* input1_data, const Dims<4>& input1_dims,
         const int32* input2_data, const Dims<4>& input2_dims,
         int32* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_63(mht_63_v, 1656, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Add");

  ruy::profiler::ScopeLabel label("Add/int32");
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);

  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = std::numeric_limits<int32>::min();
  op_params.quantized_activation_max = std::numeric_limits<int32>::max();
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAdd(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_64(mht_64_v, 1681, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BroadcastAdd");

  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }

  tflite::ArithmeticParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  BroadcastAdd4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
void Add(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_65(mht_65_v, 1718, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Add");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_66(mht_66_v, 1737, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BroadcastAdd");

  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  BroadcastAdd4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAddFivefold(
    int y0, int y1, int y2, int y3, int y4, int left_shift,
    const uint8* input1_data, const Dims<4>& input1_dims, int32 input1_offset,
    int32 input1_multiplier, int input1_shift, const uint8* input2_data,
    const Dims<4>& input2_dims, int32 input2_offset, int32 input2_multiplier,
    int input2_shift, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_67(mht_67_v, 1757, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BroadcastAddFivefold");

  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  tflite::ArithmeticParams op_params;
  op_params.broadcast_category =
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.broadcast_shape[4] = y0;
  op_params.broadcast_shape[3] = y1;
  op_params.broadcast_shape[2] = y2;
  op_params.broadcast_shape[1] = y3;
  op_params.broadcast_shape[0] = y4;
  BroadcastAddFivefold(op_params, DimsToShape(input1_dims), input1_data,
                       DimsToShape(input2_dims), input2_data,
                       DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  BroadcastAdd(input1_data, input1_dims, input2_data, input2_dims,
               output_activation_min, output_activation_max, output_data,
               output_dims);
}

template <FusedActivationFunctionType Ac>
inline void Add(const int16* input1_data, const Dims<4>& input1_dims,
                int input1_shift, const int16* input2_data,
                const Dims<4>& input2_dims, int input2_shift,
                int16 output_activation_min, int16 output_activation_max,
                int16* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_68(mht_68_v, 1815, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Add");

  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, -32768);
    TFLITE_DCHECK_EQ(output_activation_max, 32767);
  }

  tflite::ArithmeticParams op_params;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void Sub(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_69(mht_69_v, 1842, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Sub");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(FusedActivationFunctionType::kNone,
                      &output_activation_min, &output_activation_max);
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  Sub(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void Sub(const T* input1_data, const Dims<4>& input1_dims, const T* input2_data,
         const Dims<4>& input2_dims, T* output_data,
         const Dims<4>& output_dims) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_70(mht_70_v, 1860, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Sub");

  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = std::numeric_limits<T>::min();
  op_params.quantized_activation_max = std::numeric_limits<T>::max();
  Sub(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline bool AveragePool(const float* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int kwidth, int kheight,
                        float output_activation_min,
                        float output_activation_max, float* output_data,
                        const Dims<4>& output_dims) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_71(mht_71_v, 1877, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "AveragePool");

  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = kheight;
  params.filter_width = kwidth;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  return AveragePool(params, DimsToShape(input_dims), input_data,
                     DimsToShape(output_dims), output_data);
}

// Transitional version that will be moved shortly to legacy_reference_ops, as
// part of RuntimeShape revisions.
inline void BroadcastMul4DSlow(const uint8* input1_data,
                               const Dims<4>& input1_dims, int32 input1_offset,
                               const uint8* input2_data,
                               const Dims<4>& input2_dims, int32 input2_offset,
                               int32 output_offset, int32 output_multiplier,
                               int output_shift, int32 output_activation_min,
                               int32 output_activation_max, uint8* output_data,
                               const Dims<4>& output_dims) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_72(mht_72_v, 1903, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BroadcastMul4DSlow");

  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);
  op_params.input1_offset = input1_offset;
  op_params.input2_offset = input2_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_73(mht_73_v, 1926, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BroadcastMul");

  BroadcastMul4DSlow(
      input1_data, input1_dims, input1_offset, input2_data, input2_dims,
      input2_offset, output_offset, output_multiplier,
      //
      kReverseShift * output_shift,
      //
      output_activation_min, output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_74(mht_74_v, 1947, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BroadcastMul");

  BroadcastMul(input1_data, input1_dims, input1_offset, input2_data,
               input2_dims, input2_offset, output_offset, output_multiplier,
               output_shift, output_activation_min, output_activation_max,
               output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
bool AveragePool(const float* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int kwidth, int kheight, float* output_data,
                 const Dims<4>& output_dims) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_75(mht_75_v, 1962, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "AveragePool");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  return AveragePool(input_data, input_dims, stride_width, stride_height,
                     pad_width, pad_height, kwidth, kheight,
                     output_activation_min, output_activation_max, output_data,
                     output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
bool AveragePool(const float* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, float* output_data,
                 const Dims<4>& output_dims) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_76(mht_76_v, 1980, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "AveragePool");

  return AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width,
                         pad_height, filter_width, filter_height, output_data,
                         output_dims);
}

inline bool AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int filter_width, int filter_height,
                        int32 output_activation_min,
                        int32 output_activation_max, uint8* output_data,
                        const Dims<4>& output_dims) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_77(mht_77_v, 1994, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "AveragePool");

  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;
  return AveragePool(params, DimsToShape(input_dims), input_data,
                     DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
bool AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int filter_width, int filter_height,
                 int32 output_activation_min, int32 output_activation_max,
                 uint8* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_78(mht_78_v, 2017, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "AveragePool");

  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  return AveragePool(input_data, input_dims, stride_width, stride_height,
                     pad_width, pad_height, filter_width, filter_height,
                     output_activation_min, output_activation_max, output_data,
                     output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
bool AveragePool(const uint8* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_79(mht_79_v, 2042, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "AveragePool");

  return AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width,
                         pad_height, filter_width, filter_height,
                         output_activation_min, output_activation_max,
                         output_data, output_dims);
}

inline void MaxPool(const float* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int kwidth, int kheight,
                    float output_activation_min, float output_activation_max,
                    float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_80(mht_80_v, 2056, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "MaxPool");

  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = kheight;
  params.filter_width = kwidth;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  MaxPool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
          output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int kwidth, int kheight, float* output_data,
             const Dims<4>& output_dims) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_81(mht_81_v, 2078, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "MaxPool");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, kwidth, kheight, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_82(mht_82_v, 2093, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "MaxPool");

  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_data, output_dims);
}

inline void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int filter_width, int filter_height,
                    int32 output_activation_min, int32 output_activation_max,
                    uint8* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_83(mht_83_v, 2105, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "MaxPool");

  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;
  MaxPool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
          output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int filter_width, int filter_height, int32 output_activation_min,
             int32 output_activation_max, uint8* output_data,
             const Dims<4>& output_dims) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_84(mht_84_v, 2128, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "MaxPool");

  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, filter_width, filter_height, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             int32 output_activation_min, int32 output_activation_max,
             uint8* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_85(mht_85_v, 2151, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "MaxPool");

  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_activation_min,
              output_activation_max, output_data, output_dims);
}

inline void L2Pool(const float* input_data, const Dims<4>& input_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int filter_width, int filter_height,
                   float output_activation_min, float output_activation_max,
                   float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_86(mht_86_v, 2164, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "L2Pool");

  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  L2Pool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
         output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims,
            int stride_width, int stride_height, int pad_width, int pad_height,
            int filter_width, int filter_height, float* output_data,
            const Dims<4>& output_dims) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_87(mht_87_v, 2186, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "L2Pool");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  L2Pool(input_data, input_dims, stride_width, stride_height, pad_width,
         pad_height, filter_width, filter_height, output_activation_min,
         output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int filter_width, int filter_height,
            float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_88(mht_88_v, 2201, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "L2Pool");

  L2Pool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
             filter_width, filter_height, output_data, output_dims);
}

inline void Softmax(const float* input_data, const Dims<4>& input_dims,
                    float beta, float* output_data,
                    const Dims<4>& output_dims) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_89(mht_89_v, 2211, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Softmax");

  Softmax(input_data, DimsToShape(input_dims), beta, output_data,
          DimsToShape(output_dims));
}

inline void Softmax(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const Dims<4>& output_dims) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_90(mht_90_v, 2222, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Softmax");

  Softmax(input_data, DimsToShape(input_dims), input_beta_multiplier,
          input_beta_left_shift, diff_min, output_data,
          DimsToShape(output_dims));
}

inline void LogSoftmax(const float* input_data, const Dims<4>& input_dims,
                       float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_91(mht_91_v, 2232, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "LogSoftmax");

  LogSoftmax(input_data, DimsToShape(input_dims), output_data,
             DimsToShape(output_dims));
}

inline void LogSoftmax(const uint8* input_data, const Dims<4>& input_dims,
                       int32 input_multiplier, int32 input_left_shift,
                       int32 reverse_scaling_divisor,
                       int32 reverse_scaling_right_shift, int diff_min,
                       uint8* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_92(mht_92_v, 2244, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "LogSoftmax");

  LogSoftmax(input_data, DimsToShape(input_dims), input_multiplier,
             input_left_shift, reverse_scaling_divisor,
             reverse_scaling_right_shift, diff_min, output_data,
             DimsToShape(output_dims));
}

inline void Logistic(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_93(mht_93_v, 2255, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Logistic");

  Logistic(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
           output_data);
}

inline void Logistic(const uint8* input_data, const Dims<4>& input_dims,
                     int32 input_zero_point, int32 input_range_radius,
                     int32 input_multiplier, int input_left_shift,
                     uint8* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_94(mht_94_v, 2266, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Logistic");

  Logistic(input_data, DimsToShape(input_dims), input_zero_point,
           input_range_radius, input_multiplier, input_left_shift, output_data,
           DimsToShape(output_dims));
}

inline void Logistic(const int16* input_data, const Dims<4>& input_dims,
                     int16* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_95(mht_95_v, 2276, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Logistic");

  Logistic(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
           output_data);
}

inline void Tanh(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_96(mht_96_v, 2285, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Tanh");

  Tanh(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

inline void Tanh(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_97(mht_97_v, 2296, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Tanh");

  Tanh(input_data, DimsToShape(input_dims), input_zero_point,
       input_range_radius, input_multiplier, input_left_shift, output_data,
       DimsToShape(output_dims));
}

inline void Tanh(const int16* input_data, const Dims<4>& input_dims,
                 int input_left_shift, int16* output_data,
                 const Dims<4>& output_dims) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_98(mht_98_v, 2307, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Tanh");

  Tanh(input_data, DimsToShape(input_dims), input_left_shift, output_data,
       DimsToShape(output_dims));
}

template <typename T>
inline void DepthToSpace(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_99(mht_99_v, 2318, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "DepthToSpace");

  tflite::DepthToSpaceParams op_params;
  op_params.block_size = block_size;

  DepthToSpace(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

template <typename T>
inline void SpaceToDepth(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_100(mht_100_v, 2332, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "SpaceToDepth");

  tflite::SpaceToDepthParams op_params;
  op_params.block_size = block_size;

  SpaceToDepth(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

template <typename T>
inline void Mul(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T output_activation_min, T output_activation_max,
                T* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_101(mht_101_v, 2347, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Mul");

  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Mul(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_102(mht_102_v, 2363, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Mul");

  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_103(mht_103_v, 2382, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BroadcastMul");

  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int16* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_104(mht_104_v, 2412, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Mul");

  tflite::ArithmeticParams op_params;
  // No params in this version.

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int32 output_offset, int32 output_activation_min,
                int32 output_activation_max, uint8* output_data,
                const Dims<4>& output_dims) {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_105(mht_105_v, 2428, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Mul");

  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.output_offset = output_offset;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void LocalResponseNormalization(const float* input_data,
                                       const Dims<4>& input_dims, int range,
                                       float bias, float alpha, float beta,
                                       float* output_data,
                                       const Dims<4>& output_dims) {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_106(mht_106_v, 2446, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "LocalResponseNormalization");

  tflite::LocalResponseNormalizationParams op_params;
  op_params.range = range;
  op_params.bias = bias;
  op_params.alpha = alpha;
  op_params.beta = beta;

  LocalResponseNormalization(op_params, DimsToShape(input_dims), input_data,
                             DimsToShape(output_dims), output_data);
}

template <typename SrcT, typename DstT>
void Cast(const SrcT* input_data, const Dims<4>& input_dims, DstT* output_data,
          const Dims<4>& output_dims) {
  Cast(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

inline void Floor(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_107(mht_107_v, 2468, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Floor");

  Floor(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
        output_data);
}

template <typename T>
inline void ResizeBilinear(const T* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, T* output_data,
                           const Dims<4>& output_dims, bool align_corners) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_108(mht_108_v, 2480, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "ResizeBilinear");

  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = align_corners;
  op_params.half_pixel_centers = false;
  ResizeBilinear(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(output_size_dims), output_size_data,
                 DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_109(mht_109_v, 2496, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "ResizeBilinear");

  ResizeBilinear<float>(input_data, input_dims, output_size_data,
                        output_size_dims, output_data, output_dims,
                        /*align_corners=*/false);
}

inline void ResizeBilinear(const uint8* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, uint8* output_data,
                           const Dims<4>& output_dims) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_110(mht_110_v, 2508, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "ResizeBilinear");

  ResizeBilinear<uint8>(input_data, input_dims, output_size_data,
                        output_size_dims, output_data, output_dims,
                        /*align_corners=*/false);
}

template <typename T>
inline void SpaceToBatchND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* paddings_data,
                           const Dims<4>& paddings_dims, T* output_data,
                           const Dims<4>& output_dims,
                           const int32_t pad_value) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_111(mht_111_v, 2524, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "SpaceToBatchND");

  tflite::SpaceToBatchParams op_params;
  op_params.output_offset = pad_value;

  SpaceToBatchND(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(block_shape_dims), block_shape_data,
                 DimsToShape(paddings_dims), paddings_data,
                 DimsToShape(output_dims), output_data);
}

template <typename T>
inline void SpaceToBatchND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* paddings_data,
                           const Dims<4>& paddings_dims, T* output_data,
                           const Dims<4>& output_dims) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_112(mht_112_v, 2543, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "SpaceToBatchND");

  tflite::SpaceToBatchParams op_params;
  op_params.output_offset = 0;

  SpaceToBatchND(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(block_shape_dims), block_shape_data,
                 DimsToShape(paddings_dims), paddings_data,
                 DimsToShape(output_dims), output_data);
}

template <typename T>
inline void BatchToSpaceND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* crops_data, const Dims<4>& crops_dims,
                           T* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_113(mht_113_v, 2561, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BatchToSpaceND");

  BatchToSpaceND(DimsToShape(input_dims), input_data,
                 DimsToShape(block_shape_dims), block_shape_data,
                 DimsToShape(crops_dims), crops_data, DimsToShape(output_dims),
                 output_data);
}

// Legacy signature, function covered both Pad and PadV2.
template <typename T>
inline void PadV2(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& left_paddings,
                  const std::vector<int>& right_paddings, T* output_data,
                  const Dims<4>& output_dims, const T pad_value) {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_114(mht_114_v, 2576, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "PadV2");

  TFLITE_DCHECK_EQ(left_paddings.size(), 4);
  TFLITE_DCHECK_EQ(right_paddings.size(), 4);
  tflite::PadParams op_params;
  op_params.left_padding_count = 4;
  op_params.right_padding_count = 4;
  for (int i = 0; i < 4; ++i) {
    op_params.left_padding[i] = left_paddings[3 - i];
    op_params.right_padding[i] = right_paddings[3 - i];
  }
  // SetFloatOrInt(pad_value, &op_params.pad_value);
  const T pad_value_copy = pad_value;

  Pad(op_params, DimsToShape(input_dims), input_data, &pad_value_copy,
      DimsToShape(output_dims), output_data);
}

// Old Pad that calls legacy PadV2.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims, const int32_t pad_value) {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_115(mht_115_v, 2601, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Pad");

  const T converted_pad_value = static_cast<T>(pad_value);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, converted_pad_value);
}

// Old Pad that only padded with 0.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims) {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_116(mht_116_v, 2615, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Pad");

  const T pad_value = static_cast<T>(0);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, pad_value);
}

template <typename T>
void TensorFlowMinimum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_117(mht_117_v, 2627, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "TensorFlowMinimum");

  Minimum(DimsToShape(input1_dims), input1_data, input2_data,
          DimsToShape(output_dims), output_data);
}

template <typename T>
void TensorFlowMaximum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_118(mht_118_v, 2638, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "TensorFlowMaximum");

  Maximum(DimsToShape(input1_dims), input1_data, input2_data,
          DimsToShape(output_dims), output_data);
}

template <typename T, typename Op>
void TensorFlowMaximumMinimum(const T* input1_data, const Dims<4>& input1_dims,
                              const T* input2_data, const Dims<4>& input2_dims,
                              T* output_data, const Dims<4>& output_dims,
                              Op op) {
  MaximumMinimumBroadcastSlow(DimsToShape(input1_dims), input1_data,
                              DimsToShape(input2_dims), input2_data,
                              DimsToShape(output_dims), output_data, op);
}

template <typename T1, typename T2, typename T3>
void ArgMax(const T3* axis, const T1* input_data,
            const tflite::Dims<4>& input_dims, T2* output_data,
            const tflite::Dims<4>& output_dims) {
  // Assumes the input always has 4 dimensions, and therefore,
  // output always has three dimensions.
  auto output_shape = RuntimeShape(
      {output_dims.sizes[2], output_dims.sizes[1], output_dims.sizes[0]});
  // Another way to interpret this is that output_dims.sizes[4] is always 1.
  TFLITE_DCHECK_EQ(output_shape.FlatSize(),
                   DimsToShape(output_dims).FlatSize());
  // Legacy path only supported this.
  TFLITE_DCHECK_EQ(axis[0], 3);
  ArgMinMax(DimsToShape(input_dims), input_data, axis, output_shape,
            output_data, std::greater<T1>());
}

template <typename T1, typename T2, typename T3, typename Cmp>
void ArgMinMax(const T3* axis, const T1* input_data, const Dims<4>& input_dims,
               T2* output_data, const Dims<4>& output_dims, const Cmp& cmp) {
  ArgMinMax(axis, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
            output_data, cmp);
}

template <typename T>
inline void Pow(const T* input1_data, const Dims<4>& input1_dims,
                const T* input2_data, const Dims<4>& input2_dims,
                T* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_119(mht_119_v, 2683, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Pow");

  Pow(DimsToShape(input1_dims), input1_data, DimsToShape(input2_dims),
      input2_data, DimsToShape(output_dims), output_data);
}

template <typename T>
inline void BroadcastPow(const T* input1_data, const Dims<4>& input1_dims,
                         const T* input2_data, const Dims<4>& input2_dims,
                         T* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_120(mht_120_v, 2694, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "BroadcastPow");

  BroadcastPow4DSlow(DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

// R: Result type. T1: Input 1 type. T2: Input 2 type.
template <typename R, typename T1, typename T2>
inline void BroadcastBinaryFunction(const T1* input1_data,
                                    const Dims<4>& input1_dims,
                                    const T2* input2_data,
                                    const Dims<4>& input2_dims, R* output_data,
                                    const Dims<4>& output_dims,
                                    R (*func)(T1, T2)) {
  BroadcastBinaryFunction(DimsToShape(input1_dims), input1_data,
                          DimsToShape(input2_dims), input2_data,
                          DimsToShape(output_dims), output_data, func);
}

// R: Result type. T1: Input 1 type. T2: Input 2 type.
template <typename R, typename T1, typename T2>
inline void BinaryFunction(const T1* input1_data, const Dims<4>& input1_dims,
                           const T2* input2_data, const Dims<4>& input2_dims,
                           R* output_data, const Dims<4>& output_dims,
                           R (*func)(T1, T2)) {
  BinaryFunction(DimsToShape(input1_dims), input1_data,
                 DimsToShape(input2_dims), input2_data,
                 DimsToShape(output_dims), output_data, func);
}

template <typename T>
inline void Slice(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& begin, const std::vector<int>& size,
                  T* output_data, const Dims<4>& output_dims) {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlegacy_reference_opsDTh mht_121(mht_121_v, 2730, "", "./tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h", "Slice");

  tflite::SliceParams op_params;
  op_params.begin_count = 4;
  op_params.size_count = 4;
  for (int i = 0; i < 4; ++i) {
    op_params.begin[i] = begin[3 - i];
    op_params.size[i] = size[3 - i];
  }

  Slice(op_params, DimsToShape(input_dims), input_data,
        DimsToShape(output_dims), output_data);
}

}  // namespace reference_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEGACY_REFERENCE_OPS_H_
