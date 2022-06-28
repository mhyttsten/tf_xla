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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_CONSTANTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_CONSTANTS_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTh() {
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


#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

template <DataType S, typename T>
void RearrangeWeightsForConvConstants(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int d = 0; d < dst_depth; ++d) {
          const int channels_count = std::min(4, weights.shape.i - s * 4);
          T filters[4];
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < channels_count; ++j) {
              const int s_ch = s * 4 + j;
              const int d_ch = d * 4 + i;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index =
                    weights.shape.LinearIndex({d_ch, y, x, s_ch});
                filters[j][i] = weights.data[f_index];
              } else {
                filters[j][i] = 0.0f;
              }
            }
          }
          for (int i = 0; i < channels_count; ++i) {
            dst[counter++] = filters[i];
          }
        }
      }
    }
  }
}

template <DataType S, typename T>
void RearrangeWeightsForConvConstantsDot(
    const tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int d = 0; d < dst_depth; ++d) {
          const int channels_count = std::min(4, weights.shape.o - d * 4);
          T filters[4];
          for (int j = 0; j < channels_count; ++j) {
            for (int i = 0; i < 4; ++i) {
              const int s_ch = s * 4 + i;
              const int d_ch = d * 4 + j;
              if (s_ch < weights.shape.i && d_ch < weights.shape.o) {
                const int f_index =
                    weights.shape.LinearIndex({d_ch, y, x, s_ch});
                filters[j][i] = weights.data[f_index];
              } else {
                filters[j][i] = 0.0f;
              }
            }
          }
          for (int i = 0; i < channels_count; ++i) {
            dst[counter++] = filters[i];
          }
        }
      }
    }
  }
}

template <DataType T>
void UploadWeightsForConvConstants(const tflite::gpu::Tensor<OHWI, T>& weights,
                                   CalculationsPrecision precision,
                                   bool use_dot_conv, GPUOperation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_constantsDTh mht_0(mht_0_v, 279, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_constants.h", "UploadWeightsForConvConstants");

  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const bool f32_weights = precision == CalculationsPrecision::F32;
  const int float_size = f32_weights ? 4 : 2;
  const int aligned_ch_count = use_dot_conv ? weights.shape.o * src_depth * 4
                                            : weights.shape.i * dst_depth * 4;
  const int float_count = aligned_ch_count * kernel_x * kernel_y;

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type = MemoryType::CONSTANT;
  desc.size = float_size * float_count;
  desc.data.resize(desc.size);

  if (f32_weights) {
    float4* ptr = reinterpret_cast<float4*>(desc.data.data());
    if (use_dot_conv) {
      RearrangeWeightsForConvConstantsDot(weights,
                                          absl::MakeSpan(ptr, float_count / 4));
    } else {
      RearrangeWeightsForConvConstants(weights,
                                       absl::MakeSpan(ptr, float_count / 4));
    }
  } else {
    half4* ptr = reinterpret_cast<half4*>(desc.data.data());
    if (use_dot_conv) {
      RearrangeWeightsForConvConstantsDot(weights,
                                          absl::MakeSpan(ptr, float_count / 4));
    } else {
      RearrangeWeightsForConvConstants(weights,
                                       absl::MakeSpan(ptr, float_count / 4));
    }
  }

  op->args_.AddObject("weights",
                      absl::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvConstantsSupported(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr);

GPUOperation CreateConvConstants(const GpuInfo& gpu_info,
                                 const OperationDef& definition,
                                 const Convolution2DAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_CONSTANTS_H_
