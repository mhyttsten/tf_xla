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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_DEPTHWISE_CONV_3X3_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_DEPTHWISE_CONV_3X3_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTh() {
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


#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class DepthwiseConv3x3 : public GPUOperation {
 public:
  DepthwiseConv3x3() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  int3 GetGridSize() const override;

  // Move only
  DepthwiseConv3x3(DepthwiseConv3x3&& operation);
  DepthwiseConv3x3& operator=(DepthwiseConv3x3&& operation);
  DepthwiseConv3x3(const DepthwiseConv3x3&) = delete;
  DepthwiseConv3x3& operator=(const DepthwiseConv3x3&) = delete;

 private:
  explicit DepthwiseConv3x3(const OperationDef& definition,
                            bool weights_are_buffer, bool local_mem_uploads,
                            const GpuInfo& gpu_info);
  template <DataType T>
  void UploadWeightsAndBiases(const tflite::gpu::Tensor<OHWI, T>& weights,
                              const tflite::gpu::Tensor<Linear, T>& biases,
                              bool weights_are_buffer);

  friend DepthwiseConv3x3 CreateDepthwiseConv3x3(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const DepthwiseConvolution2DAttributes& attr);

  template <DataType S, typename T>
  void RearrangeWeightsAndBiasesData(
      const tflite::gpu::Tensor<OHWI, S>& weights,
      const tflite::gpu::Tensor<Linear, S>& biases, absl::Span<T> dst);

  std::string GenerateDepthwiseConvCode(const GpuInfo& gpu_info,
                                        const OperationDef& op_def,
                                        bool weights_are_buffer,
                                        bool local_mem_uploads);

  bool local_mem_uploads_;
};

template <DataType T>
void DepthwiseConv3x3::UploadWeightsAndBiases(
    const tflite::gpu::Tensor<OHWI, T>& weights,
    const tflite::gpu::Tensor<Linear, T>& biases, bool weights_are_buffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSdepthwise_conv_3x3DTh mht_0(mht_0_v, 249, "", "./tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3.h", "DepthwiseConv3x3::UploadWeightsAndBiases");

  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  int texture_width = 10;  // 3x3 kernel + 1 bias
  int texture_height = src_depth;
  const int elements_count = texture_width * texture_height;
  const bool fp32_weights = definition_.precision == CalculationsPrecision::F32;
  const int float4_size = fp32_weights ? 16 : 8;

  std::vector<uint8_t> data(float4_size * elements_count);
  if (fp32_weights) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    RearrangeWeightsAndBiasesData(weights, biases,
                                  absl::MakeSpan(ptr, elements_count));
  } else {
    half4* ptr = reinterpret_cast<half4*>(data.data());
    RearrangeWeightsAndBiasesData(weights, biases,
                                  absl::MakeSpan(ptr, elements_count));
  }

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 4;
    desc.size = float4_size * elements_count;
    desc.data = std::move(data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    Texture2DDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.size = int2(texture_width, texture_height);
    desc.data = std::move(data);
    args_.AddObject("weights",
                    absl::make_unique<Texture2DDescriptor>(std::move(desc)));
  }
}

template <DataType S, typename T>
void DepthwiseConv3x3::RearrangeWeightsAndBiasesData(
    const tflite::gpu::Tensor<OHWI, S>& weights,
    const tflite::gpu::Tensor<Linear, S>& biases, absl::Span<T> dst) {
  const int src_depth = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  for (int s = 0; s < src_depth; ++s) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        T filter_val;
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + i;
          if (s_ch < weights.shape.i) {
            const int f_index = weights.shape.LinearIndex({0, y, x, s_ch});
            filter_val[i] = weights.data[f_index];
          } else {
            filter_val[i] = 0.0f;
          }
        }
        dst[counter++] = filter_val;
      }
    }

    T bias_val;
    for (int i = 0; i < 4; ++i) {
      const int dst_ch = s * 4 + i;
      bias_val[i] = dst_ch >= biases.shape.v ? 0.0f : biases.data[dst_ch];
    }
    dst[counter++] = bias_val;
  }
}

bool IsDepthwiseConv3x3Supported(const GpuInfo& gpu_info,
                                 const DepthwiseConvolution2DAttributes& attr);

DepthwiseConv3x3 CreateDepthwiseConv3x3(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_DEPTHWISE_CONV_3X3_H_
