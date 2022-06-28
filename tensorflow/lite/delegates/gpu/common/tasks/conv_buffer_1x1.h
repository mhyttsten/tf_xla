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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_BUFFER_1X1_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_BUFFER_1X1_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_buffer_1x1DTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_buffer_1x1DTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_buffer_1x1DTh() {
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


#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

namespace tflite {
namespace gpu {

class ConvBuffer1x1 : public GPUOperation {
 public:
  ConvBuffer1x1() = default;

  // Move only
  ConvBuffer1x1(ConvBuffer1x1&& operation);
  ConvBuffer1x1& operator=(ConvBuffer1x1&& operation);
  ConvBuffer1x1(const ConvBuffer1x1&) = delete;
  ConvBuffer1x1& operator=(const ConvBuffer1x1&) = delete;

  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  int3 GetGridSize() const override;

  WeightsDescription GetWeightsDescription() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_buffer_1x1DTh mht_0(mht_0_v, 223, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_buffer_1x1.h", "GetWeightsDescription");

    WeightsDescription desc;
    desc.type = DeduceDataTypeFromPrecision(definition_.precision);
    desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    desc.output_group_size = conv_params_.block_size.z;
    return desc;
  }

  struct ConvParams {
    int3 block_size = int3(1, 1, 1);
    int element_size = 4;  // can be 4, 8 or 16

    // By default in 2d convolution we have the same weights for WH dims, but in
    // some cases we need separate weights for H dimension and convolution
    // kernel requires very small modifications to support it.
    bool different_weights_for_height = false;
  };

 private:
  ConvBuffer1x1(const OperationDef& definition, const ConvParams& conv_params,
                const GpuInfo& gpu_info);
  friend ConvBuffer1x1 CreateConvBuffer1x1(const GpuInfo& gpu_info,
                                           const OperationDef& definition,
                                           const Convolution2DAttributes& attr,
                                           const BHWC* shape);
  friend ConvBuffer1x1 CreateConvBuffer1x1(const GpuInfo& gpu_info,
                                           const OperationDef& definition,
                                           const FullyConnectedAttributes& attr,
                                           const BHWC* shape);
  friend ConvBuffer1x1 CreateConvBuffer1x1Wino4x4To6x6(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Convolution2DAttributes& attr, const BHWC* shape);
  friend ConvBuffer1x1 CreateConvBuffer1x1DynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Convolution2DAttributes& attr, const BHWC& weights_shape,
      const BHWC* dst_shape);

  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                  const tflite::gpu::Tensor<Linear, T>& biases);
  template <DataType T>
  void UploadDataForWinograd4x4To6x6(
      const tflite::gpu::Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadBiases(const tflite::gpu::Tensor<Linear, T>& biases);

  std::string GenerateConvBuffer1x1(
      const OperationDef& op_def, const ConvBuffer1x1::ConvParams& conv_params,
      const GpuInfo& gpu_info, Arguments* args);

  ConvParams conv_params_;
};

template <DataType T>
void ConvBuffer1x1::UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                               const tflite::gpu::Tensor<Linear, T>& biases) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_buffer_1x1DTh mht_1(mht_1_v, 285, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_buffer_1x1.h", "ConvBuffer1x1::UploadData");

  UploadWeights(weights);
  UploadBiases(biases);
}

template <DataType T>
void ConvBuffer1x1::UploadDataForWinograd4x4To6x6(
    const tflite::gpu::Tensor<OHWI, T>& weights) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_buffer_1x1DTh mht_2(mht_2_v, 295, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_buffer_1x1.h", "ConvBuffer1x1::UploadDataForWinograd4x4To6x6");

  tflite::gpu::Tensor<OHWI, T> wino_weights;
  RearrangeWeightsToWinograd4x4To6x6Weights(weights, &wino_weights);
  UploadWeights(wino_weights);
  tflite::gpu::Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape = Linear(weights.shape.o);
  bias.data.resize(weights.shape.o, 0.0f);
  UploadBiases(bias);
}

template <DataType T>
void ConvBuffer1x1::UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_buffer_1x1DTh mht_3(mht_3_v, 309, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_buffer_1x1.h", "ConvBuffer1x1::UploadWeights");

  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor desc;
  desc.element_type = weights_desc.type;
  desc.element_size = 16;
  desc.memory_type = MemoryType::GLOBAL;
  desc.size = flt_count * SizeOf(desc.element_type);
  desc.data.resize(desc.size);

  RearrangeWeights(weights, weights_desc, absl::MakeSpan(desc.data));

  args_.AddObject("weights",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType T>
void ConvBuffer1x1::UploadBiases(const tflite::gpu::Tensor<Linear, T>& biases) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_buffer_1x1DTh mht_4(mht_4_v, 331, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_buffer_1x1.h", "ConvBuffer1x1::UploadBiases");

  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::BUFFER;
  desc.element_type = definition_.GetDataType();
  int depth = AlignByN(biases.shape.v, 4 * conv_params_.block_size.z) / 4;
  desc.UploadLinearData(biases, depth);
  args_.AddObject("biases",
                  absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
}

bool IsConvBuffer1x1Supported(const OperationDef& definition,
                              const Convolution2DAttributes& attr);

bool IsConvBuffer1x1Supported(const OperationDef& definition,
                              const BHWC& weights_shape,
                              const Convolution2DAttributes& attr);

ConvBuffer1x1 CreateConvBuffer1x1(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  const Convolution2DAttributes& attr,
                                  const BHWC* shape = nullptr);

ConvBuffer1x1 CreateConvBuffer1x1(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  const FullyConnectedAttributes& attr,
                                  const BHWC* shape = nullptr);

ConvBuffer1x1 CreateConvBuffer1x1DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC& weights_shape,
    const BHWC* dst_shape = nullptr);

ConvBuffer1x1 CreateConvBuffer1x1Wino4x4To6x6(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Convolution2DAttributes& attr, const BHWC* shape = nullptr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_BUFFER_1X1_H_
