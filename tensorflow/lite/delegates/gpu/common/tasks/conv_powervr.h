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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_POWERVR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_POWERVR_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh() {
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


#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

namespace tflite {
namespace gpu {

class ConvPowerVR : public GPUOperation {
 public:
  ConvPowerVR() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  WeightsDescription GetWeightsDescription() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh mht_0(mht_0_v, 221, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h", "GetWeightsDescription");

    WeightsDescription desc;
    desc.type = conv_params_.weights_data_type;
    desc.layout = conv_params_.weights_layout;
    desc.output_group_size = conv_params_.block_size.w;
    return desc;
  }

  // Move only
  ConvPowerVR(ConvPowerVR&& operation);
  ConvPowerVR& operator=(ConvPowerVR&& operation);
  ConvPowerVR(const ConvPowerVR&) = delete;
  ConvPowerVR& operator=(const ConvPowerVR&) = delete;

 private:
  enum class WeightsUploadType {
    LOCAL_MEM_ASYNC_SUBGROUP,  // we use it for PowerVR with workgroup size = 32
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
    CONSTANT_MEM,
    PRIVATE_MEM_SIMD_BROADCAST,
    TEXTURES_MEM_X4,  // 4 textures for weights
  };

  struct ConvParams {
    // Usually we use this combinations for CalculationPrecision:
    // F32: all F32
    // F16: all F16
    // F32_F16: all besides accumulator is F16, including weights
    // But for PowerVR we can achieve better performance in F32_F16 with F32
    // weights, so for PowerVR in this kernel we have F32 weights for
    // F32_F16 precision mode
    DataType weights_data_type;  // used for weights and biases
    int4 block_size;             // WHDS
    bool fixed_work_group_size;
    bool linear_spatial;  // spatial dimensions are Width/Height/Depth
    bool linear_all;  // linear_spatial & linear_all can not be used together,
                      // linear_all can not be used with WeightsUploadTypes
                      // that use workgroups(subgroups) for
                      // uploading(LOCAL_MEM_BY_THREADS for example).
    bool different_weights_for_height;
    bool groups_support = false;  // convolution groups
    int src_depth_loop_size;
    WeightsUploadType weights_upload_type;
    bool x_kernel_is_1 = false;
    bool y_kernel_is_1 = false;
    bool z_kernel_is_1 = false;
    WeightsLayout weights_layout;

    // used only with PRIVATE_MEM_SIMD_BROADCAST
    int simd_size = 1;

    bool AreWeightsBuffer() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh mht_1(mht_1_v, 276, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h", "AreWeightsBuffer");

      return weights_upload_type != WeightsUploadType::TEXTURES_MEM_X4;
    }

    bool IsPrivateMemBroadcast() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh mht_2(mht_2_v, 283, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h", "IsPrivateMemBroadcast");

      return weights_upload_type ==
             WeightsUploadType::PRIVATE_MEM_SIMD_BROADCAST;
    }
  };

  ConvPowerVR(const OperationDef& definition,
              const Convolution2DAttributes& attr, const GpuInfo& gpu_info,
              const BHWC* dst_shape = nullptr);
  ConvPowerVR(const OperationDef& definition,
              const Convolution2DAttributes& attr, const BHWC& weights_shape,
              const GpuInfo& gpu_info, const BHWC* dst_shape = nullptr);
  ConvPowerVR(const OperationDef& definition,
              const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
              const BHWC* dst_shape = nullptr);
  explicit ConvPowerVR(const OperationDef& definition);
  ConvPowerVR(const OperationDef& definition,
              const Convolution3DAttributes& attr, const GpuInfo& gpu_info,
              const BHWDC* dst_shape = nullptr);

  void GenerateCode(const GpuInfo& gpu_info);

  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                  const tflite::gpu::Tensor<Linear, T>& biases);
  template <DataType T>
  void UploadDataForWinograd4x4To6x6(
      const tflite::gpu::Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights);

  template <DataType T>
  void UploadWeights(const tflite::gpu::Tensor<OHWDI, T>& weights);

  template <DataType T>
  void UploadBias(const tflite::gpu::Tensor<Linear, T>& bias);

  friend ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const Convolution2DAttributes& attr,
                                       const BHWC* dst_shape);

  friend ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const FullyConnectedAttributes& attr,
                                       const BHWC* dst_shape);

  friend ConvPowerVR CreateConvPowerVRDynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Convolution2DAttributes& attr, const BHWC& weights_shape,
      const BHWC* dst_shape);

  friend ConvPowerVR CreateConvPowerVRWino4x4To6x6(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Convolution2DAttributes& attr, const BHWC* dst_shape);

  friend ConvPowerVR CreateConvPowerVR3D(const GpuInfo& gpu_info,
                                         const OperationDef& definition,
                                         const Convolution3DAttributes& attr,
                                         const BHWDC* dst_shape);

  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition,
                             const Convolution2DAttributes& attr,
                             const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition,
                             const Convolution2DAttributes& attr,
                             const BHWC& weights_shape,
                             const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition,
                             const FullyConnectedAttributes& attr,
                             const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParamsWinograd(const GpuInfo& gpu_info,
                                     const OperationDef& definition,
                                     const Convolution2DAttributes& attr,
                                     const BHWC* dst_shape = nullptr);
  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition,
                             const Convolution3DAttributes& attr,
                             const BHWDC* dst_shape = nullptr);
  ConvParams GuessBestParams(const GpuInfo& gpu_info,
                             const OperationDef& definition, int src_depth,
                             int dst_depth, bool x_kernel_is_1,
                             bool y_kernel_is_1,
                             bool different_weights_for_height,
                             const BHWC* dst_shape = nullptr);

  std::string GenerateConv(const GpuInfo& gpu_info, const OperationDef& op_def,
                           bool stride_correction,
                           const ConvParams& conv_params);

  int4 stride_;
  int4 padding_;
  int4 kernel_size_;
  int4 dilation_;
  ConvParams conv_params_;
};

template <DataType T>
void ConvPowerVR::UploadData(const tflite::gpu::Tensor<OHWI, T>& weights,
                             const tflite::gpu::Tensor<Linear, T>& biases) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh mht_3(mht_3_v, 389, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h", "ConvPowerVR::UploadData");

  UploadWeights(weights);
  UploadBias(biases);
}

template <DataType T>
void ConvPowerVR::UploadDataForWinograd4x4To6x6(
    const tflite::gpu::Tensor<OHWI, T>& weights) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh mht_4(mht_4_v, 399, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h", "ConvPowerVR::UploadDataForWinograd4x4To6x6");

  tflite::gpu::Tensor<OHWI, T> wino_weights;
  RearrangeWeightsToWinograd4x4To6x6Weights(weights, &wino_weights);
  UploadWeights(wino_weights);
  tflite::gpu::Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(weights.shape.o);
  biases.data.resize(weights.shape.o, 0.0f);
  UploadBias(biases);
}

template <DataType T>
void ConvPowerVR::UploadBias(const tflite::gpu::Tensor<Linear, T>& bias) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh mht_5(mht_5_v, 413, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h", "ConvPowerVR::UploadBias");

  BufferDescriptor desc;
  desc.element_type = conv_params_.weights_data_type;
  desc.element_size = 4;
  desc.memory_type = conv_params_.weights_upload_type ==
                             ConvPowerVR::WeightsUploadType::CONSTANT_MEM
                         ? MemoryType::CONSTANT
                         : MemoryType::GLOBAL;
  const int float_size = conv_params_.weights_data_type == DataType::FLOAT32
                             ? sizeof(float)
                             : sizeof(half);
  int aligned_channels = AlignByN(bias.shape.v, 4 * conv_params_.block_size.w);
  desc.size = float_size * aligned_channels;
  desc.data.resize(desc.size);
  if (conv_params_.weights_data_type == DataType::FLOAT32) {
    float* gpu_data = reinterpret_cast<float*>(desc.data.data());
    for (int i = 0; i < aligned_channels; ++i) {
      gpu_data[i] = i < bias.shape.v ? bias.data[i] : 0.0f;
    }
  } else {
    half* gpu_data = reinterpret_cast<half*>(desc.data.data());
    for (int i = 0; i < aligned_channels; ++i) {
      gpu_data[i] = i < bias.shape.v ? bias.data[i] : 0.0f;
    }
  }
  args_.AddObject("biases",
                  absl::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType T>
void ConvPowerVR::UploadWeights(const tflite::gpu::Tensor<OHWI, T>& weights) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh mht_6(mht_6_v, 446, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h", "ConvPowerVR::UploadWeights");

  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (conv_params_.AreWeightsBuffer()) {
    BufferDescriptor desc;
    desc.element_type = weights_desc.type;
    desc.element_size = 4;
    desc.memory_type = conv_params_.weights_upload_type ==
                               ConvPowerVR::WeightsUploadType::CONSTANT_MEM
                           ? MemoryType::CONSTANT
                           : MemoryType::GLOBAL;
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      Texture2DDescriptor desc;
      desc.element_type = weights_desc.type;
      desc.size = int2(tex_size.x, tex_size.y);
      desc.data.resize(sub_size);
      memcpy(desc.data.data(), weights_data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name,
                      absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

template <DataType T>
void ConvPowerVR::UploadWeights(const tflite::gpu::Tensor<OHWDI, T>& weights) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSconv_powervrDTh mht_7(mht_7_v, 486, "", "./tensorflow/lite/delegates/gpu/common/tasks/conv_powervr.h", "ConvPowerVR::UploadWeights");

  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (conv_params_.AreWeightsBuffer()) {
    BufferDescriptor desc;
    desc.element_type = weights_desc.type;
    desc.element_size = 4;
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    absl::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      Texture2DDescriptor desc;
      desc.element_type = weights_desc.type;
      desc.size = int2(tex_size.x, tex_size.y);
      desc.data.resize(sub_size);
      memcpy(desc.data.data(), weights_data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name,
                      absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const Convolution2DAttributes& attr,
                              const BHWC* dst_shape = nullptr);

ConvPowerVR CreateConvPowerVR(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              const FullyConnectedAttributes& attr,
                              const BHWC* dst_shape = nullptr);

ConvPowerVR CreateConvPowerVRDynamicWeights(const GpuInfo& gpu_info,
                                            const OperationDef& definition,
                                            const Convolution2DAttributes& attr,
                                            const BHWC& weights_shape,
                                            const BHWC* dst_shape = nullptr);

ConvPowerVR CreateConvPowerVRWino4x4To6x6(const GpuInfo& gpu_info,
                                          const OperationDef& definition,
                                          const Convolution2DAttributes& attr,
                                          const BHWC* dst_shape = nullptr);

ConvPowerVR CreateConvPowerVR3D(const GpuInfo& gpu_info,
                                const OperationDef& definition,
                                const Convolution3DAttributes& attr,
                                const BHWDC* dst_shape = nullptr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_POWERVR_H_
