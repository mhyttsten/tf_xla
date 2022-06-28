/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the Licensgoe is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_SPATIAL_TENSOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_SPATIAL_TENSOR_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh() {
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


#import <Metal/Metal.h>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_tensor.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"
#include "tensorflow/lite/delegates/gpu/metal/gpu_object.h"

namespace tflite {
namespace gpu {
namespace metal {

class MetalSpatialTensor : public GPUObject, public GpuSpatialTensor {
 public:
  MetalSpatialTensor()
      : memory_(nullptr),
        texture_mem_(nullptr),
        memory_owner_(true),
        texture_mem_owner_(true) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "MetalSpatialTensor");
}
  MetalSpatialTensor(id<MTLBuffer> buffer, id<MTLTexture> texture,
                     bool memory_owner, bool texture_mem_owner,
                     const BHWC& shape, const TensorDescriptor& descriptor);
  MetalSpatialTensor(id<MTLBuffer> buffer, id<MTLTexture> texture,
                     bool memory_owner, bool texture_mem_owner,
                     const BHWDC& shape, const TensorDescriptor& descriptor);

  // Move only
  MetalSpatialTensor(MetalSpatialTensor&& tensor);
  MetalSpatialTensor& operator=(MetalSpatialTensor&& tensor);
  MetalSpatialTensor(const MetalSpatialTensor&) = delete;
  MetalSpatialTensor& operator=(const MetalSpatialTensor&) = delete;

  ~MetalSpatialTensor() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_1(mht_1_v, 224, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "~MetalSpatialTensor");
 Release(); }

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  int Width() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_2(mht_2_v, 232, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "Width");
 return shape_.w; }
  int Height() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_3(mht_3_v, 236, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "Height");
 return shape_.h; }
  int Depth() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_4(mht_4_v, 240, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "Depth");
 return shape_.d; }
  int Channels() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_5(mht_5_v, 244, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "Channels");
 return shape_.c; }
  int Slices() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_6(mht_6_v, 248, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "Slices");
 return DivideRoundUp(shape_.c, 4); }
  int Batch() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_7(mht_7_v, 252, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "Batch");
 return shape_.b; }

  TensorDescriptor GetDescriptor() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_8(mht_8_v, 257, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "GetDescriptor");
 return descriptor_; }
  DataType GetDataType() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_9(mht_9_v, 261, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "GetDataType");
 return descriptor_.data_type; }
  TensorStorageType GetStorageType() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_10(mht_10_v, 265, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "GetStorageType");
 return descriptor_.storage_type; }

  uint64_t GetMemorySizeInBytes() const;

  absl::Status WriteData(
      id<MTLDevice> device,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src);
  absl::Status WriteData(
      id<MTLDevice> device,
      const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src);
  template <DataType T>
  absl::Status WriteData(id<MTLDevice> device, const tflite::gpu::Tensor<BHWC, T>& src);
  template <DataType T>
  absl::Status WriteData(id<MTLDevice> device, const tflite::gpu::Tensor<BHWDC, T>& src);
  template <DataType T>
  absl::Status ReadData(id<MTLDevice> device, tflite::gpu::Tensor<BHWC, T>* dst) const;
  template <DataType T>
  absl::Status ReadData(id<MTLDevice> device, tflite::gpu::Tensor<BHWDC, T>* dst) const;

  absl::Status CreateFromDescriptor(const TensorDescriptor& desc,
                                    id<MTLDevice> device);
  absl::Status ToDescriptor(TensorDescriptor* desc, id<MTLDevice> device) const;

  absl::Status SetBufferHandle(id<MTLBuffer> buffer);
  id<MTLBuffer> GetBufferHandle() const;

 private:
  friend absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                               const TensorDescriptor& descriptor,
                                               MetalSpatialTensor* result, uint64_t buffer_offset);

  friend absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      int row_bytes_alignment,
                                                      MetalSpatialTensor* result,
                                                      uint64_t buffer_offset);

  absl::Status IsValid(const BHWC& shape) const;
  absl::Status IsValid(const BHWDC& shape) const;

  template <typename T>
  absl::Status WriteDataBHWDC(id<MTLDevice> device, const T* in);
  absl::Status WriteData(id<MTLDevice> device, const void* ptr);
  template <typename T>
  absl::Status ReadDataBHWDC(id<MTLDevice> device, T* out) const;
  absl::Status ReadData(id<MTLDevice> device, void* ptr) const;

  int GetAlignedChannels() const;
  int3 GetFullTensorRegion() const;
  void Release();

  id<MTLBuffer> memory_;
  id<MTLTexture> texture_mem_;
  bool memory_owner_;
  bool texture_mem_owner_;
  BHWDC shape_;
  TensorDescriptor descriptor_;
  // for use with TEXTURE_2D and when texture created from buffer.
  int aligned_texture_width_;
  // used when created from shared buffer
  uint64_t buffer_offset_ = 0;
};

absl::Status CreateTensor(id<MTLDevice> device, const BHWC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result);

absl::Status CreateTensor(id<MTLDevice> device, const BHWDC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result);

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result, uint64_t buffer_offset = 0);

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result, uint64_t buffer_offset = 0);

absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer, const BHWC& shape,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment, MetalSpatialTensor* result,
                                             uint64_t buffer_offset = 0);

absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment, MetalSpatialTensor* result,
                                             uint64_t buffer_offset = 0);

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info);

template <DataType T>
absl::Status MetalSpatialTensor::WriteData(id<MTLDevice> device,
                                           const tflite::gpu::Tensor<BHWC, T>& src) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_11(mht_11_v, 361, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "MetalSpatialTensor::WriteData");

  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(device, src.data.data());
}

template <DataType T>
absl::Status MetalSpatialTensor::WriteData(id<MTLDevice> device,
                                           const tflite::gpu::Tensor<BHWDC, T>& src) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_12(mht_12_v, 371, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "MetalSpatialTensor::WriteData");

  RETURN_IF_ERROR(IsValid(src.shape));
  return WriteDataBHWDC(device, src.data.data());
}

template <DataType T>
absl::Status MetalSpatialTensor::ReadData(id<MTLDevice> device,
                                          tflite::gpu::Tensor<BHWC, T>* dst) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_13(mht_13_v, 381, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "MetalSpatialTensor::ReadData");

  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(device, dst->data.data());
}

template <DataType T>
absl::Status MetalSpatialTensor::ReadData(id<MTLDevice> device,
                                          tflite::gpu::Tensor<BHWDC, T>* dst) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_14(mht_14_v, 391, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "MetalSpatialTensor::ReadData");

  RETURN_IF_ERROR(IsValid(dst->shape));
  return ReadDataBHWDC(device, dst->data.data());
}

template <typename T>
absl::Status MetalSpatialTensor::WriteDataBHWDC(id<MTLDevice> device, const T* in) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_15(mht_15_v, 400, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "MetalSpatialTensor::WriteDataBHWDC");

  std::unique_ptr<uint8_t[]> data_copy;
  data_copy.reset(new uint8_t[GetMemorySizeInBytes()]);
  if (descriptor_.data_type == DataType::FLOAT16) {
    // rearrangement and conversion from float32 to float16
    DataFromBHWDC(reinterpret_cast<const float*>(in), shape_, descriptor_,
                  reinterpret_cast<half*>(data_copy.get()));
  } else {
    // rearrangement
    DataFromBHWDC(in, shape_, descriptor_, reinterpret_cast<T*>(data_copy.get()));
  }

  return WriteData(device, data_copy.get());
}

template <typename T>
absl::Status MetalSpatialTensor::ReadDataBHWDC(id<MTLDevice> device, T* out) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTh mht_16(mht_16_v, 419, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h", "MetalSpatialTensor::ReadDataBHWDC");

  std::unique_ptr<uint8_t[]> data_copy;
  data_copy.reset(new uint8_t[GetMemorySizeInBytes()]);

  RETURN_IF_ERROR(ReadData(device, data_copy.get()));

  if (descriptor_.data_type == DataType::FLOAT16) {
    // rearrangement and conversion from float32 to float16
    DataToBHWDC(reinterpret_cast<half*>(data_copy.get()), shape_, descriptor_,
                reinterpret_cast<float*>(out));
  } else {
    // rearrangement
    DataToBHWDC(reinterpret_cast<T*>(data_copy.get()), shape_, descriptor_, out);
  }

  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_SPATIAL_TENSOR_H_
