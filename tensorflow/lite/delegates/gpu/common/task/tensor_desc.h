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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_DESC_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_DESC_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh() {
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


#include <cstddef>
#include <string>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

enum class AddressMode {
  kDontCare,
  kZero,
};

enum class TensorStorageType {
  UNKNOWN,
  BUFFER,
  IMAGE_BUFFER,
  TEXTURE_2D,
  TEXTURE_3D,
  TEXTURE_ARRAY,
  SINGLE_TEXTURE_2D
};

struct TensorDescriptor : public GPUObjectDescriptor {
  TensorDescriptor() = default;
  TensorDescriptor(DataType dt, TensorStorageType st, Layout l)
      : data_type(dt), storage_type(st), layout(l) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_0(mht_0_v, 219, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "TensorDescriptor");
}

  TensorDescriptor(const TensorDescriptor&) = default;
  TensorDescriptor& operator=(const TensorDescriptor&) = default;
  TensorDescriptor(TensorDescriptor&& desc);
  TensorDescriptor& operator=(TensorDescriptor&& desc);

  bool operator==(const TensorDescriptor& d) const {
    return data_type == d.data_type && storage_type == d.storage_type &&
           layout == d.layout;
  }

  bool operator!=(const TensorDescriptor& d) const { return !(*this == d); }

  absl::Status PerformConstExpr(const GpuInfo& gpu_info,
                                const std::string& const_expr,
                                std::string* result) const override;

  absl::Status PerformSelector(const GpuInfo& gpu_info,
                               const std::string& selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources(const GpuInfo& gpu_info) const override;

  void Release() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_1(mht_1_v, 248, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "Release");
 data.clear(); }
  uint64_t GetSizeInBytes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_2(mht_2_v, 252, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "GetSizeInBytes");
 return data.size(); };
  size_t GetSizeInBytesForShape(const BHWDC& shape5d) const;

  bool HasAxis(Axis axis) const;
  void SetAddressMode(AddressMode mode);
  int GetWidthSize(BHWDC shape) const;
  int GetSliceStrideSize(BHWDC shape) const;

  absl::Status GetLinkingContextFromWriteSelector(
      const std::vector<std::string>& args, std::string* value_name,
      std::string* x_coord, std::string* y_coord, std::string* s_coord) const;

  template <DataType T>
  void UploadData(const tflite::gpu::Tensor<BHWC, T>& src);
  template <DataType T>
  void DownloadData(tflite::gpu::Tensor<BHWC, T>* dst);

  void UploadData(const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src);
  void UploadData(const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src);

  int GetLinearIndex(const BHWDC& shape5d, int b, int x, int y, int d, int s,
                     int sub_c) const;

  bool SupportsZeroClamp(const Axis& axis) const;
  bool CanReadOutOfBorder(const Axis& axis) const;
  bool IsLinear() const;

  // applicable only for types that: IsLinear -> true.
  // In this case for address we have 1d component - addr (int)
  // If for addr == -1 this linear storage type returns FLT4(0.0), this function
  // returns true, otherwise false
  bool ReturnsZeroForNegOneRead() const;

  absl::Status CanCreateTensorWithShape(const GpuInfo& gpu_info,
                                        const BHWDC& shape) const;

  absl::Status CanCreateTensorWithShape(const GpuInfo& gpu_info,
                                        const BHWC& shape) const;

  DataType data_type = DataType::UNKNOWN;
  TensorStorageType storage_type = TensorStorageType::UNKNOWN;
  // This field describes logical layout, actual(physical) GPU layout can be
  // totally different.
  Layout layout =
      Layout::UNKNOWN;  // Supported layouts is HWC, BHWC, HWDC, BHWDC

  void SetBHWCShape(const BHWC& new_shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_3(mht_3_v, 301, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "SetBHWCShape");

    shape = BHWDC(new_shape.b, new_shape.h, new_shape.w, 1, new_shape.c);
  }
  void SetBHWDCShape(const BHWDC& new_shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_4(mht_4_v, 307, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "SetBHWDCShape");
 shape = new_shape; }
  BHWC GetBHWCShape() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_5(mht_5_v, 311, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "GetBHWCShape");
 return BHWC(shape.b, shape.h, shape.w, shape.c); }
  BHWDC GetBHWDCShape() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_6(mht_6_v, 315, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "GetBHWDCShape");
 return shape; }
  void SetData(std::vector<uint8_t>&& new_data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_7(mht_7_v, 319, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "SetData");
 data = new_data; }
  const std::vector<uint8_t>& GetData() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_8(mht_8_v, 323, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "GetData");
 return data; }

  // applicable only for TEXTURE_2D.
  // When Texture 2d created from buffer, we can use it as texture or as buffer.
  // This option allows to use texture 2d as buffer when we use it as dst
  // tensor(write only).
  // Currently supported only for Metal/OpenCL.
  // By default false.
  bool use_buffer_for_write_only_2d_texture = false;

  // applicable only for IMAGE_BUFFER.
  // We can use image buffer as image or as buffer.
  // This option allows to use image buffer as buffer when we use it as dst
  // tensor(write only).
  // Currently supported only for Metal/OpenCL.
  // By default true.
  bool use_buffer_for_write_only_image_buffer = true;

 private:
  absl::Status PerformReadSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;
  absl::Status PerformReadNearestSelector(const GpuInfo& gpu_info,
                                          const std::vector<std::string>& args,
                                          std::string* result) const;
  absl::Status PerformReadBilinearSelector(const GpuInfo& gpu_info,
                                           const std::vector<std::string>& args,
                                           std::string* result) const;
  absl::Status PerformReadPerChannelSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  absl::Status PerformGetAddressSelector(const std::vector<std::string>& args,
                                         std::string* result) const;

  absl::Status PerformGetPtrWithSliceOffsetSelector(
      const std::vector<std::string>& args, std::string* result) const;

  absl::Status PerformGetWHOffsetSelector(const std::vector<std::string>& args,
                                          std::string* result) const;

  absl::Status PerformGetHandleSelector(const std::vector<std::string>& args,
                                        std::string* result) const;

  std::string DeclareAddress(const std::string& var_name,
                             const std::string& address) const;

  std::string StorageTypeToAddressType() const;

  absl::Status PerformWriteSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  absl::Status PerformWriteLinearSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  absl::Status PerformWrite2DSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  std::string Read(const GpuInfo& gpu_info, DataType read_as_type,
                   const std::vector<std::string>& coords) const;
  std::string Write(const GpuInfo& gpu_info, DataType write_type,
                    const std::string& var_name,
                    const std::vector<std::string>& coords) const;

  bool IsBatchedWidth() const;

  AddressMode AddressModeFromState() const;

  absl::Status MaybeGetDataTypeFromTemplateArgs(
      const std::vector<std::string>& template_args, DataType* result) const;

  std::string GetGlobalAddressNoDeclaration(const std::string& xc,
                                            const std::string& yc,
                                            const std::string& zc,
                                            const std::string& sc,
                                            const std::string& bc) const;

  std::vector<std::string> GetPhysicalCoordsWHS(const std::string& x,
                                                const std::string& y,
                                                const std::string& s) const;
  std::vector<std::string> GetPhysicalCoordsWHSB(const std::string& x,
                                                 const std::string& y,
                                                 const std::string& s,
                                                 const std::string& b) const;
  std::vector<std::string> GetPhysicalCoordsWHDS(const std::string& x,
                                                 const std::string& y,
                                                 const std::string& z,
                                                 const std::string& s) const;
  std::vector<std::string> GetPhysicalCoordsWHDSB(const std::string& x,
                                                  const std::string& y,
                                                  const std::string& z,
                                                  const std::string& s,
                                                  const std::string& b) const;
  std::vector<std::string> GetPhysicalCoords(const std::string& xc,
                                             const std::string& yc,
                                             const std::string& zc,
                                             const std::string& sc,
                                             const std::string& bc) const;

  bool ParseCoordsFromArgs(const std::vector<std::string>& args, int offset,
                           std::string* xc, std::string* yc, std::string* zc,
                           std::string* sc, std::string* bc) const;

  template <typename T>
  void UploadData(const T* src);
  template <typename T>
  void DownloadData(T* dst);

  // optional
  BHWDC shape;
  std::vector<uint8_t> data;
};

template <DataType T>
void TensorDescriptor::UploadData(const tflite::gpu::Tensor<BHWC, T>& src) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_9(mht_9_v, 443, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "TensorDescriptor::UploadData");

  shape = BHWDC(src.shape.b, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.data.data());
}

template <DataType T>
void TensorDescriptor::DownloadData(tflite::gpu::Tensor<BHWC, T>* dst) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_10(mht_10_v, 452, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "TensorDescriptor::DownloadData");

  dst->shape = BHWC(shape.b, shape.h, shape.w, shape.c);
  dst->data.resize(dst->shape.DimensionsProduct(), 0.0f);
  DownloadData(dst->data.data());
}

template <typename T>
void TensorDescriptor::UploadData(const T* src) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_11(mht_11_v, 462, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "TensorDescriptor::UploadData");

  data.resize(GetSizeInBytesForShape(shape));
  if (data_type == DataType::FLOAT16) {
    half* gpu_data = reinterpret_cast<half*>(data.data());
    DataFromBHWDC(src, shape, *this, gpu_data);
  } else {
    T* gpu_data = reinterpret_cast<T*>(data.data());
    DataFromBHWDC(src, shape, *this, gpu_data);
  }
}

template <typename T>
void TensorDescriptor::DownloadData(T* dst) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStensor_descDTh mht_12(mht_12_v, 477, "", "./tensorflow/lite/delegates/gpu/common/task/tensor_desc.h", "TensorDescriptor::DownloadData");

  data.resize(GetSizeInBytesForShape(shape));
  if (data_type == DataType::FLOAT16) {
    half* gpu_data = reinterpret_cast<half*>(data.data());
    DataToBHWDC(gpu_data, shape, *this, dst);
  } else {
    T* gpu_data = reinterpret_cast<T*>(data.data());
    DataToBHWDC(gpu_data, shape, *this, dst);
  }
}

template <typename FromType, typename ToType>
void DataFromBHWDC(const FromType* src, const BHWDC& shape,
                   const TensorDescriptor& desc, ToType* dst) {
  const int channels_alignment =
      desc.storage_type == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c : 4;
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              FromType value;
              if (s * 4 + c < shape.c) {
                const int cpu_index =
                    shape.LinearIndex({b, y, x, d, s * 4 + c});
                value = src[cpu_index];
              } else {
                value = 0;
              }
              int gpu_index = desc.GetLinearIndex(shape, b, x, y, d, s, c);
              dst[gpu_index] = value;
            }
          }
        }
      }
    }
  }
}

template <typename FromType, typename ToType>
void DataToBHWDC(const FromType* src, const BHWDC& shape,
                 const TensorDescriptor& desc, ToType* dst) {
  const int channels_alignment =
      desc.storage_type == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c : 4;
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              if (s * 4 + c >= shape.c) {
                continue;
              }
              int cpu_index = shape.LinearIndex({b, y, x, d, s * 4 + c});
              int gpu_index = desc.GetLinearIndex(shape, b, x, y, d, s, c);
              dst[cpu_index] = src[gpu_index];
            }
          }
        }
      }
    }
  }
}

std::string ToString(TensorStorageType type);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TENSOR_DESC_H_
