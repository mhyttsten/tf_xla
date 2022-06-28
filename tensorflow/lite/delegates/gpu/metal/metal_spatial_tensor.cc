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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc() {
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

#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

#include <memory>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

absl::Status CreateTextureBuffer(id<MTLBuffer> buffer, uint64_t buffer_offset,
                                 const BHWDC& shape,
                                 const TensorDescriptor& descriptor,
                                 id<MTLTexture>* texture) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "CreateTextureBuffer");

  if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    const int slices = DivideRoundUp(shape.c, 4);
    const size_t flt4_count = shape.b * shape.w * shape.h * shape.d * slices;
    const size_t data_size = flt4_count * 4 * SizeOf(descriptor.data_type);
    MTLTextureDescriptor* texture_desc = [[MTLTextureDescriptor alloc] init];
    texture_desc.width = flt4_count;
    texture_desc.pixelFormat =
        DataTypeToRGBAPixelFormat(descriptor.data_type, false);
    texture_desc.textureType = MTLTextureTypeTextureBuffer;
    texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    texture_desc.storageMode = buffer.storageMode;
    *texture = [buffer newTextureWithDescriptor:texture_desc
                                         offset:buffer_offset
                                    bytesPerRow:data_size];
    if (!*texture) {
      return absl::UnknownError("Failed to allocate id<MTLTexture>");
    }
  } else {
    return absl::UnknownError(
        "TensorStorageType::IMAGE_BUFFER available only in iOS 12/tvOS "
        "12/macOS 10.14 and higher.");
  }
  return absl::OkStatus();
}

absl::Status AllocateTensorMemory(id<MTLDevice> device, const BHWDC& shape,
                                  const TensorDescriptor& descriptor,
                                  const void* data_ptr, id<MTLBuffer>* buffer,
                                  id<MTLTexture>* texture) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "AllocateTensorMemory");

  const int slices = DivideRoundUp(shape.c, 4);
  switch (descriptor.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      const size_t data_size = shape.b * shape.w * shape.h * shape.d * slices *
                               4 * SizeOf(descriptor.data_type);
      if (data_ptr) {
        *buffer = [device newBufferWithBytes:data_ptr
                                      length:data_size
                                     options:MTLResourceStorageModeShared];
      } else {
        *buffer = [device newBufferWithLength:data_size
                                      options:MTLResourceStorageModeShared];
      }
      if (!*buffer) {
        return absl::UnknownError("Failed to allocate id<MTLBuffer>");
      }
      if (descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
        RETURN_IF_ERROR(
            CreateTextureBuffer(*buffer, 0, shape, descriptor, texture));
      }
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_2D: {
      MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:DataTypeToRGBAPixelFormat(
                                                 descriptor.data_type, false)
                                       width:shape.w * shape.b * shape.d
                                      height:shape.h * slices
                                   mipmapped:NO];
      texture_desc.textureType = MTLTextureType2D;
      texture_desc.usage =
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      texture_desc.storageMode = MTLStorageModePrivate;

      *texture = [device newTextureWithDescriptor:texture_desc];
      if (!*texture) {
        return absl::UnknownError("Failed to allocate id<MTLTexture>");
      }
      if (data_ptr) {
        WriteDataToTexture2D(*texture, device, data_ptr);
      }
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_3D: {
      MTLTextureDescriptor* texture_desc = [[MTLTextureDescriptor alloc] init];
      texture_desc.width = shape.w * shape.b;
      texture_desc.height = shape.h;
      texture_desc.depth = slices * shape.d;
      texture_desc.pixelFormat =
          DataTypeToRGBAPixelFormat(descriptor.data_type, false);
      texture_desc.textureType = MTLTextureType3D;
      texture_desc.usage =
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      texture_desc.storageMode = MTLStorageModePrivate;

      *texture = [device newTextureWithDescriptor:texture_desc];
      if (!*texture) {
        return absl::UnknownError("Failed to allocate id<MTLTexture>");
      }
      if (data_ptr) {
        WriteDataToTexture3D(*texture, device, data_ptr);
      }
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      MTLTextureDescriptor* texture_desc = [[MTLTextureDescriptor alloc] init];
      texture_desc.width = shape.w * shape.b;
      texture_desc.height = shape.h;
      texture_desc.arrayLength = slices * shape.d;
      texture_desc.pixelFormat =
          DataTypeToRGBAPixelFormat(descriptor.data_type, false);
      texture_desc.textureType = MTLTextureType2DArray;
      texture_desc.usage =
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      texture_desc.storageMode = MTLStorageModePrivate;

      *texture = [device newTextureWithDescriptor:texture_desc];
      if (!*texture) {
        return absl::UnknownError("Failed to allocate id<MTLTexture>");
      }
      if (data_ptr) {
        WriteDataToTexture2DArray(*texture, device, data_ptr);
      }
      return absl::OkStatus();
    }
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
}

absl::Status CreateTensor(id<MTLDevice> device, const BHWDC& shape,
                          const TensorDescriptor& descriptor,
                          id<MTLBuffer> buffer, id<MTLTexture> texture,
                          MetalSpatialTensor* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_2(mht_2_v, 332, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "CreateTensor");

  const bool user_provided = buffer != nullptr || texture != nullptr;
  const bool memory_owner = !user_provided;
  if (memory_owner) {
    RETURN_IF_ERROR(AllocateTensorMemory(device, shape, descriptor, nullptr,
                                         &buffer, &texture));
  }

  *result = MetalSpatialTensor(buffer, texture, memory_owner, memory_owner,
                               shape, descriptor);
  return absl::OkStatus();
}
}  // namespace

MetalSpatialTensor::MetalSpatialTensor(id<MTLBuffer> buffer,
                                       id<MTLTexture> texture,
                                       bool memory_owner,
                                       bool texture_mem_owner,
                                       const BHWC& shape,
                                       const TensorDescriptor& descriptor)
    : memory_(buffer),
      texture_mem_(texture),
      memory_owner_(memory_owner),
      texture_mem_owner_(texture_mem_owner),
      shape_(shape.b, shape.h, shape.w, 1, shape.c),
      descriptor_(descriptor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_3(mht_3_v, 360, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::MetalSpatialTensor");
}

MetalSpatialTensor::MetalSpatialTensor(id<MTLBuffer> buffer,
                                       id<MTLTexture> texture,
                                       bool memory_owner,
                                       bool texture_mem_owner,
                                       const BHWDC& shape,
                                       const TensorDescriptor& descriptor)
    : memory_(buffer),
      texture_mem_(texture),
      memory_owner_(memory_owner),
      texture_mem_owner_(texture_mem_owner),
      shape_(shape),
      descriptor_(descriptor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_4(mht_4_v, 376, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::MetalSpatialTensor");
}

MetalSpatialTensor::MetalSpatialTensor(MetalSpatialTensor&& tensor)
    : memory_(tensor.memory_),
      texture_mem_(tensor.texture_mem_),
      memory_owner_(tensor.memory_owner_),
      texture_mem_owner_(tensor.texture_mem_owner_),
      shape_(tensor.shape_),
      descriptor_(tensor.descriptor_),
      aligned_texture_width_(tensor.aligned_texture_width_),
      buffer_offset_(tensor.buffer_offset_) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_5(mht_5_v, 389, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::MetalSpatialTensor");

  tensor.memory_ = nullptr;
}

MetalSpatialTensor& MetalSpatialTensor::operator=(MetalSpatialTensor&& tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_6(mht_6_v, 396, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "=");

  if (this != &tensor) {
    Release();
    std::swap(memory_, tensor.memory_);
    std::swap(texture_mem_, tensor.texture_mem_);
    std::swap(memory_owner_, tensor.memory_owner_);
    std::swap(texture_mem_owner_, tensor.texture_mem_owner_);
    std::swap(shape_, tensor.shape_);
    std::swap(descriptor_, tensor.descriptor_);
    std::swap(aligned_texture_width_, tensor.aligned_texture_width_);
    std::swap(buffer_offset_, tensor.buffer_offset_);
  }
  return *this;
}

void MetalSpatialTensor::Release() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_7(mht_7_v, 414, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::Release");

  if (memory_owner_ && memory_) {
    memory_ = nullptr;
  }
  if (texture_mem_owner_ && texture_mem_) {
    texture_mem_ = nullptr;
  }
}

absl::Status MetalSpatialTensor::GetGPUResources(
    const GPUObjectDescriptor* obj_ptr,
    GPUResourcesWithValue* resources) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_8(mht_8_v, 428, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::GetGPUResources");

  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(obj_ptr);
  if (buffer_desc) {
    if (descriptor_.storage_type != TensorStorageType::BUFFER) {
      return absl::InvalidArgumentError(
          "Tensor can be used with BufferDescriptor only wtih "
          "TensorStorageType::BUFFER.");
    }
    resources->buffers.push_back({"buffer", {memory_, buffer_offset_}});
    return absl::OkStatus();
  }
  const auto* texture2d_desc =
      dynamic_cast<const Texture2DDescriptor*>(obj_ptr);
  if (texture2d_desc) {
    if (descriptor_.storage_type != TensorStorageType::TEXTURE_2D) {
      return absl::InvalidArgumentError(
          "Tensor can be used with Texture2DDescriptor only wtih "
          "TensorStorageType::TEXTURE_2D.");
    }
    resources->images2d.push_back({"tex2d", texture_mem_});
    return absl::OkStatus();
  }
  const auto* tensor_desc = dynamic_cast<const TensorDescriptor*>(obj_ptr);
  if (!tensor_desc) {
    return absl::InvalidArgumentError("Expected TensorDescriptor on input.");
  }
  resources->ints.push_back(
      {"slice_stride", tensor_desc->GetSliceStrideSize(shape_)});
  if (descriptor_.HasAxis(Axis::WIDTH)) {
    resources->ints.push_back({"width", tensor_desc->GetWidthSize(shape_)});
  }
  if (descriptor_.HasAxis(Axis::HEIGHT)) {
    resources->ints.push_back({"height", Height()});
  }
  if (descriptor_.HasAxis(Axis::CHANNELS)) {
    resources->ints.push_back({"slices", Slices()});
    resources->ints.push_back({"channels", Channels()});
  }
  if (descriptor_.HasAxis(Axis::BATCH)) {
    resources->ints.push_back({"batch", Batch()});
  }
  if (descriptor_.HasAxis(Axis::DEPTH)) {
    resources->ints.push_back({"depth", Depth()});
  }

  if (descriptor_.storage_type == TensorStorageType::BUFFER) {
    resources->buffers.push_back({"buffer", {memory_, buffer_offset_}});
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_2D) {
    if (obj_ptr->GetAccess() == AccessType::WRITE &&
        tensor_desc->use_buffer_for_write_only_2d_texture) {
      resources->ints.push_back(
          {"aligned_texture_width", aligned_texture_width_});
      resources->buffers.push_back({"buffer", {memory_, buffer_offset_}});
    } else {
      resources->images2d.push_back({"image2d", texture_mem_});
    }
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_3D) {
    resources->images3d.push_back({"image3d", texture_mem_});
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_ARRAY) {
    resources->image2d_arrays.push_back({"image2d_array", texture_mem_});
  } else if (descriptor_.storage_type == TensorStorageType::IMAGE_BUFFER) {
    if (obj_ptr->GetAccess() == AccessType::WRITE &&
        tensor_desc->use_buffer_for_write_only_image_buffer) {
      resources->buffers.push_back({"buffer", {memory_, buffer_offset_}});
    } else {
      resources->image_buffers.push_back({"image_buffer", texture_mem_});
    }
  }

  return absl::OkStatus();
}

int3 MetalSpatialTensor::GetFullTensorRegion() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_9(mht_9_v, 503, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::GetFullTensorRegion");

  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::IMAGE_BUFFER:
      return {shape_.w * shape_.b, shape_.h, shape_.d * Slices()};
    case TensorStorageType::TEXTURE_2D:
      return {shape_.w * shape_.b * shape_.d, shape_.h * Slices(), 1};
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {shape_.w * shape_.b * shape_.d, shape_.h, 1};
    case TensorStorageType::UNKNOWN:
      return {-1, -1, -1};
  }
}

absl::Status MetalSpatialTensor::IsValid(const BHWC& shape) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_10(mht_10_v, 522, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::IsValid");

  if (shape.b != shape_.b) {
    return absl::InvalidArgumentError(
        "Shape batch does not match tensor batch");
  }
  if (shape.w != shape_.w) {
    return absl::InvalidArgumentError(
        "Shape width does not match tensor width");
  }
  if (shape.h != shape_.h) {
    return absl::InvalidArgumentError(
        "Shape height does not match tensor height");
  }
  if (shape.c != shape_.c) {
    return absl::InvalidArgumentError(
        "Shape channels does not match tensor channels");
  }
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::IsValid(const BHWDC& shape) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_11(mht_11_v, 545, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::IsValid");

  if (shape.b != shape_.b) {
    return absl::InvalidArgumentError(
        "Shape batch does not match tensor batch");
  }
  if (shape.w != shape_.w) {
    return absl::InvalidArgumentError(
        "Shape width does not match tensor width");
  }
  if (shape.h != shape_.h) {
    return absl::InvalidArgumentError(
        "Shape height does not match tensor height");
  }
  if (shape.d != shape_.d) {
    return absl::InvalidArgumentError(
        "Shape depth does not match tensor depth");
  }
  if (shape.c != shape_.c) {
    return absl::InvalidArgumentError(
        "Shape channels does not match tensor channels");
  }
  return absl::OkStatus();
}

uint64_t MetalSpatialTensor::GetMemorySizeInBytes() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_12(mht_12_v, 572, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::GetMemorySizeInBytes");

  const int flt_size = SizeOf(descriptor_.data_type);
  const int flt4_size = 4 * flt_size;
  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
      return flt4_size * shape_.b * shape_.w * shape_.h * shape_.d * Slices();
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return flt_size * shape_.w * shape_.h * shape_.c * shape_.b * shape_.d;
    default:
      return 0;
  }
}

int MetalSpatialTensor::GetAlignedChannels() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_13(mht_13_v, 592, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::GetAlignedChannels");

  return descriptor_.storage_type == TensorStorageType::SINGLE_TEXTURE_2D
             ? shape_.c
             : AlignByN(shape_.c, 4);
}

absl::Status MetalSpatialTensor::WriteData(
    id<MTLDevice> device,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_14(mht_14_v, 603, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::WriteData");

  return WriteDataBHWDC(device, src.data.data());
}

absl::Status MetalSpatialTensor::WriteData(
    id<MTLDevice> device,
    const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_15(mht_15_v, 612, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::WriteData");

  return WriteDataBHWDC(device, src.data.data());
}

absl::Status MetalSpatialTensor::CreateFromDescriptor(
    const TensorDescriptor& desc, id<MTLDevice> device) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_16(mht_16_v, 620, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::CreateFromDescriptor");

  shape_ = desc.GetBHWDCShape();
  descriptor_.data_type = desc.data_type;
  descriptor_.storage_type = desc.storage_type;
  descriptor_.layout = desc.layout;
  memory_owner_ = true;
  const uint8_t* data_ptr =
      desc.GetData().empty() ? nullptr : desc.GetData().data();
  id<MTLBuffer> buffer;
  id<MTLTexture> texture;
  RETURN_IF_ERROR(AllocateTensorMemory(device, shape_, descriptor_, data_ptr,
                                       &buffer, &texture));
  memory_ = buffer;
  texture_mem_ = texture;
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::ToDescriptor(TensorDescriptor* desc,
                                              id<MTLDevice> device) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_17(mht_17_v, 641, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::ToDescriptor");

  *desc = descriptor_;
  desc->SetBHWDCShape(shape_);
  std::vector<uint8_t> data(GetMemorySizeInBytes());
  RETURN_IF_ERROR(ReadData(device, data.data()));
  desc->SetData(std::move(data));
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::WriteData(id<MTLDevice> device,
                                           const void* ptr) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_18(mht_18_v, 654, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::WriteData");

  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      std::memcpy(
          reinterpret_cast<uint8_t*>([memory_ contents]) + buffer_offset_, ptr,
          GetMemorySizeInBytes());
      break;
    case TensorStorageType::TEXTURE_2D:
      WriteDataToTexture2D(texture_mem_, device, ptr);
      break;
    case TensorStorageType::TEXTURE_3D:
      WriteDataToTexture3D(texture_mem_, device, ptr);
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      WriteDataToTexture2DArray(texture_mem_, device, ptr);
      break;
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::ReadData(id<MTLDevice> device,
                                          void* ptr) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_19(mht_19_v, 682, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::ReadData");

  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      std::memcpy(
          ptr, reinterpret_cast<uint8_t*>([memory_ contents]) + buffer_offset_,
          GetMemorySizeInBytes());
      break;
    case TensorStorageType::TEXTURE_2D:
      ReadDataFromTexture2D(texture_mem_, device, ptr);
      break;
    case TensorStorageType::TEXTURE_3D:
      ReadDataFromTexture3D(texture_mem_, device, ptr);
      break;
    case TensorStorageType::TEXTURE_ARRAY:
      ReadDataFromTexture2DArray(texture_mem_, device, ptr);
      break;
    case TensorStorageType::SINGLE_TEXTURE_2D:
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status MetalSpatialTensor::SetBufferHandle(id<MTLBuffer> buffer) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_20(mht_20_v, 709, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::SetBufferHandle");

  if (memory_owner_) {
    return absl::InvalidArgumentError(
        "SetBufferHandle can be used only with shared "
        "Tensors(CreateSharedBufferTensor).");
  }
  if (memory_ == buffer) {
    return absl::OkStatus();
  }
  memory_ = buffer;
  if (descriptor_.storage_type == TensorStorageType::IMAGE_BUFFER) {
    id<MTLTexture> texture_buffer = nullptr;
    RETURN_IF_ERROR(
        CreateTextureBuffer(memory_, 0, shape_, descriptor_, &texture_buffer));
    texture_mem_ = texture_buffer;
  }
  return absl::OkStatus();
}

id<MTLBuffer> MetalSpatialTensor::GetBufferHandle() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_21(mht_21_v, 731, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "MetalSpatialTensor::GetBufferHandle");
 return memory_; }

absl::Status CreateTensor(id<MTLDevice> device, const BHWC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_22(mht_22_v, 738, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "CreateTensor");

  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CreateTensor(device, shape5D, descriptor, nullptr, nullptr, result);
}

absl::Status CreateTensor(id<MTLDevice> device, const BHWDC& shape,
                          const TensorDescriptor& descriptor,
                          MetalSpatialTensor* result) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_23(mht_23_v, 748, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "CreateTensor");

  return CreateTensor(device, shape, descriptor, nullptr, nullptr, result);
}

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result,
                                      uint64_t buffer_offset) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_24(mht_24_v, 758, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "CreateSharedBufferTensor");

  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CreateSharedBufferTensor(buffer, shape5D, descriptor, result,
                                  buffer_offset);
}

absl::Status CreateSharedBufferTensor(id<MTLBuffer> buffer, const BHWDC& shape,
                                      const TensorDescriptor& descriptor,
                                      MetalSpatialTensor* result,
                                      uint64_t buffer_offset) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_25(mht_25_v, 770, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "CreateSharedBufferTensor");

  id<MTLTexture> texture_buffer = nullptr;
  if (buffer && descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
    RETURN_IF_ERROR(CreateTextureBuffer(buffer, buffer_offset, shape,
                                        descriptor, &texture_buffer));
  }
  *result = MetalSpatialTensor(buffer, texture_buffer, false, true, shape,
                               descriptor);
  result->buffer_offset_ = buffer_offset;
  return absl::OkStatus();
}

absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer,
                                             const BHWC& shape,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment,
                                             MetalSpatialTensor* result,
                                             uint64_t buffer_offset) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_26(mht_26_v, 790, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "CreateSharedImage2DBufferTensor");

  const BHWDC shape5D = BHWDC(shape.b, shape.h, shape.w, 1, shape.c);
  return CreateSharedImage2DBufferTensor(
      buffer, shape5D, descriptor, row_bytes_alignment, result, buffer_offset);
}

absl::Status CreateSharedImage2DBufferTensor(id<MTLBuffer> buffer,
                                             const BHWDC& shape,
                                             const TensorDescriptor& descriptor,
                                             int row_bytes_alignment,
                                             MetalSpatialTensor* result,
                                             uint64_t buffer_offset) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_27(mht_27_v, 804, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "CreateSharedImage2DBufferTensor");

  const int width = shape.b * shape.w * shape.d;
  const int height = shape.h * DivideRoundUp(shape.c, 4);
  const int channels =
      descriptor.storage_type == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c
                                                                      : 4;
  MTLTextureDescriptor* texture_desc = [[MTLTextureDescriptor alloc] init];
  texture_desc.width = width;
  texture_desc.height = height;
  texture_desc.depth = 1;
  texture_desc.textureType = MTLTextureType2D;
  texture_desc.arrayLength = 1;
  texture_desc.mipmapLevelCount = 1;
  texture_desc.sampleCount = 1;
  texture_desc.pixelFormat =
      DataTypeToRGBAPixelFormat(descriptor.data_type, false);
  texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  texture_desc.storageMode = buffer.storageMode;
  const size_t pixel_size = channels * SizeOf(descriptor.data_type);
  const size_t bytes_per_row = width * pixel_size;
  const size_t bytes_per_row_aligned =
      AlignByN(bytes_per_row, row_bytes_alignment);
  id<MTLTexture> texture_buffer =
      [buffer newTextureWithDescriptor:texture_desc
                                offset:buffer_offset
                           bytesPerRow:bytes_per_row_aligned];
  if (!texture_buffer) {
    return absl::UnknownError("Failed to allocate id<MTLTexture>.");
  }
  if (bytes_per_row_aligned % pixel_size != 0) {
    return absl::UnknownError("Alignment mismatch.");
  }
  *result = MetalSpatialTensor(buffer, texture_buffer, false, true, shape,
                               descriptor);
  result->aligned_texture_width_ = bytes_per_row_aligned / pixel_size;
  result->buffer_offset_ = buffer_offset;
  return absl::OkStatus();
}

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_spatial_tensorDTcc mht_28(mht_28_v, 846, "", "./tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.cc", "GetFastestStorageType");

  const bool a7_or_a8 =
      gpu_info.IsApple() && (gpu_info.apple_info.IsA7GenerationGpu() ||
                             gpu_info.apple_info.IsA8GenerationGpu());
  if (a7_or_a8) {
    return TensorStorageType::TEXTURE_2D;
  } else {
    return TensorStorageType::BUFFER;
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
