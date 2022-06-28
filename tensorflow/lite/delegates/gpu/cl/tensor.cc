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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/tensor.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_image_format.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
absl::Status AllocateTensorMemory(const CLContext& context, const BHWDC& shape,
                                  const TensorDescriptor& descriptor,
                                  const void* data_ptr, CLMemory* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "AllocateTensorMemory");

  const int slices = DivideRoundUp(shape.c, 4);
  cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
  if (data_ptr) {
    mem_flags |= CL_MEM_COPY_HOST_PTR;
  }
  switch (descriptor.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      const size_t data_size = shape.b * shape.w * shape.h * shape.d * slices *
                               4 * SizeOf(descriptor.data_type);
      cl_int error_code;
      cl_mem memory = clCreateBuffer(context.context(), mem_flags, data_size,
                                     const_cast<void*>(data_ptr), &error_code);
      if (!memory) {
        return absl::UnknownError(
            absl::StrCat("Failed to allocate device memory (clCreateBuffer): ",
                         CLErrorCodeToString(error_code)));
      }
      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_2D: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = shape.w * shape.b * shape.d;
      desc.image_height = shape.h * slices;
      desc.image_depth = 0;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type =
          DataTypeToChannelType(descriptor.data_type);

      cl_int error_code;
      cl_mem memory =
          CreateImage2DLegacy(context.context(), mem_flags, &format, &desc,
                              const_cast<void*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(
            absl::StrCat("Failed to create 2D texture (clCreateImage): ",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_3D: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE3D;
      desc.image_width = shape.w * shape.b;
      desc.image_height = shape.h;
      desc.image_depth = slices * shape.d;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type =
          DataTypeToChannelType(descriptor.data_type);

      cl_int error_code;
      cl_mem memory =
          CreateImage3DLegacy(context.context(), mem_flags, &format, &desc,
                              const_cast<void*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(
            absl::StrCat("Failed to create 3D texture (clCreateImage): ",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
      desc.image_width = shape.w * shape.b;
      desc.image_height = shape.h;
      desc.image_depth = 0;
      desc.image_array_size = slices * shape.d;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type =
          DataTypeToChannelType(descriptor.data_type);

      cl_int error_code;
      cl_mem memory =
          clCreateImage(context.context(), mem_flags, &format, &desc,
                        const_cast<void*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(
            absl::StrCat("Failed to create 2D texture array (clCreateImage): ",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }

    case TensorStorageType::SINGLE_TEXTURE_2D: {
      if (slices != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "SINGLE_TEXTURE_2D support only channels in range [1-4], but ",
            shape.c, "was provided"));
      }
      cl_image_desc desc;
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = shape.w * shape.b * shape.d;
      desc.image_height = shape.h;
      desc.image_depth = 0;
      desc.image_row_pitch = 0;
      desc.image_slice_pitch = 0;
      desc.num_mip_levels = 0;
      desc.num_samples = 0;
      desc.buffer = nullptr;

      cl_image_format format;
      if (context.IsFloatTexture2DSupported(shape.c, descriptor.data_type)) {
        format.image_channel_order = ToChannelOrder(shape.c);
        format.image_channel_data_type =
            DataTypeToChannelType(descriptor.data_type);
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "This device doesn't support ", shape.c, "-channel textures."));
      }

      cl_int error_code;
      cl_mem memory =
          CreateImage2DLegacy(context.context(), mem_flags, &format, &desc,
                              const_cast<void*>(data_ptr), &error_code);
      if (error_code != CL_SUCCESS) {
        return absl::UnknownError(
            absl::StrCat("Failed to create single 2D texture (clCreateImage): ",
                         CLErrorCodeToString(error_code)));
      }

      *result = CLMemory(memory, true);
      return absl::OkStatus();
    }

    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
}

absl::Status CreateImageBufferFromBuffer(const CLContext& context,
                                         cl_mem memory, DataType data_type,
                                         int width, cl_mem* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_1(mht_1_v, 371, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateImageBufferFromBuffer");

  cl_image_format format;
  cl_image_desc desc;
  std::memset(&desc, 0, sizeof(desc));
  desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  desc.image_width = width;
  desc.mem_object = memory;

  format.image_channel_data_type = DataTypeToChannelType(data_type);
  format.image_channel_order = CL_RGBA;

  cl_int error_code;
  *result = clCreateImage(context.context(), CL_MEM_READ_WRITE, &format, &desc,
                          nullptr, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create Image from Buffer (clCreateImage): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateImage2DFromBuffer(const CLContext& context, cl_mem memory,
                                     DataType data_type, int width, int height,
                                     int channels, int width_pixel_alignment,
                                     cl_mem* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_2(mht_2_v, 399, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateImage2DFromBuffer");

  if (!context.IsFloatTexture2DSupported(channels, data_type)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "This device doesn't support ", channels, "-channel textures."));
  }

  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = width;
  desc.image_height = height;
  desc.image_depth = 0;
  const size_t width_aligned = AlignByN(width, width_pixel_alignment);
  desc.image_row_pitch = width_aligned * channels * SizeOf(data_type);
  desc.image_slice_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;
  desc.mem_object = memory;

  cl_image_format format;
  format.image_channel_order = ToChannelOrder(channels);
  format.image_channel_data_type = DataTypeToChannelType(data_type);

  cl_int error_code;
  *result = CreateImage2DLegacy(context.context(), CL_MEM_READ_WRITE, &format,
                                &desc, nullptr, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create Image2D from Buffer (clCreateImage): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateTensor(const CLContext& context, const BHWDC& shape,
                          const TensorDescriptor& descriptor, cl_mem memory,
                          Tensor* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_3(mht_3_v, 437, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateTensor");

  const bool memory_owner = memory == nullptr;
  if (memory_owner) {
    CLMemory mem;
    RETURN_IF_ERROR(
        AllocateTensorMemory(context, shape, descriptor, nullptr, &mem));
    memory = mem.Release();
  }
  if (descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
    cl_mem image_memory;
    RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        context, memory, descriptor.data_type,
        shape.b * shape.w * shape.h * shape.d * DivideRoundUp(shape.c, 4),
        &image_memory));
    *result = Tensor(memory, memory_owner, image_memory, shape, descriptor);
  } else {
    *result = Tensor(memory, memory_owner, shape, descriptor);
  }
  return absl::OkStatus();
}

absl::Status CreateTensorShared(const CLContext& context, const BHWDC& shape,
                                const TensorDescriptor& descriptor,
                                cl_mem memory, Tensor* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_4(mht_4_v, 463, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateTensorShared");

  const bool memory_owner = false;
  if (descriptor.storage_type == TensorStorageType::IMAGE_BUFFER) {
    cl_mem image_memory;
    RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        context, memory, descriptor.data_type,
        shape.b * shape.w * shape.h * shape.d * DivideRoundUp(shape.c, 4),
        &image_memory));
    *result = Tensor(memory, memory_owner, image_memory, shape, descriptor);
  } else {
    *result = Tensor(memory, memory_owner, shape, descriptor);
  }
  return absl::OkStatus();
}

}  // namespace

Tensor::Tensor(cl_mem memory, bool memory_owner, const BHWC& shape,
               const TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(nullptr),
      memory_owner_(memory_owner),
      shape_(shape.b, shape.h, shape.w, 1, shape.c),
      descriptor_(descriptor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_5(mht_5_v, 489, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::Tensor");
}

Tensor::Tensor(cl_mem memory, bool memory_owner, const BHWDC& shape,
               const TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(nullptr),
      memory_owner_(memory_owner),
      shape_(shape),
      descriptor_(descriptor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_6(mht_6_v, 500, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::Tensor");
}

Tensor::Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory,
               const BHWC& shape, const TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(image_buffer_memory),
      memory_owner_(memory_owner),
      shape_(shape.b, shape.h, shape.w, 1, shape.c),
      descriptor_(descriptor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_7(mht_7_v, 511, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::Tensor");

  if (image_buffer_memory &&
      (descriptor.storage_type == TensorStorageType::TEXTURE_2D ||
       descriptor.storage_type == TensorStorageType::SINGLE_TEXTURE_2D)) {
    buffer_based_ = true;
  }
}

Tensor::Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory,
               const BHWDC& shape, const TensorDescriptor& descriptor)
    : memory_(memory),
      image_buffer_memory_(image_buffer_memory),
      memory_owner_(memory_owner),
      shape_(shape),
      descriptor_(descriptor) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_8(mht_8_v, 528, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::Tensor");

  if (image_buffer_memory &&
      (descriptor.storage_type == TensorStorageType::TEXTURE_2D ||
       descriptor.storage_type == TensorStorageType::SINGLE_TEXTURE_2D)) {
    buffer_based_ = true;
  }
}

Tensor::Tensor(Tensor&& tensor)
    : memory_(tensor.memory_),
      image_buffer_memory_(tensor.image_buffer_memory_),
      memory_owner_(tensor.memory_owner_),
      buffer_based_(tensor.buffer_based_),
      shape_(tensor.shape_),
      descriptor_(tensor.descriptor_),
      aligned_texture_width_(tensor.aligned_texture_width_) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_9(mht_9_v, 546, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::Tensor");

  tensor.memory_ = nullptr;
  tensor.image_buffer_memory_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& tensor) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_10(mht_10_v, 554, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "=");

  if (this != &tensor) {
    Release();
    std::swap(memory_, tensor.memory_);
    std::swap(image_buffer_memory_, tensor.image_buffer_memory_);
    std::swap(memory_owner_, tensor.memory_owner_);
    std::swap(buffer_based_, tensor.buffer_based_);
    std::swap(shape_, tensor.shape_);
    std::swap(descriptor_, tensor.descriptor_);
    std::swap(aligned_texture_width_, tensor.aligned_texture_width_);
  }
  return *this;
}

void Tensor::Release() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_11(mht_11_v, 571, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::Release");

  // image_buffer_memory_ always owned by object
  if (image_buffer_memory_) {
    clReleaseMemObject(image_buffer_memory_);
    image_buffer_memory_ = nullptr;
  }
  if (memory_owner_ && memory_) {
    clReleaseMemObject(memory_);
    memory_ = nullptr;
  }
}

absl::Status Tensor::GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                                     GPUResourcesWithValue* resources) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_12(mht_12_v, 587, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::GetGPUResources");

  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(obj_ptr);
  if (buffer_desc) {
    if (descriptor_.storage_type != TensorStorageType::BUFFER) {
      return absl::InvalidArgumentError(
          "Tensor can be used with BufferDescriptor only with "
          "TensorStorageType::BUFFER.");
    }
    resources->buffers.push_back({"buffer", memory_});
    return absl::OkStatus();
  }
  const auto* texture2d_desc =
      dynamic_cast<const Texture2DDescriptor*>(obj_ptr);
  if (texture2d_desc) {
    if (descriptor_.storage_type != TensorStorageType::TEXTURE_2D) {
      return absl::InvalidArgumentError(
          "Tensor can be used with Texture2DDescriptor only with "
          "TensorStorageType::TEXTURE_2D.");
    }
    cl_mem mem = buffer_based_ ? image_buffer_memory_ : memory_;
    resources->images2d.push_back({"tex2d", mem});
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
    resources->buffers.push_back({"buffer", memory_});
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_2D ||
             descriptor_.storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
    if (obj_ptr->GetAccess() == AccessType::WRITE &&
        tensor_desc->use_buffer_for_write_only_2d_texture) {
      resources->ints.push_back(
          {"aligned_texture_width", aligned_texture_width_});
      resources->buffers.push_back({"buffer", memory_});
    } else {
      cl_mem mem = buffer_based_ ? image_buffer_memory_ : memory_;
      resources->images2d.push_back({"image2d", mem});
    }
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_ARRAY) {
    resources->image2d_arrays.push_back({"image2d_array", memory_});
  } else if (descriptor_.storage_type == TensorStorageType::TEXTURE_3D) {
    resources->images3d.push_back({"image3d", memory_});
  } else if (descriptor_.storage_type == TensorStorageType::IMAGE_BUFFER) {
    if (obj_ptr->GetAccess() == AccessType::WRITE &&
        tensor_desc->use_buffer_for_write_only_image_buffer) {
      resources->buffers.push_back({"buffer", memory_});
    } else {
      resources->image_buffers.push_back(
          {"image_buffer", image_buffer_memory_});
    }
  }

  return absl::OkStatus();
}

int3 Tensor::GetFullTensorRegion() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_13(mht_13_v, 666, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::GetFullTensorRegion");

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

absl::Status Tensor::IsValid(const BHWC& shape) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_14(mht_14_v, 685, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::IsValid");

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

absl::Status Tensor::IsValid(const BHWDC& shape) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_15(mht_15_v, 708, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::IsValid");

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

int Tensor::GetAlignedChannels() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_16(mht_16_v, 735, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::GetAlignedChannels");

  return descriptor_.storage_type == TensorStorageType::SINGLE_TEXTURE_2D
             ? shape_.c
             : AlignByN(shape_.c, 4);
}

uint64_t Tensor::GetMemorySizeInBytes() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_17(mht_17_v, 744, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::GetMemorySizeInBytes");

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

cl_mem Tensor::GetMemoryPtr() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_18(mht_18_v, 764, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::GetMemoryPtr");

  if (buffer_based_) {
    return image_buffer_memory_;
  } else {
    return descriptor_.storage_type == TensorStorageType::IMAGE_BUFFER
               ? image_buffer_memory_
               : memory_;
  }
}

cl_mem Tensor::GetMemoryPtrForWriting() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_19(mht_19_v, 777, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::GetMemoryPtrForWriting");

  if (buffer_based_) {
    return image_buffer_memory_;
  } else {
    return memory_;
  }
}

absl::Status Tensor::WriteData(
    CLCommandQueue* queue,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& src) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_20(mht_20_v, 790, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::WriteData");

  return WriteDataBHWDC(src.data.data(), queue);
}

absl::Status Tensor::WriteData(
    CLCommandQueue* queue,
    const tflite::gpu::Tensor<HWC, DataType::FLOAT32>& src) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_21(mht_21_v, 799, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::WriteData");

  return WriteDataBHWDC(src.data.data(), queue);
}

absl::Status Tensor::CreateFromDescriptor(const TensorDescriptor& desc,
                                          CLContext* context) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_22(mht_22_v, 807, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::CreateFromDescriptor");

  shape_ = desc.GetBHWDCShape();
  descriptor_.data_type = desc.data_type;
  descriptor_.storage_type = desc.storage_type;
  descriptor_.layout = desc.layout;
  memory_owner_ = true;
  CLMemory memory;
  const uint8_t* data_ptr =
      desc.GetData().empty() ? nullptr : desc.GetData().data();
  RETURN_IF_ERROR(
      AllocateTensorMemory(*context, shape_, descriptor_, data_ptr, &memory));
  memory_ = memory.Release();
  if (desc.storage_type == TensorStorageType::IMAGE_BUFFER) {
    RETURN_IF_ERROR(CreateImageBufferFromBuffer(
        *context, memory_, desc.data_type,
        shape_.b * shape_.w * shape_.h * shape_.d * DivideRoundUp(shape_.c, 4),
        &image_buffer_memory_));
  }
  return absl::OkStatus();
}

absl::Status Tensor::ToDescriptor(TensorDescriptor* desc,
                                  CLCommandQueue* queue) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_23(mht_23_v, 832, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::ToDescriptor");

  *desc = descriptor_;
  desc->SetBHWDCShape(shape_);
  std::vector<uint8_t> data(GetMemorySizeInBytes());
  RETURN_IF_ERROR(ReadData(data.data(), queue));
  desc->SetData(std::move(data));
  return absl::OkStatus();
}

absl::Status Tensor::WriteData(const void* ptr, CLCommandQueue* queue) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_24(mht_24_v, 844, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::WriteData");

  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      RETURN_IF_ERROR(
          queue->EnqueueWriteBuffer(memory_, GetMemorySizeInBytes(), ptr));
      break;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      cl_mem mem = buffer_based_ ? image_buffer_memory_ : memory_;
      RETURN_IF_ERROR(
          queue->EnqueueWriteImage(mem, GetFullTensorRegion(), ptr));
      break;
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status Tensor::ReadData(void* ptr, CLCommandQueue* queue) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_25(mht_25_v, 869, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "Tensor::ReadData");

  switch (descriptor_.storage_type) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      RETURN_IF_ERROR(
          queue->EnqueueReadBuffer(memory_, GetMemorySizeInBytes(), ptr));
      break;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      cl_mem mem = buffer_based_ ? image_buffer_memory_ : memory_;
      RETURN_IF_ERROR(queue->EnqueueReadImage(mem, GetFullTensorRegion(), ptr));
      break;
    }
    default:
      return absl::InternalError("Unsupported tensor storage type");
  }
  return absl::OkStatus();
}

absl::Status CreateTensor(const CLContext& context, const BHWC& shape,
                          const TensorDescriptor& descriptor, Tensor* result) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_26(mht_26_v, 894, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateTensor");

  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CreateTensor(context, shape5D, descriptor, nullptr, result);
}

absl::Status CreateTensor(const CLContext& context, const BHWDC& shape,
                          const TensorDescriptor& descriptor, Tensor* result) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_27(mht_27_v, 903, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateTensor");

  return CreateTensor(context, shape, descriptor, nullptr, result);
}

absl::Status CreateSharedTensor(const CLContext& context, cl_mem memory,
                                const BHWC& shape,
                                const TensorDescriptor& descriptor,
                                Tensor* result) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_28(mht_28_v, 913, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateSharedTensor");

  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CreateTensorShared(context, shape5D, descriptor, memory, result);
}

absl::Status CreateSharedTensor(const CLContext& context, cl_mem memory,
                                const BHWDC& shape,
                                const TensorDescriptor& descriptor,
                                Tensor* result) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_29(mht_29_v, 924, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateSharedTensor");

  return CreateTensorShared(context, shape, descriptor, memory, result);
}

absl::Status CreateSharedImage2DBufferTensor(const CLContext& context,
                                             cl_mem memory, const BHWC& shape,
                                             const TensorDescriptor& descriptor,
                                             int width_pixel_alignment,
                                             Tensor* result) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_30(mht_30_v, 935, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateSharedImage2DBufferTensor");

  BHWDC shape5d(shape.b, shape.h, shape.w, 1, shape.c);
  return CreateSharedImage2DBufferTensor(context, memory, shape5d, descriptor,
                                         width_pixel_alignment, result);
}

absl::Status CreateSharedImage2DBufferTensor(const CLContext& context,
                                             cl_mem memory, const BHWDC& shape,
                                             const TensorDescriptor& descriptor,
                                             int width_pixel_alignment,
                                             Tensor* result) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_31(mht_31_v, 948, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "CreateSharedImage2DBufferTensor");

  const int width = shape.b * shape.w * shape.d;
  const int height =
      descriptor.storage_type == TensorStorageType::SINGLE_TEXTURE_2D
          ? shape.h
          : shape.h * DivideRoundUp(shape.c, 4);
  const int channels =
      descriptor.storage_type == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c
                                                                      : 4;
  cl_mem image_memory;
  RETURN_IF_ERROR(CreateImage2DFromBuffer(
      context, memory, descriptor.data_type, width, height, channels,
      width_pixel_alignment, &image_memory));
  *result = Tensor(memory, false, image_memory, shape, descriptor);
  result->aligned_texture_width_ = AlignByN(width, width_pixel_alignment);
  return absl::OkStatus();
}

absl::Status AllocateTensorMemory(const CLContext& context, const BHWC& shape,
                                  const TensorDescriptor& descriptor,
                                  CLMemory* result) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_32(mht_32_v, 971, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "AllocateTensorMemory");

  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return AllocateTensorMemory(context, shape5D, descriptor, nullptr, result);
}

absl::Status AllocateTensorMemory(const CLContext& context, const BHWDC& shape,
                                  const TensorDescriptor& descriptor,
                                  CLMemory* result) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStensorDTcc mht_33(mht_33_v, 981, "", "./tensorflow/lite/delegates/gpu/cl/tensor.cc", "AllocateTensorMemory");

  return AllocateTensorMemory(context, shape, descriptor, nullptr, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
