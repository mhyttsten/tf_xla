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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSlinear_storageDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSlinear_storageDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSlinear_storageDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/linear_storage.h"

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"

namespace tflite {
namespace gpu {
namespace metal {

void LinearStorage::Release() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSlinear_storageDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/gpu/metal/linear_storage.cc", "LinearStorage::Release");

  if (buffer_) {
    buffer_ = nullptr;
  }
  if (texture_) {
    texture_ = nullptr;
  }
}

LinearStorage::LinearStorage(LinearStorage&& storage)
    : GPUObject(std::move(storage)),
      buffer_(storage.buffer_),
      texture_(storage.texture_),
      depth_(storage.depth_),
      storage_type_(storage.storage_type_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSlinear_storageDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/delegates/gpu/metal/linear_storage.cc", "LinearStorage::LinearStorage");

  storage.buffer_ = nullptr;
  storage.texture_ = nullptr;
}

LinearStorage& LinearStorage::operator=(LinearStorage&& storage) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSlinear_storageDTcc mht_2(mht_2_v, 220, "", "./tensorflow/lite/delegates/gpu/metal/linear_storage.cc", "=");

  if (this != &storage) {
    Release();
    std::swap(buffer_, storage.buffer_);
    std::swap(texture_, storage.texture_);
    std::swap(depth_, storage.depth_);
    std::swap(storage_type_, storage.storage_type_);
    GPUObject::operator=(std::move(storage));
  }
  return *this;
}

absl::Status LinearStorage::GetGPUResources(
    const GPUObjectDescriptor* obj_ptr,
    GPUResourcesWithValue* resources) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSlinear_storageDTcc mht_3(mht_3_v, 237, "", "./tensorflow/lite/delegates/gpu/metal/linear_storage.cc", "LinearStorage::GetGPUResources");

  const auto* linear_desc =
      dynamic_cast<const TensorLinearDescriptor*>(obj_ptr);
  if (!linear_desc) {
    return absl::InvalidArgumentError(
        "Expected TensorLinearDescriptor on input.");
  }

  resources->ints.push_back({"length", depth_});

  if (storage_type_ == LinearStorageType::BUFFER) {
    resources->buffers.push_back({"buffer", {buffer_, 0}});
  } else {
    resources->images2d.push_back({"tex2d", texture_});
  }

  return absl::OkStatus();
}

absl::Status LinearStorage::CreateFromTensorLinearDescriptor(
    const TensorLinearDescriptor& desc, id<MTLDevice> device) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSlinear_storageDTcc mht_4(mht_4_v, 260, "", "./tensorflow/lite/delegates/gpu/metal/linear_storage.cc", "LinearStorage::CreateFromTensorLinearDescriptor");

  storage_type_ = desc.storage_type;
  depth_ = desc.size;
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());
  const int float4_size = desc.element_type == DataType::FLOAT32
                              ? sizeof(float) * 4
                              : sizeof(half) * 4;
  if (storage_type_ == LinearStorageType::BUFFER) {
    bool read_only = desc.memory_type == MemoryType::CONSTANT;
    uint8_t* data_ptr = desc.data.empty()
                            ? nullptr
                            : const_cast<unsigned char*>(desc.data.data());
    buffer_ = [device newBufferWithBytes:data_ptr
                                  length:depth_ * float4_size
                                 options:MTLResourceStorageModeShared];
    if (!buffer_) {
      return absl::UnknownError("Failed to allocate id<MTLBuffer>");
    }

    return absl::OkStatus();
  } else {
    MTLPixelFormat pixel_format = desc.element_type == DataType::FLOAT32
                                      ? MTLPixelFormatRGBA32Float
                                      : MTLPixelFormatRGBA16Float;
    MTLTextureDescriptor* texture_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
                                                           width:depth_
                                                          height:1
                                                       mipmapped:NO];
    texture_desc.textureType = MTLTextureType2D;
    texture_desc.usage = MTLTextureUsageShaderRead;
    texture_desc.storageMode = MTLStorageModePrivate;

    texture_ = [device newTextureWithDescriptor:texture_desc];
    if (!texture_) {
      return absl::UnknownError("Failed to allocate id<MTLTexture>");
    }

    WriteDataToTexture2D(texture_, device, data_ptr);

    return absl::OkStatus();
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
