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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/texture2d.h"

#include "tensorflow/lite/delegates/gpu/cl/cl_image_format.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// Creates new 4-channel 2D texture with cl_channel_type elements
absl::Status CreateTexture2D(int width, int height, DataType type, void* data,
                             CLContext* context, Texture2D* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "CreateTexture2D");

  cl_mem texture;
  cl_channel_type channel_type = DataTypeToChannelType(type);
  RETURN_IF_ERROR(CreateRGBAImage2D(context->context(), width, height,
                                    channel_type, data, &texture));
  *result = Texture2D(texture, width, height, channel_type);

  return absl::OkStatus();
}
}  // namespace

Texture2D::Texture2D(cl_mem texture, int width, int height,
                     cl_channel_type type)
    : texture_(texture), width_(width), height_(height), channel_type_(type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "Texture2D::Texture2D");
}

Texture2D::Texture2D(Texture2D&& texture)
    : texture_(texture.texture_),
      width_(texture.width_),
      height_(texture.height_),
      channel_type_(texture.channel_type_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_2(mht_2_v, 221, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "Texture2D::Texture2D");

  texture.texture_ = nullptr;
  texture.width_ = 0;
  texture.height_ = 0;
}

Texture2D& Texture2D::operator=(Texture2D&& texture) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_3(mht_3_v, 230, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "=");

  if (this != &texture) {
    Release();
    std::swap(channel_type_, texture.channel_type_);
    std::swap(width_, texture.width_);
    std::swap(height_, texture.height_);
    std::swap(texture_, texture.texture_);
  }
  return *this;
}

void Texture2D::Release() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_4(mht_4_v, 244, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "Texture2D::Release");

  if (texture_) {
    clReleaseMemObject(texture_);
    texture_ = nullptr;
    width_ = 0;
    height_ = 0;
  }
}

absl::Status Texture2D::GetGPUResources(
    const GPUObjectDescriptor* obj_ptr,
    GPUResourcesWithValue* resources) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_5(mht_5_v, 258, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "Texture2D::GetGPUResources");

  const auto* texture_desc = dynamic_cast<const Texture2DDescriptor*>(obj_ptr);
  if (!texture_desc) {
    return absl::InvalidArgumentError("Expected Texture2DDescriptor on input.");
  }

  resources->images2d.push_back({"tex2d", texture_});
  return absl::OkStatus();
}

absl::Status Texture2D::CreateFromTexture2DDescriptor(
    const Texture2DDescriptor& desc, CLContext* context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_6(mht_6_v, 272, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "Texture2D::CreateFromTexture2DDescriptor");

  width_ = desc.size.x;
  height_ = desc.size.y;
  channel_type_ = DataTypeToChannelType(desc.element_type, desc.normalized);
  uint8_t* data_ptr = desc.data.empty()
                          ? nullptr
                          : const_cast<unsigned char*>(desc.data.data());
  return CreateRGBAImage2D(context->context(), desc.size.x, desc.size.y,
                           channel_type_, data_ptr, &texture_);
}

// Creates new 4-channel 2D texture with f32 elements
absl::Status CreateTexture2DRGBA32F(int width, int height, CLContext* context,
                                    Texture2D* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_7(mht_7_v, 288, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "CreateTexture2DRGBA32F");

  return CreateTexture2D(width, height, DataType::FLOAT32, nullptr, context,
                         result);
}

// Creates new 4-channel 2D texture with f16 elements
absl::Status CreateTexture2DRGBA16F(int width, int height, CLContext* context,
                                    Texture2D* result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_8(mht_8_v, 298, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "CreateTexture2DRGBA16F");

  return CreateTexture2D(width, height, DataType::FLOAT16, nullptr, context,
                         result);
}

absl::Status CreateTexture2DRGBA(DataType type, int width, int height,
                                 CLContext* context, Texture2D* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_9(mht_9_v, 307, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "CreateTexture2DRGBA");

  return CreateTexture2D(width, height, type, nullptr, context, result);
}

absl::Status CreateTexture2DRGBA(DataType type, int width, int height,
                                 void* data, CLContext* context,
                                 Texture2D* result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStexture2dDTcc mht_10(mht_10_v, 316, "", "./tensorflow/lite/delegates/gpu/cl/texture2d.cc", "CreateTexture2DRGBA");

  return CreateTexture2D(width, height, type, data, context, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
