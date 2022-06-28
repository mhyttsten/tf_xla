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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_image_format.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::vector<cl_image_format> GetSupportedImage2DFormats(cl_context context,
                                                        cl_mem_flags flags) {
  cl_uint num_image_formats;
  cl_int error = clGetSupportedImageFormats(
      context, flags, CL_MEM_OBJECT_IMAGE2D, 0, nullptr, &num_image_formats);
  if (error != CL_SUCCESS) {
    return {};
  }

  std::vector<cl_image_format> result(num_image_formats);
  error = clGetSupportedImageFormats(context, flags, CL_MEM_OBJECT_IMAGE2D,
                                     num_image_formats, &result[0], nullptr);
  if (error != CL_SUCCESS) {
    return {};
  }
  return result;
}

bool IsEqualToImageFormat(cl_image_format image_format, DataType data_type,
                          int num_channels) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "IsEqualToImageFormat");

  return image_format.image_channel_data_type ==
             DataTypeToChannelType(data_type) &&
         image_format.image_channel_order == ToChannelOrder(num_channels);
}

void AddSupportedImageFormats(cl_context context, GpuInfo* info) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_1(mht_1_v, 225, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "AddSupportedImageFormats");

  auto supported_formats =
      GetSupportedImage2DFormats(context, CL_MEM_READ_WRITE);
  for (auto format : supported_formats) {
    info->opencl_info.supports_r_f16_tex2d =
        info->opencl_info.supports_r_f16_tex2d ||
        IsEqualToImageFormat(format, DataType::FLOAT16, 1);
    info->opencl_info.supports_rg_f16_tex2d =
        info->opencl_info.supports_rg_f16_tex2d ||
        IsEqualToImageFormat(format, DataType::FLOAT16, 2);
    info->opencl_info.supports_rgb_f16_tex2d =
        info->opencl_info.supports_rgb_f16_tex2d ||
        IsEqualToImageFormat(format, DataType::FLOAT16, 3);
    info->opencl_info.supports_rgba_f16_tex2d =
        info->opencl_info.supports_rgba_f16_tex2d ||
        IsEqualToImageFormat(format, DataType::FLOAT16, 4);
    info->opencl_info.supports_r_f32_tex2d =
        info->opencl_info.supports_r_f32_tex2d ||
        IsEqualToImageFormat(format, DataType::FLOAT32, 1);
    info->opencl_info.supports_rg_f32_tex2d =
        info->opencl_info.supports_rg_f32_tex2d ||
        IsEqualToImageFormat(format, DataType::FLOAT32, 2);
    info->opencl_info.supports_rgb_f32_tex2d =
        info->opencl_info.supports_rgb_f32_tex2d ||
        IsEqualToImageFormat(format, DataType::FLOAT32, 3);
    info->opencl_info.supports_rgba_f32_tex2d =
        info->opencl_info.supports_rgba_f32_tex2d ||
        IsEqualToImageFormat(format, DataType::FLOAT32, 4);
  }
}

absl::Status CreateCLContext(const CLDevice& device,
                             cl_context_properties* properties,
                             CLContext* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_2(mht_2_v, 261, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "CreateCLContext");

  int error_code;
  cl_device_id device_id = device.id();
  cl_context context =
      clCreateContext(properties, 1, &device_id, nullptr, nullptr, &error_code);
  if (!context) {
    return absl::UnknownError(
        absl::StrCat("Failed to create a compute context - ",
                     CLErrorCodeToString(error_code)));
  }
  AddSupportedImageFormats(context, &device.info_);

  *result = CLContext(context, true);
  return absl::OkStatus();
}

}  // namespace

CLContext::CLContext(cl_context context, bool has_ownership)
    : context_(context), has_ownership_(has_ownership) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_3(mht_3_v, 283, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "CLContext::CLContext");
}

CLContext::CLContext(CLContext&& context)
    : context_(context.context_), has_ownership_(context.has_ownership_) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_4(mht_4_v, 289, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "CLContext::CLContext");

  context.context_ = nullptr;
}

CLContext& CLContext::operator=(CLContext&& context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_5(mht_5_v, 296, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "=");

  if (this != &context) {
    Release();
    std::swap(context_, context.context_);
    has_ownership_ = context.has_ownership_;
  }
  return *this;
}

CLContext::~CLContext() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_6(mht_6_v, 308, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "CLContext::~CLContext");
 Release(); }

void CLContext::Release() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_7(mht_7_v, 313, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "CLContext::Release");

  if (has_ownership_ && context_) {
    clReleaseContext(context_);
    context_ = nullptr;
  }
}

bool CLContext::IsFloatTexture2DSupported(int num_channels, DataType data_type,
                                          cl_mem_flags flags) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_8(mht_8_v, 324, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "CLContext::IsFloatTexture2DSupported");

  auto supported_formats = GetSupportedImage2DFormats(context_, flags);
  for (auto format : supported_formats) {
    if (format.image_channel_data_type == DataTypeToChannelType(data_type) &&
        format.image_channel_order == ToChannelOrder(num_channels)) {
      return true;
    }
  }

  return false;
}

absl::Status CreateCLContext(const CLDevice& device, CLContext* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_9(mht_9_v, 339, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "CreateCLContext");

  return CreateCLContext(device, nullptr, result);
}

absl::Status CreateCLGLContext(const CLDevice& device,
                               cl_context_properties egl_context,
                               cl_context_properties egl_display,
                               CLContext* result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_contextDTcc mht_10(mht_10_v, 349, "", "./tensorflow/lite/delegates/gpu/cl/cl_context.cc", "CreateCLGLContext");

  if (!device.GetInfo().SupportsExtension("cl_khr_gl_sharing")) {
    return absl::UnavailableError("Device doesn't support CL-GL sharing.");
  }
  cl_context_properties platform =
      reinterpret_cast<cl_context_properties>(device.platform());
  cl_context_properties props[] = {CL_GL_CONTEXT_KHR,
                                   egl_context,
                                   CL_EGL_DISPLAY_KHR,
                                   egl_display,
                                   CL_CONTEXT_PLATFORM,
                                   platform,
                                   0};
  return CreateCLContext(device, props, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
