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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStexture2d_descDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStexture2d_descDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStexture2d_descDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"

#include <string>

#include "absl/strings/str_cat.h"

namespace tflite {
namespace gpu {

void Texture2DDescriptor::Release() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStexture2d_descDTcc mht_0(mht_0_v, 194, "", "./tensorflow/lite/delegates/gpu/common/task/texture2d_desc.cc", "Texture2DDescriptor::Release");
 data.clear(); }

GPUResources Texture2DDescriptor::GetGPUResources(
    const GpuInfo& gpu_info) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStexture2d_descDTcc mht_1(mht_1_v, 200, "", "./tensorflow/lite/delegates/gpu/common/task/texture2d_desc.cc", "Texture2DDescriptor::GetGPUResources");

  GPUResources resources;
  GPUImage2DDescriptor desc;
  desc.data_type = element_type;
  desc.normalized = normalized;
  desc.normalized_type = normalized_type;
  desc.access_type = access_type_;
  resources.images2d.push_back({"tex2d", desc});
  if (gpu_info.IsApiOpenGl() && gpu_info.opengl_info.major_version < 3) {
    resources.floats.push_back("inv_tex_width");
    resources.floats.push_back("inv_tex_height");
  }
  return resources;
}

absl::Status Texture2DDescriptor::PerformSelector(
    const GpuInfo& gpu_info, const std::string& selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("selector: \"" + selector + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStexture2d_descDTcc mht_2(mht_2_v, 222, "", "./tensorflow/lite/delegates/gpu/common/task/texture2d_desc.cc", "Texture2DDescriptor::PerformSelector");

  if (selector == "Read") {
    return PerformReadSelector(gpu_info, args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "Texture2DDescriptor don't have selector with name - ", selector));
  }
}

absl::Status Texture2DDescriptor::PerformReadSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPStexture2d_descDTcc mht_3(mht_3_v, 236, "", "./tensorflow/lite/delegates/gpu/common/task/texture2d_desc.cc", "Texture2DDescriptor::PerformReadSelector");

  if (args.size() != 2) {
    return absl::NotFoundError(
        absl::StrCat("Texture2DDescriptor Read require two arguments, but ",
                     args.size(), " was passed"));
  }
  if (gpu_info.IsApiMetal()) {
    *result =
        absl::StrCat("tex2d.read(ushort2(", args[0], ", " + args[1] + "))");
    return absl::OkStatus();
  } else if (gpu_info.IsApiOpenCl()) {
    std::string read;
    switch (element_type) {
      case DataType::FLOAT32:
        read = "read_imagef";
        break;
      case DataType::FLOAT16:
        read = "read_imageh";
        break;
      case DataType::INT8:
      case DataType::INT16:
      case DataType::INT32:
        if (normalized) {
          read = normalized_type == DataType::FLOAT16 ? "read_imageh"
                                                      : "read_imagef";
        } else {
          read = "read_imagei";
        }
        break;
      case DataType::UINT8:
      case DataType::UINT16:
      case DataType::UINT32:
        if (normalized) {
          read = normalized_type == DataType::FLOAT16 ? "read_imageh"
                                                      : "read_imagef";
        } else {
          read = "read_imageui";
        }
        break;
      default:
        read = "unknown_type";
        break;
    }
    *result = absl::StrCat(read, "(tex2d, smp_none, (int2)(", args[0],
                           ", " + args[1] + "))");
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    if (gpu_info.IsApiOpenGl() && gpu_info.opengl_info.major_version < 3) {
      *result = absl::StrCat("texture2D(tex2d, vec2(float(", args[0],
                             ") * inv_tex_width, float(", args[1],
                             ") * inv_tex_height))");
      return absl::OkStatus();
    } else {
      *result = "texelFetch(tex2d, ivec2(" + args[0] + ", " + args[1] + "), 0)";
      if (element_type == DataType::FLOAT16 &&
          gpu_info.IsGlslSupportsExplicitFp16()) {
        *result = "f16vec4(" + *result + ")";
      }
      return absl::OkStatus();
    }
  } else {
    return absl::UnimplementedError(
        "No implementation of Texture2D.Read for this API.");
  }
}

}  // namespace gpu
}  // namespace tflite
