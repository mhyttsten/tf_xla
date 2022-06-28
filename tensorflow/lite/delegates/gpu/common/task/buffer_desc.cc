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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSbuffer_descDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSbuffer_descDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSbuffer_descDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {

void BufferDescriptor::Release() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSbuffer_descDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/delegates/gpu/common/task/buffer_desc.cc", "BufferDescriptor::Release");
 data.clear(); }

GPUResources BufferDescriptor::GetGPUResources(const GpuInfo& gpu_info) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSbuffer_descDTcc mht_1(mht_1_v, 203, "", "./tensorflow/lite/delegates/gpu/common/task/buffer_desc.cc", "BufferDescriptor::GetGPUResources");

  GPUResources resources;
  GPUBufferDescriptor desc;
  desc.data_type = element_type;
  desc.access_type = access_type_;
  desc.element_size = element_size;
  desc.memory_type = memory_type;
  desc.attributes = attributes;
  if (gpu_info.IsGlsl() && memory_type == tflite::gpu::MemoryType::CONSTANT) {
    desc.attributes.push_back(
        std::to_string(size / (element_size * SizeOf(element_type))));
  }
  resources.buffers.push_back({"buffer", desc});
  return resources;
}

absl::Status BufferDescriptor::PerformSelector(
    const GpuInfo& gpu_info, const std::string& selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("selector: \"" + selector + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSbuffer_descDTcc mht_2(mht_2_v, 226, "", "./tensorflow/lite/delegates/gpu/common/task/buffer_desc.cc", "BufferDescriptor::PerformSelector");

  if (selector == "Read") {
    return PerformReadSelector(gpu_info, args, result);
  } else if (selector == "GetPtr") {
    return PerformGetPtrSelector(args, template_args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "BufferDescriptor don't have selector with name - ", selector));
  }
}

absl::Status BufferDescriptor::PerformReadSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSbuffer_descDTcc mht_3(mht_3_v, 242, "", "./tensorflow/lite/delegates/gpu/common/task/buffer_desc.cc", "BufferDescriptor::PerformReadSelector");

  if (args.size() != 1) {
    return absl::NotFoundError(
        absl::StrCat("BufferDescriptor Read require one argument, but ",
                     args.size(), " was passed"));
  }
  if (gpu_info.IsGlsl()) {
    if (element_type == DataType::FLOAT16 &&
        !gpu_info.IsGlslSupportsExplicitFp16()) {
      if (memory_type == MemoryType::CONSTANT) {
        bool is_kernel_global_space = false;
        for (const auto& attribute : attributes) {
          if (attribute == "kernel_global_space") {
            is_kernel_global_space = true;
            break;
          }
        }
        if (is_kernel_global_space) {
          *result = absl::StrCat("buffer[", args[0], "]");
          return absl::OkStatus();
        }
        const std::string arg0 = "(" + args[0] + ")";
        *result =
            absl::StrCat("vec4(unpackHalf2x16(buffer[", arg0, " / 2][", arg0,
                         " % 2 == 0 ? 0 : 2]), unpackHalf2x16(buffer[", arg0,
                         " / 2][", arg0, " % 2 == 0 ? 1 : 3]))");
      } else {
        if (element_size == 4) {
          *result =
              absl::StrCat("vec4(unpackHalf2x16(buffer[", args[0],
                           "].x), unpackHalf2x16(buffer[", args[0], "].y))");
        } else if (element_size == 16) {
          const std::string vec0 = absl::Substitute(
              "vec4(unpackHalf2x16(buffer[$0].a.x), "
              "unpackHalf2x16(buffer[$0].a.y))",
              args[0]);
          const std::string vec1 = absl::Substitute(
              "vec4(unpackHalf2x16(buffer[$0].a.z), "
              "unpackHalf2x16(buffer[$0].a.w))",
              args[0]);
          const std::string vec2 = absl::Substitute(
              "vec4(unpackHalf2x16(buffer[$0].b.x), "
              "unpackHalf2x16(buffer[$0].b.y))",
              args[0]);
          const std::string vec3 = absl::Substitute(
              "vec4(unpackHalf2x16(buffer[$0].b.z), "
              "unpackHalf2x16(buffer[$0].b.w))",
              args[0]);
          *result = absl::Substitute("mat4x4($0, $1, $2, $3)", vec0, vec1, vec2,
                                     vec3);
        }
      }
    } else {
      *result = absl::StrCat("buffer[", args[0], "]");
    }
    return absl::OkStatus();
  } else {
    *result = absl::StrCat("buffer[", args[0], "]");
    return absl::OkStatus();
  }
}

absl::Status BufferDescriptor::PerformGetPtrSelector(
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSbuffer_descDTcc mht_4(mht_4_v, 309, "", "./tensorflow/lite/delegates/gpu/common/task/buffer_desc.cc", "BufferDescriptor::PerformGetPtrSelector");

  if (args.size() > 1) {
    return absl::NotFoundError(absl::StrCat(
        "BufferDescriptor GetPtr require one or zero arguments, but ",
        args.size(), " was passed"));
  }
  if (template_args.size() > 1) {
    return absl::NotFoundError(
        absl::StrCat("BufferDescriptor GetPtr require one or zero teemplate "
                     "arguments, but ",
                     template_args.size(), " was passed"));
  }
  std::string conversion;
  if (template_args.size() == 1) {
    const std::string type_name = ToCLDataType(element_type, element_size);
    if (type_name != template_args[0]) {
      conversion = absl::StrCat("(", MemoryTypeToCLType(memory_type), " ",
                                template_args[0], "*)&");
    }
  }
  if (args.empty()) {
    *result = absl::StrCat(conversion, "buffer");
  } else if (conversion.empty()) {
    *result = absl::StrCat("(buffer + ", args[0], ")");
  } else {
    *result = absl::StrCat(conversion, "buffer[", args[0], "]");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
