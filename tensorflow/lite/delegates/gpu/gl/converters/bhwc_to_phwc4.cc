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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSconvertersPSbhwc_to_phwc4DTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSconvertersPSbhwc_to_phwc4DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSconvertersPSbhwc_to_phwc4DTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/converters/bhwc_to_phwc4.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/converters/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

absl::Status ConverterBhwcToPhwc4::Create(ConverterBhwcToPhwc4* converter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSconvertersPSbhwc_to_phwc4DTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/delegates/gpu/gl/converters/bhwc_to_phwc4.cc", "ConverterBhwcToPhwc4::Create");

  uint3 workgroup_size = uint3(4, 4, 4);
  std::string shader_source = GetShaderHeader(workgroup_size) + R"(
    layout(std430) buffer;

    precision highp float;

    layout(binding = 0) readonly buffer B0 {
      float elements[];
    } input_data;

    layout(binding = 1) writeonly buffer B1 {
      vec4 elements[];
    } output_data;

    uniform ivec4 sizes_;

    void main() {
      ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
      if (gid.x >= sizes_.x || gid.y >= sizes_.y || gid.z >= sizes_.z) {
        return;
      }
      vec4 v = vec4(0);
      int dst_channel = gid.z * 4;
      int index = (gid.y * sizes_.x + gid.x) * sizes_.w + dst_channel;
      for (int i = 0; i < 4; ++i, ++index, ++dst_channel) {
        if (dst_channel >= sizes_.w) break;
        v[i] = input_data.elements[index];
      }
      output_data.elements[(gid.z * sizes_.y + gid.y) * sizes_.x + gid.x] = v;
    })";

  GlShader shader;
  RETURN_IF_ERROR(
      GlShader::CompileShader(GL_COMPUTE_SHADER, shader_source, &shader));
  GlProgram program;
  RETURN_IF_ERROR(GlProgram::CreateWithShader(shader, &program));
  *converter = ConverterBhwcToPhwc4(std::move(program), workgroup_size);
  return absl::OkStatus();
}

absl::Status ConverterBhwcToPhwc4::Convert(const BHWC& shape,
                                           const GlBuffer& source,
                                           CommandQueue* command_queue,
                                           GlBuffer* destination) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSconvertersPSbhwc_to_phwc4DTcc mht_1(mht_1_v, 250, "", "./tensorflow/lite/delegates/gpu/gl/converters/bhwc_to_phwc4.cc", "ConverterBhwcToPhwc4::Convert");

  if (source.bytes_size() < BytesForBHWC(shape)) {
    return absl::InvalidArgumentError(
        "BhwcToPhwc4: Input data size does not match expected size.");
  }
  if (destination->bytes_size() < BytesForPHWC4(shape)) {
    return absl::InvalidArgumentError(
        "BhwcToPhwc4: output data size does not match expected size.");
  }
  if (shape.b != 1) {
    return absl::UnimplementedError(
        "BhwcToPhwc4: Batch size is not equal to 1.");
  }
  uint3 workload = uint3(shape.w, shape.h, DivideRoundUp(shape.c, 4));
  uint3 num_workgroups = DivideRoundUp(workload, workgroup_size_);

  RETURN_IF_ERROR(program_.SetParameter(
      {"sizes_",
       int4(static_cast<int32_t>(workload.x), static_cast<int32_t>(workload.y),
            static_cast<int32_t>(workload.z), static_cast<int32_t>(shape.c))}));
  RETURN_IF_ERROR(source.BindToIndex(0));
  RETURN_IF_ERROR(destination->BindToIndex(1));
  if (command_queue) {
    return command_queue->Dispatch(program_, num_workgroups);
  }
  return program_.Dispatch(num_workgroups);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
