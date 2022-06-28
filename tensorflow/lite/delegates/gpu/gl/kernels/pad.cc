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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpadDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpadDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpadDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/pad.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Pad : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSpadDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/gpu/gl/kernels/pad.cc", "GenerateCode");

    const auto& attr = absl::any_cast<const PadAttributes&>(ctx.op_attr);

    if (attr.type != PaddingContentType::ZEROS &&
        attr.type != PaddingContentType::REFLECT) {
      return absl::UnimplementedError(
          "Only ZERO and REFLECT padding types are supported.");
    }
    if (attr.appended.h < 0 || attr.appended.w < 0 || attr.appended.c < 0 ||
        attr.prepended.h < 0 || attr.prepended.w < 0 || attr.prepended.c < 0) {
      return absl::UnimplementedError("Negative padding is not supported.");
    }
    if (attr.appended.b != 0 || attr.prepended.b != 0) {
      return absl::UnimplementedError("Padding for BATCH is not supported.");
    }
    std::vector<Variable> parameters = {
        {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
        {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])},
        {"input_data_0_c", static_cast<int>(ctx.input_shapes[0][3])},
        {"prepended",
         int4(attr.prepended.w, attr.prepended.h, attr.prepended.c, 0)},
    };
    std::string source;
    if (attr.type == PaddingContentType::REFLECT) {
      source = R"(
  int src_x = gid.x - $prepended.x$;
  src_x = abs(src_x);
  src_x = $input_data_0_w$ - 1 - abs(src_x - $input_data_0_w$ + 1);

  int src_y = gid.y - $prepended.y$;
  src_y = abs(src_y);
  src_y = $input_data_0_h$ - 1 - abs(src_y - $input_data_0_h$ + 1);
)";
      if (attr.prepended.c == 0 && attr.appended.c == 0) {
        // optimized case
        source += "  value_0 = $input_data_0[src_x, src_y, gid.z]$;\n";
      } else {
        source += R"(
  int start_channel = gid.z * 4;
  for (int i = 0; i < 4; ++i) {
    int channel = start_channel + i;
    int src_z = channel - $prepended.z$;
    src_z = abs(src_z);
    src_z = $input_data_0_c$ - 1 - abs(src_z - $input_data_0_c$ + 1);
    // We need additional clamp for z, so that we use alignment for channels
    // and can proceed extra channels that can lead to reading out of
    // resource.
    src_z = clamp(src_z, 0, $input_data_0_c$ - 1);
    value_0[i] = $input_data_0[src_x, src_y, src_z / 4]$[src_z % 4];
  }
)";
      }
    } else {
      source = R"(
  int src_x = gid.x - $prepended.x$;
  int src_y = gid.y - $prepended.y$;
  if (src_x >= 0 && src_x < $input_data_0_w$ && src_y >= 0 && src_y < $input_data_0_h$) {
)";
      if (attr.prepended.c == 0 && attr.appended.c == 0) {
        // optimized case
        source += "    value_0 = $input_data_0[src_x, src_y, gid.z]$;\n";
      } else if (attr.prepended.c % 4 == 0) {
        parameters.push_back(
            {"src_slices",
             DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)});
        source += R"(
    int src_z = gid.z - $prepended.z$ / 4;
    if (src_z >= 0 && src_z < $src_slices$) {
      value_0 = $input_data_0[src_x, src_y, src_z]$;
    }
)";
      } else {
        source += R"(
    int start_channel = gid.z * 4;
    for (int i = 0; i < 4; ++i) {
      int channel = start_channel + i;
      int src_z = channel - $prepended.z$;
      if (src_z >= 0 && src_z < $input_data_0_c$) {
        value_0[i] = $input_data_0[src_x, src_y, src_z / 4]$[src_z % 4];
      }
    }
)";
      }
      source += "  }\n";
    }
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewPadNodeShader() {
  return absl::make_unique<Pad>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
