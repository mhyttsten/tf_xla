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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSresizeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSresizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSresizeDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/resize.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Resize : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSresizeDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/gpu/gl/kernels/resize.cc", "GenerateCode");

    const auto& attr = absl::any_cast<const Resize2DAttributes&>(ctx.op_attr);

    if (ctx.input_shapes[0][2] > ctx.output_shapes[0][2] ||
        ctx.input_shapes[0][1] > ctx.output_shapes[0][1]) {
      return absl::InvalidArgumentError("Output size is less than input size.");
    }
    if (ctx.output_shapes[0][2] != attr.new_shape.w ||
        ctx.output_shapes[0][1] != attr.new_shape.h) {
      return absl::InvalidArgumentError(
          "Output size does not match new_size in attributes.");
    }
    if (ctx.input_shapes[0][3] != ctx.output_shapes[0][3]) {
      return absl::InvalidArgumentError("Input/output channels mismatch.");
    }
    if (ctx.input_shapes[0][1] == 1 && ctx.input_shapes[0][2] == 1) {
      // Copy a single element from input.
      *generated_code = {
          /*parameters=*/{},
          /*objects=*/{},
          /*shared_variables=*/{},
          /*workload=*/uint3(),
          /*workgroup=*/uint3(),
          /*source_code=*/"value_0 = $input_data_0[0, 0, gid.z]$;",
          /*input=*/IOStructure::ONLY_DEFINITIONS,
          /*output=*/IOStructure::AUTO,
      };
      return absl::OkStatus();
    }
    std::vector<Variable> parameters = {
        {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
        {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])},
        {"scale_factor",
         float2(CalculateResizeScale(ctx.input_shapes[0][2],
                                     ctx.output_shapes[0][2], attr),
                CalculateResizeScale(ctx.input_shapes[0][1],
                                     ctx.output_shapes[0][1], attr))},
    };

    std::string source;
    if (attr.type == SamplingType::BILINEAR) {
      if (attr.half_pixel_centers) {
        source = "vec2 coord = (vec2(gid.xy) + 0.5) * $scale_factor$ - 0.5;";
      } else {
        source = "vec2 coord = vec2(gid.xy) * $scale_factor$;";
      }
      source += R"(
      vec2 coord_floor = floor(coord);
      ivec2 icoord_floor = ivec2(coord_floor);
      ivec2 borders = ivec2($input_data_0_w$, $input_data_0_h$) - ivec2(1, 1);
      ivec4 st;
      st.xy = max(icoord_floor, ivec2(0, 0));
      st.zw = min(icoord_floor + ivec2(1, 1), borders);

      vec2 t = coord - coord_floor; // interpolating factors

      vec4 tex11 = $input_data_0[st.x, st.y, gid.z]$;
      vec4 tex21 = $input_data_0[st.z, st.y, gid.z]$;
      vec4 tex12 = $input_data_0[st.x, st.w, gid.z]$;
      vec4 tex22 = $input_data_0[st.z, st.w, gid.z]$;

      value_0 = mix(mix(tex11, tex21, t.x), mix(tex12, tex22, t.x), t.y);)";
    } else if (attr.type == SamplingType::NEAREST) {
      std::string fxc;
      std::string fyc;
      if (attr.half_pixel_centers) {
        fxc = "(float(gid.x) + 0.5) * $scale_factor.x$";
        fyc = "(float(gid.y) + 0.5) * $scale_factor.y$";
      } else {
        fxc = "float(gid.x) * $scale_factor.x$";
        fyc = "float(gid.y) * $scale_factor.y$";
      }
      if (attr.align_corners) {
        fxc += " + 0.5";
        fyc += " + 0.5";
      }
      source += "  ivec2 coord;\n";
      source += "  coord.x = int(" + fxc + ");\n";
      source += "  coord.y = int(" + fyc + ");\n";
      source += "  coord.x = max(0, coord.x);\n";
      source += "  coord.y = max(0, coord.y);\n";
      source += "  coord.x = min(coord.x, $input_data_0_w$ - 1);\n";
      source += "  coord.y = min(coord.y, $input_data_0_h$ - 1);\n";
      source += R"(
      value_0 = $input_data_0[coord.x, coord.y, gid.z]$;
      )";
    } else {
      return absl::InvalidArgumentError("Unknown sampling type");
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

std::unique_ptr<NodeShader> NewResizeNodeShader() {
  return absl::make_unique<Resize>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
