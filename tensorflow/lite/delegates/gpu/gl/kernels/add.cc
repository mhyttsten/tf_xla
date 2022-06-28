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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSaddDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSaddDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSaddDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/add.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Add : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSaddDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/delegates/gpu/gl/kernels/add.cc", "GenerateCode");

    const auto& attr =
        absl::any_cast<const ElementwiseAttributes&>(ctx.op_attr);
    auto adds = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.param);
    auto scalar = absl::get_if<float>(&attr.param);

    const auto* hwc_tensor =
        absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.param);

    if (hwc_tensor) {
      std::string code;
      const std::string x_coord = hwc_tensor->shape.w == 1 ? "0" : "gid.x";
      const std::string y_coord = hwc_tensor->shape.h == 1 ? "0" : "gid.y";
      const std::string s_coord = hwc_tensor->shape.c == 1 ? "0" : "gid.z";
      code = absl::StrCat("vec4 second_val = $hwc_buffer[", x_coord, ", ",
                          y_coord, ", ", s_coord, "]$;\n");
      if (hwc_tensor->shape.c == 1) {
        code += "  second_val.y = second_val.x;\n";
        code += "  second_val.z = second_val.x;\n";
        code += "  second_val.w = second_val.x;\n";
      }
      code += "  value_0 += second_val;\n";
      *generated_code = {
          /*parameters=*/{},
          /*objects=*/
          {{"hwc_buffer",
            MakeReadonlyObject(
                uint3(hwc_tensor->shape.w, hwc_tensor->shape.h,
                      DivideRoundUp(hwc_tensor->shape.c, 4)),
                ConvertToPHWC4(
                    absl::get<Tensor<HWC, DataType::FLOAT32>>(attr.param)))}},
          /*shared_variables=*/{},
          // Declare workload explicitly because shader depends on gid.z.
          /*workload=*/
          uint3(static_cast<int>(ctx.input_shapes[0][2]),
                static_cast<int>(ctx.input_shapes[0][1]),
                DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)),
          /*workgroup=*/uint3(),
          /*source_code=*/std::move(code),
          /*input=*/IOStructure::AUTO,
          /*output=*/IOStructure::AUTO,
      };
      return absl::OkStatus();
    }

    if (!adds && !scalar) {
      // check if it is a broadcast
      if (ctx.input_shapes.size() == 2 &&
          ctx.input_shapes[0] != ctx.input_shapes[1] &&
          ctx.input_shapes[1][1] == 1 && ctx.input_shapes[1][2] == 1 &&
          ctx.input_shapes[0][3] == ctx.input_shapes[1][3]) {
        // TODO(b/147771327): investigate why input_data_1[gid.z] worked before
        *generated_code = {
            /*parameters=*/{},
            /*objects=*/{},
            /*shared_variables=*/{},
            /*workload=*/uint3(),
            /*workgroup=*/uint3(),
            /*source_code=*/
            "value_0 = $input_data_0[gid.x, gid.y, gid.z]$ + "
            "          $input_data_1[0, 0, gid.z]$;",
            /*input=*/IOStructure::ONLY_DEFINITIONS,
            /*output=*/IOStructure::AUTO,
        };
        return absl::OkStatus();
      }

      std::string code = "value_0 = value_0";
      for (int index = 1; index < ctx.input_shapes.size(); ++index) {
        if (ctx.input_shapes[index] != ctx.input_shapes[0]) {
          return absl::InvalidArgumentError("Shapes are not equal");
        }
        absl::StrAppend(&code, " + value_", index);
      }
      absl::StrAppend(&code, ";");
      *generated_code = {
          /*parameters=*/{},
          /*objects=*/{},
          /*shared_variables=*/{},
          /*workload=*/uint3(),
          /*workgroup=*/uint3(),
          /*source_code=*/std::move(code),
          /*input=*/IOStructure::AUTO,
          /*output=*/IOStructure::AUTO,
      };
      return absl::OkStatus();
    }

    if (scalar) {
      *generated_code = {
          /*parameters=*/{{"scalar", *scalar}},
          /*objects=*/{},
          /*shared_variables=*/{},
          /*workload=*/uint3(),
          /*workgroup=*/uint3(),
          /*source_code=*/"value_0 += $scalar$;",
          /*input=*/IOStructure::AUTO,
          /*output=*/IOStructure::AUTO,
      };
      return absl::OkStatus();
    }

    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{{"add_buffer", MakeReadonlyObject(adds->data)}},
        /*shared_variables=*/{},
        // Declare workload explicitly because shader depends on gid.z.
        /*workload=*/
        uint3(ctx.input_shapes[0][2], ctx.input_shapes[0][1],
              DivideRoundUp(ctx.input_shapes[0][3], 4)),
        /*workgroup=*/uint3(),
        /*source_code=*/"value_0 += $add_buffer[gid.z]$;",
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewAddNodeShader() {
  return absl::make_unique<Add>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
