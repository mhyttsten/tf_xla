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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmulDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmulDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmulDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/mul.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {

namespace {

bool IsApplyMaskSupported(const NodeShader::GenerationContext& ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmulDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mul.cc", "IsApplyMaskSupported");

  if (ctx.input_shapes.size() != 2) return false;

  // [H, W, C] x [H, W, 0][0]
  if (ctx.input_shapes[0][1] == ctx.input_shapes[1][1] &&
      ctx.input_shapes[0][2] == ctx.input_shapes[1][2] &&
      ctx.input_shapes[1][3] == 1) {
    return true;
  }

  // [H, W, C] x [H, W, C]
  if (ctx.input_shapes[0] == ctx.input_shapes[1]) return true;

  // [H, W, C] x [0, 0, C]
  return ctx.input_shapes[1][1] == 1 && ctx.input_shapes[1][2] == 1 &&
         ctx.input_shapes[0][3] == ctx.input_shapes[1][3];
}

absl::Status GenerateApplyMaskCode(const NodeShader::GenerationContext& ctx,
                                   GeneratedCode* generated_code) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmulDTcc mht_1(mht_1_v, 227, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mul.cc", "GenerateApplyMaskCode");

  std::string source = "value_0 = $input_data_0[gid.x, gid.y, gid.z]$ * ";
  if (ctx.input_shapes[1][3] == 1) {
    // [H, W, C] x [H, W, 0][0]
    absl::StrAppend(&source, "$input_data_1[gid.x, gid.y, 0]$.x;");
  } else if (ctx.input_shapes[0][1] == ctx.input_shapes[1][1] &&
             ctx.input_shapes[0][2] == ctx.input_shapes[1][2]) {
    // [H, W, C] x [H, W, C]
    absl::StrAppend(&source, "$input_data_1[gid.x, gid.y, gid.z]$;");
  } else {
    // [H, W, C] x [0, 0, C]
    absl::StrAppend(&source, "$input_data_1[0, 0, gid.z]$;");
  }

  *generated_code = {
      /*parameters=*/{},
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

absl::Status GenerateMultiplyScalarCode(
    const NodeShader::GenerationContext& ctx, GeneratedCode* generated_code) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmulDTcc mht_2(mht_2_v, 258, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mul.cc", "GenerateMultiplyScalarCode");

  const auto& attr = absl::any_cast<const ElementwiseAttributes&>(ctx.op_attr);

  if (absl::holds_alternative<float>(attr.param)) {
    *generated_code = {
        /*parameters=*/{{"scalar", absl::get<float>(attr.param)}},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/"value_0 *= $scalar$;",
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

  if (absl::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(attr.param)) {
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/
        {{"mul_buffer",
          MakeReadonlyObject(
              absl::get<Tensor<Linear, DataType::FLOAT32>>(attr.param).data)}},
        /*shared_variables=*/{},
        // Declare workload explicitly because shader depends on gid.z.
        /*workload=*/
        uint3(static_cast<int>(ctx.input_shapes[0][2]),
              static_cast<int>(ctx.input_shapes[0][1]),
              DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)),
        /*workgroup=*/uint3(),
        /*source_code=*/"value_0 *= $mul_buffer[gid.z]$;",
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

  if (absl::holds_alternative<Tensor<HWC, DataType::FLOAT32>>(attr.param)) {
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/
        {{"hwc_buffer",
          MakeReadonlyObject(
              uint3(static_cast<int>(ctx.input_shapes[0][2]),
                    static_cast<int>(ctx.input_shapes[0][1]),
                    DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)),
              ConvertToPHWC4(
                  absl::get<Tensor<HWC, DataType::FLOAT32>>(attr.param)))}},
        /*shared_variables=*/{},
        // Declare workload explicitly because shader depends on gid.z.
        /*workload=*/
        uint3(static_cast<int>(ctx.input_shapes[0][2]),
              static_cast<int>(ctx.input_shapes[0][1]),
              DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)),
        /*workgroup=*/uint3(),
        /*source_code=*/"value_0 *= $hwc_buffer[gid.x, gid.y, gid.z]$;",
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Unsupported Multiplication case.");
}

class Multiply : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmulDTcc mht_3(mht_3_v, 330, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mul.cc", "GenerateCode");

    if (IsApplyMaskSupported(ctx)) {
      return GenerateApplyMaskCode(ctx, generated_code);
    } else {
      return GenerateMultiplyScalarCode(ctx, generated_code);
    }
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewMultiplyNodeShader() {
  return absl::make_unique<Multiply>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
