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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSelementwiseDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSelementwiseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSelementwiseDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/elementwise.h"

#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ElementwiseOneArgument : public NodeShader {
 public:
  explicit ElementwiseOneArgument(OperationType operation_type)
      : operation_type_(operation_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSelementwiseDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/delegates/gpu/gl/kernels/elementwise.cc", "ElementwiseOneArgument");
}
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSelementwiseDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/delegates/gpu/gl/kernels/elementwise.cc", "GenerateCode");

    std::string source;
    switch (operation_type_) {
      case OperationType::ABS:
        source = "value_0 = abs(value_0);";
        break;
      case OperationType::COS:
        source = "value_0 = cos(value_0);";
        break;
      case OperationType::COPY:
        source = "value_0 = value_0;";
        break;
      case OperationType::ELU:
        source = R"(
            value_0.x = value_0.x < 0.0 ? exp(value_0.x) - 1.0 : value_0.x;
            value_0.y = value_0.y < 0.0 ? exp(value_0.y) - 1.0 : value_0.y;
            value_0.z = value_0.z < 0.0 ? exp(value_0.z) - 1.0 : value_0.z;
            value_0.w = value_0.w < 0.0 ? exp(value_0.w) - 1.0 : value_0.w;
        )";
        break;
      case OperationType::EXP:
        source = "value_0 = exp(value_0);";
        break;
      case tflite::gpu::OperationType::FLOOR:
        source = "value_0 = floor(value_0);";
        break;
      case OperationType::HARD_SWISH:
        source =
            "value_0 *= clamp(value_0 / 6.0 + vec4(0.5), vec4(0.0), "
            "vec4(1.0));";
        break;
      case OperationType::LOG:
        source = R"(
            const float nan = normalize(vec4(0, 0, 0, 0)).x;
            value_0.x = value_0.x > 0.0 ? log(value_0.x) : nan;
            value_0.y = value_0.y > 0.0 ? log(value_0.y) : nan;
            value_0.z = value_0.z > 0.0 ? log(value_0.z) : nan;
            value_0.w = value_0.w > 0.0 ? log(value_0.w) : nan;
        )";
        break;
      case OperationType::NEG:
        source = "value_0 = -(value_0);";
        break;
      case OperationType::RSQRT:
        source = R"(
            const float nan = normalize(vec4(0, 0, 0, 0)).x;
            value_0.x = value_0.x > 0.0 ? 1.0 / sqrt(value_0.x) : nan;
            value_0.y = value_0.y > 0.0 ? 1.0 / sqrt(value_0.y) : nan;
            value_0.z = value_0.z > 0.0 ? 1.0 / sqrt(value_0.z) : nan;
            value_0.w = value_0.w > 0.0 ? 1.0 / sqrt(value_0.w) : nan;
        )";
        break;
      case OperationType::SIGMOID:
        source = "value_0 = 1.0 / (1.0 + exp(-1.0 * value_0));";
        break;
      case OperationType::SIN:
        source = "value_0 = sin(value_0);";
        break;
      case OperationType::SQRT:
        source = R"(
            const float nan = normalize(vec4(0, 0, 0, 0)).x;
            value_0.x = value_0.x >= 0.0 ? sqrt(value_0.x) : nan;
            value_0.y = value_0.y >= 0.0 ? sqrt(value_0.y) : nan;
            value_0.z = value_0.z >= 0.0 ? sqrt(value_0.z) : nan;
            value_0.w = value_0.w >= 0.0 ? sqrt(value_0.w) : nan;
        )";
        break;
      case OperationType::SQUARE:
        source = "value_0 = value_0 * value_0;";
        break;
      case OperationType::TANH:
        source = "value_0 = tanh(value_0);";
        break;
      default:
        return absl::InvalidArgumentError(
            "Incorrect elementwise operation type.");
    }
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        source,
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

 private:
  OperationType operation_type_;
};

class ElementwiseTwoArguments : public NodeShader {
 public:
  explicit ElementwiseTwoArguments(OperationType operation_type)
      : operation_type_(operation_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSelementwiseDTcc mht_2(mht_2_v, 307, "", "./tensorflow/lite/delegates/gpu/gl/kernels/elementwise.cc", "ElementwiseTwoArguments");
}

  inline bool IsElementwiseSupported(const GenerationContext& ctx) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSelementwiseDTcc mht_3(mht_3_v, 312, "", "./tensorflow/lite/delegates/gpu/gl/kernels/elementwise.cc", "IsElementwiseSupported");

    return ctx.input_shapes.size() == 2 &&
           ctx.input_shapes[0] == ctx.input_shapes[1];
  }

  inline bool IsBroadcastSupported(const GenerationContext& ctx) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSelementwiseDTcc mht_4(mht_4_v, 320, "", "./tensorflow/lite/delegates/gpu/gl/kernels/elementwise.cc", "IsBroadcastSupported");

    return ctx.input_shapes.size() == 2 && ctx.input_shapes[1][1] == 1 &&
           ctx.input_shapes[1][2] == 1 &&
           ctx.input_shapes[0][3] == ctx.input_shapes[1][3];
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSelementwiseDTcc mht_5(mht_5_v, 330, "", "./tensorflow/lite/delegates/gpu/gl/kernels/elementwise.cc", "GenerateCode");

    std::vector<Variable> parameters;
    std::vector<std::pair<std::string, Object>> objects;
    std::string argument0, argument1;
    if (IsElementwiseSupported(ctx)) {
      argument0 = "value_0";
      argument1 = "value_1";
    } else if (IsBroadcastSupported(ctx)) {
      argument0 = "$input_data_0[gid.x, gid.y, gid.z]$";
      argument1 = "$input_data_1[0, 0, gid.z]$";
    } else {  // Scalar of const vector case
      const auto& attr =
          absl::any_cast<const ElementwiseAttributes&>(ctx.op_attr);
      const auto* tensor =
          absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.param);
      const auto* scalar = absl::get_if<float>(&attr.param);
      if (!tensor && !scalar) {
        return absl::InvalidArgumentError(
            "Couldn't read scalar of const vector data from the attributes.");
      }

      argument0 = "value_0";
      if (tensor) {
        argument1 = "$const_data[gid.z]$";
        objects.push_back({"const_data", MakeReadonlyObject(tensor->data)});
      } else {
        argument1 = "vec4($const_data$)";
        parameters.push_back({"const_data", *scalar});
      }
    }

    std::string source;
    switch (operation_type_) {
      case OperationType::DIV: {
        source = "value_0 = $0/$1;";
        break;
      }
      case tflite::gpu::OperationType::FLOOR_DIV:
        source = "value_0 = floor($0 / $1);";
        break;
      case tflite::gpu::OperationType::FLOOR_MOD:
        source = "value_0 = $0 - floor($0 / $1) * $1;";
        break;
      case OperationType::MAXIMUM: {
        source = "value_0 = max($0, $1);";
        break;
      }
      case OperationType::MINIMUM: {
        source = "value_0 = min($0, $1);";
        break;
      }
      case OperationType::SQUARED_DIFF: {
        source = "value_0 = ($0 - $1) * ($0 - $1);";
        break;
      }
      case OperationType::SUB: {
        source = "value_0 = $0 - $1;";
        break;
      }
      case OperationType::POW: {
        source = "value_0 = pow($0, $1);";
        break;
      }
      default:
        return absl::InvalidArgumentError(
            "Incorrect elementwise with scalar operation type.");
    }
    source = absl::Substitute(source, argument0, argument1);
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/source,
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

 private:
  OperationType operation_type_;
};

}  // namespace

std::unique_ptr<NodeShader> NewElementwiseNodeShader(
    OperationType operation_type) {
  switch (operation_type) {
    case OperationType::ABS:
    case OperationType::COS:
    case OperationType::COPY:
    case OperationType::ELU:
    case OperationType::EXP:
    case OperationType::FLOOR:
    case OperationType::HARD_SWISH:
    case OperationType::LOG:
    case OperationType::NEG:
    case OperationType::RSQRT:
    case OperationType::SIGMOID:
    case OperationType::SIN:
    case OperationType::SQRT:
    case OperationType::SQUARE:
    case OperationType::TANH:
      return absl::make_unique<ElementwiseOneArgument>(operation_type);
    case OperationType::DIV:
    case OperationType::FLOOR_DIV:
    case OperationType::FLOOR_MOD:
    case OperationType::MAXIMUM:
    case OperationType::MINIMUM:
    case OperationType::POW:
    case OperationType::SQUARED_DIFF:
    case OperationType::SUB:
      return absl::make_unique<ElementwiseTwoArguments>(operation_type);
    default:
      return nullptr;
  }
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
