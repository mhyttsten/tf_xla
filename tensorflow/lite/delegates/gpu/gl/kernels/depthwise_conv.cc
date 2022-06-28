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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSdepthwise_convDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSdepthwise_convDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSdepthwise_convDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/depthwise_conv.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/ideal_workgroup_picker.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class DepthwiseConvolution : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSdepthwise_convDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/delegates/gpu/gl/kernels/depthwise_conv.cc", "GenerateCode");

    if (ctx.input_shapes.size() != 1) {
      return absl::UnimplementedError(
          "DepthWise Convolution does not support more than 1 runtime tensor");
    }
    const auto& attr =
        absl::any_cast<const DepthwiseConvolution2DAttributes&>(ctx.op_attr);
    auto weights = attr.weights.shape;
    const int offsets_count = weights.h * weights.w;
    const bool offsets_count_too_large = offsets_count > kMaxConstArraySize;
    std::vector<Variable> parameters;
    if (offsets_count_too_large) {
      parameters = {
          {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
          {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])},
          {"padding_w", attr.padding.prepended.w},
          {"padding_h", attr.padding.prepended.h},
          {"dilation_w", attr.dilations.w},
          {"dilation_h", attr.dilations.h},
          {"kernel_w", weights.w},
          {"kernel_h", weights.h},
          {"src_depth", DivideRoundUp(weights.i, 4)},
          {"channel_multiplier", weights.o},
          {"stride", int2(attr.strides.w, attr.strides.h)},
      };
    } else {
      std::vector<int2> offsets;
      for (int h = 0; h < weights.h; ++h) {
        for (int w = 0; w < weights.w; ++w) {
          offsets.emplace_back(w * attr.dilations.w - attr.padding.prepended.w,
                               h * attr.dilations.h - attr.padding.prepended.h);
        }
      }
      parameters = {
          {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
          {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])},
          {"offsets_count", offsets_count},
          {"offsets", offsets},
          {"src_depth", DivideRoundUp(weights.i, 4)},
          {"channel_multiplier", weights.o},
          {"stride", int2(attr.strides.w, attr.strides.h)},
      };
    }
    bool non_empty_padding =
        attr.padding.appended.h != 0 || attr.padding.appended.w != 0 ||
        attr.padding.prepended.h != 0 || attr.padding.prepended.w != 0;

    std::vector<std::pair<std::string, Object>> objects = {
        {"weights", MakeReadonlyObject(ConvertToPIOHW4(attr.weights))}};

    std::string source;
    if (offsets_count_too_large) {
      source = R"(
        int offsets_count = $kernel_w$ * $kernel_h$;
        int src_layer_offset = (gid.z % $channel_multiplier$) * 4;
        int i = 0;
        for (int ky = 0; ky < $kernel_h$; ky++) {
          for (int kx = 0; kx < $kernel_w$; kx++, i++) {
            ivec2 coord = gid.xy * $stride$ + ivec2(kx * $dilation_w$ - $padding_w$, ky * $dilation_h$ - $padding_h$);)";
    } else {
      source = R"(
        int offsets_count = $offsets_count$;
        int src_layer_offset = (gid.z % $channel_multiplier$) * 4;
        for (int i = 0; i < offsets_count; ++i) {
          ivec2 coord = gid.xy * $stride$ + $offsets[i]$;)";
    }
    if (non_empty_padding) {
      source += R"(
        if (coord.x < 0 || coord.y < 0 ||
            coord.x >= $input_data_0_w$ || coord.y >= $input_data_0_h$) {
          continue;
        })";
    }
    source += R"(
        int src_layer = gid.z / $channel_multiplier$;
        vec4 input_ = $input_data_0[coord.x, coord.y, src_layer]$;
        vec4 input_shifted = vec4(
          input_[(src_layer_offset + 0) / $channel_multiplier$],
          input_[(src_layer_offset + 1) / $channel_multiplier$],
          input_[(src_layer_offset + 2) / $channel_multiplier$],
          input_[(src_layer_offset + 3) / $channel_multiplier$]
        );
        value_0 += input_shifted * $weights[gid.z * offsets_count + i]$;
      }
)";
    if (offsets_count_too_large) {
      source += R"(
      }
)";
    }
    if (!attr.bias.data.empty()) {
      source += "value_0 += $bias[gid.z]$;\n";
      objects.push_back({"bias", MakeReadonlyObject(attr.bias.data)});
    }
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/
        GetIdealWorkgroupIfPossible(
            *ctx.gpu_info, OperationType::DEPTHWISE_CONVOLUTION,
            HW(attr.weights.shape.h, attr.weights.shape.w), attr.strides,
            OHWI(attr.weights.shape.o, ctx.input_shapes[0][1],
                 ctx.input_shapes[0][2], ctx.input_shapes[0][3])),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewDepthwiseConvolutionNodeShader() {
  return absl::make_unique<DepthwiseConvolution>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
