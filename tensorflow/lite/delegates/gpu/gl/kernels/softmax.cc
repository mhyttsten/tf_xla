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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSsoftmaxDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSsoftmaxDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSsoftmaxDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/softmax.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

float4 GetMask(int num_channels) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSsoftmaxDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/delegates/gpu/gl/kernels/softmax.cc", "GetMask");

  float4 mask(0.0f);
  const int remainder = num_channels % 4 == 0 ? 4 : num_channels % 4;
  for (int i = 0; i < remainder; ++i) mask[i] = 1.0f;
  return mask;
}

class Softmax : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSsoftmaxDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/delegates/gpu/gl/kernels/softmax.cc", "GenerateCode");

    const auto& attr = absl::any_cast<const SoftmaxAttributes&>(ctx.op_attr);
    if (ctx.input_shapes[0] != ctx.output_shapes[0]) {
      return absl::InvalidArgumentError(
          "Input and output shapes do not match.");
    }
    if (attr.axis != Axis::CHANNELS) {
      return absl::UnimplementedError(
          "Softmax is only supported for channels axis.");
    }
    return ctx.input_shapes[0][1] == 1 && ctx.input_shapes[0][2] == 1
               ? GenerateCodeFor1x1(ctx, generated_code)
               : GenerateCodeGeneral(ctx, generated_code);
  }

 private:
  absl::Status GenerateCodeFor1x1(const GenerationContext& ctx,
                                  GeneratedCode* generated_code) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSsoftmaxDTcc mht_2(mht_2_v, 237, "", "./tensorflow/lite/delegates/gpu/gl/kernels/softmax.cc", "GenerateCodeFor1x1");

    const int depth = DivideRoundUp(ctx.output_shapes[0][3], 4);
    std::vector<Variable> shared_variables = {
        {"partial_sum", std::vector<float4>(8)},
    };
    std::vector<Variable> uniform_parameters = {
        {"depth", depth},
        {"mask", GetMask(ctx.output_shapes[0][3])},
    };
    std::string source_code = R"(
  highp vec4 kOnes = vec4(1.0);
  int tid = int(gl_LocalInvocationID.x);
  highp vec4 maxx4 = $input_data_0[0, 0, 0]$;
  maxx4.y = maxx4.x;
  maxx4.z = maxx4.x;
  maxx4.w = maxx4.x;
  for (int s = tid; s < $depth$; s += 32) {
    highp vec4 mask_a = s == $depth$ - 1 ? $mask$ : kOnes;
    highp vec4 mask_b = kOnes - mask_a;
    highp vec4 src = $input_data_0[0, 0, s]$;
    src = src * mask_a + mask_b * src.x;
    maxx4 = max(maxx4, src);
  }
  highp float maximum = max(maxx4.x, maxx4.y);
  maximum = max(maximum, maxx4.z);
  maximum = max(maximum, maxx4.w);
  partial_sum[tid / 4][tid % 4] = maximum;

  memoryBarrierShared();
  barrier();

  if (tid == 0) {
    maxx4 = max(partial_sum[0], partial_sum[1]);
    maxx4 = max(maxx4, partial_sum[2]);
    maxx4 = max(maxx4, partial_sum[3]);
    maxx4 = max(maxx4, partial_sum[4]);
    maxx4 = max(maxx4, partial_sum[5]);
    maxx4 = max(maxx4, partial_sum[6]);
    maxx4 = max(maxx4, partial_sum[7]);
    maximum = max(maxx4.x, maxx4.y);
    maximum = max(maximum, maxx4.z);
    maximum = max(maximum, maxx4.w);
    partial_sum[0][0] = maximum;
  }

  memoryBarrierShared();
  barrier();

  maximum = partial_sum[0][0];

  highp float sum = 0.0;
  for (int s = tid; s < $depth$; s += 32) {
    highp vec4 mask_temp = s == $depth$ - 1 ? $mask$ : kOnes;
    highp vec4 src = $input_data_0[0, 0, s]$ - vec4(maximum);
    sum += dot(mask_temp, exp(src));
  }

  memoryBarrierShared();
  barrier();

  partial_sum[tid / 4][tid % 4] = sum;

  memoryBarrierShared();
  barrier();

  if (tid == 0) {
    sum = dot(kOnes, partial_sum[0]);
    sum += dot(kOnes, partial_sum[1]);
    sum += dot(kOnes, partial_sum[2]);
    sum += dot(kOnes, partial_sum[3]);
    sum += dot(kOnes, partial_sum[4]);
    sum += dot(kOnes, partial_sum[5]);
    sum += dot(kOnes, partial_sum[6]);
    sum += dot(kOnes, partial_sum[7]);
    partial_sum[0][0] = 1.0 / sum;
  }

  memoryBarrierShared();
  barrier();

  sum = partial_sum[0][0];

  int dst_s = int(gl_GlobalInvocationID.x);
  if (dst_s < $depth$) {
    highp vec4 src = $input_data_0[0, 0, dst_s]$ - vec4(maximum);
    highp vec4 temp = exp(src) * sum;
    $output_data_0[0, 0, dst_s] = temp$;
  }
)";

    *generated_code = {
        /*parameters=*/std::move(uniform_parameters),
        /*objects=*/{},
        /*shared_variables=*/std::move(shared_variables),
        /*workload=*/uint3(depth, 1, 1),
        /*workgroup=*/uint3(32, 1, 1),
        /*source_code=*/std::move(source_code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return absl::OkStatus();
  }

  absl::Status GenerateCodeGeneral(const GenerationContext& ctx,
                                   GeneratedCode* generated_code) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSsoftmaxDTcc mht_3(mht_3_v, 344, "", "./tensorflow/lite/delegates/gpu/gl/kernels/softmax.cc", "GenerateCodeGeneral");

    std::vector<Variable> parameters = {
        {"src_depth",
         DivideRoundUp(static_cast<int>(ctx.output_shapes[0][3]), 4)},
        {"mask", GetMask(ctx.output_shapes[0][3])},
    };

    std::string source_code = R"(
  highp vec4 kOnes = vec4(1.0);
  highp float sum = 0.0;
  highp float maximum = $input_data_0[gid.x, gid.y, 0]$.x;
  for (int d = 0; d < $src_depth$; ++d) {
    highp vec4 mask_a = d == $src_depth$ - 1 ? $mask$ : kOnes;
    highp vec4 mask_b = kOnes - mask_a;
    highp vec4 src = $input_data_0[gid.x, gid.y, d]$;
    src = src * mask_a + mask_b * src.x;
    maximum = max(maximum, src.x);
    maximum = max(maximum, src.y);
    maximum = max(maximum, src.z);
    maximum = max(maximum, src.w);
  }
  for (int d = 0; d < $src_depth$; ++d) {
    highp vec4 mask_temp = d == $src_depth$ - 1 ? $mask$ : kOnes;
    highp vec4 src = $input_data_0[gid.x, gid.y, d]$ - vec4(maximum);
    sum += dot(mask_temp, exp(src));
  }
  for (int d = 0; d < $src_depth$; ++d) {
    highp vec4 src = $input_data_0[gid.x, gid.y, d]$ - vec4(maximum);
    highp vec4 temp_sum = exp(src) / sum;
    $output_data_0[gid.x, gid.y, d] = temp_sum$;
  }
)";
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/
        uint3(static_cast<int>(ctx.output_shapes[0][2]),
              static_cast<int>(ctx.output_shapes[0][1]), 1),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source_code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewSoftmaxNodeShader() {
  return absl::make_unique<Softmax>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
