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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSfully_connectedDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSfully_connectedDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSfully_connectedDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/fully_connected.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class FullyConnectedBuffers : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSfully_connectedDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/gpu/gl/kernels/fully_connected.cc", "GenerateCode");

    const auto& attr =
        absl::any_cast<const FullyConnectedAttributes&>(ctx.op_attr);

    const int src_depth = DivideRoundUp(attr.weights.shape.i, 4);
    const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);

    // This shader can work with any workgroup size, the values below work well
    // for OpenGL.
    constexpr int kWorkgroupHintX = 4;
    constexpr int kWorkgroupHintY = 4;

    // TODO(akulik): check that input has h,w == 1,1
    std::vector<Variable> parameters = {
        {"src_depth", src_depth},
        {"dst_depth", dst_depth},
    };

    // TODO(akulik): refactor indexed access to weights.
    std::vector<std::pair<std::string, Object>> objects = {
        {"weights", MakeReadonlyObject(ConvertToPHWO4I4(attr.weights))}};

    std::string source = R"(
  const int threads = int(gl_WorkGroupSize.y);
  const int workers = int(gl_WorkGroupSize.x);
  ivec3 tid = ivec3(gl_LocalInvocationID);

  if (gid.x < $dst_depth$) {
    int offset = 4 * gid.x * $src_depth$ + 4 * tid.y;
    for (int d = tid.y; d < $src_depth$; d += threads, offset += 4 * threads) {
      vec4 src = $input_data_0[0, 0, d]$;
      value_0.x += dot(src, $weights[offset + 0]$);
      value_0.y += dot(src, $weights[offset + 1]$);
      value_0.z += dot(src, $weights[offset + 2]$);
      value_0.w += dot(src, $weights[offset + 3]$);
    }
    sh_mem[workers * tid.y + tid.x] = value_0;
  }
  memoryBarrierShared();
  barrier();

  if (tid.y > 0 || gid.x >= $dst_depth$) {
    return;
  }

  for (int t = 1; t < threads; t++) {
    value_0 += sh_mem[workers * t + tid.x];
  }
)";
    if (!attr.bias.data.empty()) {
      source += "  value_0 += $bias[gid.x]$;\n";
      objects.push_back({"bias", MakeReadonlyObject(attr.bias.data)});
    }
    source += "  $output_data_0[0, 0, gid.x] = value_0$;";

    std::vector<Variable> shared_variables = {
#ifdef __APPLE__
        // MoltenVK has problems with shared memory sized using the workgroup
        // size. Fortunately with Metal a fixed workgroup size of 32 seems to
        // give optimal results.
        {"sh_mem", std::vector<float4>(32)},
#else
        // The actual size of sh_mem depends on the WorkgroupSize
        {"sh_mem", std::vector<float4>(0)},
#endif
    };

    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*shared_variables=*/std::move(shared_variables),
        /*workload=*/uint3(dst_depth, kWorkgroupHintY, 1),
        /*workgroup=*/uint3(kWorkgroupHintX, kWorkgroupHintY, 1),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewFullyConnectedNodeShader() {
  return absl::make_unique<FullyConnectedBuffers>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
