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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmeanDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmeanDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmeanDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/mean.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

bool UseSubgroupBasedImpl(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmeanDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mean.cc", "UseSubgroupBasedImpl");

  return gpu_info.IsApiVulkan() &&
         (gpu_info.vulkan_info.api_version_major > 1 ||
          gpu_info.vulkan_info.api_version_minor >= 1) &&
         gpu_info.vulkan_info.subgroup_size >= 32 &&
         gpu_info.vulkan_info.supports_subgroup_arithmetic;
}

// An implementation of Mean for desktop GPUs and some phones with recent
// Vulkan drivers. It is more parallel than the trivial Mean operation, but
// still limited to using a single work group.
void GenerateSubgroupBasedMean(const NodeShader::GenerationContext& ctx,
                               GeneratedCode* generated_code) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmeanDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mean.cc", "GenerateSubgroupBasedMean");

  int height = ctx.input_shapes[0][1];
  int width = ctx.input_shapes[0][2];
  int depth = ctx.input_shapes[0][3];
  std::vector<Variable> parameters = {
      {"input_data_0_h", height},
      {"input_data_0_w", width},
      {"output_data_0_h", 1},
      {"output_data_0_w", 1},
  };

  std::string source = R"(
  // Round columns and rows per invocation up, to ensure that we read the
  // entire input.
  const uint columns_per_invocation =
      ($input_data_0_w$ + (gl_WorkGroupSize.x - 1))/gl_WorkGroupSize.x;
  const uint rows_per_invocation =
      ($input_data_0_h$ + (gl_WorkGroupSize.y - 1))/gl_WorkGroupSize.y;
  const uint first_row = gl_GlobalInvocationID.y*rows_per_invocation;
  const uint first_col = gl_GlobalInvocationID.x*columns_per_invocation;
  const uint last_row_exclusive =
      min(first_row+rows_per_invocation, $input_data_0_h$);
  const uint last_column_exclusive =
      min(first_col+columns_per_invocation, $input_data_0_w$);
  vec4 value = vec4(0);
  for (uint h = first_row; h < last_row_exclusive; ++h) {
    for (uint w = first_col; w < last_column_exclusive; ++w) {
      value += $input_data_0[w, h, gid.z]$;
    }
  }
  highp vec4 subgroup_sum = subgroupAdd(value);
  if(subgroupElect()) {
    subgroup_sums[gl_SubgroupID] = subgroup_sum;
  }

  memoryBarrierShared();
  barrier();
  // Do the final reduction in the first subgroup.
  if(gl_SubgroupID == 0) {
    highp vec4 subtotal = vec4(0);
    if (gl_SubgroupInvocationID < gl_NumSubgroups) {
      subtotal = subgroup_sums[gl_SubgroupInvocationID];
    }
    highp vec4 grand_total = subgroupAdd(subtotal);
    if(subgroupElect()) {
      highp vec4 result = grand_total / $input_data_0_w$ / $input_data_0_h$;
      $output_data_0[0, 0, gid.z] = result$;
    }
  }
  )";

  const uint32_t subgroup_size = ctx.gpu_info->vulkan_info.subgroup_size;
  const uint32_t max_wg_size_x = ctx.gpu_info->GetMaxWorkGroupSizeForX();
  const uint32_t max_wg_size_y = ctx.gpu_info->GetMaxWorkGroupSizeForY();
  // Due to the design of the shader, at most subgroup_size subgroups can be
  // launched. This may limit the maximal workgroup size.
  const uint32_t max_wg_size =
      std::min(static_cast<uint32_t>(ctx.gpu_info->GetMaxWorkGroupTotalSize()),
               subgroup_size * subgroup_size);
  const uint32_t max_number_of_subgroups = max_wg_size / subgroup_size;
  uint32_t wg_size_x = 0;
  uint32_t wg_size_y = 0;
  if (width * height <= max_wg_size && width <= max_wg_size_x &&
      height <= max_wg_size_y) {
    wg_size_x = width;
    wg_size_y = height;
  } else {
    // Approximately square workgroup. Also make sure to limit by driver limit
    // and input size.
    wg_size_x = std::min({static_cast<uint32_t>(std::sqrt(max_wg_size)),
                          max_wg_size_x, static_cast<uint32_t>(width)});
    wg_size_y = std::min({max_wg_size / wg_size_x, max_wg_size_y,
                          static_cast<uint32_t>(height)});
  }

  std::vector<Variable> shared_variables = {
      {"subgroup_sums", std::vector<float4>(max_number_of_subgroups)},
  };

  *generated_code = {
      /*parameters=*/std::move(parameters),
      /*objects=*/{},
      /*shared_variables=*/{std::move(shared_variables)},
      // Make sure we get one dispatch of size wg_size_x*wg_size_y*1 per layer.
      /*workload=*/
      uint3(wg_size_x, wg_size_y, uint32_t(DivideRoundUp(depth, 4))),
      /*workgroup=*/uint3(wg_size_x, wg_size_y, 1u),
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::ONLY_DEFINITIONS,
  };
}

void GenerateTrivialMean(const NodeShader::GenerationContext& ctx,
                         GeneratedCode* generated_code) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmeanDTcc mht_2(mht_2_v, 316, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mean.cc", "GenerateTrivialMean");

  std::vector<Variable> parameters = {
      {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
      {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])}};

  // Shaders may be compiled with a precision hint mediump, which means that
  // GLSL compiler may drop the size of float data type from 32 to 16 bits.
  // If "sum" and "size" variables are 16bit floats, their values range
  // become not enough for providing a good results accuracy. That is why
  // their precision is forced to be 32bit by using highp qualifier.
  std::string source = R"(
    highp vec4 sum = vec4(0.0);
    highp float size = float($input_data_0_w$ * $input_data_0_h$);
    for (int w = 0; w < $input_data_0_w$; w++) {
      for (int h = 0; h < $input_data_0_h$; h++) {
        sum += $input_data_0[w, h, gid.z]$;
      }
    }
    value_0 = sum / size;
  )";
  *generated_code = {
      /*parameters=*/std::move(parameters),
      /*objects=*/{},
      /*shared_variables=*/{},
      /*workload=*/uint3(),
      /*workgroup=*/uint3(1, 1, 4),
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::AUTO,
  };
}

// Tiled implementation.

constexpr uint3 kTileSize = {8, 8, 1};

inline bool UseTiledImpl(const NodeShader::GenerationContext& ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmeanDTcc mht_3(mht_3_v, 355, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mean.cc", "UseTiledImpl");

  const int h = ctx.input_shapes[0][1];
  const int w = ctx.input_shapes[0][2];
  const int c = ctx.input_shapes[0][3];
  return h % kTileSize.y == 0 && w % kTileSize.x == 0 && c % 4 == 0 &&
         (h / kTileSize.y) * (w / kTileSize.x) * c * sizeof(float) <=
             32768;  // required min value for GL_MAX_COMPUTE_SHARED_MEMORY_SIZE
}

void GenerateTiledMean(const NodeShader::GenerationContext& ctx,
                       GeneratedCode* generated_code) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmeanDTcc mht_4(mht_4_v, 368, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mean.cc", "GenerateTiledMean");

  const int h = ctx.input_shapes[0][1];
  const int w = ctx.input_shapes[0][2];
  const int s = DivideRoundUp(ctx.input_shapes[0][3], 4);

  std::vector<Variable> parameters = {
      {"input_data_0_h", h},
      {"input_data_0_w", w},
      {"tile_size_h", kTileSize.y},
      {"tile_size_w", kTileSize.x},
  };

  std::vector<Variable> shared_variables = {
      {"tile_sum",
       std::vector<float4>((w / kTileSize.x) * (h / kTileSize.y) * s)}};

  std::string source = R"(
  ivec2 tile_size = ivec2($tile_size_w$, $tile_size_h$);
  ivec2 num_tiles = ivec2($input_data_0_w$, $input_data_0_h$) / tile_size;

  highp vec4 partial_sum = vec4(0.0);
  for (int x = gid.x * tile_size.x; x < (gid.x + 1) * tile_size.x; ++x) {
    for (int y = gid.y * tile_size.y; y < (gid.y + 1) * tile_size.y; ++y) {
      partial_sum += $input_data_0[x, y, gid.z]$;
    }
  }
  $tile_sum$[num_tiles.x * num_tiles.y * gid.z + num_tiles.x * gid.y + gid.x] = partial_sum;

  memoryBarrierShared(); barrier();

  if (gid.x == 0 && gid.y == 0) {
    highp vec4 sum = vec4(0.0);
    for (int i = 0; i < num_tiles.x * num_tiles.y; ++i) {
      sum += $tile_sum$[num_tiles.x * num_tiles.y * gid.z + i];
    }
    highp vec4 mean = sum / float($input_data_0_w$ * $input_data_0_h$);
    $output_data_0[0, 0, gid.z] = mean$;
  }
)";
  *generated_code = {
      /*parameters=*/std::move(parameters),
      /*objects=*/{},
      /*shared_variables=*/std::move(shared_variables),
      /*workload=*/uint3(kTileSize.x, kTileSize.y, static_cast<uint32_t>(s)),
      /*workgroup=*/kTileSize,
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::ONLY_DEFINITIONS,
  };
}

class Mean : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSmeanDTcc mht_5(mht_5_v, 425, "", "./tensorflow/lite/delegates/gpu/gl/kernels/mean.cc", "GenerateCode");

    const auto& attr = absl::any_cast<const MeanAttributes&>(ctx.op_attr);
    if (attr.dims != std::set<Axis>({Axis::HEIGHT, Axis::WIDTH})) {
      return absl::InvalidArgumentError(
          "Mean calculation is supported only for height and width.");
    }

    if (!(ctx.input_shapes.size() == 1 && ctx.output_shapes.size() == 1 &&
          ctx.output_shapes[0][1] == 1 && ctx.output_shapes[0][2] == 1 &&
          ctx.output_shapes[0][3] == ctx.input_shapes[0][3])) {
      return absl::InvalidArgumentError(
          "Mean calculation is supported for one input and one 1x1 output with "
          "the same channel count.");
    }

    if (UseSubgroupBasedImpl(*ctx.gpu_info)) {
      GenerateSubgroupBasedMean(ctx, generated_code);
    } else if (UseTiledImpl(ctx)) {
      GenerateTiledMean(ctx, generated_code);
    } else {
      GenerateTrivialMean(ctx, generated_code);
    }
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewMeanNodeShader() {
  return absl::make_unique<Mean>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
