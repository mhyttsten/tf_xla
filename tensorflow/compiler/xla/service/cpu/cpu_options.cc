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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_optionsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_optionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_optionsDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace {

const char* const kXlaOptimizeForSizeCpuOption = "xla_cpu_optimize_for_size";
const char* const kLlvmIrDotTilingFactor = "xla_llvm_dot_tiling_factor";
const char* const kXlaForceEnableExperimentalLlvmIrGemm =
    "xla_force_enable_experimental_llvm_ir_gemm";
const char* const kLlvmIrGemmTileSize = "xla_llvm_ir_gemm_tile_size";

}  // namespace

namespace xla {
namespace cpu {
namespace options {

bool OptimizeForSizeRequested(const HloModuleConfig& config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_optionsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/cpu/cpu_options.cc", "OptimizeForSizeRequested");

  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kXlaOptimizeForSizeCpuOption) > 0;
}

bool VectorizedReduceDisabled(const HloModuleConfig& config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_optionsDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/xla/service/cpu/cpu_options.cc", "VectorizedReduceDisabled");

  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kXlaOptimizeForSizeCpuOption) > 0;
}

absl::optional<int64_t> LlvmIrGemvTilingFactor(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  auto it = extra_options_map.find(kLlvmIrDotTilingFactor);
  int64_t tiling_factor;
  if (it != extra_options_map.end() &&
      absl::SimpleAtoi(it->second, &tiling_factor)) {
    return tiling_factor;
  }
  return absl::nullopt;
}

bool ForceEnableExperimentalLlvmIrGemm(const HloModuleConfig& config) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_optionsDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/xla/service/cpu/cpu_options.cc", "ForceEnableExperimentalLlvmIrGemm");

  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kXlaForceEnableExperimentalLlvmIrGemm) > 0;
}

static absl::string_view RemoveSuffix(absl::string_view str,
                                      absl::string_view suffix) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   mht_3_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_optionsDTcc mht_3(mht_3_v, 247, "", "./tensorflow/compiler/xla/service/cpu/cpu_options.cc", "RemoveSuffix");

  CHECK_GE(str.size(), suffix.size());
  CHECK_EQ(str.substr(str.size() - suffix.size()), suffix);
  return str.substr(0, str.size() - suffix.size());
}

absl::optional<std::tuple<int64_t, int64_t, int64_t>> LlvmIrGemmTileSize(
    const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  auto it = extra_options_map.find(kLlvmIrGemmTileSize);
  if (it == extra_options_map.end()) {
    return absl::nullopt;
  }

  std::vector<std::string> tile_components = absl::StrSplit(it->second, ':');
  CHECK_EQ(tile_components.size(), 3);

  int64_t tile_size_m;
  int64_t tile_size_k;
  int64_t tile_size_n_in_vector_width;

  CHECK(absl::SimpleAtoi(tile_components[0], &tile_size_m));
  CHECK(absl::SimpleAtoi(tile_components[1], &tile_size_k));

  absl::string_view tile_size_n_in_vector_width_str =
      RemoveSuffix(tile_components[2], "*vectwidth");

  CHECK(absl::SimpleAtoi(tile_size_n_in_vector_width_str,
                         &tile_size_n_in_vector_width));

  return std::tuple<int64_t, int64_t, int64_t>(tile_size_m, tile_size_k,
                                               tile_size_n_in_vector_width);
}

}  // namespace options
}  // namespace cpu
}  // namespace xla
