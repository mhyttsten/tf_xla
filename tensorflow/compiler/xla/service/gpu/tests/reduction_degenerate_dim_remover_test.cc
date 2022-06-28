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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSreduction_degenerate_dim_remover_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSreduction_degenerate_dim_remover_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSreduction_degenerate_dim_remover_testDTcc() {
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

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

namespace {

class ReductionDegenerateDimRemoverTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSreduction_degenerate_dim_remover_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/gpu/tests/reduction_degenerate_dim_remover_test.cc", "GetDebugOptionsForTest");

    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.add_xla_disable_hlo_passes("reduction-layout-normalizer");
    debug_options.add_xla_disable_hlo_passes("reduction-dimension-grouper");
    debug_options.add_xla_disable_hlo_passes("reduction-splitter");
    debug_options.add_xla_disable_hlo_passes("gpu-tree-reduction-rewriter");
    return debug_options;
  }
};

TEST_F(ReductionDegenerateDimRemoverTest, ReductionWithDegenerateDimensions) {
  const char* hlo_text = R"(
HloModule ReduceWithDegenerateDimensions

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1,3,1,4,1,5,1] parameter(0)
  zero = f32[] constant(0)

  ROOT out = f32[1,1,1,1] reduce(input, zero), dimensions={1,3,5}, to_apply=add
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: f32[] reduce(f32[3,4,5]{2,1,0} {{.+}}, f32[] {{.+}}), dimensions={0,1,2}, to_apply=%add
      )");
}

TEST_F(ReductionDegenerateDimRemoverTest,
       ReductionWithDegenerateDimensionsVariadic) {
  const char* hlo_text = R"(
HloModule ReduceWithDegenerateDimensions

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  input = f32[1,3,1,4,1,5,1] parameter(0)
  idxs = u32[1,3,1,4,1,5,1] parameter(1)
  zero = f32[] constant(0)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[1,1,1,1], u32[1,1,1,1]) reduce(input, idxs, zero, zero_idx), dimensions={1,3,5}, to_apply=argmax
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: (f32[], u32[]) reduce(f32[3,4,5]{2,1,0} %bitcast.5, u32[3,4,5]{2,1,0} %bitcast.4, f32[] %zero_1, u32[] %zero_idx_1), dimensions={0,1,2}
)");
}

TEST_F(ReductionDegenerateDimRemoverTest, DegenerateWithEmptyDimension) {
  const char* hlo_text = R"(
HloModule ReduceWithDegenerateDimensions

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1,3,1,4,1,5,1] parameter(0)
  zero = f32[] constant(0)

  ROOT out = f32[3,4,5,1] reduce(input, zero), dimensions={0,2,4}, to_apply=add
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  // Copy instruction is added after bitcast because of copy-insertion pass,
  // so we check the entire hlo module to verify there is no reduce instruction
  // in this case.
  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: ENTRY %main (input: f32[1,3,1,4,1,5,1]) -> f32[3,4,5,1] {
// CHECK:   %input = f32[1,3,1,4,1,5,1]{6,5,4,3,2,1,0} parameter(0)
// CHECK:   %bitcast{{.+}} = f32[3,4,5,1]{3,2,1,0} bitcast(f32[1,3,1,4,1,5,1]{6,5,4,3,2,1,0} %input)
// CHECK:   ROOT %copy{{.+}} = f32[3,4,5,1]{3,2,1,0} copy(f32[3,4,5,1]{3,2,1,0} %bitcast{{.+}})
      )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
