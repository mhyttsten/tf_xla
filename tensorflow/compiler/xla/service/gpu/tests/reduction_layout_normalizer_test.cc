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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSreduction_layout_normalizer_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSreduction_layout_normalizer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSreduction_layout_normalizer_testDTcc() {
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
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {

namespace {

// TODO(b/210165681): The tests in this file are fragile to HLO op names.

class ReductionLayoutNormalizerTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSreduction_layout_normalizer_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/gpu/tests/reduction_layout_normalizer_test.cc", "GetDebugOptionsForTest");

    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.add_xla_disable_hlo_passes("reduction-dimension-grouper");
    debug_options.add_xla_disable_hlo_passes("reduction-splitter");
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    debug_options.add_xla_disable_hlo_passes("gpu-tree-reduction-rewriter");
    return debug_options;
  }
};

TEST_F(ReductionLayoutNormalizerTest, LayoutCanonicalizerTest) {
  const char* hlo_text = R"(
HloModule ReduceWithLayoutChange

add {
  x0 = f32[] parameter(0)
  y0 = f32[] parameter(1)
  ROOT add0 = f32[] add(x0, y0)
}

ENTRY main {
  arg0 = f32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(0)
  constant0 = f32[] constant(0)
  ROOT reduce0 = f32[4,5,16,12,12]{4,3,2,1,0} reduce(arg0, constant0),
    dimensions={1,6,7}, to_apply=add
}

)";

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: f32[4,12,12,16,5]{2,1,3,4,0} reduce(f32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} {{.+}}, f32[] {{.+}}), dimensions={0,1,2}, to_apply=%add
      )");
}

TEST_F(ReductionLayoutNormalizerTest, LayoutCanonicalizerTestVariadic) {
  const char* hlo_text = R"(
HloModule ReduceWithLayoutChangeVariadic


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
  arg0 = f32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(0)
  idxs = u32[4,5,5,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[4,5,16,12,12]{4,3,2,1,0},
      u32[4,5,16,12,12]{4,3,2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={1,6,7}, to_apply=argmax
}


)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: %reduce.1 = (f32[4,12,12,16,5]{2,1,3,4,0}, u32[4,12,12,16,5]{2,1,3,4,0}) reduce(f32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} %bitcast.5, u32[5,3,3,4,12,12,16,5]{7,6,5,4,3,2,1,0} %bitcast.4, f32[] %constant0_1, u32[] %constant1_1), dimensions={0,1,2}, to_apply=%argmax
//
      )");
}

TEST_F(ReductionLayoutNormalizerTest,
       LayoutCanonicalizerTestVariadicDifferentLayouts) {
  const char* hlo_text = R"(
HloModule ReduceWithLayoutChangeVariadicDifferent

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
  arg0 = f32[2,3,4,7]{2,1,0,3}  parameter(0)
  idxs = u32[2,3,4,7]{3,2,1,0}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[2,3,4]{2,1,0},
      u32[2,3,4]{2,1,0}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}


)";

  MatchOptimizedHloWithShapes(hlo_text,
                              R"(
// CHECK: %fused_computation (param_0.1: u32[2,3,4,7]) -> u32[7,2,3,4] {
// CHECK:  %param_0.1 = u32[2,3,4,7]{3,2,1,0} parameter(0)
// CHECK:  %copy.1 = u32[2,3,4,7]{2,1,0,3} copy(u32[2,3,4,7]{3,2,1,0} %param_0.1)
// CHECK:  ROOT %bitcast.2 = u32[7,2,3,4]{3,2,1,0} bitcast(u32[2,3,4,7]{2,1,0,3} %copy.1)
// CHECK: }
//
// CHECK: ENTRY %main (arg0: f32[2,3,4,7], idxs: u32[2,3,4,7]) -> (f32[2,3,4], u32[2,3,4]) {
// CHECK:  %arg0 = f32[2,3,4,7]{2,1,0,3} parameter(0)
// CHECK:  %bitcast = f32[7,2,3,4]{3,2,1,0} bitcast(f32[2,3,4,7]{2,1,0,3} %arg0)
// CHECK:  %idxs = u32[2,3,4,7]{3,2,1,0} parameter(1)
// CHECK:  %fusion = u32[7,2,3,4]{3,2,1,0} fusion(u32[2,3,4,7]{3,2,1,0} %idxs), kind=kLoop, calls=%fused_computation
// CHECK:  %constant0 = f32[] constant(0)
// CHECK:  %constant1 = u32[] constant(0)
// CHECK:  ROOT %reduce0 = (f32[2,3,4]{2,1,0}, u32[2,3,4]{2,1,0}) reduce(f32[7,2,3,4]{3,2,1,0} %bitcast, u32[7,2,3,4]{3,2,1,0} %fusion, f32[] %constant0, u32[] %constant1), dimensions={0}, to_apply=%argmax
// CHECK: }
      )");
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
