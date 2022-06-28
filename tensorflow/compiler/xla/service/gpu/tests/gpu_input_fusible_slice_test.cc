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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_input_fusible_slice_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_input_fusible_slice_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_input_fusible_slice_testDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuSliceInputFusionTest : public GpuCodegenTest {
 protected:
  GpuSliceInputFusionTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_input_fusible_slice_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/gpu/tests/gpu_input_fusible_slice_test.cc", "GpuSliceInputFusionTest");
}

  HloModuleConfig ConfigWithoutLayoutAssignment() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_input_fusible_slice_testDTcc mht_1(mht_1_v, 204, "", "./tensorflow/compiler/xla/service/gpu/tests/gpu_input_fusible_slice_test.cc", "ConfigWithoutLayoutAssignment");

    HloModuleConfig config;
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    // Disable the layout_assignment pass to use the preassigned layouts;
    // otherwise, the pass throws away the layouts in the fusion computation.
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    config.set_debug_options(debug_options);
    return config;
  }
};

TEST_F(GpuSliceInputFusionTest, InputFusionWithATupleOfSlices) {
  const char *const kHloString = R"(
  HloModule input_fusion_with_a_tuple_of_slices

  fused_computation {
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1024,512]{1,0} parameter(1)
    mul.1 = f16[1024,512]{1,0} multiply(arg.1, arg.2)
    add.1 = f16[1024,512]{1,0} add(mul.1, arg.2)
    slice.1 = f16[512,511]{1,0} slice(arg.1), slice={[512:1024], [1:512]}
    slice.2 = f16[0,512]{1,0} slice(add.1), slice={[512:512], [0:512]}
    slice.3 = f16[1,1]{1,0} slice(add.1), slice={[512:513], [511:512]}
    ROOT tuple.1 = (f16[512,511]{1,0}, f16[0,512]{1,0}, f16[1,1]{1,0})
        tuple(slice.1, slice.2, slice.3)
  }

  ENTRY kernel_entry {
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1024,512]{1,0} parameter(1)
    ROOT fusion = (f16[512,511]{1,0}, f16[0,512]{1,0}, f16[1,1]{1,0})
        fusion(arg.1, arg.2), kind=kInput, calls=fused_computation
  })";

  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @fusion
; CHECK: slice2
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @fusion
; CHECK: slice2
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/false);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0, 0}));
}

TEST_F(GpuSliceInputFusionTest, ConcatThenSplit) {
  const char *const kHloString = R"(
  HloModule input_fusion_with_a_tuple_of_slices

  fused_computation {
    arg.1 = f16[1024]{0} parameter(0)
    arg.2 = f16[1024]{0} parameter(1)
    arg.3 = f16[1023]{0} parameter(2)
    arg.4 = f16[1023]{0} parameter(3)
    mul.1 = f16[1024]{0} multiply(arg.1, arg.2)
    add.1 = f16[1023]{0} add(arg.3, arg.4)
    concat.1 = f16[2047]{0} concatenate(mul.1, add.1), dimensions={0}
    slice.1 = f16[1024]{0} slice(concat.1), slice={[0:1024]}
    slice.2 = f16[1023]{0} slice(concat.1), slice={[1024:2047]}
    slice.3 = f16[0]{0} slice(concat.1), slice={[2047:2047]}
    ROOT tuple.1 = (f16[1024]{0}, f16[1023]{0}, f16[0]{0})
        tuple(slice.1, slice.2, slice.3)
  }

  ENTRY kernel_entry {
    arg.1 = f16[1024]{0} parameter(0)
    arg.2 = f16[1024]{0} parameter(1)
    arg.3 = f16[1023]{0} parameter(2)
    arg.4 = f16[1023]{0} parameter(3)
    ROOT fusion = (f16[1024]{0}, f16[1023]{0}, f16[0]{0})
        fusion(arg.1, arg.2, arg.3, arg.4), kind=kInput, calls=fused_computation
  })";

  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @fusion
; CHECK: slice2
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @fusion
; CHECK: slice2
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/false);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0, 0}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
