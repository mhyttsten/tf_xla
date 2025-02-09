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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_kernel_tiling_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_kernel_tiling_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_kernel_tiling_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuKernelTilingTest : public GpuCodegenTest {
 protected:
  GpuKernelTilingTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_kernel_tiling_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/service/gpu/tests/gpu_kernel_tiling_test.cc", "GpuKernelTilingTest");
}

  // Most tests in this file want to skip layout assignment, but a few need it
  // enabled.
  HloModuleConfig ConfigWithLayoutAssignment() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_kernel_tiling_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/xla/service/gpu/tests/gpu_kernel_tiling_test.cc", "ConfigWithLayoutAssignment");

    return GetModuleConfigForTest();
  }

  HloModuleConfig ConfigWithoutLayoutAssignment() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSgpu_kernel_tiling_testDTcc mht_2(mht_2_v, 214, "", "./tensorflow/compiler/xla/service/gpu/tests/gpu_kernel_tiling_test.cc", "ConfigWithoutLayoutAssignment");

    HloModuleConfig config;
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    // Disable layout_assignment to use the preassigned layouts.
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    config.set_debug_options(debug_options);
    return config;
  }
};

TEST_F(GpuKernelTilingTest, UnnestedTransposeWithProperDimensionsTiled) {
  const char *const kHloString = R"(
    HloModule unnested_transpose_1

    ENTRY unnested_transpose_1 {
      para0 = f16[32,3,64]{2,1,0} parameter(0)
      ROOT copy1 = f16[32,3,64]{1,0,2} copy(para0)
    })";

  // Check that a call to llvm.nvvm.barrier0 is generated.
  //
  // We must enable layout assignment in order for this test to work correctly.
  // AlgebraicSimplifier removes copy1; it's added back by layout assignment,
  // which respects the module's entry computation layout.  But if we don't run
  // layout assignment...well, nobody else adds the copy back.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithLayoutAssignment())
          .ValueOrDie();

  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @copy
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @copy
; CHECK: call void @llvm.nvvm.barrier0()
; CHECK: }
)";

  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest, UnnestedTransposeWithSmallDimensionsNotTiled) {
  const char *const kHloString = R"(
    HloModule unnested_transpose_2

    ENTRY unnested_transpose_2 {
      para0 = f16[2,3,64]{2,1,0} parameter(0)
      ROOT copy1 = f16[2,3,64]{1,0,2} copy(para0)
    })";

  // Check that a call to llvm.nvvm.barrier0 is not generated.  As in
  // UnnestedTransposeWithProperDimensionsTiled, we must run layout assignment
  // here.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @copy
; CHECK-NOT: call void @llvm.amdgcn.s.barrier()
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @copy
; CHECK-NOT: call void @llvm.nvvm.barrier0()
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuKernelTilingTest, UnnestedTransposeC128TypeRun) {
  const char *const kHloString = R"(
    HloModule unnested_transpose_3

    ENTRY unnested_transpose_3 {
      para0 = c128[65,65]{1,0} parameter(0)
      ROOT copy1 = c128[65,65]{0,1} copy(para0)
    })";

  // With the current implementation for the available hardwares, we bail out
  // from the tiled transpose implementation at the last minute. Instead of
  // checking the transpose is not tiled, we only check the module compiled and
  // run in this test.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest, SimpleFusionWithTransposeTiled) {
  const char *const kHloString = R"(
    HloModule multiple_output_fusion_1
    fused_computation.1 {
      param0 = f32[4,5,6,7,8]{4,3,2,1,0} parameter(0)
      copy = f32[4,5,6,7,8]{2,1,4,3,0} copy(param0)
      ROOT convert = f16[4,5,6,7,8]{2,1,4,3,0} convert(copy)
    }

    ENTRY copy_in_fusion_run_without_hlo_passes {
      para0 = f32[4,5,6,7,8]{4,3,2,1,0} parameter(0)
      ROOT fusion.1 = f16[4,5,6,7,8]{2,1,4,3,0} fusion(para0), kind=kLoop,
        calls=fused_computation.1
    })";

  // Check that a call to llvm.nvvm.barrier0 is generated.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @fusion
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @fusion
; CHECK: call void @llvm.nvvm.barrier0()
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest, MultipleOutputFusionWithOnePossibleTransposeTiled) {
  const char *const kHloString = R"(
    HloModule multiple_output_fusion_1
    fused_computation.1 {
      param0 = f16[8,31,31,65]{3,2,1,0} parameter(0)
      param1 = f16[8,31,31,65]{3,2,1,0} parameter(1)
      copy0 = f16[8,31,31,65]{2,1,3,0} copy(param0)
      copy1 = f16[8,31,31,65]{2,1,3,0} copy(param1)
      ROOT tuple1 = (f16[8,31,31,65]{2,1,3,0}, f16[8,31,31,65]{2,1,3,0})
        tuple(copy0, copy1)
    }

    ENTRY multiple_output_fusion_1 {
      para0 = f16[8,31,31,65]{3,2,1,0} parameter(0)
      para1 = f16[8,31,31,65]{3,2,1,0} parameter(1)
      ROOT fusion.1 = (f16[8,31,31,65]{2,1,3,0}, f16[8,31,31,65]{2,1,3,0})
        fusion(para0,para1), kind=kLoop, calls=fused_computation.1
    })";

  // Check that a call to llvm.nvvm.barrier0 is generated.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @fusion
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @fusion
; CHECK: call void @llvm.nvvm.barrier0()
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest,
       MultipleOutputFusionWithTwoPossibleTransposesNotTiled) {
  const char *const kHloString = R"(
    HloModule multiple_output_fusion_2
    fused_computation.1 {
      param0 = f16[8,31,31,65]{3,2,1,0} parameter(0)
      param1 = f16[8,31,31,65]{1,3,2,0} parameter(1)
      copy2 = f16[8,31,31,65]{2,1,3,0} copy(param0)
      copy3 = f16[8,31,31,65]{2,1,3,0} copy(param1)
      ROOT tuple1 = (f16[8,31,31,65]{2,1,3,0}, f16[8,31,31,65]{2,1,3,0})
        tuple(copy2, copy3)
    }

    ENTRY multiple_output_fusion_2 {
      para0 = f16[8,31,31,65]{3,2,1,0} parameter(0)
      para1 = f16[8,31,31,65]{1,3,2,0} parameter(1)
      ROOT fusion1 = (f16[8,31,31,65]{2,1,3,0}, f16[8,31,31,65]{2,1,3,0})
        fusion(para0,para1), kind=kLoop, calls=fused_computation.1
    })";

  // Check that a call to llvm.nvvm.barrier0 is not generated.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @fusion
; CHECK-NOT: call void @llvm.amdgcn.s.barrier()
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @fusion
; CHECK-NOT: call void @llvm.nvvm.barrier0()
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuKernelTilingTest, TransposedInputWithUserReverseNotTiled) {
  const char *const kHloString = R"(
    HloModule FusionTransposeWithReverseNotTiled
    fused_computation.1 {
      arg0 = f32[128,64]{1,0} parameter(0)
      copy0 = f32[128,64]{0,1} copy(arg0)
      ROOT reverse0 = f32[128,64]{0,1} reverse(copy0), dimensions={0}
    }

    ENTRY reverse_break_assumption {
      param0 = f32[128,64]{1,0} parameter(0)
      ROOT fusion0 = f32[128,64]{0,1} fusion(param0), kind=kLoop,
        calls=fused_computation.1
    })";

  // Check that a call to llvm.nvvm.barrier0 is not generated.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @fusion
; CHECK-NOT: call void @llvm.amdgcn.s.barrier()
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @fusion
; CHECK-NOT: call void @llvm.nvvm.barrier0()
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuKernelTilingTest, TransposedInputWithUserBitcastNotTiled) {
  const char *const kHloString = R"(
    HloModule TransposedInputWithUserBitcast

    fused_computation {
      param_0 = f32[20,20]{1,0} parameter(0)
      ROOT bitcast = f32[20,20]{0,1} bitcast(param_0)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[20,20]{1,0} parameter(0)
      ROOT fusion = f32[20,20]{0,1} fusion(parameter.0),
        kind=kLoop, calls=fused_computation
    })";

  // Check that a call to llvm.nvvm.barrier0 is not generated.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @fusion
; CHECK-NOT: call void @llvm.amdgcn.s.barrier()
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @fusion
; CHECK-NOT: call void @llvm.nvvm.barrier0()
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest, TransposedInputWithoutUnsafeUseTiled) {
  const char *const kHloString = R"(
    HloModule TwoTransposedInputs

    fused_computation {
      param_0 = f32[64,64]{1,0} parameter(0)
      param_1 = f32[64,64]{1,0} parameter(1)
      bitcast = f32[64,64]{0,1} bitcast(param_0)
      copy = f32[64,64]{0,1} copy(param_1)
      ROOT tuple = (f32[64,64]{0,1}, f32[64,64]{0,1}) tuple(bitcast, copy)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[64,64]{1,0} parameter(0)
      parameter.1 = f32[64,64]{1,0} parameter(1)
      ROOT fusion = (f32[64,64]{0,1}, f32[64,64]{0,1})
        fusion(parameter.0, parameter.1),
        kind=kLoop, calls=fused_computation
    })";

  // Check that a call to llvm.nvvm.barrier0 is generated.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @fusion
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @fusion
; CHECK: call void @llvm.nvvm.barrier0()
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.0}));
}

TEST_F(GpuKernelTilingTest, ColumnReductionWithPowerOf2OutputElementsUnrolled) {
  const char *const kHloString = R"(
  HloModule column_reduce_powerof2

  reduction {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY kernel_entry {
    constant0 = f32[] constant(0)
    arg1 = f16[1024,512]{1,0} parameter(0)
    arg1_conv = f32[1024,512]{1,0} convert(arg1)
    ROOT reduce = f32[512]{0} reduce(arg1_conv, constant0), dimensions={0}, to_apply=reduction
  })";

  // Check that two calls to llvm.nvvm.atomic are generated.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  const char *expected_ir = R"(
; CHECK: store float %{{.*}}, float addrspace(1)
; CHECK: store float %{{.*}}, float addrspace(1)
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1.0e-5, 1.0e-5}));
}

TEST_F(GpuKernelTilingTest,
       ColumnReductionWithInputLargerThenReduceInputNotUnrolled) {
  const char *const kHloString = R"(
  HloModule larger_than_reduce_input_parameter

  reduction22 {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  fused_computation {
    constant0 = f32[] constant(0)
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1027,513]{1,0} parameter(1)
    arg1.conv = f32[1024,512]{1,0} convert(arg.1)
    arg2.conv = f32[1027,513]{1,0} convert(arg.2)
    slice2 = f32[1024,512]{1,0} slice(arg2.conv), slice={[2:1026], [1:513]}
    add2 = f32[1024,512]{1,0} add(arg1.conv, slice2)
    ROOT reduce = f32[512]{0} reduce(add2, constant0), dimensions={0},
      to_apply=reduction22
  }

  ENTRY kernel_entry {
    arg1 = f16[1024,512]{1,0} parameter(0)
    arg2 = f16[1027,513]{1,0} parameter(1)
    ROOT fusion = f32[512]{0} fusion(arg1, arg2), kind=kInput,
      calls=fused_computation
  })";

  // Check that one call to llvm.nvvm.atomic is generated.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  const char *expected_ir = R"(
; CHECK: store float %{{.*}}, float addrspace(1)
; CHECK-NOT: store float %{{.*}}, float addrspace(1)
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1.0e-5, 1.0e-5}));
}

TEST_F(GpuKernelTilingTest, ColumnReductionMOFUnrolled) {
  const char *const kHloString = R"(
  HloModule column_reduce_powerof2_mof

  reduction22 {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  fused_computation {
    constant0 = f32[] constant(0)
    arg.1 = f16[1024,512]{1,0} parameter(0)
    arg.2 = f16[1024,512]{1,0} parameter(1)
    arg1.conv = f32[1024,512]{1,0} convert(arg.1)
    arg2.conv = f32[1024,512]{1,0} convert(arg.2)
    reduce1 = f32[512]{0} reduce(arg1.conv, constant0), dimensions={0},
      to_apply=reduction22
    reduce2 = f32[512]{0} reduce(arg2.conv, constant0), dimensions={0},
      to_apply=reduction22
    add = f32[1024,512]{1,0} add(arg1.conv, arg2.conv)
    ROOT tuple = (f32[512]{0}, f32[512]{0}, f32[1024,512]{1,0})
      tuple(reduce1, reduce2, add)
  }

  ENTRY kernel_entry {
    arg1 = f16[1024,512]{1,0} parameter(0)
    arg2 = f16[1024,512]{1,0} parameter(1)
    ROOT fusion = (f32[512]{0}, f32[512]{0}, f32[1024,512]{1,0})
      fusion(arg1, arg2), kind=kInput, calls=fused_computation
  })";

  // Check that four calls to llvm.nvvm.atomic are generated.
  std::unique_ptr<VerifiedHloModule> hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  const char *expected_ir = R"(
; CHECK-LABEL: define void @fusion
; CHECK: store float %{{.*}}, float addrspace(1)
; CHECK: store float %{{.*}}, float addrspace(1)
; CHECK: store float %{{.*}}, float addrspace(1)
; CHECK: store float %{{.*}}, float addrspace(1)
; CHECK-NOT: store float %{{.*}}, float addrspace(1)
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1.0e-5, 1.0e-5}));
}

TEST_F(GpuKernelTilingTest, ColumnReductionWithLayoutChangeTiled) {
  const char *const kHloString = R"(
    HloModule reduce_with_layout_change
    reduction0 {
      x0 = f32[] parameter(0)
      y0 = f32[] parameter(1)
      ROOT add0 = f32[] add(x0, y0)
    }

    ENTRY kernel_entry {
      arg0 = f32[4,32,32,16,12,12,3,3]{2,3,5,4,0,7,6,1}  parameter(0)
      constant0 = f32[] constant(0)
      ROOT reduce0 = f32[4,32,16,12,12]{4,3,2,1,0} reduce(arg0, constant0),
        dimensions={1,6,7}, to_apply=reduction0
    })";

  // Check that the kernel is tiled by looking for llvm.nvvm.atomic.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  const char *expected_ir = R"(
; CHECK-LABEL: define void @
; CHECK: store float %{{.*}}, float addrspace(1)
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.001}));
}

TEST_F(GpuKernelTilingTest, RowReductionWithLayoutChangeTiled) {
  const char *const kHloString = R"(
    HloModule reduce_with_layout_change
    reduction0 {
      x0 = f32[] parameter(0)
      y0 = f32[] parameter(1)
      ROOT add0 = f32[] add(x0, y0)
    }

    ENTRY kernel_entry {
      arg0 = f32[8,6,64]{2,1,0}  parameter(0)
      constant0 = f32[] constant(0)
      ROOT reduce0 = f32[8,6]{0,1} reduce(arg0, constant0), dimensions={2},
        to_apply=reduction0
    })";

  // Check that the kernel is tiled by looking for llvm.nvvm.shfl.sync.down.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @reduce
; CHECK: call i32 @llvm.amdgcn.ds.bpermute
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @reduce
; CHECK: call float @llvm.nvvm.shfl.sync.down.f32
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.001}));
}

TEST_F(GpuKernelTilingTest,
       ColumnReductionResultTwoPartsWithLayoutChangeTiled) {
  const char *const kHloString = R"(
    HloModule reduce_with_no_layout_change
    reduction0 {
      x0 = f32[] parameter(0)
      y0 = f32[] parameter(1)
      ROOT add0 = f32[] add(x0, y0)
    }

    ENTRY kernel_entry {
      arg0 = f32[8,64,32]{2,1,0}  parameter(0)
      constant0 = f32[] constant(0)
      ROOT reduce0 = f32[8,32]{0,1} reduce(arg0, constant0), dimensions={1},
        to_apply=reduction0
    })";

  // Check that the kernel is tiled by looking for llvm.nvvm.atomic.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  const char *expected_ir = R"(
; CHECK-LABEL: define void @reduce
; CHECK: store float %{{.*}}, float addrspace(1)
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.001}));
}

TEST_F(GpuKernelTilingTest, ColumnReductionSmallTileSizeX) {
  const char *const kHloString = R"(
  HloModule Test

  scalar_add_computation.1 {
    scalar_lhs.1 = f32[] parameter(0)
    scalar_rhs.1 = f32[] parameter(1)
    ROOT add.6 = f32[] add(scalar_lhs.1, scalar_rhs.1)
  }
  ENTRY Test {
    param_3.241 = f16[512,2,9,9]{1,3,2,0} parameter(3)
    constant_661 = f16[] constant(0)
    broadcast.695 = f16[512,2,9,9]{1,3,2,0} broadcast(constant_661), dimensions={}
    compare.42 = pred[512,2,9,9]{1,3,2,0} compare(param_3.241, broadcast.695), direction=GT
    param_2.401 = f16[512,2,9,9]{1,3,2,0} parameter(2)
    select.40 = f16[512,2,9,9]{1,3,2,0} select(compare.42, param_2.401, broadcast.695)
    convert.196 = f32[512,2,9,9]{1,3,2,0} convert(select.40)
    param_1.809 = f16[512,2,9,9]{1,3,2,0} parameter(1)
    copy.335 = f16[512,2,9,9]{1,3,2,0} copy(param_1.809)
    convert.218 = f32[512,2,9,9]{1,3,2,0} convert(copy.335)
    param_0.668 = f32[2]{0} parameter(0)
    broadcast.687 = f32[512,2,9,9]{1,3,2,0} broadcast(param_0.668), dimensions={1}
    subtract.136 = f32[512,2,9,9]{1,3,2,0} subtract(convert.218, broadcast.687)
    multiply.579 = f32[512,2,9,9]{1,3,2,0} multiply(convert.196, subtract.136)
    constant_485 = f32[] constant(0)
    reduce.139 = f32[2]{0} reduce(multiply.579, constant_485), dimensions={0,2,3}, to_apply=scalar_add_computation.1
    reduce.140.clone.1 = f32[2]{0} reduce(convert.196, constant_485), dimensions={0,2,3}, to_apply=scalar_add_computation.1
    ROOT tuple.102 = (f32[2]{0}, f32[2]{0}) tuple(reduce.139, reduce.140.clone.1)
  })";

  // Check that no loop is generated for reduction.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  const char *expected_ir = R"(
; CHECK-NOT: reduce.0.loop_header
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompare(kHloString, ErrorSpec{1.0e-5, 1.0e-5}));
}

TEST_F(GpuKernelTilingTest, RowReductionWithSmallDimensionNotTiled) {
  const char *const kHloString = R"(
    HloModule reduction
    reduction0 {
      x0 = f32[] parameter(0)
      y0 = f32[] parameter(1)
      ROOT add0 = f32[] add(x0, y0)
    }

    ENTRY kernel_entry {
      arg0 = f32[8,6,16]{2,1,0}  parameter(0)
      constant0 = f32[] constant(0)
      ROOT reduce0 = f32[8,6]{1,0} reduce(arg0, constant0), dimensions={2},
        to_apply=reduction0
    })";

  // Check that the kernel is not tiled by looking for llvm.nvvm.shfl.sync.down.
  auto hlo_module =
      ParseAndReturnVerifiedModule(kHloString, ConfigWithoutLayoutAssignment())
          .ValueOrDie();
  auto expected_ir = is_built_with_rocm_ ? R"(
; CHECK-LABEL: define amdgpu_kernel void @reduce
; CHECK-NOT: call i32 @llvm.amdgcn.ds.bpermute
; CHECK: }
)"
                                         : R"(
; CHECK-LABEL: define void @reduce
; CHECK-NOT: call float @llvm.nvvm.shfl.sync.down.f32
; CHECK: }
)";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);

  // Check that the kernel runs correctly.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.001}));
}

TEST_F(GpuKernelTilingTest, RowReductionRequiring64BitIndex) {
  const char *const kHloString = R"(
  HloModule LargeReduction

  Sum {
    x.1 = f32[] parameter(0)
    y.1 = f32[] parameter(1)
    ROOT add.1 = f32[] add(x.1, y.1)
  }

  ENTRY reduce.1 {
    parameter = f32[3048576000] parameter(0)
    init_value = f32[] constant(0)
    ROOT out = f32[] reduce(parameter, init_value), dimensions={0}, to_apply=Sum
  }
  )";
  std::unique_ptr<VerifiedHloModule> hlo_module =
      ParseAndReturnVerifiedModule(kHloString).ValueOrDie();
  const char *expected_ir = R"(
; CHECK: i64
  )";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuKernelTilingTest, ColumnReductionVectorization) {
  const char *const kHloString = R"(
HloModule column_reduce_powerof2

reduction {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
}

ENTRY kernel_entry {
    constant0 = f32[] constant(0)
    arg1 = f32[1024,512]{1,0} parameter(0)
    ROOT reduce = f32[512]{0} reduce(arg1, constant0), dimensions={0}, to_apply=reduction
}
  )";
  auto expected_ir = R"(
; CHECK: load <2 x float>, <2 x float>
  )";
  auto hlo_module = ParseAndReturnVerifiedModule(kHloString).ValueOrDie();
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuKernelTilingTest, Hlo021CopyNoOobAccess) {
  const char *const kHloString = R"(
HloModule primitive_computation_svd.38

%fused_computation (param_0.7: f32[3,29,29], param_1.10: pred[3]) -> f32[3,29,29] {
  %param_1.10 = pred[3]{0} parameter(1)
  %broadcast.7 = pred[3,29,29]{2,1,0} broadcast(pred[3]{0} %param_1.10), dimensions={0}
  %param_0.7 = f32[3,29,29]{1,2,0} parameter(0)
  %copy.6 = f32[3,29,29]{2,1,0} copy(f32[3,29,29]{1,2,0} %param_0.7)
  %constant_1 = f32[] constant(nan)
  %broadcast.6 = f32[3,29,29]{2,1,0} broadcast(f32[] %constant_1), dimensions={}
  ROOT %select.0 = f32[3,29,29]{2,1,0} select(pred[3,29,29]{2,1,0} %broadcast.7, f32[3,29,29]{2,1,0} %copy.6, f32[3,29,29]{2,1,0} %broadcast.6)
}

ENTRY %primitive_computation_svd.38 (constant_5: f32[3,29,29], fusion.3: pred[3]) -> f32[3,29,29] {
  %constant_5 = f32[3,29,29]{1,2,0} parameter(0)
  %fusion.3 = pred[3]{0} parameter(1)
  ROOT %fusion = f32[3,29,29]{2,1,0} fusion(f32[3,29,29]{1,2,0} %constant_5, pred[3]{0} %fusion.3), kind=kLoop, calls=%fused_computation
}
  )";

  // Test against the OOB read due to a ptxas bug.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{0.001}));
}

TEST_F(GpuKernelTilingTest, RowReductionCorrectShmemUsage) {
  const char *const kHloString = R"(
  HloModule RowReduce

  Sum {
    x.1 = f32[] parameter(0)
    y.1 = f32[] parameter(1)
    ROOT add.1 = f32[] add(x.1, y.1)
  }

  ENTRY reduce.1 {
    parameter = f32[1048576] parameter(0)
    init_value = f32[] constant(0)
    ROOT reduce = f32[] reduce(parameter, init_value), dimensions={0}, to_apply=Sum
  }
  )";
  auto hlo_module = ParseAndReturnVerifiedModule(kHloString).ValueOrDie();
  auto expected_ir = R"(
; CHECK: shared_cache = private unnamed_addr addrspace({{[0-9]*}}) global [1 x [1 x [2 x float]]]
  )";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuKernelTilingTest, ReductionInputTooLarge) {
  const char *const kHloString = R"(
  HloModule RowReduce

  Sum {
    x.1 = f32[] parameter(0)
    y.1 = f32[] parameter(1)
    ROOT add.1 = f32[] add(x.1, y.1)
  }

  ENTRY reduce.1 {
    parameter = f32[4,1048576,1024,1024] parameter(0)
    init_value = f32[] constant(0)
    ROOT reduce = f32[4,1048576,1024] reduce(parameter, init_value), dimensions={3}, to_apply=Sum
  }
  )";
  auto hlo_module = ParseAndReturnVerifiedModule(kHloString).ValueOrDie();
  Status status = CompileToExecutable(std::move(hlo_module)).status();
  EXPECT_EQ(status.code(), tensorflow::error::Code::FAILED_PRECONDITION);
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr(
          "Number of physical blocks (4294967296) does not fit in an i32"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
