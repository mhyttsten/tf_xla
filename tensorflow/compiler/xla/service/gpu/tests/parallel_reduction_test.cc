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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSparallel_reduction_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSparallel_reduction_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSparallel_reduction_testDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"

namespace xla {
namespace gpu {

namespace {

class ParallelReductionTest : public GpuCodegenTest {
  DebugOptions GetDebugOptionsForTest() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStestsPSparallel_reduction_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/xla/service/gpu/tests/parallel_reduction_test.cc", "GetDebugOptionsForTest");

    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // The test contains a MOF fusion and the XLA optimizer passes
    // don't like this.
    debug_options.set_xla_disable_all_hlo_passes(true);
    return debug_options;
  }
};

TEST_F(ParallelReductionTest, TwoParallelReductions) {
  const char* hlo_text = R"(
HloModule TwoParallelReductions

%add_f32 {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

%fused_computation {
  %param0 = f32[1024] parameter(0)
  %param1 = f32[1024] parameter(1)
  %constant0 = f32[] constant(0)
  %reduce1 = f32[] reduce(%param0, %constant0), dimensions={0}, to_apply=%add_f32
  %reduce2 = f32[] reduce(%param1, %constant0), dimensions={0}, to_apply=%add_f32
  ROOT %tuple = (f32[], f32[]) tuple(%reduce1, %reduce2)
}

ENTRY %cluster {
  %param0 = f32[1024] parameter(0)
  %param1 = f32[1024] parameter(1)
  ROOT %fusion = (f32[], f32[])
      fusion(%param0, %param1), kind=kInput, calls=%fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
CHECK: reduce-group-0
CHECK: reduce-group-1
CHECK-NOT: reduce-group-2
)",
                     /*match_optimized_ir=*/false);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ParallelReductionTest, ManyParallelReductions) {
  std::unique_ptr<VerifiedHloModule> module = CreateNewVerifiedModule();
  // Simply use a number not too large to avoid long compilation time
  // and not too small for meaningful test.
  const size_t num_reduces = 32;

  HloComputation* reduce_computation;
  {
    auto embedded_builder = HloComputation::Builder("add");
    HloInstruction* lhs =
        embedded_builder.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(F32, {}), "lhs"));
    HloInstruction* rhs =
        embedded_builder.AddInstruction(HloInstruction::CreateParameter(
            1, ShapeUtil::MakeShape(F32, {}), "rhs"));
    embedded_builder.AddInstruction(
        HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
    reduce_computation =
        module->AddEmbeddedComputation(embedded_builder.Build());
  }

  Shape input_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape output_shape = ShapeUtil::MakeShape(F32, {});
  HloComputation* fusion_computation;
  {
    auto fusion_builder = HloComputation::Builder("fusion_computation");
    std::vector<HloInstruction*> outputs;
    HloInstruction* constant = fusion_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
    for (size_t i = 0; i < num_reduces; ++i) {
      HloInstruction* param = fusion_builder.AddInstruction(
          HloInstruction::CreateParameter(i, input_shape, "param"));
      HloInstruction* output =
          fusion_builder.AddInstruction(HloInstruction::CreateReduce(
              output_shape, param, constant, {0}, reduce_computation));
      outputs.push_back(output);
    }
    fusion_builder.AddInstruction(HloInstruction::CreateTuple(outputs));
    fusion_computation = module->AddEmbeddedComputation(fusion_builder.Build());
  }

  HloComputation::Builder b(TestName());
  std::vector<HloInstruction*> entry_params;
  std::vector<Shape> output_shapes;
  entry_params.reserve(num_reduces);
  output_shapes.reserve(num_reduces);
  for (size_t i = 0; i < num_reduces; ++i) {
    HloInstruction* param = b.AddInstruction(
        HloInstruction::CreateParameter(i, input_shape, "param"));
    entry_params.push_back(param);
    output_shapes.push_back(output_shape);
  }
  b.AddInstruction(HloInstruction::CreateFusion(
      ShapeUtil::MakeTupleShape(output_shapes),
      HloInstruction::FusionKind::kInput, entry_params, fusion_computation));
  module->AddEntryComputation(b.Build());

  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ParallelReductionTest, ThreeReductionGroups) {
  const char* hlo_text = R"(
HloModule ThreeReductionGroups

%add_f32 {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

%fused_computation {
  %param0 = f32[1024,128] parameter(0)
  %param1 = f32[1024,128] parameter(1)
  %param2 = f32[1024,128] parameter(2)
  %constant0 = f32[] constant(0)
  // %mul0, %reduce0, and %reduce1 should go into a group.
  %broadcast0 = f32[1024,128] broadcast(%constant0), dimensions={}
  %mul0 = f32[1024,128] multiply(param0, broadcast0)
  %reduce0 = f32[128] reduce(%mul0, %constant0), dimensions={0}, to_apply=%add_f32
  %reduce1 = f32[128] reduce(%param0, %constant0), dimensions={0}, to_apply=%add_f32
  // %reduce2 and %reduce3 should go into another group.
  %reduce2 = f32[128] reduce(%param1, %constant0), dimensions={0}, to_apply=%add_f32
  %reduce3 = f32[128] reduce(%param1, %constant0), dimensions={0}, to_apply=%add_f32
  // %reduce4 and %mul2 should go into the other group, although broadcast0 is
  // reused.
  %mul1 = f32[1024,128] multiply(param2, broadcast0)
  %reduce4 = f32[128] reduce(%mul1, %constant0), dimensions={0}, to_apply=%add_f32
  %mul2 = f32[1024,128] multiply(param2, param2)
  ROOT %tuple =
      (f32[1024, 128], f32[128], f32[128], f32[128], f32[128], f32[128], f32[1024, 128])
      tuple(%mul2, %reduce0, %reduce4, %reduce3, %reduce2, %reduce1, %mul0)
}

ENTRY %cluster {
  %param0 = f32[1024,128] parameter(0)
  %param1 = f32[1024,128] parameter(1)
  %param2 = f32[1024,128] parameter(2)
  ROOT %fusion =
      (f32[1024, 128], f32[128], f32[128], f32[128], f32[128], f32[128], f32[1024, 128])
      fusion(%param0, %param1, %param2), kind=kInput, calls=%fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
CHECK: reduce-group-0
CHECK: reduce-group-1
CHECK: reduce-group-2
CHECK-NOT: reduce-group-3
)",
                     /*match_optimized_ir=*/false);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
