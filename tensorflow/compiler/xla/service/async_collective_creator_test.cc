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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creator_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creator_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creator_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/async_collective_creator.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::NotNull;
using ::testing::SizeIs;

using AsyncAllReduceCreatorTest = HloTestBase;

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleAllReduce) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p0 = f32[8] parameter(0)
    ROOT ar = f32[8] all-reduce(p0), to_apply=add
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_reduce = [](const HloInstruction*) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creator_testDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/service/async_collective_creator_test.cc", "lambda");
 return true; };
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAllReduceDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAllReduceStart);
}

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleAllGather) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[1] parameter(0)
    ROOT ag = f32[8] all-gather(p0), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_gather = [](const HloInstruction*) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creator_testDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/xla/service/async_collective_creator_test.cc", "lambda");
 return true; };
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAllGatherDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAllGatherStart);
}

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleCollectivePermute) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    %p0 = bf16[8]{0} parameter(0)
    ROOT %collective-permute.1 = bf16[8]{0} collective-permute(bf16[8]{0} p0), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_collective_permute = [](const HloInstruction*) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creator_testDTcc mht_2(mht_2_v, 274, "", "./tensorflow/compiler/xla/service/async_collective_creator_test.cc", "lambda");

    return true;
  };
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kCollectivePermuteDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kCollectivePermuteStart);
}

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleInPlaceCollectivePermute) {
  std::string hlo_string = std::string(R"(
HloModule module

ENTRY %module_spmd () -> f32[4,4,128] {
  %constant.8 = u32[] constant(0)
  %constant.5 = u32[] constant(2)
  %tuple.1 = (u32[], u32[], u32[]) tuple(u32[] %constant.8, u32[] %constant.8, u32[] %constant.8)
  %tuple = (u32[], u32[], u32[]) tuple(u32[] %constant.5, u32[] %constant.8, u32[] %constant.8)
  %custom-call = f32[4,4,128]{2,1,0:T(4,128)} custom-call(), custom_call_target="SomeCustomCall"
  ROOT %collective-permute = f32[4,4,128]{2,1,0:T(4,128)} collective-permute(f32[4,4,128]{2,1,0:T(4,128)} %custom-call, f32[4,4,128]{2,1,0:T(4,128)} %custom-call, (u32[], u32[], u32[]) %tuple, (u32[], u32[], u32[]) %tuple.1), channel_id=958, source_target_pairs={{0,4},{4,0},{1,5},{5,1},{2,6},{6,2},{3,7},{7,3}}, slice_sizes={{2,4,128}}
}
)");

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_collective_permute = [](const HloInstruction*) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creator_testDTcc mht_3(mht_3_v, 309, "", "./tensorflow/compiler/xla/service/async_collective_creator_test.cc", "lambda");

    return true;
  };
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 7);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kCollectivePermuteDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kCollectivePermuteStart);
}

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleCollectivePermuteScheduled) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test, is_scheduled=true
  ENTRY entry {
    %p0 = bf16[8]{0} parameter(0)
    ROOT %collective-permute.1 = bf16[8]{0} collective-permute(bf16[8]{0} p0), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const int64_t original_instr_sequence_size =
      hlo_module->schedule().sequence(hlo_module->entry_computation()).size();

  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_collective_permute = [](const HloInstruction*) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSasync_collective_creator_testDTcc mht_4(mht_4_v, 342, "", "./tensorflow/compiler/xla/service/async_collective_creator_test.cc", "lambda");

    return true;
  };
  TF_ASSERT_OK(AsyncCollectiveCreator(config).Run(hlo_module.get()).status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kCollectivePermuteDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kCollectivePermuteStart);
  EXPECT_EQ(
      hlo_module->schedule().sequence(hlo_module->entry_computation()).size(),
      original_instr_sequence_size + 1);
}

}  // namespace
}  // namespace xla
