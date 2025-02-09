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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combiner_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combiner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combiner_testDTcc() {
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

#include "tensorflow/compiler/xla/service/all_gather_combiner.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::Matcher;
namespace op = xla::testing::opcode_matchers;
int64_t kMaxCombineCount = 256;

int64_t AllGatherCount(const HloModule& module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_gather_combiner_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xla/service/all_gather_combiner_test.cc", "AllGatherCount");

  int64_t count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kAllGather) {
        ++count;
      }
    }
  }
  return count;
}

using AllGatherCombinerTest = HloTestBase;

// Tests combination of several AllGather instructions.
TEST_F(AllGatherCombinerTest, CombineAllGathers) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0)
  param1 = f32[32] parameter(1)
  allgather0 = f32[128] all-gather(param0), replica_groups={}, dimensions={0}
  allgather1 = f32[128] all-gather(param1), replica_groups={}, dimensions={0}
  ROOT tuple = (f32[128], f32[128]) tuple(allgather0, allgather1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);

  Matcher<const HloInstruction*> combined_all_gather =
      op::AllGather(op::Parameter(0), op::Parameter(1));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(combined_all_gather, 0),
                        op::GetTupleElement(combined_all_gather, 1)));
}

// Tests combination of several cross replica gather instructions with
// different gather dimensions.
TEST_F(AllGatherCombinerTest, CombineAllGathersByAllGatherDimension) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[2,2] parameter(0)
  param1 = f32[2,2] parameter(1)
  param2 = f32[2,2] parameter(2)
  param3 = f32[2,2] parameter(3)
  param4 = f32[2,2] parameter(4)
  allgather0 = f32[8,2] all-gather(param0), replica_groups={}, dimensions={0}
  allgather1 = f32[8,2] all-gather(param1), replica_groups={}, dimensions={0}
  allgather2 = f32[2,8] all-gather(param2), replica_groups={}, dimensions={1}
  allgather3 = f32[2,8] all-gather(param3), replica_groups={}, dimensions={1}
  allgather4 = f32[8,2] all-gather(param4), replica_groups={}, dimensions={0}
  ROOT tuple = (f32[8,2], f32[8,2], f32[2,8], f32[2,8], f32[8,2])
    tuple(allgather0, allgather1, allgather2, allgather3, allgather4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 5);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);

  Matcher<const HloInstruction*> combined_all_gather0 =
      op::AllGather(op::Parameter(0), op::Parameter(1), op::Parameter(4));
  Matcher<const HloInstruction*> combined_all_gather1 =
      op::AllGather(op::Parameter(2), op::Parameter(3));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(combined_all_gather0, 0),
                        op::GetTupleElement(combined_all_gather0, 1),
                        op::GetTupleElement(combined_all_gather1, 0),
                        op::GetTupleElement(combined_all_gather1, 1),
                        op::GetTupleElement(combined_all_gather0, 2)));
}

// Tests that the combination threshold is respected.
TEST_F(AllGatherCombinerTest, DoNotCombineOverThreshold) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[8] parameter(0)
  param1 = f32[8] parameter(1)
  allgather0 = f32[32] all-gather(param0), replica_groups={}, dimensions={0}
  allgather1 = f32[32] all-gather(param1), replica_groups={}, dimensions={0}
  ROOT tuple = (f32[32], f32[32]) tuple(allgather0, allgather1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Run the AllGather combiner optimization pass with threshold less than
  // the combined size of the all gather ops so that the combination
  // cannot occur.
  AllGatherCombiner combine(255, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

// Tests that the combination threshold is respected.
TEST_F(AllGatherCombinerTest, CombineUpToThreshold) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[8] parameter(0)
  param1 = f32[8] parameter(1)
  allgather0 = f32[32] all-gather(param0), replica_groups={}, dimensions={0}
  allgather1 = f32[32] all-gather(param1), replica_groups={}, dimensions={0}
  ROOT tuple = (f32[32], f32[32]) tuple(allgather0, allgather1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Run the AllGather combiner optimization pass with a threshold just higher
  // than that required such that the combination can occur.
  AllGatherCombiner combine(256, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 1);
  EXPECT_TRUE(changed);
}

// Tests that dependent all gathers are not combined.
TEST_F(AllGatherCombinerTest, NoDependentCombination) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param = f32[1] parameter(0)
  allgather0 = f32[2] all-gather(param), replica_groups={}, dimensions={0}
  ROOT allgather1 = f32[4] all-gather(allgather0), replica_groups={}, dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

// Tests that AllGather ops with different groups are not combined.
TEST_F(AllGatherCombinerTest, NoDifferentReplicaGroupsCombination) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0)
  param1 = f32[32] parameter(1)
  allgather0 = f32[64] all-gather(param0), replica_groups={{0, 1}, {2, 3}},
    dimensions={0}
  allgather1 = f32[64] all-gather(param1), replica_groups={{0, 2}, {1, 3}},
    dimensions={0}
  ROOT tuple = (f32[64], f32[64]) tuple(allgather0, allgather1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

TEST_F(AllGatherCombinerTest, DomainPreventsCombining) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0), sharding={maximal device=0}
  param1 = f32[32] parameter(1), sharding={maximal device=1}
  allgather0 = f32[128] all-gather(param0),
    replica_groups={}, dimensions={0}, sharding={maximal device=0}
  allgather1 = f32[128] all-gather(param1),
    replica_groups={}, dimensions={0}, sharding={maximal device=1}
  domain0 = f32[128] domain(allgather0),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1}},
    exit={maximal device=0}}
  domain1 = f32[128] domain(allgather1),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1}},
    exit={maximal device=1}}
  ROOT tuple = (f32[128], f32[128]) tuple(domain0, domain1),
    sharding={{maximal device=0}, {maximal device=1}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

// This test checks that two AllGather instructions that are in separate domains
// but with the same domain metadata can be combined.
TEST_F(AllGatherCombinerTest, CombineFromTwoDomainsWithSameMetadata) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0), sharding={maximal device=0}
  param1 = f32[32] parameter(1), sharding={maximal device=1}
  param2 = f32[32] parameter(2), sharding={maximal device=1}
  allgather0 = f32[128] all-gather(param0),
    replica_groups={}, dimensions={0}, sharding={maximal device=0}
  allgather1 = f32[128] all-gather(param1),
    replica_groups={}, dimensions={0}, sharding={maximal device=1}
  allgather2 = f32[128] all-gather(param2),
    replica_groups={}, dimensions={0}, sharding={maximal device=0}
  domain0 = f32[128] domain(allgather0),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1},
    {maximal device=0}}, exit={maximal device=0}}
  domain1 = f32[128] domain(allgather1),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1},
    {maximal device=0}}, exit={maximal device=1}}
  domain2 = f32[128] domain(allgather2),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1},
    {maximal device=0}}, exit={maximal device=0}}
  ROOT tuple = (f32[128], f32[128], f32[128]) tuple(domain0, domain1,
  domain2),
    sharding={{maximal device=0}, {maximal device=1}, {maximal device=0}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_TRUE(changed);
}

TEST_F(AllGatherCombinerTest, DoNotCombineCrossShardAndCrossReplicaInSPMD) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0), sharding={maximal device=0}
  param1 = f32[32] parameter(1), sharding={maximal device=1}
  cross_shard_ag = f32[128] all-gather(param0),
    replica_groups={{0}}, dimensions={0}, channel_id=1
  cross_replica_ag = f32[128] all-gather(param1),
    replica_groups={{0}}, dimensions={0}, sharding={maximal device=1}
  ROOT tuple = (f32[128], f32[128]) tuple(cross_shard_ag, cross_replica_ag)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
