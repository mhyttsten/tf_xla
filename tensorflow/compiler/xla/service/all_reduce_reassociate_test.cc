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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_reassociate_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_reassociate_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_reassociate_testDTcc() {
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

#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = xla::testing::opcode_matchers;

class AllReduceSimplifierTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> RunPass(absl::string_view hlo_module,
                                               bool expect_change) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));
    auto changed = AllReduceReassociate().Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.ValueOrDie(), expect_change);
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  size_t AllReduceCount(std::unique_ptr<HloModule>& module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSall_reduce_reassociate_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/all_reduce_reassociate_test.cc", "AllReduceCount");

    return absl::c_count_if(module->entry_computation()->instructions(),
                            [](const HloInstruction* inst) {
                              return inst->opcode() == HloOpcode::kAllReduce;
                            });
  }
};

TEST_F(AllReduceSimplifierTest, Simple) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, SimpleWithChannelId) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), channel_id=1, replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), channel_id=1, replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

// Checks whether a linear chain of adds of ARs is reassociated iin a single
// pass.
TEST_F(AllReduceSimplifierTest, SimpleChain) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  p2 = f32[8] parameter(2)
  p3 = f32[8] parameter(3)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ar2 = f32[8] all-reduce(p2), replica_groups={}, to_apply=sum
  ar3 = f32[8] all-reduce(p3), replica_groups={}, to_apply=sum
  add0 = f32[8] add(ar0, ar1)
  add1 = f32[8] add(add0, ar2)
  ROOT add2 = f32[8] add(add1, ar3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      m::AllReduce(m::Add(
          m::Add(m::Add(m::Parameter(0), m::Parameter(1)), m::Parameter(2)),
          m::Parameter(3))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

// Checks whether a tree of add of ARs is reassociated in a single pass.
TEST_F(AllReduceSimplifierTest, SimpleTree) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  p2 = f32[8] parameter(2)
  p3 = f32[8] parameter(3)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ar2 = f32[8] all-reduce(p2), replica_groups={}, to_apply=sum
  ar3 = f32[8] all-reduce(p3), replica_groups={}, to_apply=sum
  add0 = f32[8] add(ar0, ar1)
  add1 = f32[8] add(ar2, ar3)
  ROOT add2 = f32[8] add(add0, add1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduce(m::Add(m::Add(m::Parameter(0), m::Parameter(1)),
                                  m::Add(m::Parameter(2), m::Parameter(3)))));
  EXPECT_EQ(AllReduceCount(module), 1);
}

TEST_F(AllReduceSimplifierTest, MismatchOp0) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] maximum(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=max
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchOp1) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

max {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT r = f32[] maximum(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=max
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=max
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchReplicaGroups) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={{0}}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchHasChannelId) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, channel_id=3, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, MismatchUseGlobalDeviceId) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={{0, 1}}, channel_id=3, use_global_device_ids=true, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={{0, 1}}, channel_id=4, to_apply=sum
  ROOT add = f32[8] add(ar0, ar1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, NotSingleUser) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[8] all-reduce(p1), replica_groups={}, to_apply=sum
  add = f32[8] add(ar0, ar1)
  ROOT t = (f32[8], f32[8]) tuple(ar0, add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(AllReduceSimplifierTest, DoubleUse) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ar0 = f32[8] all-reduce(p0), replica_groups={}, to_apply=sum
  add = f32[8] add(ar0, ar0)
  ROOT c = f32[8] copy(add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
}

}  // namespace
}  // namespace xla
