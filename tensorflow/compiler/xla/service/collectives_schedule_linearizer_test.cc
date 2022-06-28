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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollectives_schedule_linearizer_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollectives_schedule_linearizer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollectives_schedule_linearizer_testDTcc() {
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

#include "tensorflow/compiler/xla/service/collectives_schedule_linearizer.h"

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace xla {
namespace {

namespace m = match;

int64_t CountControlEdges(const HloComputation& computation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollectives_schedule_linearizer_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/collectives_schedule_linearizer_test.cc", "CountControlEdges");

  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    count += instruction->control_successors().size();
  }
  return count;
}

class CollectivesScheduleLinearizerTest : public HloTestBase {
 protected:
  void InsertCollectivesSchedule(HloModule* module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollectives_schedule_linearizer_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/compiler/xla/service/collectives_schedule_linearizer_test.cc", "InsertCollectivesSchedule");

    CollectivesScheduleLinearizer collectives_schedule_linearizer;
    ASSERT_IS_OK(collectives_schedule_linearizer.Run(module).status());
  }
};

TEST_F(CollectivesScheduleLinearizerTest, FixOrdering) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  c1 = f32[100] all-reduce(p0), replica_groups={}, to_apply=sum
  c2 = f32[100] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT out = f32[100] add(c1, c2)
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 1);
  HloInstruction *c1 = nullptr, *c2 = nullptr;
  for (HloInstruction* instr : module->entry_computation()->instructions()) {
    if (Match(instr, m::AllReduce(m::Parameter(0)))) {
      c1 = instr;
    }
    if (Match(instr, m::AllReduce(m::Parameter(1)))) {
      c2 = instr;
    }
  }
  EXPECT_TRUE(c1 != nullptr && c2 != nullptr);
  EXPECT_TRUE(absl::c_linear_search(c2->control_predecessors(), c1));
}

TEST_F(CollectivesScheduleLinearizerTest, NoFixRequired) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  c1 = f32[100] all-reduce(p0), replica_groups={}, to_apply=sum
  c2 = f32[100] all-reduce(p1), replica_groups={}, to_apply=sum, control-predecessors={c1}
  ROOT out = f32[100] add(c1, c2)
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 1);
}

TEST_F(CollectivesScheduleLinearizerTest, DependentCollectives) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  c1 = f32[100] all-reduce(p0), replica_groups={}, to_apply=sum
  c2 = f32[100] all-reduce(c1), replica_groups={}, to_apply=sum
  ROOT out = f32[100] add(c1, c2)
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 0);
}

TEST_F(CollectivesScheduleLinearizerTest, NonPostorder) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  c1 = f32[100] all-reduce(p0), replica_groups={}, to_apply=sum
  c2 = f32[100] all-reduce(p1), replica_groups={}, to_apply=sum
  c3 = f32[100] all-reduce(p1), replica_groups={}, to_apply=sum
  t = f32[100] add(c1, c2)
  ROOT out = f32[100] add(t, c3)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(
      module->entry_computation()
          ->GetInstructionWithName("c3")
          ->AddControlDependencyTo(
              module->entry_computation()->GetInstructionWithName("c1")));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 2);
}

}  // namespace
}  // namespace xla
