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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizer_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizer_testDTcc() {
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

#include "tensorflow/compiler/xla/service/loop_schedule_linearizer.h"

#include <set>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace xla {
namespace {

int64_t CountCopies(const HloComputation& computation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizer_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/loop_schedule_linearizer_test.cc", "CountCopies");

  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == HloOpcode::kCopy) {
      count++;
    }
  }
  return count;
}

int64_t CountCopies(const HloModule& module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizer_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/compiler/xla/service/loop_schedule_linearizer_test.cc", "CountCopies");

  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountCopies(*computation);
  }
  return count;
}

int64_t CountControlEdges(const HloComputation& computation) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizer_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/compiler/xla/service/loop_schedule_linearizer_test.cc", "CountControlEdges");

  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    count += instruction->control_successors().size();
  }
  return count;
}

int64_t CountControlEdges(const HloModule& module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizer_testDTcc mht_3(mht_3_v, 243, "", "./tensorflow/compiler/xla/service/loop_schedule_linearizer_test.cc", "CountControlEdges");

  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountControlEdges(*computation);
  }
  return count;
}

class LoopScheduleLinearizerTest : public HloTestBase {
 protected:
  void InsertCopies(HloModule* module) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSloop_schedule_linearizer_testDTcc mht_4(mht_4_v, 256, "", "./tensorflow/compiler/xla/service/loop_schedule_linearizer_test.cc", "InsertCopies");

    LoopScheduleLinearizer loop_schedule_linearizer;
    ASSERT_IS_OK(loop_schedule_linearizer.Run(module).status());

    CopyInsertion copy_insertion;
    ASSERT_IS_OK(copy_insertion.Run(module).status());
  }
};

TEST_F(LoopScheduleLinearizerTest, NoExtraCopiesRequired) {
  absl::string_view hlo_string = R"(
HloModule module

while_body {
  input = (s32[], s32[]) parameter(0)
  counter = s32[] get-tuple-element(input), index=0
  buffer = s32[] get-tuple-element(input), index=1

  one = s32[] constant(1)

  updated_counter = s32[] add(counter, one)

  updated_buffer = s32[] add(buffer, counter)
  ROOT out = (s32[], s32[]) tuple(updated_counter, updated_buffer)
}

while_cond {
  input = (s32[], s32[]) parameter(0)
  counter = s32[] get-tuple-element(input), index=0
  bound = s32[] constant(100)
  ROOT cmp = pred[] compare(counter, bound), direction=LT
}

ENTRY entry {
  zero = s32[] constant(0)
  buffer = s32[] parameter(0)
  while_input = (s32[], s32[]) tuple(zero, buffer)
  ROOT out = (s32[], s32[]) while(while_input), condition=while_cond, body=while_body
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(
                *module->entry_computation()->root_instruction()->while_body()),
            0);
  EXPECT_EQ(CountControlEdges(
                *module->entry_computation()->root_instruction()->while_body()),
            1);
}

}  // namespace
}  // namespace xla
