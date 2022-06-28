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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPScanonicalize_all_gather_for_cse_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPScanonicalize_all_gather_for_cse_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPScanonicalize_all_gather_for_cse_testDTcc() {
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

#include "tensorflow/compiler/xla/service/spmd/canonicalize_all_gather_for_cse.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace spmd {
namespace {

using ::testing::_;
using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;

class AllGatherCanonicalizeTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> RunPass(absl::string_view hlo_module) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(
                                         hlo_module, GetModuleConfigForTest()));
    HloPassPipeline pipeline("all-gather-cse");
    pipeline.AddPass<CanonicalizeAllGatherForCSE>();
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }
  Status RunPassOnModule(HloModule* module, int64_t distance_threshold = 100) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSspmdPScanonicalize_all_gather_for_cse_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/xla/service/spmd/canonicalize_all_gather_for_cse_test.cc", "RunPassOnModule");

    HloPassPipeline pipeline("all-gather-cse");
    pipeline.AddPass<CanonicalizeAllGatherForCSE>();
    TF_RETURN_IF_ERROR(pipeline.Run(module).status());
    return Status::OK();
  }
};

TEST_F(AllGatherCanonicalizeTest, SimpleReshape) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[1,8]{1,0} reshape(param0)
  ROOT ag = s32[2,8]{1,0} all-gather(resh), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = module_status.ConsumeValueOrDie();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape,
              AllOf(op::Reshape(op::AllGather(_)), op::Shape("s32[2,8]")));
}

TEST_F(AllGatherCanonicalizeTest, MultipleDegenerateReshapes) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[1,8]{1,0} reshape(param0)
  resh2 = s32[1,8,1,1]{3,2,1,0} reshape(resh)
  ROOT ag = s32[2,8,1,1]{3,2,1,0} all-gather(resh2), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = module_status.ConsumeValueOrDie();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape, op::Reshape(op::AllGather(op::Parameter())));
}

TEST_F(AllGatherCanonicalizeTest, MultipleDegenerateReshapes2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[8,1,1]{2,1,0} reshape(param0)
  resh2 = s32[1,8,1,1]{3,2,1,0} reshape(resh)
  ROOT ag = s32[2,8,1,1]{3,2,1,0} all-gather(resh2), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = module_status.ConsumeValueOrDie();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape, op::Reshape(op::AllGather(op::Parameter())));
}

TEST_F(AllGatherCanonicalizeTest, MultipleDegenerateReshapesNoDim0) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[8,1,1]{2,1,0} reshape(param0)
  resh2 = s32[1,8,1,1]{3,2,1,0} reshape(resh)
  ROOT ag = s32[1,16,1,1]{3,2,1,0} all-gather(resh2), replica_groups={{0,1}},
    dimensions={1}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = module_status.ConsumeValueOrDie();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape, op::Reshape(op::AllGather(op::Parameter())));
}

TEST_F(AllGatherCanonicalizeTest, NonDegenerateReshape) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[8,1,1]{2,1,0} reshape(param0)
  resh2 = s32[1,4,2,1,1]{4,3,2,1,0} reshape(resh)
  ROOT ag = s32[2,4,2,1,1]{4,3,2,1,0} all-gather(resh2), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = module_status.ConsumeValueOrDie();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape, AllOf(op::AllGather(op::Reshape(op::Reshape(_))),
                             op::Shape("s32[2,4,2,1,1]")));
}

}  // namespace
}  // namespace spmd
}  // namespace xla
