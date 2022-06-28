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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipeline_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipeline_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipeline_testDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::testing::StrEq;

class HloPassPipelineTest : public HloTestBase {
 protected:
  StatusOr<HloModuleGroup> ParseModuleGroup(
      absl::Span<const std::string> hlo_strings) {
    HloModuleGroup group(TestName());
    for (const std::string& hlo_string : hlo_strings) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
      group.push_back(std::move(module));
    }
    return std::move(group);
  }
};

// A module pass which renames instructions named 'foo' to 'bar'.
class FooToBarModulePass : public HloModulePass {
  absl::string_view name() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipeline_testDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline_test.cc", "name");
 return "foo2bar"; }

  StatusOr<bool> Run(HloModule* module) override {
    bool changed = false;
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->name() == "foo") {
          instruction->SetAndSanitizeName("bar");
          changed = true;
        }
      }
    }
    return changed;
  }
};

// A module group pass which renames instructions named 'baz' to 'qux'.
class BazToQuxModuleGroupPass : public HloModuleGroupPass {
  absl::string_view name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipeline_testDTcc mht_1(mht_1_v, 239, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline_test.cc", "name");
 return "baz2qux"; }

  StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group) override {
    bool changed = false;
    for (HloModule* module : module_group->modules()) {
      for (HloComputation* computation : module->computations()) {
        for (HloInstruction* instruction : computation->instructions()) {
          if (instruction->name() == "baz") {
            instruction->SetAndSanitizeName("qux");
            changed = true;
          }
        }
      }
    }
    return changed;
  }
};

// An invariant checker pass which returns an error if there exists an
// instruction named 'bar'.
class BarBlowerUpper : public HloModulePass {
  absl::string_view name() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_pass_pipeline_testDTcc mht_2(mht_2_v, 263, "", "./tensorflow/compiler/xla/service/hlo_pass_pipeline_test.cc", "name");
 return "bar-blower-upper"; }

  StatusOr<bool> Run(HloModule* module) override {
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->name() == "bar") {
          return InternalError("Module has instruction named bar");
        }
      }
    }
    return false;
  }
};

TEST_F(HloPassPipelineTest, ModulePassChanged) {
  // Test an HLO module pass which changes a module.
  const std::string module_str = R"(
HloModule ModulePassChanged

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT foo = f32[] multiply(a, b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<FooToBarModulePass>();

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->name(), "foo");
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(root->name(), "bar");
}

TEST_F(HloPassPipelineTest, ModulePassUnchanged) {
  // Test an HLO module pass which does not change a module.
  const std::string module_str = R"(
HloModule ModulePassUnchanged

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT blahblah = f32[] multiply(a, b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<FooToBarModulePass>();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(HloPassPipelineTest, MixedPipeline) {
  // Test a pipeline with both a module pass and a module group pass.
  const std::string module_0_str = R"(
HloModule MixedPipeline.1

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT baz = f32[] multiply(a, b)
}
)";
  const std::string module_1_str = R"(
HloModule MixedPipeline.0

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT foo = f32[] multiply(a, b)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModuleGroup module_group,
                          ParseModuleGroup({module_0_str, module_1_str}));

  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<BazToQuxModuleGroupPass>();
  pipeline.AddPass<FooToBarModulePass>();

  HloInstruction* root0 =
      module_group.module(0).entry_computation()->root_instruction();
  HloInstruction* root1 =
      module_group.module(1).entry_computation()->root_instruction();
  EXPECT_EQ(root0->name(), "baz");
  EXPECT_EQ(root1->name(), "foo");

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          pipeline.RunOnModuleGroup(&module_group));
  EXPECT_TRUE(changed);

  EXPECT_EQ(root0->name(), "qux");
  EXPECT_EQ(root1->name(), "bar");
}

TEST_F(HloPassPipelineTest, InvariantChecker) {
  const std::string module_str = R"(
HloModule InvariantChecker

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT foo = f32[] multiply(a, b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  {
    // Run a pipeline with just the invariant checker. It should not fail
    // because there is no 'bar' instruction in the module.
    HloPassPipeline pipeline(TestName());
    pipeline.AddInvariantChecker<BarBlowerUpper>();

    TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
    EXPECT_FALSE(changed);
  }

  {
    // Run a pipeline which renames 'foo' to 'bar' then an invariant checker
    // which fails if there is an instruction named 'bar'.
    HloPassPipeline pipeline(TestName());
    pipeline.AddInvariantChecker<BarBlowerUpper>();
    pipeline.AddPass<FooToBarModulePass>();

    Status status = pipeline.Run(module.get()).status();
    ASSERT_IS_NOT_OK(status);
    EXPECT_THAT(status.error_message(),
                ::testing::HasSubstr("Module has instruction named bar"));
    EXPECT_THAT(status.error_message(),
                ::testing::HasSubstr("Failed after foo2bar"));
  }

  {
    // Run the invariant-checker only pipeline again. It should fail this time.
    HloPassPipeline pipeline(TestName());
    pipeline.AddInvariantChecker<BarBlowerUpper>();

    Status status = pipeline.Run(module.get()).status();
    ASSERT_IS_NOT_OK(status);
    EXPECT_THAT(status.error_message(),
                ::testing::HasSubstr("Module has instruction named bar"));
    EXPECT_THAT(status.error_message(),
                ::testing::HasSubstr("Failed after pipeline-start"));
  }
}

TEST_F(HloPassPipelineTest, ModuleGroupPassOnModule) {
  // Running a module group pass on a module should produce an error.
  const std::string module_str = R"(
HloModule ModuleGroupPassOnModule

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT foo = f32[] multiply(a, b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<BazToQuxModuleGroupPass>();

  Status status = pipeline.Run(module.get()).status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr("Module group pass cannot be run on a module"));
}

// Test that metadata is set when a module group goes through a pass pipeline.
TEST_F(HloPassPipelineTest, SetHloModuleMetadata) {
  HloModuleGroup module_group(TestName());
  module_group.push_back(CreateNewVerifiedModule());
  module_group.push_back(CreateNewVerifiedModule());

  HloPassPipeline pipeline(TestName());
  pipeline.AddPass<BazToQuxModuleGroupPass>();
  pipeline.AddPass<FooToBarModulePass>();
  TF_ASSERT_OK(pipeline.RunOnModuleGroup(&module_group).status());
  ASSERT_THAT(module_group.modules(), SizeIs(2));

  std::vector<std::string> pass_names = {"pipeline-start", "baz2qux",
                                         "foo2bar"};
  std::string pipeline_name = std::string(pipeline.name());
  for (const HloModule* module : module_group.modules()) {
    const HloModuleMetadataProto& metadata = module->metadata().proto();
    EXPECT_EQ(metadata.canonical_module_id(), module->unique_id());
    EXPECT_EQ(metadata.module_group_name(), module_group.name());

    ASSERT_THAT(metadata.pass_metadata(), SizeIs(3));
    for (int pass = 0; pass < metadata.pass_metadata().size(); pass++) {
      const HloPassMetadata& pass_metadata = metadata.pass_metadata(pass);
      EXPECT_NE(pass_metadata.pass_id(), 0);
      EXPECT_THAT(pass_metadata.pass_name(), StrEq(pass_names[pass]));
      EXPECT_THAT(pass_metadata.pipeline_name(), StrEq(pipeline_name));
      EXPECT_FALSE(pass_metadata.module_changed());
      EXPECT_EQ(pass_metadata.module_id(), module->unique_id());
      EXPECT_THAT(pass_metadata.module_group_module_ids(),
                  ElementsAre(module_group.module(0).unique_id(),
                              module_group.module(1).unique_id()));
      EXPECT_GT(pass_metadata.start_timestamp_usec(), 0);
      EXPECT_LE(pass_metadata.start_timestamp_usec(),
                pass_metadata.end_timestamp_usec());
    }
  }
}

}  // namespace
}  // namespace xla
