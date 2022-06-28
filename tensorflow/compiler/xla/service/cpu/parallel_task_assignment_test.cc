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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignment_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignment_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignment_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"

#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features_fake.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class ParallelTaskAssignmentTest : public HloTestBase {
 protected:
  const HloCostAnalysis::ShapeSizeFunction shape_size_func_ =
      cpu::CpuExecutable::ShapeSizeBytes;

  // Use any value larger than 2 since we only test whether a module is
  // parallelized or not
  const int max_parallelism_ = 10;

  cpu::TargetMachineFeaturesWithFakeAlignmentLogic target_machine_features_;

  ParallelTaskAssignmentTest()
      : HloTestBase(), target_machine_features_([](int64_t shape_size) {
          return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
        }) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSparallel_task_assignment_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/cpu/parallel_task_assignment_test.cc", "ParallelTaskAssignmentTest");
}

  StatusOr<bool> RunParallelTaskAssigner(HloModule* module) {
    return cpu::ParallelTaskAssigner(max_parallelism_, shape_size_func_,
                                     &target_machine_features_)
        .Run(module);
  }
};

TEST_F(ParallelTaskAssignmentTest, DotOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_Dot
    ENTRY Dot {
      dot_lhs = f32[196614,2]{1,0} parameter(0)
      dot_rhs = f32[2,1]{1,0} parameter(1)
      ROOT dot = f32[196614,1]{1,0} dot(dot_lhs, dot_rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest,
       FusedComputationWithDotOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_DotNestedInFusedComp
    fused_computation.0 {
      parameter.0 = f32[196614,2]{1,0} parameter(0)
      parameter.0.1 = f32[2,1]{1,0} parameter(1)
      parameter.0.2 = f32[196614,1]{1,0} parameter(2)
      dot.0 = f32[196614,1]{1,0} dot(parameter.0, parameter.0.1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add.0 = f32[196614,1]{1,0} add(dot.0, parameter.0.2)

    }
    ENTRY DotNestedInFusedComp {
      parameter = f32[196614,2]{1,0} parameter(0)
      parameter.1 = f32[2,1]{1,0} parameter(1)
      parameter.2 = f32[196614,1]{1,0} parameter(2)
      ROOT fusion = f32[196614,1]{1,0} fusion(parameter, parameter.1,
        parameter.2), kind=kOutput, calls=fused_computation.0
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, RngOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_rng
    ENTRY Rng {
      src0 = f32[] parameter(0)
      src1 = f32[] parameter(1)
      ROOT rng0 = f32[1234567,2]{1,0} rng(f32[] src0, f32[] src1),
      distribution=rng_uniform
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, InfeedOutfeedOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_infeed_outfeed
    ENTRY InfeedOutfeed {
      token0 = token[] after-all()
      infeed0 = (u32[12345678,2]{1,0}, token[]) infeed(token0)
      infeed0.data = u32[12345678,2]{1,0} get-tuple-element((u32[12345678,2]{1,0}, token[]) infeed0), index=0
      ROOT outfeed0 = token[] outfeed(infeed0.data, token0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, InPlaceDynamicUpdateSliceNotParallelized) {
  // A dynamic-update-slice within a while loop.  This construction is an easy
  // way to make a DUS which can be run "in-place" (i.e. the input and output
  // are the same buffer, and running the DUS only writes to the updated
  // elements).
  const std::string hlo_string = R"(
  HloModule test

  body {
    zero = s32[] constant(0)
    one = s32[] constant(1)
    ten = s32[] constant(10)
    loop_carry = (s32[], u32[1,100], u32[10000,100]) parameter(0)
    i = s32[] get-tuple-element(loop_carry), index=0
    i_plus_ten = s32[] add(i, ten)
    update = u32[1,100] get-tuple-element(loop_carry), index=1
    data = u32[10000,100] get-tuple-element(loop_carry), index=2
    new_data = u32[10000,100] dynamic-update-slice(data, update, i_plus_ten, zero)
    new_i = s32[] add(i, one)
    ROOT tuple = (s32[], u32[1,100], u32[10000,100]) tuple(new_i, update, new_data)
  }

  cond {
    loop_carry = (s32[], u32[1,100], u32[10000,100]) parameter(0)
    two = s32[] constant(2)
    i = s32[] get-tuple-element(loop_carry), index=0
    ROOT less-than = pred[] compare(i, two), direction=LT
  }

  ENTRY test {
    zero = s32[] constant(0)
    initial_i = s32[] parameter(0)
    update = u32[1,100] parameter(1)
    data = u32[10000,100] parameter(2)
    tuple = (s32[], u32[1,100], u32[10000,100]) tuple(initial_i, update, data)
    ROOT while = (s32[], u32[1,100], u32[10000,100]) while(tuple), condition=cond, body=body
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, AllReduceNotParallelized) {
  constexpr char hlo_string[] = R"(
  HloModule TestTaskParallel_allreduce
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY CRS {
      input = f32[1234567] parameter(0)
      ROOT crs = f32[1234567] all-reduce(input), replica_groups={}, to_apply=add
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, ConstantNotParallelized) {
  constexpr char hlo_string[] = R"(
  HloModule TestTaskParallel_constant
    ENTRY const {
      ROOT constant = f32[1234567] constant({...})
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
