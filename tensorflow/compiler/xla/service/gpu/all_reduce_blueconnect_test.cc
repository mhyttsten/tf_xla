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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSall_reduce_blueconnect_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSall_reduce_blueconnect_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSall_reduce_blueconnect_testDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/all_reduce_blueconnect.h"

#include <memory>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/platform/status_matchers.h"

namespace xla {
namespace {

using ::tensorflow::testing::IsOkAndHolds;
using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;

using AllReduceBlueConnectTest = HloTestBase;

void SetModuleConfig(HloModule& module, size_t replica_count) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSall_reduce_blueconnect_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/gpu/all_reduce_blueconnect_test.cc", "SetModuleConfig");

  DeviceAssignment device_assignment(replica_count, /*computation_count=*/1);
  device_assignment.FillIota(0);
  module.config().set_replica_count(replica_count);
  module.config().set_static_device_assignment(device_assignment);
}

TEST_F(AllReduceBlueConnectTest, OneStage) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  ROOT crs = f32[4,4] all-reduce(p0), to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));

  // clang-format off
  std::vector<std::vector<int64_t>> scatter_gather_groups = {
      {0, 1, 2, 3}, {4, 5, 6, 7}};
  std::vector<std::vector<int64_t>> new_all_reduce_groups = {
      {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  // clang-format on

  auto bitcast = AllOf(op::Shape("f32[16]"), op::Bitcast(op::Parameter(0)));
  auto reduce_scatter = AllOf(op::Shape("f32[4]"), op::ReduceScatter(bitcast),
                              op::ReplicaGroups(scatter_gather_groups));
  auto all_reduce = AllOf(op::Shape("f32[4]"), op::AllReduce(reduce_scatter),
                          op::ReplicaGroups(new_all_reduce_groups));
  auto all_gather = AllOf(op::Shape("f32[16]"), op::AllGather(all_reduce),
                          op::ReplicaGroups(scatter_gather_groups));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Shape("f32[4,4]"), op::Bitcast(all_gather)));
}

TEST_F(AllReduceBlueConnectTest, TwoStage) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  ROOT crs = f32[4,4] all-reduce(p0), to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/16);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));

  std::vector<std::vector<int64_t>> outer_scatter_gather_groups = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  std::vector<std::vector<int64_t>> inner_scatter_gather_groups = {
      {0, 4}, {8, 12}, {1, 5}, {9, 13}, {2, 6}, {10, 14}, {3, 7}, {11, 15}};
  std::vector<std::vector<int64_t>> new_all_reduce_groups = {
      {0, 8}, {4, 12}, {1, 9}, {5, 13}, {2, 10}, {6, 14}, {3, 11}, {7, 15}};

  auto bitcast0 = AllOf(op::Shape("f32[16]"), op::Bitcast(op::Parameter(0)));
  auto reduce_scatter0 = AllOf(op::Shape("f32[4]"), op::ReduceScatter(bitcast0),
                               op::ReplicaGroups(outer_scatter_gather_groups));
  auto bitcast1 = AllOf(op::Shape("f32[4]"), op::Bitcast(reduce_scatter0));
  auto reduce_scatter1 = AllOf(op::Shape("f32[2]"), op::ReduceScatter(bitcast1),
                               op::ReplicaGroups(inner_scatter_gather_groups));
  auto all_reduce = AllOf(op::Shape("f32[2]"), op::AllReduce(reduce_scatter1),
                          op::ReplicaGroups(new_all_reduce_groups));
  auto all_gather0 = AllOf(op::Shape("f32[4]"), op::AllGather(all_reduce),
                           op::ReplicaGroups(inner_scatter_gather_groups));
  auto bitcast2 = AllOf(op::Shape("f32[4]"), op::Bitcast(all_gather0));
  auto all_gather1 = AllOf(op::Shape("f32[16]"), op::AllGather(bitcast2),
                           op::ReplicaGroups(outer_scatter_gather_groups));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Shape("f32[4,4]"), op::Bitcast(all_gather1)));
}

TEST_F(AllReduceBlueConnectTest, TwoOperands) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  p1 = f32[4,4,2] parameter(1)
  ROOT crs = (f32[4,4], f32[4,4,2]) all-reduce(p0, p1), to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));

  // clang-format off
  std::vector<std::vector<int64_t>> scatter_gather_groups = {
      {0, 1, 2, 3}, {4, 5, 6, 7}};
  std::vector<std::vector<int64_t>> new_all_reduce_groups = {
      {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  // clang-format on

  auto bitcast0 = AllOf(op::Shape("f32[16]"), op::Bitcast(op::Parameter(0)));
  auto bitcast1 = AllOf(op::Shape("f32[32]"), op::Bitcast(op::Parameter(1)));
  auto reduce_scatter = AllOf(op::Shape("(f32[4], f32[8])"),
                              op::ReduceScatter(bitcast0, bitcast1),
                              op::ReplicaGroups(scatter_gather_groups));
  auto all_reduce = AllOf(op::Shape("(f32[4], f32[8])"),
                          op::AllReduce(op::GetTupleElement(reduce_scatter, 0),
                                        op::GetTupleElement(reduce_scatter, 1)),
                          op::ReplicaGroups(new_all_reduce_groups));
  auto all_gather = AllOf(op::Shape("(f32[16], f32[32])"),
                          op::AllGather(op::GetTupleElement(all_reduce, 0),
                                        op::GetTupleElement(all_reduce, 1)),
                          op::ReplicaGroups(scatter_gather_groups));
  auto bitcast2 = AllOf(op::Shape("f32[4,4]"),
                        op::Bitcast(op::GetTupleElement(all_gather, 0)));
  auto bitcast3 = AllOf(op::Shape("f32[4,4,2]"),
                        op::Bitcast(op::GetTupleElement(all_gather, 1)));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(bitcast2, bitcast3));
}

TEST_F(AllReduceBlueConnectTest, DifferentNumLocalDevicesWithinReplicaGroup) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  ROOT crs = f32[4,4] all-reduce(p0),
    replica_groups={{0,1,2,7},{3,4,5,6}}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(AllReduceBlueConnectTest, DifferentNumLocalDevicesAcrossReplicaGroups) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  ROOT crs = f32[4,4] all-reduce(p0),
    replica_groups={{0,1,4,5},{2,3,6,7},{8,9,10,11},{12,13,14,15}}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/16);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(AllReduceBlueConnectTest, OperandIndivisible) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  p1 = f32[9] parameter(1)
  ROOT crs = (f32[4,4], f32[9]) all-reduce(p0, p1), to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
