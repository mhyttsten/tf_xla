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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignment_testDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignment_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignment_testDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/group_assignment.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace dtensor {
namespace {

mlir::DenseIntElementsAttr CreateGroupAssignmentAttr(
    mlir::MLIRContext& context,
    const std::vector<std::vector<int>>& replica_ids) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignment_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/dtensor/mlir/group_assignment_test.cc", "CreateGroupAssignmentAttr");

  int num_groups = replica_ids.size();
  int group_size = replica_ids.front().size();
  llvm::SmallVector<int32, 4> flat_replica_ids;
  flat_replica_ids.reserve(num_groups * group_size);
  for (const std::vector<int>& group : replica_ids) {
    CHECK_EQ(group.size(), group_size);
    flat_replica_ids.insert(flat_replica_ids.end(), group.begin(), group.end());
  }
  auto shaped_type = mlir::RankedTensorType::get(
      {num_groups, group_size}, mlir::IntegerType::get(&context, 32));
  return mlir::DenseIntElementsAttr::get(shaped_type, flat_replica_ids);
}

GroupAssignment CreateGroupAssignment(
    mlir::MLIRContext& context,
    const std::vector<std::vector<int>>& replica_ids, int num_slices,
    int slice_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignment_testDTcc mht_1(mht_1_v, 229, "", "./tensorflow/dtensor/mlir/group_assignment_test.cc", "CreateGroupAssignment");

  mlir::DenseIntElementsAttr group_assignment_attr =
      CreateGroupAssignmentAttr(context, replica_ids);
  StatusOr<GroupAssignment> group_assignment = GroupAssignment::FromMLIR(
      group_assignment_attr,
      GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(
          num_slices, slice_size));
  CHECK(group_assignment.ok());
  return *group_assignment;
}

GroupAssignment CreateGroupAssignment(
    mlir::MLIRContext& context,
    const std::vector<std::vector<int>>& replica_ids,
    absl::flat_hash_map<GroupAssignment::ReplicaId, GroupAssignment::DeviceId>
        map) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSgroup_assignment_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/dtensor/mlir/group_assignment_test.cc", "CreateGroupAssignment");

  mlir::DenseIntElementsAttr group_assignment_attr =
      CreateGroupAssignmentAttr(context, replica_ids);
  StatusOr<GroupAssignment> group_assignment = GroupAssignment::FromMLIR(
      group_assignment_attr,
      GroupAssignment::ReplicaToDeviceMap(std::move(map)));
  CHECK(group_assignment.ok());
  return *group_assignment;
}

TEST(DTensorGroupAssignmentTest, InputOutput) {
  mlir::MLIRContext context;

  mlir::DenseIntElementsAttr group_assignment_attr_in =
      CreateGroupAssignmentAttr(context,
                                /*replica_ids=*/{{0, 1, 2, 3, 4, 5, 6, 7}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto group_assignment,
      GroupAssignment::FromMLIR(
          group_assignment_attr_in,
          GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(
              /*num_slices=*/1, /*slice_size=*/8)));
  EXPECT_EQ(group_assignment.replica_ids(),
            std::vector<std::vector<int>>({{0, 1, 2, 3, 4, 5, 6, 7}}));

  mlir::DenseIntElementsAttr group_assignment_attr_out =
      group_assignment.GlobalToMLIR(context);
  EXPECT_EQ(group_assignment_attr_out, group_assignment_attr_in);

  group_assignment_attr_out =
      group_assignment.SliceToMLIR(context, /*slice_id=*/0).ValueOrDie();
  EXPECT_EQ(group_assignment_attr_out, group_assignment_attr_in);
}

TEST(DTensorGroupAssignmentTest, BadInput) {
  mlir::MLIRContext context;

  mlir::DenseIntElementsAttr indivisible_donut_size =
      CreateGroupAssignmentAttr(context,
                                /*replica_ids=*/{{0, 1, 2, 3, 4, 5, 6, 7, 8}});
  EXPECT_FALSE(
      GroupAssignment::FromMLIR(
          indivisible_donut_size,
          GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(
              /*num_slices=*/1, /*slice_size=*/8))
          .ok());

  mlir::DenseIntElementsAttr duplicate_replica_ids =
      CreateGroupAssignmentAttr(context,
                                /*replica_ids=*/{{0, 1, 2, 3, 4, 5, 6, 6}});
  EXPECT_FALSE(
      GroupAssignment::FromMLIR(
          duplicate_replica_ids,
          GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(
              /*num_slices=*/1, /*slice_size=*/8))
          .ok());
}

TEST(DTensorGroupAssignmentTest, Properties) {
  mlir::MLIRContext context;
  GroupAssignment group_assignment =
      CreateGroupAssignment(context,
                            /*replica_ids=*/{{0, 1, 2, 3}},
                            /*num_slices=*/1, /*slice_size=*/4);
  EXPECT_EQ(group_assignment.num_groups(), 1);
  EXPECT_EQ(group_assignment.group_size(), 4);
  EXPECT_EQ(group_assignment.num_replica_ids(), 4);
  EXPECT_EQ(group_assignment.replica_ids(),
            std::vector<std::vector<int>>({{0, 1, 2, 3}}));
}

TEST(DTensorGroupAssignmentTest, GlobalAllReduceSingleDonut) {
  mlir::MLIRContext context;
  GroupAssignment group_assignment =
      CreateGroupAssignment(context,
                            /*replica_ids=*/{{0, 1, 2, 3, 4, 5, 6, 7}},
                            /*num_slices=*/1, /*slice_size=*/8);
  EXPECT_TRUE(group_assignment.IsWithinSlices());
  EXPECT_EQ(group_assignment.replica_ids(),
            std::vector<std::vector<int>>({{0, 1, 2, 3, 4, 5, 6, 7}}));
  EXPECT_EQ(group_assignment.replica_ids(0),
            std::vector<std::vector<int>>({{0, 1, 2, 3, 4, 5, 6, 7}}));
}

TEST(DTensorGroupAssignmentTest, GlobalAllReduceTwoDonuts) {
  mlir::MLIRContext context;
  GroupAssignment group_assignment =
      CreateGroupAssignment(context,
                            /*replica_ids=*/{{1, 2, 0, 3}},
                            /*num_slices=*/2, /*slice_size=*/2);
  EXPECT_FALSE(group_assignment.IsWithinSlices());
  EXPECT_EQ(group_assignment.replica_ids(),
            std::vector<std::vector<int>>({{1, 2, 0, 3}}));
  EXPECT_EQ(group_assignment.replica_ids(0),
            std::vector<std::vector<int>>({{1, 0}}));
  EXPECT_EQ(group_assignment.replica_ids(1),
            std::vector<std::vector<int>>({{0, 1}}));
}

TEST(DTensorGroupAssignmentTest, SubgroupAllReduceFourDonuts) {
  mlir::MLIRContext context;
  std::vector<std::vector<int>> global(
      {{0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}});
  GroupAssignment group_assignment =
      CreateGroupAssignment(context,
                            /*replica_ids=*/global,
                            /*map=*/
                            {
                                {0, {0, 0}},
                                {1, {0, 1}},
                                {2, {1, 0}},
                                {3, {1, 1}},
                                {4, {0, 2}},
                                {5, {0, 3}},
                                {6, {1, 2}},
                                {7, {1, 3}},
                                {8, {2, 0}},
                                {9, {2, 1}},
                                {10, {3, 0}},
                                {11, {3, 1}},
                                {12, {2, 2}},
                                {13, {2, 3}},
                                {14, {3, 2}},
                                {15, {3, 3}},
                            });
  EXPECT_FALSE(group_assignment.IsWithinSlices());
  EXPECT_EQ(group_assignment.replica_ids(), global);
  EXPECT_EQ(group_assignment.host_replica_ids(0),
            std::vector<std::vector<int>>({{0}, {1}}));
  EXPECT_EQ(group_assignment.replica_ids(0),
            std::vector<std::vector<int>>({{0, 2}, {1, 3}}));
  EXPECT_EQ(group_assignment.host_replica_ids(1),
            std::vector<std::vector<int>>({{2}, {3}}));
  EXPECT_EQ(group_assignment.replica_ids(1),
            std::vector<std::vector<int>>({{0, 2}, {1, 3}}));
  EXPECT_EQ(group_assignment.host_replica_ids(2),
            std::vector<std::vector<int>>({{8}, {9}}));
  EXPECT_EQ(group_assignment.replica_ids(2),
            std::vector<std::vector<int>>({{0, 2}, {1, 3}}));
  EXPECT_EQ(group_assignment.host_replica_ids(3),
            std::vector<std::vector<int>>({{10}, {11}}));
  EXPECT_EQ(group_assignment.replica_ids(3),
            std::vector<std::vector<int>>({{0, 2}, {1, 3}}));
}

}  // namespace
}  // namespace dtensor
}  // namespace tensorflow
