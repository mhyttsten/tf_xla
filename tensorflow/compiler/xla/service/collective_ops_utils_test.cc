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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utils_testDTcc() {
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

#include "tensorflow/compiler/xla/service/collective_ops_utils.h"

#include <iterator>
#include <sstream>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

TEST(CollectiveOpsUtilsTest, GetParticipatingIDs_NoReplicaGroups) {
  std::vector<int> actual = GetParticipatingIDs(
                                /*current_id=*/0, /*total_participant_count=*/3,
                                /*groups=*/{})
                                .ConsumeValueOrDie();
  std::vector<int> expected = {0, 1, 2};
  EXPECT_EQ(actual, expected);
}

TEST(CollectiveOpsUtilsTest, GetParticipatingIDs_ReplicaGroups) {
  std::vector<ReplicaGroup> replica_groups(3);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(4);
  replica_groups[1].add_replica_ids(1);
  replica_groups[1].add_replica_ids(5);
  replica_groups[2].add_replica_ids(2);
  replica_groups[2].add_replica_ids(3);

  std::vector<int> actual =
      GetParticipatingIDs(
          /*current_id=*/1, /*total_participant_count=*/absl::nullopt,
          replica_groups)
          .ConsumeValueOrDie();
  std::vector<int> expected = {1, 5};
  EXPECT_EQ(actual, expected);
}

}  // namespace

// Tests for GetCollectOpGroupMode
namespace GetCollectiveOpGroupModeTest {
struct TestCase {
  bool has_channel_id;
  absl::optional<bool> use_global_device_ids;
  absl::optional<xla::CollectiveOpGroupMode> expected;

  std::string ToString() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utils_testDTcc mht_0(mht_0_v, 238, "", "./tensorflow/compiler/xla/service/collective_ops_utils_test.cc", "ToString");

    std::ostringstream s;
    s << (has_channel_id ? "chnl" : "nochnl");
    s << "_"
      << (use_global_device_ids
              ? (*use_global_device_ids ? "ugdi_true" : "ugdi_false")
              : "nougdi");
    return s.str();
  }
};

std::vector<TestCase> GetTestCases() {
  const std::vector<TestCase> test_cases = {
      // clang-format off
      // has_channel_id, use_global_device_ids, expected mode
      {false, absl::nullopt, CollectiveOpGroupMode::kCrossReplica},
      {false, false,         CollectiveOpGroupMode::kCrossReplica},
      {false, true,          absl::nullopt},
      {true,  absl::nullopt, CollectiveOpGroupMode::kCrossPartition},
      {true,  false,         CollectiveOpGroupMode::kCrossReplicaAndPartition},
      {true,  true,          CollectiveOpGroupMode::kFlattenedID},
      // clang-format on
  };
  return test_cases;
}

class GetCollectOpGroupModeTest : public testing::TestWithParam<TestCase> {};

TEST_P(GetCollectOpGroupModeTest, Test) {
  const TestCase &tc = GetParam();
  StatusOr<CollectiveOpGroupMode> actual =
      GetCollectiveOpGroupMode(tc.has_channel_id, tc.use_global_device_ids);
  if (tc.expected) {
    TF_ASSERT_OK(actual.status());
    EXPECT_EQ(*actual, *tc.expected);
  } else {
    EXPECT_FALSE(actual.ok());
  }
}

INSTANTIATE_TEST_SUITE_P(GetCollectOpGroupMode, GetCollectOpGroupModeTest,
                         testing::ValuesIn(GetTestCases()));
}  // namespace GetCollectiveOpGroupModeTest

// Tests for GetParticipatingDevices
namespace GetParticipatingDevicesTest {

// Test case for GetParticipatingDevices. Describes all the inputs to the
// function and for a given "setup", multiple "current_id" values and the
// expected output corresponding to those values.
struct TestCase {
  xla::Array2D<int> device_assignment;
  std::vector<std::vector<int>> replica_groups;
  bool has_channel_id;
  absl::optional<bool> use_global_device_ids;

  // For a given test case, its useful to test multiple 'current_id' inputs.
  struct CurrentIdAndOutput {
    int current_id;
    std::vector<int> expected_output;
  };
  std::vector<CurrentIdAndOutput> subtests;

  std::vector<std::vector<int>> participating_device_groups;
  bool expected_failure;

  std::string ToString() const;
};

// Please see the comment for GetParticipatingDevices() for a description of
// modes and their behavior.
std::string TestCase::ToString() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utils_testDTcc mht_1(mht_1_v, 312, "", "./tensorflow/compiler/xla/service/collective_ops_utils_test.cc", "TestCase::ToString");

  std::ostringstream s;
  StatusOr<CollectiveOpGroupMode> group_mode =
      GetCollectiveOpGroupMode(has_channel_id, use_global_device_ids);
  if (group_mode.ok()) {
    s << CollectiveOpGroupModeToString(*group_mode);
  } else {
    s << "Invalid";
  }

  s << "_" << device_assignment.n1() << "x" << device_assignment.n2();
  s << "_" << (replica_groups.empty() ? "NoRG" : "RG");
  s << "_" << subtests.size() << "SubTests";
  return s.str();
}

std::ostream &operator<<(std::ostream &os, const TestCase &tc) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScollective_ops_utils_testDTcc mht_2(mht_2_v, 331, "", "./tensorflow/compiler/xla/service/collective_ops_utils_test.cc", "operator<<");

  os << tc.ToString();
  return os;
}

std::vector<TestCase> GetTestCases() {
  std::vector<TestCase> test_cases;
  // clang-format off
  const std::vector<TestCase> cross_replica_test_cases = {
    // with empty replica groups, 1 partition.
    {
      {{33}, {44}, {55}},     // 3 replicas, 1 partition.
      {},                     // empty replica groups
      false,                  // has_channel_id
      false,                  // use_global_device_ids
      {                       // subtests
        // for empty replica group, any id should return all ids.
        {33, {33, 44, 55}},
        {44, {33, 44, 55}},
      },
      {{33, 44, 55}},          // participating device groups
      false                    // expected_failure
    },

    // empty replica groups, > 1 partition
    {
      {{33, 34}, {44, 45}, {55, 56}},  // 3r, 2p
      {},                              // empty replica groups
      false,                           // has_channel_id
      false,                           // use_global_device_ids
      // for empty replica group, any id should return all replicas within that
      // partition.
      {                                // subtests
        {33, {33, 44, 55}},
        {34, {34, 45, 56}},
        {45, {34, 45, 56}},
      },
      {{33, 44, 55}, {34, 45, 56}},    // participating device groups
      false                            // expected_failure
    },

    // non-empty replica groups, 1 partition.
    {
      {{33}, {44}, {55}},   // 3r, 1p.
      {{0}, {1, 2}},        // replica groups
      false,                // has_channel_id
      false,                // use_global_device_ids
      {                     // subtests
        // 33 is r0, so it's a singleton group.
        {33, {33}},
        // 44 is r1, so it should give {r1, r2}.
        {44, {44, 55}},
      },
      {{ 33 }, {44, 55}},    // participating device groups
      false                  // expected_failure
    },

    // non-empty, > 1 partition
    {
      {{33, 34}, {44, 45}, {55, 56}},   // 3r, 2p
      {{0}, {1, 2}},                    // replica groups
      false,                            // has_channel_id
      false,                            // use_global_device_ids
      {                                 // subtests
        // 33 is r0p0, so should be singleton.
        {33, {33}},
        // 34 is r0p1, so should be singleton.
        {34, {34}},
        // 45 is r1p1, so should get r1p1 and r2p1.
        {45, {45, 56}},
      },
      {{33}, {34}, {44, 55}, {45, 56}},  // participating device groups
      false                              // expected_failure
    },
  };

  // replica groups contain partition ids.
  const std::vector<TestCase> cross_partition_test_cases = {
    {
      // 3x4 device assignment
      {
        {33, 34, 35, 36}, {44, 45, 46, 47}, {55, 56, 57, 58}
      },
      {{0, 1}, {2, 3}},          // replica groups
      true,                      // has_channel_id
      absl::nullopt,             // use_global_device_ids
      {                          // subtests
        // 33 is r0p0, p0 group has p0, p1 so we get r0p0 and r0p1.
        {33, {33, 34}},
        // 35 is r0p2, so we get r0p2 and r0p3
        {35, {35, 36}},
        {45, {44, 45}},
        {47, {46, 47}},
        {58, {57, 58}},
      },
      {{33, 34}, {44, 45}, {55, 56},
       {35, 36}, {46, 47}, {57, 58}},  // participating device groups
      false                            // expected_failure
    }
  };


  const std::vector<TestCase> cross_replica_and_partition_test_cases = {
    {
      {{33, 34}, {44, 45}, {55, 56}},   // 3r, 2p
      {{0}, {1, 2}},                    // replica groups
      true,                             // has_channel_id
      false,                            // use_global_device_ids
      {                                 // subtests
        // 33 is r0p0, so should get r0 from all partitions.
        {33, {33, 34}},
        // 34 is r0p1, so should get r0 from all partitions.
        {34, {33, 34}},
        // 45 is r1p1, so should get r1, r2 from all partitions.
        {45, {44, 45, 55, 56}},
      },
      {{33, 34}, {44, 45, 55, 56}},   // participating device groups
      false
    },

    // empty replica group = all replicas, so we should get all devices.
    {
      {{33, 34}, {44, 45}, {55, 56}},   // 3r, 2p
      {},                               // replica groups
      true,                             // has_channel_id
      false,                            // use_global_device_ids
      {                                 // subtests
        {33, {33, 34, 44, 45, 55, 56}},
        {34, {33, 34, 44, 45, 55, 56}},
        {56, {33, 34, 44, 45, 55, 56}},
      },
      {{33, 34, 44, 45, 55, 56}},        // participating device groups
      false                              // expected_failure
    },
  };

  // Replica groups are flattened ids. For a 3x2 device assignment
  // used in these tests, the flattened ID and deviceId correspondence is as
  // follows:
  //   r0p0 = f#0 = d#33
  //   r0p1 = f#1 = d#34
  //   r1p0 = f#2 = d#44
  //   r1p1 = f#3 = d#45
  //   r2p0 = f#4 = d#55
  //   r2p1 = f#5 = d#56
  const std::vector<TestCase> flattened_id_test_cases = {
    {
      {{33, 34}, {44, 45}, {55, 56}},  // 3r, 2p
      {{0}, {1, 2}, {3, 4, 5}},        // replica groups
      true,                            // has_channel_id
      true,                            // use_global_device_ids
      {                                // subtests
        {33, {33}},
        {34, {34, 44}},
        {44, {34, 44}},
        {45, {45, 55, 56}},
        {55, {45, 55, 56}},
        {56, {45, 55, 56}},
      },
      {{33}, {34, 44}, {45, 55, 56}},  // participating device groups
      false                            // expected_failure
    },
    {
      {{33}},
      {},         // empty replica groups not allowed.
      true,       // has_channel_id
      true,       // use_global_device_ids
      {           // subtests
        {33, {33}},
      },
      {{33}},      // participating device groups
      true         // expected_failure
    },
  };

  const std::vector<TestCase> failure_test_cases = {
    // No channel id, use_global_device_ids = true;
    {
      {{33}, {44}, {55}},   // 3r, 1p
      {},                   // replica groups
      false,                // has_channel_id
      true,                 // use_global_device_ids
      {                     // subtests
        {33, {}},
      },
      {{33, 44, 55}},       // participating device groups
      true                  // expected_failure
    },
  };
  // clang-format on

  test_cases.insert(test_cases.end(), cross_replica_test_cases.begin(),
                    cross_replica_test_cases.end());
  // When use_global_device_ids is not present and channel_id is not present,
  // that implies cross replica mode as well.
  for (TestCase tc : cross_replica_test_cases) {
    tc.use_global_device_ids = absl::nullopt;
    test_cases.push_back(tc);
  }

  test_cases.insert(test_cases.end(), cross_partition_test_cases.begin(),
                    cross_partition_test_cases.end());
  test_cases.insert(test_cases.end(),
                    cross_replica_and_partition_test_cases.begin(),
                    cross_replica_and_partition_test_cases.end());
  test_cases.insert(test_cases.end(), flattened_id_test_cases.begin(),
                    flattened_id_test_cases.end());
  test_cases.insert(test_cases.end(), failure_test_cases.begin(),
                    failure_test_cases.end());

  return test_cases;
}

class GetParticipatingDevicesTest : public testing::TestWithParam<TestCase> {};

TEST_P(GetParticipatingDevicesTest, Test) {
  const TestCase &tc = GetParam();

  int64_t num_replicas = tc.device_assignment.n1();
  int64_t num_partitions = tc.device_assignment.n2();
  DeviceAssignment device_assignment(num_replicas, num_partitions);

  for (int64_t replica_id = 0; replica_id < num_replicas; ++replica_id) {
    for (int64_t partition_id = 0; partition_id < num_partitions;
         ++partition_id) {
      device_assignment(replica_id, partition_id) =
          tc.device_assignment(replica_id, partition_id);
    }
  }

  std::vector<ReplicaGroup> replica_groups;
  absl::c_transform(tc.replica_groups, std::back_inserter(replica_groups),
                    [](const std::vector<int> &ids) {
                      ReplicaGroup group;
                      for (int id : ids) {
                        group.add_replica_ids(id);
                      }
                      return group;
                    });

  StatusOr<CollectiveOpGroupMode> group_mode =
      GetCollectiveOpGroupMode(tc.has_channel_id, tc.use_global_device_ids);

  if (!group_mode.ok()) {
    EXPECT_TRUE(tc.expected_failure);
    return;
  }

  // Execute each sub-test.
  for (const TestCase::CurrentIdAndOutput &subtest : tc.subtests) {
    StatusOr<std::vector<GlobalDeviceId>> actual =
        GetParticipatingDevices(GlobalDeviceId(subtest.current_id),
                                device_assignment, replica_groups, *group_mode);
    if (!actual.ok()) {
      EXPECT_TRUE(tc.expected_failure);
      continue;
    }
    std::vector<GlobalDeviceId> expected;
    expected.reserve(subtest.expected_output.size());
    absl::c_transform(subtest.expected_output, std::back_inserter(expected),
                      [](int id) { return GlobalDeviceId(id); });
    EXPECT_EQ(*actual, expected);
  }

  StatusOr<std::vector<std::vector<GlobalDeviceId>>> actual_device_groups =
      GetParticipatingDevicesGroups(device_assignment, replica_groups,
                                    *group_mode);

  if (!actual_device_groups.ok()) {
    EXPECT_TRUE(tc.expected_failure);
    return;
  }

  std::vector<std::vector<GlobalDeviceId>> expect_device_groups;
  expect_device_groups.reserve(tc.participating_device_groups.size());

  for (auto subgroup : tc.participating_device_groups) {
    std::vector<GlobalDeviceId> subgroup_device_ids;
    subgroup_device_ids.reserve(subgroup.size());
    absl::c_transform(subgroup, std::back_inserter(subgroup_device_ids),
                      [](int id) { return GlobalDeviceId(id); });

    expect_device_groups.push_back(subgroup_device_ids);
  }

  EXPECT_THAT(*actual_device_groups,
              testing::UnorderedElementsAreArray(expect_device_groups));
}

INSTANTIATE_TEST_SUITE_P(GetParticipatingDevices, GetParticipatingDevicesTest,
                         testing::ValuesIn(GetTestCases()));

}  // namespace GetParticipatingDevicesTest
}  // namespace xla
