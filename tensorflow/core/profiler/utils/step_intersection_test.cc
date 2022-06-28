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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersection_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersection_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersection_testDTcc() {
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

#include "tensorflow/core/profiler/utils/step_intersection.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profiler {
namespace {

using PerHostStepDb =
    absl::flat_hash_map<uint32 /*=host_id*/, StepDatabaseResult>;

constexpr uint64 kStepDurationPs = 2000000000;
constexpr uint32 kNumStepsPerHost = 10;
constexpr uint64 kStepGapPs = 0;
constexpr uint32 kNumCoresPerHost = 8;

PerCoreStepInfo CreateOneTestStep(uint32 host_id, uint32 num_steps,
                                  uint32 step_idx, uint64 step_begin_ps) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersection_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/profiler/utils/step_intersection_test.cc", "CreateOneTestStep");

  PerCoreStepInfo result;
  uint32 step_num =
      step_idx * host_id;  // creates the situation where each host has a
                           // different step number for the same step.
  result.set_step_num(step_num);
  StepInfoResult info;
  info.set_step_num(step_num);
  if (host_id == 0 && step_idx == (num_steps - 1)) {
    // Makes the last step on host_id is little bit shorter so that host-0 will
    // be chosen as the chief.
    info.set_duration_ps(kStepDurationPs - 1);
  } else {
    info.set_duration_ps(kStepDurationPs);
  }
  info.set_begin_ps(step_begin_ps);
  // Don't care about the rest of the fields in StepInfoResult.
  for (uint32 core_id = 0; core_id < kNumCoresPerHost; core_id++) {
    (*result.mutable_step_info_per_core())[core_id] = info;
    // Don't care about the rest of the fields in PerCoreStepInfo.
  }
  return result;
}

PerHostStepDb CreateTestSteps(uint32 num_hosts, uint64 shift_ps) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersection_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/profiler/utils/step_intersection_test.cc", "CreateTestSteps");

  PerHostStepDb result;
  uint64 first_step_begin_ps = 0;
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    StepDatabaseResult step_db;
    uint64 step_begin_ps = first_step_begin_ps;
    for (uint32 step_idx = 0; step_idx < kNumStepsPerHost; step_idx++) {
      *step_db.add_step_sequence() =
          CreateOneTestStep(host_id, kNumStepsPerHost, step_idx, step_begin_ps);
      step_begin_ps += (kStepDurationPs + kStepGapPs);
    }
    result[host_id] = step_db;
    first_step_begin_ps += shift_ps;
  }
  return result;
}

PerHostStepDb CreateEmptyIntersectTestSteps() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersection_testDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/profiler/utils/step_intersection_test.cc", "CreateEmptyIntersectTestSteps");

  PerHostStepDb result;

  uint64 step_begin_ps;
  uint32 host_id;

  // Host-0
  host_id = 0;
  step_begin_ps = 0;
  uint64 host_0_num_steps = 10;
  StepDatabaseResult step_db_0;
  for (uint32 step_idx = 0; step_idx < host_0_num_steps; step_idx++) {
    *step_db_0.add_step_sequence() =
        CreateOneTestStep(host_id, host_0_num_steps, step_idx, step_begin_ps);
    step_begin_ps += (kStepDurationPs + kStepGapPs);
  }
  result[host_id] = step_db_0;

  // Host-1
  host_id = 1;
  step_begin_ps = (host_0_num_steps - 2) * (kStepDurationPs + kStepGapPs);
  uint64 host_1_num_steps = 5;
  StepDatabaseResult step_db_1;
  for (uint32 step_idx = 0; step_idx < host_1_num_steps; step_idx++) {
    *step_db_1.add_step_sequence() =
        CreateOneTestStep(host_id, host_1_num_steps, step_idx, step_begin_ps);
    step_begin_ps += (kStepDurationPs + kStepGapPs);
  }
  result[host_id] = step_db_1;

  // Host-2
  host_id = 2;
  step_begin_ps = (host_0_num_steps + host_1_num_steps - 4) *
                  (kStepDurationPs + kStepGapPs);
  uint64 host_2_num_steps = 10;
  StepDatabaseResult step_db_2;
  for (uint32 step_idx = 0; step_idx < host_2_num_steps; step_idx++) {
    *step_db_2.add_step_sequence() =
        CreateOneTestStep(host_id, host_2_num_steps, step_idx, step_begin_ps);
    step_begin_ps += (kStepDurationPs + kStepGapPs);
  }
  result[host_id] = step_db_2;

  return result;
}

PerHostStepDb CreateNoStep(uint32 num_hosts) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSstep_intersection_testDTcc mht_3(mht_3_v, 299, "", "./tensorflow/core/profiler/utils/step_intersection_test.cc", "CreateNoStep");

  PerHostStepDb result;
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    StepDatabaseResult step_db;
    result[host_id] = step_db;
  }
  return result;
}

absl::flat_hash_map<uint32 /*=host_id*/, const StepDatabaseResult*> Convert(
    const PerHostStepDb& perhost_stepdb) {
  absl::flat_hash_map<uint32 /*=host_id*/, const StepDatabaseResult*> result;
  for (const auto& hostid_stepdb : perhost_stepdb) {
    auto host_id = hostid_stepdb.first;
    const auto& step_db = hostid_stepdb.second;
    result[host_id] = &step_db;
  }
  return result;
}

TEST(StepIntersectionTest, EachHostShiftedBy1StepDuration) {
  uint32 num_hosts = 4;
  uint64 shift_ps = kStepDurationPs;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32 dst_num_steps = kNumStepsPerHost - num_hosts + 1;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  uint32 src_first_step_index = intersection.FirstStepIndex(0);
  EXPECT_EQ(src_first_step_index, num_hosts - 1);
  std::vector<uint32> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32 i = 0; i < dst_num_steps; i++) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
}

TEST(StepIntersectionTest, ExactlyNoShift) {
  uint32 num_hosts = 4;
  uint64 shift_ps = 0;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32 dst_num_steps = kNumStepsPerHost;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  std::vector<uint32> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32 i = 0; i < dst_num_steps; i++) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    uint32 src_first_step_index = intersection.FirstStepIndex(host_id);
    EXPECT_EQ(src_first_step_index, 0);
  }
}

TEST(StepIntersectionTest, EachHostShiftedByJustABit) {
  uint32 num_hosts = 4;
  uint64 shift_ps = 100;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32 dst_num_steps = kNumStepsPerHost;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  std::vector<uint32> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32 i = 0; i < dst_num_steps; i++) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    uint32 src_first_step_index = intersection.FirstStepIndex(host_id);
    EXPECT_EQ(src_first_step_index, 0);
  }
}

TEST(StepIntersectionTest, SingleHost) {
  uint32 num_hosts = 1;
  uint64 shift_ps = 0;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(kNumStepsPerHost, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  uint32 dst_num_steps = kNumStepsPerHost;
  EXPECT_EQ(intersection.NumSteps(), dst_num_steps);

  std::vector<uint32> dst_step_numbers = intersection.DstStepNumbers();
  for (uint32 i = 0; i < dst_num_steps; i++) {
    EXPECT_EQ(dst_step_numbers[i], i);
  }
  for (uint32 host_id = 0; host_id < num_hosts; host_id++) {
    uint32 src_first_step_index = intersection.FirstStepIndex(host_id);
    EXPECT_EQ(src_first_step_index, 0);
  }
}

TEST(StepIntersectionTest, WithMaxSteps) {
  uint32 num_hosts = 4;
  uint64 shift_ps = 0;
  uint32 max_steps = 3;

  PerHostStepDb perhost_stepdb = CreateTestSteps(num_hosts, shift_ps);
  StepIntersection intersection =
      StepIntersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), kNumStepsPerHost - max_steps);
  EXPECT_EQ(intersection.NumSteps(), max_steps);
}

TEST(StepIntersectionTest, NoStep) {
  uint32 num_hosts = 4;
  uint32 max_steps = 100;
  PerHostStepDb perhost_stepdb = CreateNoStep(num_hosts);
  StepIntersection intersection =
      StepIntersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.NumSteps(), 0);
  EXPECT_FALSE(intersection.EmptyIntersect());
}

TEST(StepIntersectionTest, EmptyIntersection) {
  uint32 max_steps = 100;
  PerHostStepDb perhost_stepdb = CreateEmptyIntersectTestSteps();
  StepIntersection intersection =
      StepIntersection(max_steps, Convert(perhost_stepdb));
  EXPECT_EQ(intersection.StepsDropped(), 0);
  EXPECT_EQ(intersection.NumSteps(), 0);
  EXPECT_TRUE(intersection.EmptyIntersect());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
