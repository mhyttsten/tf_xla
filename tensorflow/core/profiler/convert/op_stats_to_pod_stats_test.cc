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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_stats_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_stats_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_stats_testDTcc() {
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

#include "tensorflow/core/profiler/convert/op_stats_to_pod_stats.h"

#include "google/protobuf/any.pb.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/diagnostics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/diagnostics.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

const double kMaxError = 1e-6;
constexpr int kStepNum = 2;
constexpr int kCoreId = 1001;
constexpr int kStepTimePs = 1000;
constexpr int kHostComputePs = 50;
constexpr int kHostCompilePs = 50;
constexpr int kHostToHostPs = 50;
constexpr int kHostToDevicePs = 50;
constexpr int kHostPreparePs = 50;
constexpr int kDeviceCollectivePs = 350;
constexpr int kHostWaitInputPs = 50;
constexpr int kDeviceToDevicePs = 50;
constexpr int kDeviceToHostPs = 50;
constexpr int kDeviceCompute32Ps = 50;
constexpr int kDeviceCompute16Ps = 50;
constexpr int kDeviceWaitDevicePs = 50;
constexpr int kDeviceWaitHostPs = 50;
constexpr int kUnknownTimePs = 50;
static constexpr char kHostname[] = "host:123";

void CreateOpStats(OpStats* op_stats) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_stats_testDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/profiler/convert/op_stats_to_pod_stats_test.cc", "CreateOpStats");

  PerCoreStepInfo* info = op_stats->mutable_step_db()->add_step_sequence();
  info->set_step_num(kStepNum);
  StepInfoResult& step_info = (*info->mutable_step_info_per_core())[kCoreId];
  step_info.set_step_num(kStepNum);
  step_info.set_duration_ps(kStepTimePs);
  GenericStepBreakdown breakdown;
  auto& type_ps = *breakdown.mutable_type_ps();
  type_ps[HOST_COMPUTE] = kHostComputePs;
  type_ps[HOST_COMPILE] = kHostCompilePs;
  type_ps[HOST_TO_HOST] = kHostToHostPs;
  type_ps[HOST_TO_DEVICE] = kHostToDevicePs;
  type_ps[HOST_PREPARE] = kHostPreparePs;
  type_ps[DEVICE_COLLECTIVES] = kDeviceCollectivePs;
  type_ps[HOST_WAIT_INPUT] = kHostWaitInputPs;
  type_ps[DEVICE_TO_DEVICE] = kDeviceToDevicePs;
  type_ps[DEVICE_TO_HOST] = kDeviceToHostPs;
  type_ps[DEVICE_COMPUTE_32] = kDeviceCompute32Ps;
  type_ps[DEVICE_COMPUTE_16] = kDeviceCompute16Ps;
  type_ps[DEVICE_WAIT_DEVICE] = kDeviceWaitDevicePs;
  type_ps[DEVICE_WAIT_HOST] = kDeviceWaitHostPs;
  type_ps[UNKNOWN_TIME] = kUnknownTimePs;
  step_info.mutable_step_breakdown()->PackFrom(breakdown);
  CoreDetails& details = (*op_stats->mutable_core_id_to_details())[kCoreId];
  details.set_hostname(kHostname);
}

TEST(OpStatsToPodStats, GpuPodStats) {
  OpStats op_stats;
  CreateOpStats(&op_stats);
  PodStatsDatabase pod_stats_db = ConvertOpStatsToPodStats(op_stats);
  EXPECT_EQ(1, pod_stats_db.pod_stats_record_size());
  const PodStatsRecord& record = pod_stats_db.pod_stats_record(0);
  EXPECT_EQ(kStepNum, record.step_num());
  EXPECT_EQ(kHostname, record.host_name());
  EXPECT_NEAR(PicoToMicro(kStepTimePs), record.total_duration_us(), kMaxError);
  const auto& breakdown = record.step_breakdown_us();
  EXPECT_NEAR(PicoToMicro(kDeviceCompute32Ps + kDeviceCompute16Ps),
              breakdown.at(kDeviceCompute), kMaxError);
  EXPECT_NEAR(PicoToMicro(kDeviceToDevicePs + kDeviceWaitDevicePs),
              breakdown.at(kDeviceToDevice), kMaxError);
  EXPECT_NEAR(PicoToMicro(kDeviceCollectivePs),
              breakdown.at(kDeviceCollectives), kMaxError);
  EXPECT_NEAR(PicoToMicro(kHostComputePs), breakdown.at(kHostCompute),
              kMaxError);
  EXPECT_NEAR(PicoToMicro(kHostPreparePs), breakdown.at(kHostPrepare),
              kMaxError);
  EXPECT_NEAR(
      PicoToMicro(kHostWaitInputPs + kHostToDevicePs + kDeviceWaitHostPs),
      breakdown.at(kInput), kMaxError);
  EXPECT_NEAR(PicoToMicro(kDeviceToHostPs), breakdown.at(kOutput), kMaxError);
  EXPECT_NEAR(PicoToMicro(kHostCompilePs), breakdown.at(kCompile), kMaxError);
  EXPECT_NEAR(PicoToMicro(kUnknownTimePs), breakdown.at(kAllOthers), kMaxError);

  EXPECT_EQ(GetGenericEventTypeStr(kDeviceCollectives), record.bottleneck());
}

TEST(OpStatsToPodStats, Diagnostics) {
  OpStats op_stats;
  op_stats.mutable_step_db()->set_use_incomplete_step(true);
  PodStatsDatabase pod_stats_db = ConvertOpStatsToPodStats(op_stats);
  EXPECT_EQ(1, pod_stats_db.diagnostics().warnings_size());
  EXPECT_EQ(kErrorIncompleteStep, pod_stats_db.diagnostics().warnings(0));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
