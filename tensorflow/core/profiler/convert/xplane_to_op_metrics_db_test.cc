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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_db_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_db_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_db_testDTcc() {
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

#include "tensorflow/core/profiler/convert/xplane_to_op_metrics_db.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

void AddTensorFlowOpEvent(std::string&& tf_op_fullname,
                          int64_t start_timestamp_ns, int64_t duration_ns,
                          bool on_device, absl::string_view kernel_name,
                          XPlaneBuilder* plane, XLineBuilder* line) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_db_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/profiler/convert/xplane_to_op_metrics_db_test.cc", "AddTensorFlowOpEvent");

  absl::string_view name = on_device ? kernel_name : tf_op_fullname;
  XEventBuilder event = line->AddEvent(*plane->GetOrCreateEventMetadata(name));
  event.SetTimestampNs(start_timestamp_ns);
  event.SetDurationNs(duration_ns);
  if (!on_device) return;
  event.AddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      *plane->GetOrCreateStatMetadata(std::move(tf_op_fullname)));
}

TEST(ConvertXPlaneToOpMetricsDb, HostOpMetricsDb) {
  static constexpr char kTfOp1[] = "TfOp1";
  static constexpr char kTfOp2[] = "TfOp2";
  constexpr int64_t kTfOp1StartNs = 100000;
  constexpr int64_t kTfOp1DurationNs = 8000;
  constexpr int64_t kTfOp2StartNs = 110000;
  constexpr int64_t kTfOp2DurationNs = 10000;

  XSpace xspace;
  XPlane* xplane = GetOrCreateHostXPlane(&xspace);
  XPlaneBuilder host_plane(xplane);
  XLineBuilder thread1 = host_plane.GetOrCreateLine(/*line_id=*/10);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/false,
                       /*kernel_name=*/"", &host_plane, &thread1);
  XLineBuilder thread2 = host_plane.GetOrCreateLine(/*line_id=*/20);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/false,
                       /*kernel_name=*/"", &host_plane, &thread2);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp2, ":", kTfOp2), kTfOp2StartNs,
                       kTfOp2DurationNs, /*on_device=*/false,
                       /*kernel_name=*/"", &host_plane, &thread2);

  OpMetricsDb op_metrics = ConvertHostThreadsXPlaneToOpMetricsDb(*xplane);
  // Op1, Op2, Idle.
  EXPECT_EQ(3, op_metrics.metrics_db_size());
  uint64 total_op_duration =
      NanoToPico(kTfOp1DurationNs * 2 + kTfOp2DurationNs);
  EXPECT_EQ(total_op_duration, op_metrics.total_op_time_ps());
  uint64 total_duration = NanoToPico(kTfOp2StartNs - kTfOp1StartNs +
                                     kTfOp2DurationNs + kTfOp1DurationNs);
  EXPECT_EQ(total_duration, op_metrics.total_time_ps());

  // Verifies OpMetricsDb is built correctly.
  const OpMetrics& op_1 = op_metrics.metrics_db().at(0);
  EXPECT_EQ(kTfOp1, op_1.name());
  EXPECT_EQ(kTfOp1, op_1.category());
  EXPECT_EQ(2, op_1.occurrences());
  EXPECT_EQ(NanoToPico(kTfOp1DurationNs) * 2, op_1.time_ps());

  const OpMetrics& idle = op_metrics.metrics_db().at(1);
  EXPECT_EQ(kIdle, idle.name());
  EXPECT_EQ(kIdle, idle.category());
  // Idle time is the gap between Op2 start and the end of Op1, which is 2000ns.
  EXPECT_EQ(NanoToPico(2000), idle.time_ps());

  const OpMetrics& op_2 = op_metrics.metrics_db().at(2);
  EXPECT_EQ(kTfOp2, op_2.name());
  EXPECT_EQ(kTfOp2, op_2.category());
  EXPECT_EQ(1, op_2.occurrences());
  EXPECT_EQ(NanoToPico(kTfOp2DurationNs), op_2.time_ps());
}

TEST(ConvertXPlaneToOpMetricsDb, DeviceOpMetricsDb) {
  // TfOp1 has kernel1 and kernel2; TfOp2 has kernel3.
  static constexpr char kTfOp1[] = "TfOp1";
  static constexpr char kTfOp2[] = "TfOp2";
  static constexpr char kKernel1[] = "kernel1";
  static constexpr char kKernel2[] = "kernel2";
  static constexpr char kKernel3[] = "kernel3";
  constexpr int64_t kKernel1StartNs = 100000;
  constexpr int64_t kKernel1DurationNs = 8000;
  constexpr int64_t kKernel2StartNs = 110000;
  constexpr int64_t kKernel2DurationNs = 10000;
  constexpr int64_t kKernel3StartNs = 120000;
  constexpr int64_t kKernel3DurationNs = 10000;

  XSpace xspace;
  XPlane* xplane = GetOrCreateGpuXPlane(&xspace, /*device_ordinal=*/0);
  XPlaneBuilder device_plane(xplane);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(/*line_id=*/10);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel1StartNs,
                       kKernel1DurationNs, /*on_device=*/true, kKernel1,
                       &device_plane, &stream1);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel2StartNs,
                       kKernel2DurationNs, /*on_device=*/true, kKernel2,
                       &device_plane, &stream1);
  XLineBuilder stream2 = device_plane.GetOrCreateLine(/*line_id=*/20);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel1StartNs,
                       kKernel1DurationNs, /*on_device=*/true, kKernel1,
                       &device_plane, &stream2);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel2StartNs,
                       kKernel2DurationNs, /*on_device=*/true, kKernel2,
                       &device_plane, &stream2);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp2, ":", kTfOp2), kKernel3StartNs,
                       kKernel3DurationNs, /*on_device=*/true, kKernel3,
                       &device_plane, &stream2);

  OpMetricsDb op_metrics = ConvertDeviceTraceXPlaneToOpMetricsDb(*xplane);

  // kernel1, kernel2, kernel3, Idle.
  EXPECT_EQ(4, op_metrics.metrics_db_size());
  uint64 total_op_duration = NanoToPico(
      kKernel1DurationNs * 2 + kKernel2DurationNs * 2 + kKernel3DurationNs);
  EXPECT_EQ(total_op_duration, op_metrics.total_op_time_ps());
  // For device, the total_duration for each device is the total duration merged
  // from all GPU streams, which is from 100000 to 130000.
  uint64 total_duration =
      NanoToPico(kKernel3StartNs + kKernel3DurationNs - kKernel1StartNs);
  EXPECT_EQ(std::max(total_duration, total_op_duration),
            op_metrics.total_time_ps());

  // Verifies OpMetricsDb is built correctly.
  const OpMetrics& op_1 = op_metrics.metrics_db().at(0);
  EXPECT_EQ(absl::StrCat(kTfOp1, "/", kKernel1), op_1.name());
  EXPECT_EQ(kTfOp1, op_1.category());
  EXPECT_EQ(2, op_1.occurrences());
  EXPECT_EQ(NanoToPico(kKernel1DurationNs) * 2, op_1.time_ps());

  const OpMetrics& op_2 = op_metrics.metrics_db().at(1);
  EXPECT_EQ(absl::StrCat(kTfOp1, "/", kKernel2), op_2.name());
  EXPECT_EQ(kTfOp1, op_2.category());
  EXPECT_EQ(2, op_2.occurrences());
  EXPECT_EQ(NanoToPico(kKernel2DurationNs) * 2, op_2.time_ps());

  const OpMetrics& op_3 = op_metrics.metrics_db().at(2);
  EXPECT_EQ(absl::StrCat(kTfOp2, "/", kKernel3), op_3.name());
  EXPECT_EQ(kTfOp2, op_3.category());
  EXPECT_EQ(1, op_3.occurrences());
  EXPECT_EQ(NanoToPico(kKernel3DurationNs), op_3.time_ps());

  const OpMetrics& idle = op_metrics.metrics_db().at(3);
  EXPECT_EQ(kIdle, idle.name());
  EXPECT_EQ(kIdle, idle.category());
  // GPU is always busy in this example.
  EXPECT_EQ(NanoToPico(0), idle.time_ps());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
