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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_tf_statsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_tf_statsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_tf_statsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// The maximum number of Tensorflow Ops displayed on Tensorflow Stats page.
// 500 device side ops and 500 host side ops.
const int kMaxNumOfOps = 500;

TfStatsRecord ConvertOpMetricsToTfStatsRecord(
    bool on_device, const OpMetrics& metrics,
    double ridge_point_operational_intensity) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_tf_statsDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/profiler/convert/op_stats_to_tf_stats.cc", "ConvertOpMetricsToTfStatsRecord");

  TfStatsRecord record;
  record.set_host_or_device(on_device ? "Device" : "Host");
  record.set_is_eager(metrics.is_eager());
  record.set_op_type(metrics.category());
  record.set_op_name(metrics.name());
  SetExecutionTimes(metrics, &record);
  SetRooflineMetrics(metrics, ridge_point_operational_intensity, &record);
  return record;
}

TfStatsTable GenerateTfStatsTable(
    const OpMetricsDb& host_tf_metrics_db,
    const OpMetricsDb& device_tf_metrics_db,
    const KernelStatsByOpName& kernel_stats_by_op_name, double ridge_point,
    bool exclude_idle) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_tf_statsDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/profiler/convert/op_stats_to_tf_stats.cc", "GenerateTfStatsTable");

  TfStatsTable tf_stats_table;
  TfStatsRecord sentinel;
  sentinel.set_rank(0);
  sentinel.set_device_cumulative_total_self_time_as_fraction(0.0);
  sentinel.set_host_cumulative_total_self_time_as_fraction(0.0);
  const TfStatsRecord* prev_record = &sentinel;

  // Sets device-side TF stats.
  uint64 total_device_time_ps = TotalTimePs(device_tf_metrics_db, exclude_idle);
  double total_device_time_us = PicoToMicro(total_device_time_ps);
  for (const OpMetrics* metrics :
       SortedOpMetricsDb(device_tf_metrics_db, kMaxNumOfOps)) {
    if (exclude_idle && IsIdleOp(*metrics)) continue;
    TfStatsRecord* record = tf_stats_table.add_tf_stats_record();
    *record = ConvertOpMetricsToTfStatsRecord(
        /*on_device=*/true, *metrics, ridge_point);
    // Compute TensorCore utilization only on device side.
    auto iter = kernel_stats_by_op_name.find(record->op_name());
    if (iter != kernel_stats_by_op_name.end()) {
      record->set_gpu_tensorcore_utilization(
          SafeDivide(iter->second.tensor_core_duration_ns,
                     iter->second.total_duration_ns));
    } else {
      record->set_gpu_tensorcore_utilization(0.0);
    }
    SetRankAndDeviceTimeFractions(total_device_time_us, *prev_record, record);
    prev_record = record;
  }

  // Sets host-side TF stats.
  uint64 total_host_time_ps = TotalTimePs(host_tf_metrics_db, exclude_idle);
  double total_host_time_us = PicoToMicro(total_host_time_ps);
  for (const OpMetrics* metrics : tensorflow::profiler::SortedOpMetricsDb(
           host_tf_metrics_db, kMaxNumOfOps)) {
    if (exclude_idle && IsIdleOp(*metrics)) continue;
    TfStatsRecord* record = tf_stats_table.add_tf_stats_record();
    *record = ConvertOpMetricsToTfStatsRecord(
        /*on_device=*/false, *metrics, ridge_point);
    // Host side TensorCore utilization is always 0.0
    record->set_gpu_tensorcore_utilization(0.0);
    SetRankAndHostTimeFractions(total_host_time_us, *prev_record, record);
    prev_record = record;
  }
  return tf_stats_table;
}

}  // namespace

TfStatsDatabase ConvertOpStatsToTfStats(const OpStats& op_stats) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_tf_statsDTcc mht_2(mht_2_v, 276, "", "./tensorflow/core/profiler/convert/op_stats_to_tf_stats.cc", "ConvertOpStatsToTfStats");

  const OpMetricsDb& host_tf_metrics_db = op_stats.host_op_metrics_db();
  OpMetricsDb device_tf_metrics_db =
      CreateTfMetricsDbFromDeviceOpMetricsDb(op_stats.device_op_metrics_db());
  double ridge_point = op_stats.perf_env().ridge_point();
  KernelStatsByOpName kernel_stats_by_op_name =
      GroupKernelReportsByOpName(op_stats.kernel_stats_db());
  TfStatsDatabase tf_stats_db;
  *tf_stats_db.mutable_with_idle() = GenerateTfStatsTable(
      host_tf_metrics_db, device_tf_metrics_db, kernel_stats_by_op_name,
      ridge_point, /*exclude_idle=*/false);
  *tf_stats_db.mutable_without_idle() = GenerateTfStatsTable(
      host_tf_metrics_db, device_tf_metrics_db, kernel_stats_by_op_name,
      ridge_point, /*exclude_idle=*/true);
  tf_stats_db.set_device_type(op_stats.run_environment().device_type());
  return tf_stats_db;
}

}  // namespace profiler
}  // namespace tensorflow
