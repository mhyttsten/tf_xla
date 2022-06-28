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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc() {
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

#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

#include <algorithm>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"

namespace tensorflow {
namespace profiler {

const absl::string_view kIdle = "IDLE";

namespace {

class DeviceTfOpMetricsDbBuilder : public OpMetricsDbBuilder {
 public:
  explicit DeviceTfOpMetricsDbBuilder(OpMetricsDb* db)
      : OpMetricsDbBuilder(db) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.cc", "DeviceTfOpMetricsDbBuilder");
}

  void UpdateTfOpMetricsWithDeviceOpMetrics(
      absl::string_view tf_op_name, absl::string_view tf_op_type,
      const OpMetrics& device_op_metrics) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tf_op_name: \"" + std::string(tf_op_name.data(), tf_op_name.size()) + "\"");
   mht_1_v.push_back("tf_op_type: \"" + std::string(tf_op_type.data(), tf_op_type.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.cc", "UpdateTfOpMetricsWithDeviceOpMetrics");

    OpMetrics* tf_op_metrics = OpMetricsDbBuilder::LookupOrInsertNewOpMetrics(
        /*hlo_module_id=*/0, tf_op_name);
    if (tf_op_metrics->category().empty()) {
      tf_op_metrics->set_category(
          tf_op_type == kUnknownOp ? "Unknown" : std::string(tf_op_type));
    }
    tf_op_metrics->set_is_eager(device_op_metrics.is_eager());
    // The occurrences of a TF-op is the maximum among the occurrences of all
    // device ops that it contains.
    tf_op_metrics->set_occurrences(std::max(tf_op_metrics->occurrences(),
                                            device_op_metrics.occurrences()));
    tf_op_metrics->set_time_ps(tf_op_metrics->time_ps() +
                               device_op_metrics.time_ps());
    tf_op_metrics->set_self_time_ps(tf_op_metrics->self_time_ps() +
                                    device_op_metrics.self_time_ps());
    tf_op_metrics->set_flops(tf_op_metrics->flops() +
                             device_op_metrics.flops());
    tf_op_metrics->set_bytes_accessed(tf_op_metrics->bytes_accessed() +
                                      device_op_metrics.bytes_accessed());
  }
};

}  // namespace

OpMetricsDbBuilder::OpMetricsDbBuilder(OpMetricsDb* db) : db_(db) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.cc", "OpMetricsDbBuilder::OpMetricsDbBuilder");

  DCHECK_NE(db_, nullptr);
  DCHECK_EQ(db_->metrics_db_size(), 0);
}

OpMetrics* OpMetricsDbBuilder::LookupOrInsertNewOpMetrics(
    uint64 hlo_module_id, absl::string_view name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc mht_3(mht_3_v, 256, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.cc", "OpMetricsDbBuilder::LookupOrInsertNewOpMetrics");

  OpMetrics*& op_metrics = op_metrics_map_[hlo_module_id][name];
  if (op_metrics == nullptr) {
    op_metrics = db_->add_metrics_db();
    op_metrics->set_hlo_module_id(hlo_module_id);
    op_metrics->set_name(name.data(), name.size());
  }
  return op_metrics;
}

double IdleTimeRatio(const OpMetricsDb& db) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc mht_4(mht_4_v, 269, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.cc", "IdleTimeRatio");

  return 1.0 - SafeDivide(db.total_op_time_ps(), db.total_time_ps());
}

uint64 IdleTimePs(const OpMetricsDb& db) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc mht_5(mht_5_v, 276, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.cc", "IdleTimePs");

  DCHECK_GE(db.total_time_ps(), db.total_op_time_ps());
  return db.total_time_ps() - db.total_op_time_ps();
}

void AddIdleOp(OpMetricsDb& db) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc mht_6(mht_6_v, 284, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.cc", "AddIdleOp");

  uint64 idle_time_ps = IdleTimePs(db);
  OpMetrics* metrics = db.add_metrics_db();
  metrics->set_name(std::string(kIdle));
  metrics->set_category(std::string(kIdle));
  metrics->set_occurrences(0);
  metrics->set_time_ps(idle_time_ps);
  metrics->set_self_time_ps(idle_time_ps);
}

absl::optional<double> HostInfeedEnqueueRatio(const OpMetricsDb& db) {
  if (db.total_host_infeed_enq_start_timestamp_ps_diff() > 0) {
    // We use total_host_infeed_enq_start_timestamp_ps_diff to approximate the
    // total host time.
    return SafeDivide(db.total_host_infeed_enq_duration_ps(),
                      db.total_host_infeed_enq_start_timestamp_ps_diff());
  }
  return absl::nullopt;
}

OpMetricsDb CreateTfMetricsDbFromDeviceOpMetricsDb(
    const OpMetricsDb& device_op_metrics_db, bool with_idle) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSop_metrics_db_utilsDTcc mht_7(mht_7_v, 308, "", "./tensorflow/core/profiler/utils/op_metrics_db_utils.cc", "CreateTfMetricsDbFromDeviceOpMetricsDb");

  OpMetricsDb tf_op_metrics_db;
  DeviceTfOpMetricsDbBuilder builder(&tf_op_metrics_db);
  for (const auto& device_op_metrics : device_op_metrics_db.metrics_db()) {
    if (IsIdleOp(device_op_metrics)) {
      if (with_idle) {
        builder.UpdateTfOpMetricsWithDeviceOpMetrics(kIdle, kIdle,
                                                     device_op_metrics);
      }
    } else if (device_op_metrics.provenance().empty()) {
      builder.UpdateTfOpMetricsWithDeviceOpMetrics(
          device_op_metrics.name(), kUnknownOp, device_op_metrics);
    } else {
      TfOp tf_op = ParseTfOpFullname(device_op_metrics.provenance());
      builder.UpdateTfOpMetricsWithDeviceOpMetrics(tf_op.name, tf_op.type,
                                                   device_op_metrics);
    }
  }
  tf_op_metrics_db.set_total_op_time_ps(
      device_op_metrics_db.total_op_time_ps());

  tf_op_metrics_db.set_total_time_ps(
      with_idle ? device_op_metrics_db.total_time_ps()
                : device_op_metrics_db.total_op_time_ps());

  return tf_op_metrics_db;
}

}  // namespace profiler
}  // namespace tensorflow
