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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc() {
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

#include "tensorflow/core/profiler/convert/xplane_to_op_metrics_db.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/op_stack.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/cost_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/op_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// Type of a TensorFlow Op activity, which is either beginning or ending an Op.
enum TfActivityType { kTfOpBegin, kTfOpEnd };

// Instant activity representing the begin or end of a host-side TF Op.
struct TfActivity {
  // The timestamp in picoseconds when this activity happened.
  uint64 timestamp_ps;
  // The ID of this Op.
  uint32 tf_op_id;
  // Type of this activity.
  TfActivityType activity_type;
  // Full TF op name and type of this activity (backed by XEvent::name).
  TfOp tf_op;
  // Whether it is eagerly executed.
  bool is_eager;
};

// TF Op metrics stored as element in OpStack.
struct TfOpInfo {
  explicit TfOpInfo(uint64 ts) : start_timestamp_ps(ts) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc mht_0(mht_0_v, 236, "", "./tensorflow/core/profiler/convert/xplane_to_op_metrics_db.cc", "TfOpInfo");
}

  // Start timestamp in picoseconds.
  uint64 start_timestamp_ps;
  // Children duration in picoseconds.
  uint64 children_duration_ps = 0;
};

// Processes a TF-activity on particular core.
void ProcessOneTfActivity(const TfActivity& activity,
                          OpStack<TfOpInfo>* tf_op_stack,
                          TfMetricsDbData* tf_metrics_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc mht_1(mht_1_v, 250, "", "./tensorflow/core/profiler/convert/xplane_to_op_metrics_db.cc", "ProcessOneTfActivity");

  uint32 tf_op_id = activity.tf_op_id;
  switch (activity.activity_type) {
    case kTfOpBegin: {
      tf_op_stack->Push(tf_op_id,
                        absl::make_unique<TfOpInfo>(activity.timestamp_ps));
      break;
    }
    case kTfOpEnd: {
      std::unique_ptr<TfOpInfo> info = tf_op_stack->Pop(tf_op_id);
      if (info == nullptr) {
        // This happens if TraceMes overlap.
        VLOG(1) << "No begin event found for TF activity id=" << tf_op_id
                << " name=" << activity.tf_op.name
                << " type=" << activity.tf_op.type;
        break;
      }
      Timespan tf_op_span =
          PicoSpan(info->start_timestamp_ps, activity.timestamp_ps);
      tf_metrics_data->tf_metrics_db_builder.EnterOp(
          activity.tf_op.name, activity.tf_op.type, activity.is_eager,
          tf_op_span.duration_ps(), info->children_duration_ps);
      TfOpInfo* parent_info = tf_op_stack->Top();
      if (parent_info != nullptr) {
        parent_info->children_duration_ps += tf_op_span.duration_ps();
      }
      if (IsInfeedEnqueueOp(activity.tf_op.type)) {
        tf_metrics_data->tf_metrics_db_builder.EnterHostInfeedEnqueue(
            tf_op_span);
      }
      break;
    }
  }
}

// Processes all TF-activities on the given core.
void ProcessTfActivities(std::vector<TfActivity>* tf_activities,
                         TfMetricsDbData* tf_metrics_db_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc mht_2(mht_2_v, 290, "", "./tensorflow/core/profiler/convert/xplane_to_op_metrics_db.cc", "ProcessTfActivities");

  if (tf_activities->empty()) return;
  absl::c_stable_sort(*tf_activities,
                      [](const TfActivity& a, const TfActivity& b) {
                        return a.timestamp_ps < b.timestamp_ps;
                      });
  OpStack<TfOpInfo> tf_op_stack;
  for (const auto& tf_activity : *tf_activities) {
    ProcessOneTfActivity(tf_activity, &tf_op_stack, tf_metrics_db_data);
  }
  SetTotalTimePs(
      tf_metrics_db_data->tf_metrics_db,
      tf_activities->back().timestamp_ps - tf_activities->front().timestamp_ps);
}

void CollectTfActivities(const XLineVisitor& line,
                         const absl::flat_hash_map<int64_t, TfOp>& tf_ops,
                         std::vector<TfActivity>* tf_activities) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc mht_3(mht_3_v, 310, "", "./tensorflow/core/profiler/convert/xplane_to_op_metrics_db.cc", "CollectTfActivities");

  uint32 tf_op_id = 0;
  tf_activities->reserve(line.NumEvents() * 2);
  line.ForEachEvent([&tf_ops, &tf_op_id,
                     &tf_activities](const XEventVisitor& event) {
    const TfOp* tf_op = gtl::FindOrNull(tf_ops, event.Id());
    if (tf_op != nullptr) {
      ++tf_op_id;
      bool is_eager = false;
      if (absl::optional<XStatVisitor> stat =
              event.GetStat(StatType::kIsEager)) {
        is_eager = stat->IntValue();
      }
      Timespan span = event.GetTimespan();
      tf_activities->push_back(
          {span.begin_ps(), tf_op_id, kTfOpBegin, *tf_op, is_eager});
      tf_activities->push_back(
          {span.end_ps(), tf_op_id, kTfOpEnd, *tf_op, is_eager});
    }
  });
}

}  // namespace

absl::flat_hash_map<int64_t, TfOp> CollectTfOpsFromHostThreadsXPlane(
    const XPlane& host_trace) {
  absl::flat_hash_map<int64_t, TfOp> tf_ops;
  for (const auto& id_metadata : host_trace.event_metadata()) {
    const XEventMetadata& metadata = id_metadata.second;
    // On the host, we have added some user-specified TraceMe's in addition to
    // the TraceMe's added to every TensorFlow op by the system. These
    // user-inserted TraceMe's have "unknown" type. We don't count them in
    // Tf-stats.
    TfOp tf_op = ParseTfOpFullname(metadata.name());
    if (tf_op.category != Category::kUnknown) {
      tf_ops.try_emplace(metadata.id(), tf_op);
    }
  }
  return tf_ops;
}

TfMetricsDbData ConvertHostThreadsXLineToTfMetricsDbData(
    const XLineVisitor& line,
    const absl::flat_hash_map<int64_t, TfOp>& tf_ops) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc mht_4(mht_4_v, 356, "", "./tensorflow/core/profiler/convert/xplane_to_op_metrics_db.cc", "ConvertHostThreadsXLineToTfMetricsDbData");

  TfMetricsDbData tf_metrics_db_data;
  if (!tf_ops.empty()) {
    std::vector<TfActivity> tf_activities;
    CollectTfActivities(line, tf_ops, &tf_activities);
    ProcessTfActivities(&tf_activities, &tf_metrics_db_data);
  }
  return tf_metrics_db_data;
}

void ConsumeTfMetricsDbData(TfMetricsDbData src, OpMetricsDbCombiner* dst) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc mht_5(mht_5_v, 369, "", "./tensorflow/core/profiler/convert/xplane_to_op_metrics_db.cc", "ConsumeTfMetricsDbData");

  AddIdleOp(src.tf_metrics_db);
  dst->Combine(src.tf_metrics_db);
  src.tf_metrics_db.Clear();
}

OpMetricsDb ConvertHostThreadsXPlaneToOpMetricsDb(const XPlane& host_trace) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc mht_6(mht_6_v, 378, "", "./tensorflow/core/profiler/convert/xplane_to_op_metrics_db.cc", "ConvertHostThreadsXPlaneToOpMetricsDb");

  absl::flat_hash_map<int64_t, TfOp> tf_ops =
      CollectTfOpsFromHostThreadsXPlane(host_trace);
  OpMetricsDb result;
  OpMetricsDbCombiner combiner(&result);
  XPlaneVisitor plane = CreateTfXPlaneVisitor(&host_trace);
  plane.ForEachLine([&tf_ops, &combiner](const XLineVisitor& line) {
    ConsumeTfMetricsDbData(
        ConvertHostThreadsXLineToTfMetricsDbData(line, tf_ops), &combiner);
  });
  return result;
}

OpMetricsDb ConvertDeviceTraceXPlaneToOpMetricsDb(const XPlane& device_trace) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_op_metrics_dbDTcc mht_7(mht_7_v, 394, "", "./tensorflow/core/profiler/convert/xplane_to_op_metrics_db.cc", "ConvertDeviceTraceXPlaneToOpMetricsDb");

  OpMetricsDb result;
  DeviceOpMetricsDbBuilder device_op_metrics_db_builder(&result);

  int64_t first_op_offset_ps = kint64max;
  int64_t last_op_offset_ps = 0;

  TfOpRoofLineCostEstimator op_level_cost_estimator;
  XPlaneVisitor plane = CreateTfXPlaneVisitor(&device_trace);
  plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) return;
    line.ForEachEvent([&](const XEventVisitor& event) {
      first_op_offset_ps = std::min(first_op_offset_ps, event.OffsetPs());
      last_op_offset_ps = std::max(last_op_offset_ps, event.EndOffsetPs());

      absl::string_view tf_op_full_name;
      bool is_eager = false;
      event.ForEachStat([&](const XStatVisitor& stat) {
        if (stat.Type() == StatType::kTfOp) {
          tf_op_full_name = stat.StrOrRefValue();
        } else if (stat.Type() == StatType::kIsEager) {
          is_eager = stat.IntValue();
        }
      });
      if (tf_op_full_name.empty()) return;
      TfOp tf_op = ParseTfOpFullname(tf_op_full_name);
      TfOpRoofLineCostEstimator::OpRoofLineStats costs;
      if (tf_op.category != Category::kUnknown) {
        costs = op_level_cost_estimator.Predict(event);
      }
      device_op_metrics_db_builder.EnterOp(
          /*program_id=*/0, absl::StrCat(tf_op.name, "/", event.Name()),
          tf_op.type, tf_op_full_name, is_eager,
          /*occurrences=*/1, event.DurationPs(),
          /*children_time_ps=*/0, costs.flops, costs.bytes_accessed);
    });
  });
  SetTotalTimePs(
      result, last_op_offset_ps ? last_op_offset_ps - first_op_offset_ps : 0);
  AddIdleOp(result);
  return result;
}

}  // namespace profiler
}  // namespace tensorflow
