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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStfstreamz_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStfstreamz_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStfstreamz_utilsDTcc() {
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
#include "tensorflow/core/profiler/utils/tfstreamz_utils.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/monitoring/collected_metrics.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/tfstreamz.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"

namespace tensorflow {
namespace profiler {

namespace {

std::string ConstructXStatName(absl::string_view name,
                               const monitoring::Point& point) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStfstreamz_utilsDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/profiler/utils/tfstreamz_utils.cc", "ConstructXStatName");

  if (point.labels.empty()) {
    return std::string(name);
  }
  return absl::Substitute(
      "$0{$1}", name,
      absl::StrJoin(
          point.labels, ", ",
          [](std::string* out, const monitoring::Point::Label& label) {
            absl::StrAppend(out, label.name, "=", label.value);
          }));
}

tfstreamz::Percentiles ToProto(const monitoring::Percentiles& percentiles) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStfstreamz_utilsDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/profiler/utils/tfstreamz_utils.cc", "ToProto");

  tfstreamz::Percentiles output;
  output.set_unit_of_measure(
      static_cast<tfstreamz::UnitOfMeasure>(percentiles.unit_of_measure));
  output.set_start_nstime(percentiles.start_nstime);
  output.set_end_nstime(percentiles.end_nstime);
  output.set_min_value(percentiles.min_value);
  output.set_max_value(percentiles.max_value);
  output.set_mean(percentiles.mean);
  output.set_stddev(percentiles.stddev);
  output.set_num_samples(percentiles.num_samples);
  output.set_total_samples(percentiles.total_samples);
  output.set_accumulator(percentiles.accumulator);
  for (const auto& pp : percentiles.points) {
    auto* percentile_point = output.add_points();
    percentile_point->set_percentile(pp.percentile);
    percentile_point->set_value(pp.value);
  }
  return output;
}

}  // namespace

Status SerializeToXPlane(const std::vector<TfStreamzSnapshot>& snapshots,
                         XPlane* plane, uint64 line_start_time_ns) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPStfstreamz_utilsDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/profiler/utils/tfstreamz_utils.cc", "SerializeToXPlane");

  XPlaneBuilder xplane(plane);
  XLineBuilder line = xplane.GetOrCreateLine(0);  // This plane has single line.
  line.SetTimestampNs(line_start_time_ns);

  // For each snapshot, create a virtual event.
  for (const auto& snapshot : snapshots) {
    XEventMetadata* event_metadata =
        xplane.GetOrCreateEventMetadata("TFStreamz Snapshot");
    XEventBuilder xevent = line.AddEvent(*event_metadata);
    xevent.SetTimestampNs(snapshot.start_time_ns);
    xevent.SetEndTimestampNs(snapshot.end_time_ns);
    auto& metric_descriptor_map = snapshot.metrics->metric_descriptor_map;
    for (const auto& point_set : snapshot.metrics->point_set_map) {
      const std::string& metric_name = point_set.first;
      // Each metrics have multiple points corresponding to different labels.
      for (const auto& point : point_set.second->points) {
        // Generates one KPI metric for each point.
        std::string stat_name = ConstructXStatName(metric_name, *point);
        auto* metadata = xplane.GetOrCreateStatMetadata(stat_name);
        auto it = metric_descriptor_map.find(metric_name);
        if (it != metric_descriptor_map.end()) {
          metadata->set_description(it->second->description);
        }
        switch (point->value_type) {
          case monitoring::ValueType::kInt64:
            xevent.AddStatValue(*metadata, point->int64_value);
            break;
          case monitoring::ValueType::kBool:
            xevent.AddStatValue(*metadata, point->bool_value);
            break;
          case monitoring::ValueType::kString:
            xevent.AddStatValue(*metadata, *xplane.GetOrCreateStatMetadata(
                                               point->string_value));
            break;
          case monitoring::ValueType::kHistogram:
            xevent.AddStatValue(*metadata, point->histogram_value);
            break;
          case monitoring::ValueType::kPercentiles:
            xevent.AddStatValue(*metadata, ToProto(point->percentiles_value));
            break;
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
