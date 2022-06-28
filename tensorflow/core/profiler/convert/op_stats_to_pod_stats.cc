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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_statsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_statsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_statsDTcc() {
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
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/diagnostics.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

PodStatsRecord CreatePodStatsRecord(absl::string_view host_name,
                                    const StepInfoResult& step_info) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("host_name: \"" + std::string(host_name.data(), host_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_statsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/profiler/convert/op_stats_to_pod_stats.cc", "CreatePodStatsRecord");

  PodStatsRecord record;
  GenericStepBreakdown generic;
  bool success = step_info.step_breakdown().UnpackTo(&generic);
  DCHECK(success);
  record.set_host_name(string(host_name));
  record.set_step_num(step_info.step_num());
  record.set_total_duration_us(PicoToMicro(step_info.duration_ps()));
  auto& step_breakdown_map = *record.mutable_step_breakdown_us();
  std::vector<std::pair<uint64, absl::string_view>> metrics;

  auto add_event = [&](GenericEventType type,
                       std::initializer_list<EventType> event_list) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_statsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/profiler/convert/op_stats_to_pod_stats.cc", "lambda");

    uint64 ps = 0;
    for (const auto& event_type : event_list) {
      ps += gtl::FindWithDefault(generic.type_ps(), event_type, /*value=*/0);
    }
    step_breakdown_map[type] = PicoToMicro(ps);
    metrics.emplace_back(ps, GetGenericEventTypeStr(type));
  };

  add_event(kDeviceCompute, {DEVICE_COMPUTE_32, DEVICE_COMPUTE_16});
  add_event(kDeviceToDevice, {DEVICE_TO_DEVICE, DEVICE_WAIT_DEVICE});
  add_event(kDeviceCollectives, {DEVICE_COLLECTIVES});
  add_event(kHostCompute, {HOST_COMPUTE});
  add_event(kHostPrepare, {HOST_PREPARE});
  add_event(kInput, {HOST_WAIT_INPUT, HOST_TO_DEVICE, DEVICE_WAIT_HOST});
  add_event(kOutput, {DEVICE_TO_HOST});
  add_event(kCompile, {HOST_COMPILE});
  add_event(kAllOthers, {UNKNOWN_TIME});

  std::sort(metrics.begin(), metrics.end());
  record.set_bottleneck(metrics.back().second.data(),
                        metrics.back().second.size());
  return record;
}

}  // namespace

PodStatsDatabase ConvertOpStatsToPodStats(const OpStats& op_stats) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSop_stats_to_pod_statsDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/profiler/convert/op_stats_to_pod_stats.cc", "ConvertOpStatsToPodStats");

  PodStatsDatabase pod_stats_db;
  const auto& core_id_map = op_stats.core_id_to_details();
  for (int i = GenericEventType::kFirstGenericEventType;
       i <= GenericEventType::kLastGenericEventType; i++) {
    auto& event = *pod_stats_db.add_step_breakdown_events();
    event.set_id(i);
    absl::string_view type_str =
        GetGenericEventTypeStr(static_cast<GenericEventType>(i));
    event.set_name(type_str.data(), type_str.size());
  }

  for (const auto& step_sequence : op_stats.step_db().step_sequence()) {
    for (const auto& entry : step_sequence.step_info_per_core()) {
      const CoreDetails& details = core_id_map.at(entry.first);
      *pod_stats_db.add_pod_stats_record() =
          CreatePodStatsRecord(details.hostname(), entry.second);
    }
  }
  PopulateStepDiagnostics(op_stats, pod_stats_db.mutable_diagnostics());
  return pod_stats_db;
}

}  // namespace profiler
}  // namespace tensorflow
