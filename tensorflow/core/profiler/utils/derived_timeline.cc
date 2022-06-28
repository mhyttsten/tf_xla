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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc() {
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
#include "tensorflow/core/profiler/utils/derived_timeline.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kAnnotationDelimiter = "::";

XEvent CreateXEvent(const XEventMetadata& metadata, int64_t offset_ps,
                    int64_t duration_ps, int64_t group_id_stat_metadata_id,
                    absl::optional<int64_t> group_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "CreateXEvent");

  XEvent event;
  event.set_metadata_id(metadata.id());
  // TODO(b/150498419): Normalize with the line start time.
  event.set_offset_ps(offset_ps);
  event.set_duration_ps(duration_ps);
  if (group_id) {
    XStat* stat = event.add_stats();
    stat->set_metadata_id(group_id_stat_metadata_id);
    stat->set_int64_value(*group_id);
  }
  return event;
}

int64_t GroupIdOrInvalid(absl::optional<int64_t> group_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_1(mht_1_v, 237, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "GroupIdOrInvalid");

  if (group_id)
    return *group_id;
  else
    return DerivedXLineBuilder::kInvalidGroupId;
}

}  // namespace

void ProcessTfOpEvent(absl::string_view tf_op_full_name,
                      absl::string_view low_level_event_name, int64_t offset_ps,
                      int64_t duration_ps, absl::optional<int64_t> group_id,
                      XPlaneBuilder* plane_builder,
                      DerivedXLineBuilder* tf_name_scope_line_builder,
                      DerivedXLineBuilder* tf_op_line_builder) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tf_op_full_name: \"" + std::string(tf_op_full_name.data(), tf_op_full_name.size()) + "\"");
   mht_2_v.push_back("low_level_event_name: \"" + std::string(low_level_event_name.data(), low_level_event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "ProcessTfOpEvent");

  int64_t group_id_stat_metadata_id =
      plane_builder->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId))
          ->id();
  TfOp tf_op = ParseTfOpFullname(tf_op_full_name);
  Category category = tf_op.category;
  int64_t group_id_or_invalid = GroupIdOrInvalid(group_id);
  if (category == Category::kTensorFlow || category == Category::kJax) {
    std::vector<XEvent> name_scope_event_per_level;
    for (const auto& tf_name_scope : ParseTfNameScopes(tf_op)) {
      name_scope_event_per_level.push_back(CreateXEvent(
          *plane_builder->GetOrCreateEventMetadata(tf_name_scope), offset_ps,
          duration_ps, group_id_stat_metadata_id, group_id));
    }
    tf_name_scope_line_builder->ExpandOrAddEvents(
        name_scope_event_per_level, group_id_or_invalid, low_level_event_name);
  }
  XEventMetadata* tf_op_event_metadata =
      plane_builder->GetOrCreateEventMetadata(tf_op_full_name);
  // Set the display name to op_type so that the events of the same op_type have
  // the same color in the trace viewer.
  tf_op_event_metadata->set_display_name(TfOpEventName(tf_op));
  tf_op_line_builder->ExpandOrAddEvent(
      CreateXEvent(*tf_op_event_metadata, offset_ps, duration_ps,
                   group_id_stat_metadata_id, group_id),
      group_id_or_invalid, low_level_event_name);
}

DerivedXLineBuilder::DerivedXLineBuilder(
    XPlaneBuilder* plane, int64_t line_id, absl::string_view name,
    int64_t timestamp_ns, std::vector<DerivedXLineBuilder*> dependent_lines)
    : line_(plane->GetOrCreateLine(line_id)) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_3(mht_3_v, 291, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "DerivedXLineBuilder::DerivedXLineBuilder");

  line_.SetName(name);
  line_.SetTimestampNs(timestamp_ns);
  dependent_lines_ = std::move(dependent_lines);
  level_stats_ = plane->GetOrCreateStatMetadata("l");
}

void DerivedXLineBuilder::ExpandOrAddLevelEvent(
    const XEvent& event, int64_t group_id,
    absl::string_view low_level_event_name, int level) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("low_level_event_name: \"" + std::string(low_level_event_name.data(), low_level_event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_4(mht_4_v, 304, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "DerivedXLineBuilder::ExpandOrAddLevelEvent");

  int64_t offset_ps = event.offset_ps();
  int64_t duration_ps = event.duration_ps();
  auto& last_event = last_event_by_level_[level];
  // If last_event is not nullptr, its offset must be less than or equal to
  // the given event's offset.
  DCHECK(!last_event || last_event->OffsetPs() <= offset_ps);
  auto& last_eventinfo = last_eventinfo_by_level_[level];
  bool merge_last_event = false;
  if (last_event && last_event->MetadataId() == event.metadata_id()) {
    // If last_event is not nullptr and metadata is same, merge the given
    // event into last_event.
    DCHECK(last_eventinfo);  // last_eventinfo must be valid as well.
    // Merges event with last_event if (1) they have the same group_id
    // and (2) low_level_event_name hasn't been seen before. If
    // low_level_event has been seen before, event and last_event are actually
    // different invocations of the same Op, and so they shouldn't be merged.
    merge_last_event =
        (group_id == last_eventinfo->group_id) &&
        !last_eventinfo->low_level_event_names.contains(low_level_event_name);
  }
  if (merge_last_event) {
    // Merge event with last_event.
    last_event->SetDurationPs((offset_ps + duration_ps) -
                              last_event->OffsetPs());
    if (!low_level_event_name.empty()) {
      // One more low_level_event_name associated with last_event.
      last_eventinfo->low_level_event_names.insert(
          std::string(low_level_event_name));
    }
  } else {
    // Otherwise, reset the last events lower than or equal to the given level.
    ResetLastEvents(level);
    // And create a new event for the given level.
    last_event = line_.AddEvent(event);
    last_event->AddStatValue(*level_stats_, level);
    // Also create a new XEventInfo for this level.
    last_eventinfo = XEventInfo(group_id, low_level_event_name);
  }
}

void DerivedXLineBuilder::ResetLastEvents(int level) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_5(mht_5_v, 348, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "DerivedXLineBuilder::ResetLastEvents");

  for (int i = level, end = last_event_by_level_.size(); i < end; ++i) {
    last_event_by_level_[i] = absl::nullopt;
    last_eventinfo_by_level_[i] = absl::nullopt;
  }
  if (level == 0) ResetDependentLines();
}

void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 const GroupMetadataMap& group_metadata_map,
                                 XPlane* device_trace, bool step_info_only) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_6(mht_6_v, 361, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "DeriveEventsFromAnnotations");

  // Merge and sort events by Timespan as they come from different lines.
  std::vector<XEventVisitor> events;
  uint64 start_timestamp_ns = 0;
  XPlaneVisitor device_plane = CreateTfXPlaneVisitor(device_trace);
  device_plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) return;  // Skip overhead line.
    start_timestamp_ns = line.TimestampNs();
    line.ForEachEvent(
        [&](const XEventVisitor& event) { events.push_back(event); });
  });
  absl::c_sort(events);

  XPlaneBuilder plane(device_trace);
  DerivedXLineBuilder tf_ops(&plane, kThreadIdTfOp, kTensorFlowOpLineName,
                             start_timestamp_ns, {});
  DerivedXLineBuilder tf_name_scope(&plane, kThreadIdTfNameScope,
                                    kTensorFlowNameScopeLineName,
                                    start_timestamp_ns, {&tf_ops});
  DerivedXLineBuilder hlo_ops(&plane, kThreadIdHloOp, kXlaOpLineName,
                              start_timestamp_ns, {});
  DerivedXLineBuilder hlo_modules(&plane, kThreadIdHloModule,
                                  kXlaModuleLineName, start_timestamp_ns,
                                  {&tf_name_scope, &hlo_ops});
  DerivedXLineBuilder steps(&plane, kThreadIdStepInfo, kStepLineName,
                            start_timestamp_ns, {&hlo_modules});
  DerivedXLineBuilder source(&plane, kThreadIdSource, kSourceLineName,
                             start_timestamp_ns, {});

  int64_t group_id_stat_metadata_id =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId))->id();
  int64_t step_name_stat_metadata_id =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kStepName))->id();

  // Process events in order by start time.
  for (const XEventVisitor& event : events) {
    int64_t offset_ps = event.OffsetPs();
    int64_t duration_ps = event.DurationPs();
    absl::string_view tf_op_full_name;
    absl::string_view hlo_module_name;
    absl::optional<uint64_t> program_id;
    std::vector<absl::string_view> hlo_op_names;
    absl::optional<int64_t> group_id;
    bool is_kernel = false;
    event.ForEachStat([&](const XStatVisitor& stat) {
      if (!stat.Type().has_value()) return;
      switch (stat.Type().value()) {
        case StatType::kGroupId:
          group_id = stat.IntValue();
          break;
        case StatType::kTfOp:
          tf_op_full_name = stat.StrOrRefValue();
          break;
        case StatType::kHloOp:
          hlo_op_names =
              absl::StrSplit(stat.StrOrRefValue(), kAnnotationDelimiter);
          break;
        case StatType::kHloModule:
          hlo_module_name = stat.StrOrRefValue();
          break;
        case StatType::kProgramId:
          program_id = stat.IntOrUintValue();
          break;
        case StatType::kKernelDetails:
          is_kernel = true;
          break;
        default:
          break;
      }
    });
    int64_t group_id_or_invalid = GroupIdOrInvalid(group_id);
    if (group_id) {
      XEvent step_event = CreateXEvent(
          *plane.GetOrCreateEventMetadata(absl::StrCat(*group_id)), offset_ps,
          duration_ps, group_id_stat_metadata_id, group_id);
      if (auto group_metadata =
              gtl::FindOrNull(group_metadata_map, *group_id)) {
        XStat* stat = step_event.add_stats();
        stat->set_metadata_id(step_name_stat_metadata_id);
        stat->set_str_value(group_metadata->name);
      }
      steps.ExpandOrAddEvent(step_event, group_id_or_invalid);
    }

    if (step_info_only) continue;

    // For HLO/TF op lines, only use kernel events (i.e. excluding memcpy or
    // allocation events).
    if (!is_kernel) continue;

    if (!hlo_module_name.empty()) {
      std::string name(hlo_module_name);
      if (program_id.has_value()) {
        absl::StrAppend(&name, "(", program_id.value(), ")");
      }
      hlo_modules.ExpandOrAddEvent(
          CreateXEvent(*plane.GetOrCreateEventMetadata(name), offset_ps,
                       duration_ps, group_id_stat_metadata_id, group_id));
    }

    if (!hlo_op_names.empty()) {  // GPU kernel compiled by XLA
      DCHECK(!hlo_module_name.empty());
      std::vector<XEvent> hlo_op_event_per_level;
      for (absl::string_view hlo_op_name : hlo_op_names) {
        DCHECK(!hlo_op_name.empty());
        hlo_op_event_per_level.push_back(CreateXEvent(
            *plane.GetOrCreateEventMetadata(hlo_op_name), offset_ps,
            duration_ps, group_id_stat_metadata_id, group_id));
      }
      hlo_ops.ExpandOrAddEvents(hlo_op_event_per_level, group_id_or_invalid);
      auto symbol =
          symbol_resolver(program_id, hlo_module_name, hlo_op_names.back());
      if (!symbol.tf_op_name.empty()) {
        ProcessTfOpEvent(symbol.tf_op_name,
                         /*low_level_event_name=*/event.Name(), offset_ps,
                         duration_ps, group_id, &plane, &tf_name_scope,
                         &tf_ops);
      }
      if (!symbol.source_info.empty()) {
        source.ExpandOrAddEvent(CreateXEvent(
            *plane.GetOrCreateEventMetadata(symbol.source_info), offset_ps,
            duration_ps, group_id_stat_metadata_id, group_id));
      }
    } else if (!tf_op_full_name.empty()) {  // GPU kernel not compiled by XLA
      ProcessTfOpEvent(tf_op_full_name,
                       /*low_level_event_name=*/event.Name(), offset_ps,
                       duration_ps, group_id, &plane, &tf_name_scope, &tf_ops);
    }
  }
  RemoveEmptyLines(device_trace);
}

void DeriveEventsFromHostTrace(const XPlane* host_trace,
                               const GroupMetadataMap& group_metadata_map,
                               std::vector<XPlane*> device_traces) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_7(mht_7_v, 498, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "DeriveEventsFromHostTrace");

  struct GroupLaunchInfo {  // "Group" normally means step.
    Timespan timespan;
    int32 num_launches = 0;
    uint64 max_launch_time_ps = 0ULL;
    uint64 total_launch_time_ps = 0ULL;
  };
  typedef absl::flat_hash_map<int64_t /*group_id*/, GroupLaunchInfo>
      DeviceLaunchInfo;

  int num_devices = device_traces.size();
  std::vector<DeviceLaunchInfo> per_device_launch_info(num_devices);

  XPlaneVisitor host_plane = CreateTfXPlaneVisitor(host_trace);
  host_plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) return;
    line.ForEachEvent([&](const XEventVisitor& event) {
      absl::optional<int64_t> group_id;
      absl::optional<int64_t> device_id;
      absl::optional<int64_t> correlation_id;
      // Filter out API calls for cuEventRecord/cuEventQuery/cuCtxSynchronize
      // etc for now. TODO: find a better way to filter out only the memcpy and
      // kernel launch events.
      if (absl::StartsWith(event.Name(), "cu")) return;
      event.ForEachStat([&](const XStatVisitor& stat) {
        if (stat.Type() == StatType::kGroupId) {
          group_id = stat.IntValue();
        } else if (stat.Type() == StatType::kDeviceId) {
          device_id = stat.IntOrUintValue();
        } else if (stat.Type() == StatType::kCorrelationId) {
          correlation_id = stat.IntValue();
        }
      });
      if (group_id && device_id && correlation_id && *device_id >= 0 &&
          *device_id < num_devices) {
        // This is a launch event on a known device.
        GroupLaunchInfo& group_launch_info =
            per_device_launch_info[*device_id][*group_id];
        Timespan& group_span = group_launch_info.timespan;
        Timespan event_span = event.GetTimespan();
        if (group_launch_info.num_launches) {  // Existing group.
          group_span.ExpandToInclude(event_span);
        } else {
          group_span = event_span;
        }
        ++group_launch_info.num_launches;
        group_launch_info.max_launch_time_ps = std::max(
            group_launch_info.max_launch_time_ps, event_span.duration_ps());
        group_launch_info.total_launch_time_ps += event_span.duration_ps();
      }
    });
  });

  uint64 host_plane_start = GetStartTimestampNs(*host_trace);
  for (int i = 0; i < num_devices; ++i) {
    if (per_device_launch_info[i].empty()) continue;
    uint64 device_plane_start = GetStartTimestampNs(*device_traces[i]);
    XPlaneBuilder device_plane(device_traces[i]);
    XLineBuilder launch_line =
        device_plane.GetOrCreateLine(kThreadIdKernelLaunch);
    launch_line.SetName(kKernelLaunchLineName);
    launch_line.SetTimestampNs(std::min(device_plane_start, host_plane_start));
    for (const auto& kv : per_device_launch_info[i]) {
      int64_t group_id = kv.first;
      const GroupLaunchInfo& group_info = kv.second;
      if (auto group_metadata = gtl::FindOrNull(group_metadata_map, group_id)) {
        XEventBuilder device_event =
            launch_line.AddEvent(*device_plane.GetOrCreateEventMetadata(
                absl::StrCat("Launch Stats for ", group_metadata->name)));
        device_event.SetTimestampNs(host_plane_start +
                                    PicoToNano(group_info.timespan.begin_ps()));
        device_event.SetDurationPs(group_info.timespan.duration_ps());
        device_event.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                      GetStatTypeStr(StatType::kGroupId)),
                                  group_id);
        device_event.AddStatValue(
            *device_plane.GetOrCreateStatMetadata("num_launches"),
            group_info.num_launches);
        device_event.AddStatValue(
            *device_plane.GetOrCreateStatMetadata("max_launch_time_us"),
            PicoToMicro(group_info.max_launch_time_ps));
        device_event.AddStatValue(
            *device_plane.GetOrCreateStatMetadata("avg_launch_time_us"),
            PicoToMicro(group_info.total_launch_time_ps /
                        group_info.num_launches));
      }
    }
  }
}

void GenerateDerivedTimeLines(const GroupMetadataMap& group_metadata_map,
                              XSpace* space, bool step_info_only) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_8(mht_8_v, 592, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "GenerateDerivedTimeLines");

  // TODO(profiler): Once we capture HLO protos for xla/gpu, we should use that
  // to look up tensorflow op name from hlo_module/hlo_op.
  auto dummy_symbol_resolver =
      [](absl::optional<uint64_t> program_id, absl::string_view hlo_module,
         absl::string_view hlo_op) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("hlo_module: \"" + std::string(hlo_module.data(), hlo_module.size()) + "\"");
   mht_9_v.push_back("hlo_op: \"" + std::string(hlo_op.data(), hlo_op.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTcc mht_9(mht_9_v, 602, "", "./tensorflow/core/profiler/utils/derived_timeline.cc", "lambda");
 return tensorflow::profiler::Symbol(); };
  std::vector<XPlane*> device_traces =
      FindMutablePlanesWithPrefix(space, kGpuPlanePrefix);
  for (XPlane* plane : device_traces) {
    DeriveEventsFromAnnotations(dummy_symbol_resolver, group_metadata_map,
                                plane, step_info_only);
  }
}

}  // namespace profiler
}  // namespace tensorflow
