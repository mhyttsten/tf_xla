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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc() {
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
#include "tensorflow/core/profiler/utils/xplane_utils.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/context_types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// Returns the index of the first element in array for which pred is true.
// Returns -1 if no such element is found.
template <typename T, typename Pred>
int Find(const protobuf::RepeatedPtrField<T>& array, const Pred& pred) {
  for (int i = 0; i < array.size(); ++i) {
    if (pred(&array.Get(i))) return i;
  }
  return -1;
}

// Returns the indices of all elements in array for which pred is true.
template <typename T, typename Pred>
std::vector<int> FindAll(const protobuf::RepeatedPtrField<T>& array,
                         const Pred& pred) {
  std::vector<int> indices;
  for (int i = 0; i < array.size(); ++i) {
    if (pred(&array.Get(i))) indices.push_back(i);
  }
  return indices;
}

template <typename T>
void RemoveAt(protobuf::RepeatedPtrField<T>* array,
              const std::vector<int>& indices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_0(mht_0_v, 234, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "RemoveAt");

  if (indices.empty()) return;
  if (array->size() == indices.size()) {
    // Assumes that 'indices' consists of [0 ... N-1].
    array->Clear();
    return;
  }
  auto remove_iter = indices.begin();
  int i = *(remove_iter++);
  for (int j = i + 1; j < array->size(); ++j) {
    if (remove_iter != indices.end() && *remove_iter == j) {
      ++remove_iter;
    } else {
      array->SwapElements(j, i++);
    }
  }
  array->DeleteSubrange(i, array->size() - i);
}

// Removes the given element from array.
template <typename T>
void Remove(protobuf::RepeatedPtrField<T>* array, const T* elem) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_1(mht_1_v, 258, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "Remove");

  int i = Find(*array, [elem](const T* e) { return elem == e; });
  RemoveAt(array, {i});
}

template <typename T, typename Pred>
void RemoveIf(protobuf::RepeatedPtrField<T>* array, Pred&& pred) {
  std::vector<int> indices = FindAll(*array, pred);
  RemoveAt(array, indices);
}

}  // namespace

const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_2(mht_2_v, 275, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "FindPlaneWithName");

  int i = Find(space.planes(),
               [name](const XPlane* plane) { return plane->name() == name; });
  return (i != -1) ? &space.planes(i) : nullptr;
}

std::vector<const XPlane*> FindPlanesWithNames(
    const XSpace& space, const std::vector<absl::string_view>& names) {
  absl::flat_hash_set<absl::string_view> names_set(names.begin(), names.end());
  std::vector<int> indices =
      FindAll(space.planes(), [&names_set](const XPlane* plane) {
        return names_set.contains(plane->name());
      });
  std::vector<const XPlane*> planes;
  planes.reserve(indices.size());
  for (int i : indices) {
    planes.push_back(&space.planes(i));
  }
  return planes;
}

XPlane* FindMutablePlaneWithName(XSpace* space, absl::string_view name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_3(mht_3_v, 300, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "FindMutablePlaneWithName");

  int i = Find(space->planes(),
               [name](const XPlane* plane) { return plane->name() == name; });
  return (i != -1) ? space->mutable_planes(i) : nullptr;
}

XPlane* FindOrAddMutablePlaneWithName(XSpace* space, absl::string_view name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_4(mht_4_v, 310, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "FindOrAddMutablePlaneWithName");

  XPlane* plane = FindMutablePlaneWithName(space, name);
  if (plane == nullptr) {
    plane = space->add_planes();
    plane->set_name(name.data(), name.size());
  }
  return plane;
}

std::vector<const XPlane*> FindPlanesWithPrefix(const XSpace& space,
                                                absl::string_view prefix) {
  std::vector<const XPlane*> result;
  for (const XPlane& plane : space.planes()) {
    if (absl::StartsWith(plane.name(), prefix)) result.push_back(&plane);
  }
  return result;
}

std::vector<XPlane*> FindMutablePlanesWithPrefix(XSpace* space,
                                                 absl::string_view prefix) {
  std::vector<XPlane*> result;
  for (XPlane& plane : *space->mutable_planes()) {
    if (absl::StartsWith(plane.name(), prefix)) result.push_back(&plane);
  }
  return result;
}

const XLine* FindLineWithId(const XPlane& plane, int64_t id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_5(mht_5_v, 340, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "FindLineWithId");

  int i =
      Find(plane.lines(), [id](const XLine* line) { return line->id() == id; });
  return (i != -1) ? &plane.lines(i) : nullptr;
}

XStat* FindOrAddMutableStat(const XStatMetadata& stat_metadata, XEvent* event) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_6(mht_6_v, 349, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "FindOrAddMutableStat");

  for (auto& stat : *event->mutable_stats()) {
    if (stat.metadata_id() == stat_metadata.id()) {
      return &stat;
    }
  }
  XStat* stat = event->add_stats();
  stat->set_metadata_id(stat_metadata.id());
  return stat;
}

void RemovePlane(XSpace* space, const XPlane* plane) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_7(mht_7_v, 363, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "RemovePlane");

  DCHECK(plane != nullptr);
  Remove(space->mutable_planes(), plane);
}

void RemovePlanes(XSpace* space, const std::vector<const XPlane*>& planes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_8(mht_8_v, 371, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "RemovePlanes");

  absl::flat_hash_set<const XPlane*> planes_set(planes.begin(), planes.end());
  RemoveIf(space->mutable_planes(), [&planes_set](const XPlane* plane) {
    return planes_set.contains(plane);
  });
}

void RemoveLine(XPlane* plane, const XLine* line) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_9(mht_9_v, 381, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "RemoveLine");

  DCHECK(line != nullptr);
  Remove(plane->mutable_lines(), line);
}

void RemoveEvents(XLine* line,
                  const absl::flat_hash_set<const XEvent*>& events) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_10(mht_10_v, 390, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "RemoveEvents");

  RemoveIf(line->mutable_events(),
           [&](const XEvent* event) { return events.contains(event); });
}

void RemoveEmptyPlanes(XSpace* space) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_11(mht_11_v, 398, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "RemoveEmptyPlanes");

  RemoveIf(space->mutable_planes(),
           [&](const XPlane* plane) { return plane->lines().empty(); });
}

void RemoveEmptyLines(XPlane* plane) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_12(mht_12_v, 406, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "RemoveEmptyLines");

  RemoveIf(plane->mutable_lines(),
           [&](const XLine* line) { return line->events().empty(); });
}

bool XEventsComparator::operator()(const XEvent* a, const XEvent* b) const {
  return XEventTimespan(*a) < XEventTimespan(*b);
}

void SortXPlane(XPlane* plane) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_13(mht_13_v, 418, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "SortXPlane");

  for (XLine& line : *plane->mutable_lines()) {
    auto& events = *line.mutable_events();
    std::sort(events.pointer_begin(), events.pointer_end(),
              XEventsComparator());
  }
}

void SortXSpace(XSpace* space) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_14(mht_14_v, 429, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "SortXSpace");

  for (XPlane& plane : *space->mutable_planes()) SortXPlane(&plane);
}

// Normalize the line's timestamp in this XPlane.
// NOTE: This can be called multiple times on the same plane. Only the first
// call will do the normalization, subsequent calls will do nothing.
// The assumption is that both line's timestamp_ns and start_time_ns are
// nano-seconds from epoch time, the different of these values is much
// smaller than these value.
void NormalizeTimestamps(XPlane* plane, uint64 start_time_ns) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_15(mht_15_v, 442, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "NormalizeTimestamps");

  for (XLine& line : *plane->mutable_lines()) {
    if (line.timestamp_ns() >= static_cast<int64_t>(start_time_ns)) {
      line.set_timestamp_ns(line.timestamp_ns() - start_time_ns);
    }
  }
}

void NormalizeTimestamps(XSpace* space, uint64 start_time_ns) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_16(mht_16_v, 453, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "NormalizeTimestamps");

  for (XPlane& plane : *space->mutable_planes()) {
    NormalizeTimestamps(&plane, start_time_ns);
  }
}

void MergePlanes(const XPlane& src_plane, XPlane* dst_plane) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_17(mht_17_v, 462, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "MergePlanes");

  RemoveEmptyLines(dst_plane);
  XPlaneVisitor src(&src_plane);
  XPlaneBuilder dst(dst_plane);
  src.ForEachStat([&](const tensorflow::profiler::XStatVisitor& stat) {
    XStatMetadata* stat_metadata = dst.GetOrCreateStatMetadata(stat.Name());
    // Use SetOrAddStat to avoid duplicating stats in dst_plane.
    dst.SetOrAddStat(*stat_metadata, stat.RawStat(), src_plane);
  });
  src.ForEachLine([&](const XLineVisitor& line) {
    XLineBuilder dst_line = dst.GetOrCreateLine(line.Id());
    int64_t time_offset_ps = 0LL;
    if (dst_line.NumEvents() == 0) {
      // Since we RemoveEmptyLines above, this could only mean that current
      // line only exist in src plane.
      dst_line.SetTimestampNs(line.TimestampNs());
      dst_line.SetName(line.Name());
      dst_line.SetDisplayNameIfEmpty(line.DisplayName());
    } else {
      if (line.TimestampNs() <= dst_line.TimestampNs()) {
        dst_line.SetTimestampNsAndAdjustEventOffsets(line.TimestampNs());
      } else {
        time_offset_ps =
            NanoToPico(line.TimestampNs() - dst_line.TimestampNs());
      }
      dst_line.SetNameIfEmpty(line.Name());
      // Don't override dst_line's display name because if both lines have name,
      // but no display name, line's name will became display name of dst_line.
    }

    line.ForEachEvent([&](const XEventVisitor& event) {
      const XEventMetadata* src_event_metadata = event.metadata();
      XEventMetadata* dst_event_metadata =
          dst.GetOrCreateEventMetadata(event.Name());
      if (dst_event_metadata->display_name().empty() &&
          !src_event_metadata->display_name().empty()) {
        dst_event_metadata->set_display_name(
            src_event_metadata->display_name());
      }
      if (dst_event_metadata->metadata().empty() &&
          !src_event_metadata->metadata().empty()) {
        dst_event_metadata->set_metadata(src_event_metadata->metadata());
      }
      XEventBuilder dst_event = dst_line.AddEvent(*dst_event_metadata);
      dst_event.SetOffsetPs(event.OffsetPs() + time_offset_ps);
      dst_event.SetDurationPs(event.DurationPs());
      if (event.NumOccurrences()) {
        dst_event.SetNumOccurrences(event.NumOccurrences());
      }
      event.ForEachStat([&](const XStatVisitor& stat) {
        // Here we can call AddStat instead of SetOrAddStat because dst_event
        // was just added.
        dst_event.AddStat(*dst.GetOrCreateStatMetadata(stat.Name()),
                          stat.RawStat(), src_plane);
      });
    });
  });
}

void MergePlanes(const std::vector<const XPlane*>& src_planes,
                 XPlane* dst_plane) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_18(mht_18_v, 525, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "MergePlanes");

  for (const XPlane* src_plane : src_planes) {
    MergePlanes(*src_plane, dst_plane);
  }
}

uint64 GetStartTimestampNs(const XPlane& plane) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_19(mht_19_v, 534, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "GetStartTimestampNs");

  int64_t plane_timestamp = 0;
  for (const auto& line : plane.lines()) {
    plane_timestamp = std::min<int64_t>(plane_timestamp, line.timestamp_ns());
  }
  return plane_timestamp;
}

bool IsEmpty(const XSpace& space) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_20(mht_20_v, 545, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "IsEmpty");

  for (const auto& plane : space.planes()) {
    for (const auto& line : plane.lines()) {
      if (!line.events().empty()) {
        return false;
      }
    }
  }
  return true;
}

void AddFlowsToXplane(int32_t host_id, bool is_host_plane, bool connect_traceme,
                      XPlane* xplane) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTcc mht_21(mht_21_v, 560, "", "./tensorflow/core/profiler/utils/xplane_utils.cc", "AddFlowsToXplane");

  if (!xplane) return;
  XPlaneBuilder plane(xplane);
  XStatMetadata* correlation_id_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kCorrelationId));
  XStatMetadata* producer_type_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kProducerType));
  XStatMetadata* consumer_type_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kConsumerType));
  XStatMetadata* producer_id_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kProducerId));
  XStatMetadata* consumer_id_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kConsumerId));
  XStatMetadata* flow_stats_metadata =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kFlow));
  XFlow::FlowDirection direction = is_host_plane
                                       ? XFlow::FlowDirection::kFlowOut
                                       : XFlow::FlowDirection::kFlowIn;

  plane.ForEachLine([&](XLineBuilder line) {
    line.ForEachEvent([&](XEventBuilder event) {
      absl::optional<uint64_t> correlation_id;
      absl::optional<uint64_t> producer_type;
      absl::optional<uint64_t> consumer_type;
      absl::optional<uint64_t> producer_id;
      absl::optional<uint64_t> consumer_id;
      event.ForEachStat([&](XStat* stat) {
        if (correlation_id_stats_metadata &&
            stat->metadata_id() == correlation_id_stats_metadata->id()) {
          correlation_id = stat->uint64_value();
        } else if (connect_traceme) {
          if (producer_type_stats_metadata &&
              stat->metadata_id() == producer_type_stats_metadata->id()) {
            producer_type = XStatsBuilder<XPlane>::IntOrUintValue(*stat);
          } else if (consumer_type_stats_metadata &&
                     stat->metadata_id() ==
                         consumer_type_stats_metadata->id()) {
            consumer_type = XStatsBuilder<XPlane>::IntOrUintValue(*stat);
          } else if (producer_id_stats_metadata &&
                     stat->metadata_id() == producer_id_stats_metadata->id()) {
            producer_id = XStatsBuilder<XPlane>::IntOrUintValue(*stat);
          } else if (consumer_id_stats_metadata &&
                     stat->metadata_id() == consumer_id_stats_metadata->id()) {
            consumer_id = XStatsBuilder<XPlane>::IntOrUintValue(*stat);
          }
        }
      });
      if (correlation_id) {
        XFlow flow(XFlow::GetFlowId(host_id, *correlation_id), direction,
                   ContextType::kGpuLaunch);
        event.AddStatValue(*flow_stats_metadata, flow.ToStatValue());
      }
      if (connect_traceme) {
        if (producer_type && producer_id) {
          auto context_type = GetSafeContextType(*producer_type);
          XFlow flow(XFlow::GetFlowId(host_id, *producer_id, context_type),
                     XFlow::FlowDirection::kFlowOut, context_type);
          event.AddStatValue(*flow_stats_metadata, flow.ToStatValue());
        }
        if (consumer_type && consumer_id) {
          auto context_type = GetSafeContextType(*consumer_type);
          XFlow flow(XFlow::GetFlowId(host_id, *consumer_id, context_type),
                     XFlow::FlowDirection::kFlowIn, context_type);
          event.AddStatValue(*flow_stats_metadata, flow.ToStatValue());
        }
      }
    });
  });
}

}  // namespace profiler
}  // namespace tensorflow
