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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc() {
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
#include "tensorflow/core/profiler/utils/event_span.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

namespace {

// Representing a boundary of an event.
struct EventBoundary {
  // Time at this boundary.
  uint64 time_ps;
  // Type of the event.
  EventType type;
  // True if this is the start of the event; False if this is the end.
  bool is_start;
  EventBoundary(uint64 time_ps, EventType type, bool is_start)
      : time_ps(time_ps), type(type), is_start(is_start) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/profiler/utils/event_span.cc", "EventBoundary");
}
};

// Returns true if EventBoundary a should appear before EventBoundary b.
bool CmpEventBoundaries(const EventBoundary& a, const EventBoundary& b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/profiler/utils/event_span.cc", "CmpEventBoundaries");

  if (a.time_ps == b.time_ps) {
    if (a.is_start == b.is_start) {
      // Puts the higher-priority type before the lower-priority type if they
      // have the same time and same boundary type.
      return a.type > b.type;
    } else {
      // Puts the "end" bounary before the "start" boundary if they have the
      // same time.
      return !a.is_start;
    }
  }
  // In ascending order of time.
  return a.time_ps < b.time_ps;
}

// Generates vector of event boundaries from the given overlapped_events.
std::vector<EventBoundary> GenerateEventBoundaries(
    const std::vector<EventTypeSpan>& overlapped_events) {
  std::vector<EventBoundary> boundaries;
  boundaries.reserve(2 * overlapped_events.size());
  for (const auto& event : overlapped_events) {
    boundaries.push_back(
        {event.span.begin_ps(), event.type, /*is_start=*/true});
    boundaries.push_back({event.span.end_ps(), event.type, /*is_start=*/false});
  }
  absl::c_sort(boundaries, CmpEventBoundaries);
  return boundaries;
}

// A class to track the highest priority that an event should be assigned.
class PriorityTracker {
 private:
  // The current maximum priority.
  EventType current_max_priority_;
  // A count for each possible priority.
  std::vector<int64_t> priority_count_;

 public:
  PriorityTracker() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_2(mht_2_v, 262, "", "./tensorflow/core/profiler/utils/event_span.cc", "PriorityTracker");

    current_max_priority_ = UNKNOWN_TIME;
    priority_count_.resize(LAST_EVENT_TYPE + 1, 0);
  }
  // Updates current_max_priority_ and priority_count_[] given the boundary.
  // Returns the new current_max_priority_.
  EventType Update(const EventBoundary& boundary) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_3(mht_3_v, 271, "", "./tensorflow/core/profiler/utils/event_span.cc", "Update");

    EventType event_type = boundary.type;
    bool is_start = boundary.is_start;
    if (is_start) {
      priority_count_[event_type]++;
      if (event_type > current_max_priority_) {
        current_max_priority_ = event_type;
      }
    } else {
      priority_count_[event_type]--;
      if (event_type == current_max_priority_ &&
          priority_count_[event_type] == 0) {
        // Reduces current_max_priority_ to the first event type (starting from
        // the highest priority) that has a non-zero count.
        bool found = false;
        for (int i = event_type - 1; i >= 0; i--) {
          if (priority_count_[i] > 0) {
            current_max_priority_ = static_cast<EventType>(i);
            found = true;
            break;
          }
        }
        if (!found) current_max_priority_ = UNKNOWN_TIME;
      }
    }
    return current_max_priority_;
  }
};

constexpr int kNumGenericEventTypes = GenericEventType::kLastGenericEventType -
                                      GenericEventType::kFirstGenericEventType +
                                      1;

using GenericEventTypeStrMap =
    absl::flat_hash_map<GenericEventType, absl::string_view>;

const GenericEventTypeStrMap& GetGenericEventTypeStrMap() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_4(mht_4_v, 310, "", "./tensorflow/core/profiler/utils/event_span.cc", "GetGenericEventTypeStrMap");

  static const auto* generic_event_type_str_map = new GenericEventTypeStrMap({
      {kDeviceCompute, "Device compute"},
      {kDeviceToDevice, "Device to device"},
      {kDeviceCollectives, "Device collective communication"},
      {kHostCompute, "Host compute"},
      {kHostPrepare, "Kernel launch"},
      {kInput, "Input"},
      {kOutput, "Output"},
      {kCompile, "Compilation"},
      {kAllOthers, "All others"},
  });
  DCHECK_EQ(generic_event_type_str_map->size(), kNumGenericEventTypes);
  return *generic_event_type_str_map;
}

}  // namespace

absl::string_view GetGenericEventTypeStr(GenericEventType event_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_5(mht_5_v, 331, "", "./tensorflow/core/profiler/utils/event_span.cc", "GetGenericEventTypeStr");

  return GetGenericEventTypeStrMap().at(event_type);
}

std::string PrintEventType(EventType event_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_6(mht_6_v, 338, "", "./tensorflow/core/profiler/utils/event_span.cc", "PrintEventType");

  switch (event_type) {
    case UNKNOWN_TIME:
      return "unknown_time";
    case HOST_COMPUTE:
      return "host_compute";
    case HOST_COMPILE:
      return "host_compile";
    case HOST_TO_HOST:
      return "host_to_host";
    case HOST_TO_DEVICE:
      return "host_to_device";
    case HOST_PREPARE:
      return "host_prepare";
    case DEVICE_COLLECTIVES:
      return "device_collectives";
    case HOST_WAIT_INPUT:
      return "host_wait_input";
    case DEVICE_TO_DEVICE:
      return "device_to_device";
    case DEVICE_TO_HOST:
      return "device_to_host";
    case DEVICE_COMPUTE_32:
      return "device_compute_32";
    case DEVICE_COMPUTE_16:
      return "device_compute_16";
    case DEVICE_WAIT_DEVICE:
      return "device_wait_device";
    case DEVICE_WAIT_HOST:
      return "device_wait_host";
    default:
      return "unexpected";
  }
}

std::string PrintEventTypeSpan(const EventTypeSpan& event_type_span) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_7(mht_7_v, 376, "", "./tensorflow/core/profiler/utils/event_span.cc", "PrintEventTypeSpan");

  return absl::StrCat("(", PrintEventType(event_type_span.type), ", ",
                      event_type_span.span.DebugString(), ")");
}

absl::string_view PrintStepMarkerType(StepMarkerType type) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_8(mht_8_v, 384, "", "./tensorflow/core/profiler/utils/event_span.cc", "PrintStepMarkerType");

  switch (type) {
    case StepMarkerType::kExplicitHostStepMarker:
      return "ExplicitHostStepMarker";
    case StepMarkerType::kImplicitHostStepMarker:
      return "ImplicitHostStepMarker";
    case StepMarkerType::kDeviceStepMarker:
      return "DeviceStepMarker";
  }
}

std::string PrintStepMarker(const StepMarker& step_marker) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_9(mht_9_v, 398, "", "./tensorflow/core/profiler/utils/event_span.cc", "PrintStepMarker");

  return absl::StrCat("(", PrintStepMarkerType(step_marker.type), ", ",
                      step_marker.event_name, ", ",
                      step_marker.span.DebugString(), ")");
}

std::string PrintStepEvents(const StepEvents& step_events) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_10(mht_10_v, 407, "", "./tensorflow/core/profiler/utils/event_span.cc", "PrintStepEvents");

  std::vector<int64_t> step_ids;
  step_ids.reserve(step_events.size());
  for (const auto& id_details : step_events) {
    step_ids.push_back(id_details.first);
  }
  absl::c_sort(step_ids);
  std::string result = "{";
  for (auto id : step_ids) {
    absl::StrAppend(&result, "\n");
    auto* details = gtl::FindOrNull(step_events, id);
    std::string details_str = details ? details->DebugString() : "()";
    absl::StrAppend(&result, id, ":", details_str);
  }
  return absl::StrCat(result, "\n}");
}

void CombineStepEvents(const StepEvents& src, StepEvents* dst) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_11(mht_11_v, 427, "", "./tensorflow/core/profiler/utils/event_span.cc", "CombineStepEvents");

  for (const auto& step_details : src) {
    int64_t step_id = step_details.first;
    const StepDetails& src_details = step_details.second;
    StepDetails* dst_details = &(*dst)[step_id];
    dst_details->Combine(src_details);
  }
}

std::vector<EventTypeSpan> ToNonOverlappedEvents(
    const std::vector<EventTypeSpan>& overlapped_events) {
  std::vector<EventBoundary> event_boundaries =
      GenerateEventBoundaries(overlapped_events);
  std::vector<EventTypeSpan> result;
  if (event_boundaries.empty()) return result;
  result.reserve(event_boundaries.size());
  PriorityTracker priority_tracker;
  for (int64_t i = 0, end = (event_boundaries.size() - 1); i < end; i++) {
    EventType highest_priority = priority_tracker.Update(event_boundaries[i]);
    result.push_back({highest_priority, Timespan::FromEndPoints(
                                            event_boundaries[i].time_ps,
                                            event_boundaries[i + 1].time_ps)});
  }
  return result;
}

// Converts from overlapped step-events to non-overlapped step-events.
StepEvents ToNonOverlappedStepEvents(const StepEvents& overlapped_step_events) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_12(mht_12_v, 457, "", "./tensorflow/core/profiler/utils/event_span.cc", "ToNonOverlappedStepEvents");

  StepEvents non_overlapped_step_events;
  for (const auto& step_events : overlapped_step_events) {
    const auto& step_id = step_events.first;
    const auto& step_details = step_events.second;
    non_overlapped_step_events.try_emplace(step_id,
                                           step_details.ToNonOverlapped());
  }
  return non_overlapped_step_events;
}

void StepDetails::AddMarker(const StepMarker& m) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_13(mht_13_v, 471, "", "./tensorflow/core/profiler/utils/event_span.cc", "StepDetails::AddMarker");
 markers_.push_back(m); }

void StepDetails::AddEvent(const EventTypeSpan& e) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_14(mht_14_v, 476, "", "./tensorflow/core/profiler/utils/event_span.cc", "StepDetails::AddEvent");
 events_.push_back(e); }

void StepDetails::AggregateDeviceMemoryTransfers(
    const std::vector<DeviceMemoryTransfer> device_memory_transfers) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_15(mht_15_v, 482, "", "./tensorflow/core/profiler/utils/event_span.cc", "StepDetails::AggregateDeviceMemoryTransfers");

  if (device_memory_transfers.size() != device_memory_transfers_.size()) {
    return;  // Sanity check.
  }
  for (size_t i = 0; i < device_memory_transfers.size(); ++i) {
    device_memory_transfers_[i].set_occurrence(
        device_memory_transfers_[i].occurrence() +
        device_memory_transfers[i].occurrence());
    device_memory_transfers_[i].set_bytes_transferred(
        device_memory_transfers_[i].bytes_transferred() +
        device_memory_transfers[i].bytes_transferred());
    device_memory_transfers_[i].set_time_us(
        device_memory_transfers_[i].time_us() +
        device_memory_transfers[i].time_us());
  }
}

void StepDetails::AddCollectiveOpEvent(uint64 core_id, const AllReduceInfo& e) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_16(mht_16_v, 502, "", "./tensorflow/core/profiler/utils/event_span.cc", "StepDetails::AddCollectiveOpEvent");

  *collectives_[core_id].add_all_reduce_info() = e;
}

void StepDetails::AddDeviceMemoryTransferEvent(EventType event_type,
                                               const Timespan& time_span,
                                               uint64 bytes) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_17(mht_17_v, 511, "", "./tensorflow/core/profiler/utils/event_span.cc", "StepDetails::AddDeviceMemoryTransferEvent");

  int index = 0;
  switch (event_type) {
    case HOST_TO_DEVICE:
      index = 0;
      break;
    case DEVICE_TO_HOST:
      index = 1;
      break;
    case DEVICE_TO_DEVICE:
      index = 2;
      break;
    default:
      return;
  }
  device_memory_transfers_[index].set_occurrence(
      device_memory_transfers_[index].occurrence() + 1);
  device_memory_transfers_[index].set_time_us(
      device_memory_transfers_[index].time_us() +
      time_span.duration_ps() / 1000000.0);
  device_memory_transfers_[index].set_bytes_transferred(
      device_memory_transfers_[index].bytes_transferred() + bytes);
}

Timespan StepDetails::StepTime() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_18(mht_18_v, 538, "", "./tensorflow/core/profiler/utils/event_span.cc", "StepDetails::StepTime");

  Timespan max_host_step_time;
  Timespan max_device_step_time;
  for (const auto& marker : markers_) {
    Timespan& cur_max_step_time =
        marker.type == StepMarkerType::kDeviceStepMarker ? max_device_step_time
                                                         : max_host_step_time;
    const Timespan& new_step_time = marker.span;
    if (new_step_time.duration_ps() > cur_max_step_time.duration_ps())
      cur_max_step_time = new_step_time;
  }
  // CPU-only profile.
  if (max_device_step_time.Empty()) {
    return max_host_step_time;
  }

  // If the host step time includes the device step time, use the host step
  // time. This covers the case where the device is synchronized at the end of
  // each step.
  if (max_host_step_time.Includes(max_device_step_time)) {
    return max_host_step_time;
  }
  return max_device_step_time;
}

StepDetails StepDetails::ToNonOverlapped() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_19(mht_19_v, 566, "", "./tensorflow/core/profiler/utils/event_span.cc", "StepDetails::ToNonOverlapped");

  StepDetails non_overlapped_step_details;
  non_overlapped_step_details.markers_ = markers_;
  non_overlapped_step_details.events_ = ToNonOverlappedEvents(events_);
  non_overlapped_step_details.collectives_ = collectives_;
  non_overlapped_step_details.device_memory_transfers_ =
      device_memory_transfers_;
  non_overlapped_step_details.step_name_ = step_name_;
  return non_overlapped_step_details;
}

void StepDetails::Combine(const StepDetails& other) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_20(mht_20_v, 580, "", "./tensorflow/core/profiler/utils/event_span.cc", "StepDetails::Combine");

  markers_.insert(markers_.end(), other.markers_.begin(), other.markers_.end());
  events_.insert(events_.end(), other.events_.begin(), other.events_.end());
  collectives_.insert(other.collectives_.begin(), other.collectives_.end());
  AggregateDeviceMemoryTransfers(other.device_memory_transfers_);
  if (step_name_.empty()) step_name_ = other.step_name_;
}

std::string StepDetails::DebugString() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_21(mht_21_v, 591, "", "./tensorflow/core/profiler/utils/event_span.cc", "StepDetails::DebugString");

  std::string result = "([";
  for (int i = 0, end = markers_.size(); i < end; i++) {
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, PrintStepMarker(markers_[i]));
  }
  absl::StrAppend(&result, "], [");
  for (int i = 0, end = events_.size(); i < end; i++) {
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, PrintEventTypeSpan(events_[i]));
  }
  return absl::StrCat(result, "])");
}

bool StepDetails::operator==(const StepDetails& other) const {
  const auto& other_markers = other.Markers();
  if (markers_.size() != other_markers.size()) return false;
  for (uint64 i = 0; i < markers_.size(); i++) {
    if (markers_[i] != other_markers[i]) return false;
  }
  const auto& other_events = other.Events();
  if (events_.size() != other_events.size()) return false;
  for (uint64 i = 0; i < events_.size(); i++) {
    if (events_[i] != other_events[i]) return false;
  }
  return true;
}

bool operator==(const StepEvents& a, const StepEvents& b) {
  if (a.size() != b.size()) return false;
  for (const auto& id_details : a) {
    const auto a_id = id_details.first;
    const auto& a_details = id_details.second;
    const auto* b_details = gtl::FindOrNull(b, a_id);
    if (b_details == nullptr) return false;
    if (a_details != *b_details) return false;
  }
  return true;
}

PrecisionStats ComputePrecisionStats(
    const StepEvents& nonoverlapped_step_events) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTcc mht_22(mht_22_v, 635, "", "./tensorflow/core/profiler/utils/event_span.cc", "ComputePrecisionStats");

  int64_t compute_32bit_ps = 0;
  int64_t compute_16bit_ps = 0;
  for (const auto& id_details : nonoverlapped_step_events) {
    for (const auto& event : id_details.second.Events()) {
      switch (event.type) {
        case DEVICE_COMPUTE_32:
          compute_32bit_ps += event.span.duration_ps();
          break;
        case DEVICE_COMPUTE_16:
          compute_16bit_ps += event.span.duration_ps();
          break;
        default:
          break;
      }
    }
  }
  PrecisionStats precision_stats;
  precision_stats.set_compute_32bit_ps(compute_32bit_ps);
  precision_stats.set_compute_16bit_ps(compute_16bit_ps);
  return precision_stats;
}

}  // namespace profiler
}  // namespace tensorflow
