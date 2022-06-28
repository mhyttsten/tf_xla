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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_EVENT_SPAN_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_EVENT_SPAN_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh() {
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


#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

// The various event types. Enumerations are numbered such that a bigger number
// has a higher priority than a smaller number when used in execution-time
// breakdown.
enum EventType {
  // No event associated with the time. It could be that the machine was idle or
  // executing some events which were not traced.
  UNKNOWN_TIME = 0,
  // Host is computing.
  HOST_COMPUTE = 10,
  // Host is preprocessing the data before the execution on device.
  HOST_PREPROCESS = 20,
  // Host is postprocessing the data after the execution on device.
  HOST_POSTPROCESS = 30,
  // Host is batching data (for inference).
  HOST_BATCH_FORMATION = 40,
  // Host runtime, like memory allocation and etc.
  HOST_RUNTIME = 50,
  // Host is compiling.
  HOST_COMPILE = 60,
  // Host-to-host communication.
  HOST_TO_HOST = 70,
  // Host-to-device communication.
  HOST_TO_DEVICE = 80,
  // Host is preparing to launch a computation on device.
  HOST_PREPARE = 90,
  // Assigns a smaller priority to DEVICE_COLLECTIVES than HOST_WAIT_INPUT,
  // because if an all-reduce event is overlapped with an host-wait-input event,
  // we want to count it as waiting for input.
  // Collective Ops such as All-Reduce.
  DEVICE_COLLECTIVES = 100,
  // Host is waiting for input.
  HOST_WAIT_INPUT = 110,
  // Device-to-device communication.
  DEVICE_TO_DEVICE = 120,
  // Device-to-host communication.
  DEVICE_TO_HOST = 130,
  // Device is computing with 32-bit precision.
  DEVICE_COMPUTE_32 = 140,
  // Device is computing with 16-bit precision.
  DEVICE_COMPUTE_16 = 150,
  // Device is waiting for another device.
  DEVICE_WAIT_DEVICE = 160,
  // Device is waiting for host.
  DEVICE_WAIT_HOST = 170,
  LAST_EVENT_TYPE = DEVICE_WAIT_HOST
};

// Generic event types that shown to the user.
enum GenericEventType {
  kFirstGenericEventType = 1,
  // Device is computing.
  kDeviceCompute = kFirstGenericEventType,
  // Device-to-device communication.
  kDeviceToDevice,
  // Collective Ops such as All-Reduce and NCCL.
  kDeviceCollectives,
  // Host is computing.
  kHostCompute,
  // Host is preparing to launch a computation on device.
  kHostPrepare,
  // Device waiting for input from the host.
  kInput,
  // Device sending output to the host.
  kOutput,
  // Host is compling.
  kCompile,
  // No recognized event associated with the time.
  kAllOthers,
  kLastGenericEventType = kAllOthers,
};

// Contains the type and timespan of an event.
struct EventTypeSpan {
  EventType type;  // type of this event.
  Timespan span;   // timespan of this event.
  EventTypeSpan(EventType t, Timespan s) : type(t), span(s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh mht_0(mht_0_v, 276, "", "./tensorflow/core/profiler/utils/event_span.h", "EventTypeSpan");
}
  // Equality test.
  bool operator==(const EventTypeSpan& other) const {
    return type == other.type && span == other.span;
  }
  // Inequality test.
  bool operator!=(const EventTypeSpan& other) const {
    return !(*this == other);
  }
};

enum class StepMarkerType {
  // "TraceContext" TraceMe events.
  kExplicitHostStepMarker,
  // Identified by group_events (e.g., FunctionRun, SessionRun).
  kImplicitHostStepMarker,
  // Derived from the result of group_events. A device step marker starts with
  // the first device event of the group and ends with the last event of the
  // group.
  kDeviceStepMarker,
};

// Record of an event that is used as a step marker.
struct StepMarker {
  StepMarkerType type;
  std::string event_name;  // name of this event.
  std::string step_name;
  Timespan span;           // timespan of this event.
  StepMarker(StepMarkerType step_marker_type, absl::string_view name,
             Timespan s)
      : type(step_marker_type), event_name(name), span(s) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh mht_1(mht_1_v, 310, "", "./tensorflow/core/profiler/utils/event_span.h", "StepMarker");
}
  // Equality test.
  bool operator==(const StepMarker& other) const {
    return type == other.type && event_name == other.event_name &&
           span == other.span;
  }
  // Inequality test.
  bool operator!=(const StepMarker& other) const { return !(*this == other); }
};

// Details of a step. Note that this could be the result of combining the
// StepDetails of the same step executed on different cores.
class StepDetails {
 public:
  StepDetails() : device_memory_transfers_(3) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh mht_2(mht_2_v, 327, "", "./tensorflow/core/profiler/utils/event_span.h", "StepDetails");
}

  const std::vector<StepMarker>& Markers() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh mht_3(mht_3_v, 332, "", "./tensorflow/core/profiler/utils/event_span.h", "Markers");
 return markers_; }
  const std::vector<EventTypeSpan>& Events() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh mht_4(mht_4_v, 336, "", "./tensorflow/core/profiler/utils/event_span.h", "Events");
 return events_; }
  const absl::flat_hash_map<uint32, AllReduceDbResult>& Collectives() const {
    return collectives_;
  }
  const std::vector<DeviceMemoryTransfer>& DeviceMemoryTransfers() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh mht_5(mht_5_v, 343, "", "./tensorflow/core/profiler/utils/event_span.h", "DeviceMemoryTransfers");

    return device_memory_transfers_;
  }
  // Returns the step time.
  Timespan StepTime() const;
  // Adds a step-marker to this step.
  void AddMarker(const StepMarker& m);
  // Adds an EventTypeSpan to this step.
  void AddEvent(const EventTypeSpan& e);
  // Adds a collective op to this step.
  void AddCollectiveOpEvent(uint64 core_id, const AllReduceInfo& e);
  // Appends device memory transfer events to this step.
  // Only event type of HOST_TO_DEVICE/DEVICE_TO_DEVICE/DEVICE_TO_HOST are
  // allowed.
  void AddDeviceMemoryTransferEvent(EventType event_type,
                                    const Timespan& time_span, uint64 bytes);
  // Returns the step name.
  std::string StepName() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh mht_6(mht_6_v, 363, "", "./tensorflow/core/profiler/utils/event_span.h", "StepName");
 return step_name_; }
  // Sets the name of this step.
  void SetStepName(std::string step_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("step_name: \"" + step_name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSevent_spanDTh mht_7(mht_7_v, 369, "", "./tensorflow/core/profiler/utils/event_span.h", "SetStepName");
 step_name_ = step_name; }

  // Converts from overlapped events to non-overlapped events.
  StepDetails ToNonOverlapped() const;

  // Combines other.
  void Combine(const StepDetails& other);

  // Equality test.
  bool operator==(const StepDetails& other) const;
  // Inequality test.
  bool operator!=(const StepDetails& other) const { return !(*this == other); }

  // Returns a string that prints the content of this object.
  std::string DebugString() const;

 private:
  // Accumulates the device memory transfers from another step to this step.
  void AggregateDeviceMemoryTransfers(
      const std::vector<DeviceMemoryTransfer> device_memory_transfers);

  // All step-markers found for marking this step in the traces. There could be
  // multiple step-markers for a single step for different reasons. One such
  // reason is that there may be one step-marker for the same step on each core;
  // so after combining the StepDetails from multiple cores, there would be
  // multiple step-markers for the same step.
  std::vector<StepMarker> markers_;
  // All events belonging to this step.
  std::vector<EventTypeSpan> events_;
  // Collective operation related events such as all-reduce etc.
  absl::flat_hash_map<uint32, AllReduceDbResult> collectives_;
  // Device memory transfers (including time and bytes involved).
  // TODO(jiesun): Consider to use IntervalSet instead of just sum up the event
  // durations.
  std::vector<DeviceMemoryTransfer> device_memory_transfers_;
  std::string step_name_;
};

// Map from step_id to the events happened in that step.
using StepEvents = absl::flat_hash_map<int64_t /*step_id*/, StepDetails>;

// Equality test for StepEvents.
bool operator==(const StepEvents& a, const StepEvents& b);

// Returns the name of the given EventType.
std::string PrintEventType(EventType event_type);

// Returns the string of the given GenericEventType.
absl::string_view GetGenericEventTypeStr(GenericEventType event_type);

// Returns a string that prints the given EventTypeSpan.
std::string PrintEventTypeSpan(const EventTypeSpan& event_type_span);

// Returns a string that prints the given StepMarker.
std::string PrintStepMarker(const StepMarker& step_marker);

// Returns a string that prints the given StepEvents.
std::string PrintStepEvents(const StepEvents& step_events);

// Combines the src StepEvents into dst.
void CombineStepEvents(const StepEvents& src, StepEvents* dst);

// Converts from overlapped events to non-overlapped events.
std::vector<EventTypeSpan> ToNonOverlappedEvents(
    const std::vector<EventTypeSpan>& overlapped_events);

// Converts from overlapped step-events to non-overlapped step events.
StepEvents ToNonOverlappedStepEvents(const StepEvents& overlapped_step_events);

// Returns the precision stats of the given non-overlapped step events.
PrecisionStats ComputePrecisionStats(
    const StepEvents& nonoverlapped_step_events);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_EVENT_SPAN_H_
