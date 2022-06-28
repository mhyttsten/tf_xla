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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc() {
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

#include "tensorflow/core/profiler/convert/xplane_to_step_events.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

inline bool IsExplicitHostStepMarker(absl::string_view event_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("event_name: \"" + std::string(event_name.data(), event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "IsExplicitHostStepMarker");

  return (absl::StartsWith(event_name, "train") ||
          absl::StartsWith(event_name, "test") ||
          absl::StartsWith(event_name, "TraceContext")) &&
         !absl::StrContains(event_name, "/");
}

// Returns true if the given event_name should be considered as real computation
// on CPU.
inline bool IsRealCpuCompute(absl::string_view event_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("event_name: \"" + std::string(event_name.data(), event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "IsRealCpuCompute");

  bool not_real = absl::StartsWith(event_name, "EagerExecute") ||
                  absl::StartsWith(event_name, "EagerLocalExecute") ||
                  absl::StartsWith(event_name, "EagerKernelExecute") ||
                  absl::StartsWith(event_name, "FunctionRun") ||
                  IsExplicitHostStepMarker(event_name);
  return !not_real;
}

uint64 ParseNumBytesFromMemcpyDetail(absl::string_view memcpy_detail) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("memcpy_detail: \"" + std::string(memcpy_detail.data(), memcpy_detail.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "ParseNumBytesFromMemcpyDetail");

  const std::vector<absl::string_view> params =
      absl::StrSplit(memcpy_detail, absl::ByAnyChar(":\n"));

  // Processes value pairs.
  for (uint32 ii = 0; ii < params.size(); ii += 2) {
    if (params[ii] != "num_bytes") continue;
    uint64 value = 0;
    if (absl::SimpleAtoi(params[ii + 1], &value)) return value;
    break;
  }
  return 0ULL;
}

EventType ClassifyGpuCompute(absl::string_view event_name,
                             absl::string_view tensor_shapes) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("event_name: \"" + std::string(event_name.data(), event_name.size()) + "\"");
   mht_3_v.push_back("tensor_shapes: \"" + std::string(tensor_shapes.data(), tensor_shapes.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_3(mht_3_v, 256, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "ClassifyGpuCompute");

  if (tensor_shapes.empty()) {
    // Deduces the precision from the name.
    return (absl::StrContains(event_name, "half") ||
            absl::StrContains(event_name, "fp16"))
               ? DEVICE_COMPUTE_16
               : DEVICE_COMPUTE_32;
  } else {
    // Deduces the precision from the shapes.
    return (absl::StrContains(tensor_shapes, "half")) ? DEVICE_COMPUTE_16
                                                      : DEVICE_COMPUTE_32;
  }
}

EventType ClassifyGpuEvent(absl::string_view event_name,
                           absl::string_view tensor_shapes) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("event_name: \"" + std::string(event_name.data(), event_name.size()) + "\"");
   mht_4_v.push_back("tensor_shapes: \"" + std::string(tensor_shapes.data(), tensor_shapes.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_4(mht_4_v, 276, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "ClassifyGpuEvent");

  TfOp tf_op = ParseTfOpFullname(event_name);
  if (IsMemcpyHToDOp(tf_op)) {
    return HOST_TO_DEVICE;
  } else if (IsMemcpyDToHOp(tf_op)) {
    return DEVICE_TO_HOST;
  } else if (IsMemcpyDToDOp(tf_op)) {
    return DEVICE_TO_DEVICE;
  } else if (absl::StartsWithIgnoreCase(event_name, "nccl")) {
    return DEVICE_COLLECTIVES;
  } else {
    return ClassifyGpuCompute(event_name, tensor_shapes);
  }
}

EventType ClassifyCpuEvent(absl::string_view event_name, bool has_device,
                           bool has_correlation_id) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("event_name: \"" + std::string(event_name.data(), event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_5(mht_5_v, 296, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "ClassifyCpuEvent");

  TfOp tf_op = ParseTfOpFullname(event_name);
  if (IsInfeedEnqueueOp(tf_op) || IsMemcpyHToDOp(tf_op)) {
    return HOST_TO_DEVICE;
  } else if (IsMemcpyHToHOp(tf_op)) {
    return HOST_TO_HOST;
  } else if (has_device && (has_correlation_id ||
                            absl::StartsWithIgnoreCase(
                                event_name, "ExecutorState::Process"))) {
    // TODO(b/150420972): Separate runtime overhead from actual compute for
    // CPU-only.
    return HOST_PREPARE;
  } else if (absl::StartsWithIgnoreCase(event_name, "IteratorGetNext")) {
    return HOST_WAIT_INPUT;
  } else {
    return HOST_COMPUTE;
  }
}

}  // namespace

StepEvents ConvertHostThreadsXLineToStepEvents(
    const XLineVisitor& line, const StepEvents* device_step_events) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_6(mht_6_v, 321, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "ConvertHostThreadsXLineToStepEvents");

  StepEvents result;
  line.ForEachEvent([&](const XEventVisitor& event) {
    int64_t correlation_id = -1;
    int64_t group_id = -1;
    absl::string_view step_name;
    event.ForEachStat([&](const XStatVisitor& stat) {
      if (!stat.Type().has_value()) return;
      switch (stat.Type().value()) {
        case StatType::kCorrelationId:
          correlation_id = stat.IntValue();
          break;
        case StatType::kGroupId:
          group_id = stat.IntValue();
          break;
        case StatType::kStepName:
          step_name = stat.StrOrRefValue();
          break;
      }
    });
    if (group_id < 0) return;
    // Don't add CPU events when (1) it includes device step events and (2) it
    // doesn't have a device and that the group_id (i.e. step number) already
    // appears on the device. This will filter out all cpu events that do not
    // correspond to any steps executed on the device.
    bool has_device = (device_step_events != nullptr);
    if (has_device && !device_step_events->contains(group_id)) return;
    if (IsExplicitHostStepMarker(event.Name())) {
      result[group_id].AddMarker(
          StepMarker(StepMarkerType::kExplicitHostStepMarker, event.Name(),
                     event.GetTimespan()));
    } else if (!step_name.empty()) {
      // Grouping adds a step_name stat to implicit host step markers.
      result[group_id].AddMarker(
          StepMarker(StepMarkerType::kImplicitHostStepMarker, event.Name(),
                     event.GetTimespan()));
    } else if (IsRealCpuCompute(event.Name())) {
      result[group_id].AddEvent(EventTypeSpan(
          ClassifyCpuEvent(event.Name(), has_device, correlation_id >= 0),
          event.GetTimespan()));
    }
    if (!step_name.empty()) {
      result[group_id].SetStepName(std::string(step_name));
    }
  });
  return result;
}

StepEvents ConvertHostThreadsXPlaneToStepEvents(
    const XPlane& host_trace, const StepEvents* device_step_events) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_7(mht_7_v, 373, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "ConvertHostThreadsXPlaneToStepEvents");

  StepEvents host_step_events;
  XPlaneVisitor plane = CreateTfXPlaneVisitor(&host_trace);
  plane.ForEachLine([&](const XLineVisitor& line) {
    StepEvents thread_step_events =
        ConvertHostThreadsXLineToStepEvents(line, device_step_events);
    CombineStepEvents(thread_step_events, &host_step_events);
  });
  return host_step_events;
}

StepEvents ConvertDeviceStepInfoToStepMarkers(const XLineVisitor& line) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_8(mht_8_v, 387, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "ConvertDeviceStepInfoToStepMarkers");

  StepEvents result;
  line.ForEachEvent([&](const XEventVisitor& event) {
    if (absl::optional<XStatVisitor> stat = event.GetStat(StatType::kGroupId)) {
      result[stat->IntValue()].AddMarker(
          StepMarker(StepMarkerType::kDeviceStepMarker, event.Name(),
                     event.GetTimespan()));
    }
  });
  return result;
}

StepEvents ConvertDeviceTraceXLineToStepEvents(const uint64 device_id,
                                               const XLineVisitor& line) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_9(mht_9_v, 403, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "ConvertDeviceTraceXLineToStepEvents");

  StepEvents result;
  line.ForEachEvent([&](const XEventVisitor& event) {
    int64_t correlation_id = -1;
    int64_t group_id = -1;
    absl::string_view tensor_shapes;
    absl::string_view memcpy_details;
    event.ForEachStat([&](const XStatVisitor& stat) {
      if (!stat.Type().has_value()) return;
      switch (stat.Type().value()) {
        case StatType::kCorrelationId:
          correlation_id = stat.IntValue();
          break;
        case StatType::kGroupId:
          group_id = stat.IntValue();
          break;
        case StatType::kTensorShapes:
          tensor_shapes = stat.StrOrRefValue();
          break;
        case StatType::kMemcpyDetails:
          memcpy_details = stat.StrOrRefValue();
          break;
      }
    });

    if (correlation_id >= 0 && group_id >= 0) {
      EventType event_type = ClassifyGpuEvent(event.Name(), tensor_shapes);
      EventTypeSpan event_type_span(event_type, event.GetTimespan());
      result[group_id].AddEvent(event_type_span);
      switch (event_type) {
        case DEVICE_COLLECTIVES: {
          AllReduceInfo collective_ops;
          collective_ops.set_start_time_ps(event.TimestampPs());
          collective_ops.set_end_time_ps(event.EndOffsetPs());
          // TODO(jiesun): figure out how to get size info etc.
          result[group_id].AddCollectiveOpEvent(device_id, collective_ops);
          break;
        }
        case HOST_TO_DEVICE:
        case DEVICE_TO_DEVICE:
        case DEVICE_TO_HOST: {
          // TODO(jiesun): not all memcpy events are grouped, figure out a
          // better way to attribute them to steps.
          uint64 bytes_transferred =
              ParseNumBytesFromMemcpyDetail(memcpy_details);
          result[group_id].AddDeviceMemoryTransferEvent(
              event_type, event.GetTimespan(), bytes_transferred);
          break;
        }
        default:
          return;
      }
    }
  });
  return result;
}

StepEvents ConvertDeviceTraceXPlaneToStepEvents(const XPlane& device_trace) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_step_eventsDTcc mht_10(mht_10_v, 463, "", "./tensorflow/core/profiler/convert/xplane_to_step_events.cc", "ConvertDeviceTraceXPlaneToStepEvents");

  StepEvents device_step_events;
  XPlaneVisitor plane = CreateTfXPlaneVisitor(&device_trace);
  plane.ForEachLine([&](const XLineVisitor& line) {
    int64_t line_id = line.Id();
    if (line_id == kThreadIdStepInfo) {
      StepEvents step_marker_events = ConvertDeviceStepInfoToStepMarkers(line);
      CombineStepEvents(step_marker_events, &device_step_events);
    } else if (IsDerivedThreadId(line_id)) {
      return;
    } else {
      StepEvents stream_step_events =
          ConvertDeviceTraceXLineToStepEvents(plane.Id(), line);
      CombineStepEvents(stream_step_events, &device_step_events);
    }
  });
  return device_step_events;
}

}  // namespace profiler
}  // namespace tensorflow
