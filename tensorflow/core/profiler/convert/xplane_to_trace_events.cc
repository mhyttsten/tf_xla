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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_eventsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_eventsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_eventsDTcc() {
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

#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"

#include <stddef.h>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

namespace {

void BuildDeviceAndResources(uint32 device_id, const XPlaneVisitor& plane,
                             Device* device) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_eventsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/profiler/convert/xplane_to_trace_events.cc", "BuildDeviceAndResources");

  device->set_name(std::string(plane.Name()));
  device->set_device_id(device_id);

  bool sort_by_ordinal = (device_id == kHostThreadsDeviceId);
  int ordinal = 0;
  plane.ForEachLine([&](const XLineVisitor& line) {
    uint32 resource_id = line.DisplayId();
    Resource& resource = (*device->mutable_resources())[resource_id];
    resource.set_resource_id(resource_id);
    resource.set_name(std::string(line.DisplayName()));
    if (sort_by_ordinal) {
      // When sort_index is absent (i.e. 0), resource id will be used.
      // Therefore sort_index starts with 1.
      resource.set_sort_index(++ordinal);
    }
  });
}

void ConvertXPlaneToTraceEvents(uint32 device_id, const XPlaneVisitor& xplane,
                                Trace* trace) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_eventsDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/profiler/convert/xplane_to_trace_events.cc", "ConvertXPlaneToTraceEvents");

  // Convert devices and resources.
  BuildDeviceAndResources(device_id, xplane,
                          &(*trace->mutable_devices())[device_id]);

  // Convert events.
  xplane.ForEachLine([device_id, trace](const XLineVisitor& xline) {
    uint32 resource_id = xline.DisplayId();
    xline.ForEachEvent(
        [device_id, resource_id, trace](const XEventVisitor& xevent) {
          int64_t event_type =
              xevent.Type().value_or(HostEventType::kUnknownHostEventType);
          if (IsInternalEvent(event_type)) return;
          auto* event = trace->add_trace_events();
          auto& args = *event->mutable_args();
          event->set_device_id(device_id);
          event->set_resource_id(resource_id);
          if (xevent.HasDisplayName()) {
            event->set_name(std::string(xevent.DisplayName()));
            args["long_name"] = std::string(xevent.Name());
          } else {
            event->set_name(std::string(xevent.Name()));
          }
          event->set_timestamp_ps(xevent.TimestampPs());
          event->set_duration_ps(xevent.DurationPs());

          auto for_each_stat = [&](const XStatVisitor& stat) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_eventsDTcc mht_2(mht_2_v, 263, "", "./tensorflow/core/profiler/convert/xplane_to_trace_events.cc", "lambda");

            if (stat.ValueCase() == XStat::VALUE_NOT_SET) return;
            if (IsInternalStat(stat.Type())) return;
            if (stat.Type() == StatType::kStepName) {
              event->set_name(stat.ToString());
            }
            args[std::string(stat.Name())] = stat.ToString();
          };
          // The metadata stats should appear before the per-occurrence stats.
          xevent.Metadata().ForEachStat(for_each_stat);
          xevent.ForEachStat(for_each_stat);
        });
  });
}

}  // namespace

void MaybeDropEventsForTraceViewer(Trace* trace, uint32 limit) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_eventsDTcc mht_3(mht_3_v, 283, "", "./tensorflow/core/profiler/convert/xplane_to_trace_events.cc", "MaybeDropEventsForTraceViewer");

  auto* trace_events = trace->mutable_trace_events();
  size_t trace_event_size = trace_events->size();
  if (trace_event_size <= limit) return;  // Nothing to do.
  // Sort the events according to start time.
  std::vector<uint64> timestamps;
  timestamps.reserve(trace_event_size);
  for (const auto& event : *trace_events) {
    timestamps.push_back(event.timestamp_ps());
  }
  std::partial_sort(timestamps.begin(), timestamps.begin() + limit,
                    timestamps.end(), std::less<uint64>());
  uint64 cutoff_timestamp = timestamps[limit - 1];
  trace_events->erase(std::remove_if(trace_events->begin(), trace_events->end(),
                                     [&](const TraceEvent& event) {
                                       return event.timestamp_ps() >
                                              cutoff_timestamp;
                                     }),
                      trace_events->end());
}

void ConvertXSpaceToTraceEvents(const XSpace& xspace, Trace* trace) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_eventsDTcc mht_4(mht_4_v, 307, "", "./tensorflow/core/profiler/convert/xplane_to_trace_events.cc", "ConvertXSpaceToTraceEvents");

  const XPlane* host_plane = FindPlaneWithName(xspace, kHostThreadsPlaneName);
  if (host_plane != nullptr) {
    XPlaneVisitor xplane = CreateTfXPlaneVisitor(host_plane);
    ConvertXPlaneToTraceEvents(kHostThreadsDeviceId, xplane, trace);
  }
  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(xspace, kGpuPlanePrefix);
  // We don't expect GPU and TPU planes and custom devices to be present in the
  // same XSpace.
  if (device_planes.empty()) {
    device_planes = FindPlanesWithPrefix(xspace, kTpuPlanePrefix);
  }
  if (device_planes.empty()) {
    device_planes = FindPlanesWithPrefix(xspace, kCustomPlanePrefix);
  }
  for (const XPlane* device_plane : device_planes) {
    XPlaneVisitor xplane = CreateTfXPlaneVisitor(device_plane);
    uint32 device_id = kFirstDeviceId + xplane.Id();
    ConvertXPlaneToTraceEvents(device_id, xplane, trace);
  }

  // Trace viewer (non-streaming) has scalability issues, we need to drop
  // events to avoid loading failure for trace viewer.
  constexpr uint64 kMaxEvents = 1000000;
  MaybeDropEventsForTraceViewer(trace, kMaxEvents);
}

void ConvertXSpaceToTraceEventsString(const XSpace& xspace,
                                      std::string* content) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_trace_eventsDTcc mht_5(mht_5_v, 339, "", "./tensorflow/core/profiler/convert/xplane_to_trace_events.cc", "ConvertXSpaceToTraceEventsString");

  Trace trace;
  ConvertXSpaceToTraceEvents(xspace, &trace);
  trace.SerializeToString(content);
}

}  // namespace profiler
}  // namespace tensorflow
