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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_jsonDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_jsonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_jsonDTcc() {
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

/* Copyright 2020 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/trace_events_to_json.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
#include "tensorflow/core/profiler/utils/format_utils.h"
#include "tensorflow/core/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// Converts the given time from picoseconds to microseconds and then to a string
// using maximum precision.
inline std::string PicosToMicrosString(uint64 ps) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_jsonDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/profiler/convert/trace_events_to_json.cc", "PicosToMicrosString");

  return MaxPrecision(PicoToMicro(ps));
}

// Escapes and quotes the given string.
inline std::string JsonString(const std::string& s) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_jsonDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/profiler/convert/trace_events_to_json.cc", "JsonString");

  return Json::valueToQuotedString(s.c_str());
}

// Returns a vector of pointers to the elements in the given map, sorted by key.
template <typename Map>
std::vector<const typename Map::value_type*> SortByKey(const Map& m) {
  std::vector<const typename Map::value_type*> pairs;
  pairs.reserve(m.size());
  for (const auto& pair : m) {
    pairs.push_back(&pair);
  }
  absl::c_sort(pairs, [](const typename Map::value_type* a,
                         const typename Map::value_type* b) {
    return a->first < b->first;
  });
  return pairs;
}

inline void AddDeviceMetadata(uint32 device_id, const Device& device,
                              std::string* json) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_jsonDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/profiler/convert/trace_events_to_json.cc", "AddDeviceMetadata");

  if (!device.name().empty()) {
    absl::StrAppend(json, R"({"ph":"M","pid":)", device_id,
                    R"(,"name":"process_name","args":{"name":)",
                    JsonString(device.name()), "}},");
  }
  absl::StrAppend(json, R"({"ph":"M","pid":)", device_id,
                  R"(,"name":"process_sort_index","args":{"sort_index":)",
                  device_id, "}},");
}

inline void AddResourceMetadata(uint32 device_id, uint32 resource_id,
                                const Resource& resource, std::string* json) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_jsonDTcc mht_3(mht_3_v, 253, "", "./tensorflow/core/profiler/convert/trace_events_to_json.cc", "AddResourceMetadata");

  if (!resource.name().empty()) {
    absl::StrAppend(json, R"({"ph":"M","pid":)", device_id, R"(,"tid":)",
                    resource_id, R"(,"name":"thread_name","args":{"name":)",
                    JsonString(resource.name()), "}},");
  }
  uint32 sort_index =
      resource.sort_index() ? resource.sort_index() : resource_id;
  absl::StrAppend(json, R"({"ph":"M","pid":)", device_id, R"(,"tid":)",
                  resource_id, R"(,"name":"thread_sort_index")",
                  R"(,"args":{"sort_index":)", sort_index, "}},");
}

inline void AddTraceEvent(const TraceEvent& event, string* json) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_jsonDTcc mht_4(mht_4_v, 269, "", "./tensorflow/core/profiler/convert/trace_events_to_json.cc", "AddTraceEvent");

  auto duration_ps = std::max(event.duration_ps(), protobuf_uint64{1});
  absl::StrAppend(json, R"({"ph":"X","pid":)", event.device_id(), R"(,"tid":)",
                  event.resource_id(), R"(,"ts":)",
                  PicosToMicrosString(event.timestamp_ps()), R"(,"dur":)",
                  PicosToMicrosString(duration_ps), R"(,"name":)",
                  JsonString(event.name()));
  if (!event.args().empty()) {
    absl::StrAppend(json, R"(,"args":{)");
    for (const auto* arg : SortByKey(event.args())) {
      absl::StrAppend(json, JsonString(arg->first), ":",
                      JsonString(arg->second), ",");
    }
    // Replace trailing comma with closing brace.
    json->back() = '}';
  }
  absl::StrAppend(json, "},");
}

}  // namespace

std::string TraceEventsToJson(const Trace& trace) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPStrace_events_to_jsonDTcc mht_5(mht_5_v, 293, "", "./tensorflow/core/profiler/convert/trace_events_to_json.cc", "TraceEventsToJson");

  std::string json =
      R"({"displayTimeUnit":"ns","metadata":{"highres-ticks":true},)"
      R"("traceEvents":[)";
  for (const auto* id_and_device : SortByKey(trace.devices())) {
    uint32 device_id = id_and_device->first;
    const Device& device = id_and_device->second;
    AddDeviceMetadata(device_id, device, &json);
    for (const auto* id_and_resource : SortByKey(device.resources())) {
      uint32 resource_id = id_and_resource->first;
      const Resource& resource = id_and_resource->second;
      AddResourceMetadata(device_id, resource_id, resource, &json);
    }
  }
  for (const TraceEvent& event : trace.trace_events()) {
    AddTraceEvent(event, &json);
  }
  // Add one fake event to avoid dealing with no-trailing-comma rule.
  absl::StrAppend(&json, "{}]}");
  return json;
}

}  // namespace profiler
}  // namespace tensorflow
