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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTh() {
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


#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"

namespace tensorflow {
namespace profiler {

// Returns a Timespan from an XEvent.
// WARNING: This should only be used when comparing events from the same XLine.
inline Timespan XEventTimespan(const XEvent& event) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTh mht_0(mht_0_v, 202, "", "./tensorflow/core/profiler/utils/xplane_utils.h", "XEventTimespan");

  return Timespan(event.offset_ps(), event.duration_ps());
}

// Returns the plane with the given name or nullptr if not found.
const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name);
XPlane* FindMutablePlaneWithName(XSpace* space, absl::string_view name);

// Returns the planes with the given names, if found.
std::vector<const XPlane*> FindPlanesWithNames(
    const XSpace& space, const std::vector<absl::string_view>& names);

// Returns the plane with the given name in the container. If necessary, adds a
// new plane to the container.
XPlane* FindOrAddMutablePlaneWithName(XSpace* space, absl::string_view name);

// Returns all the planes with a given prefix.
std::vector<const XPlane*> FindPlanesWithPrefix(const XSpace& space,
                                                absl::string_view prefix);
std::vector<XPlane*> FindMutablePlanesWithPrefix(XSpace* space,
                                                 absl::string_view prefix);

// Returns the plane with the given id or nullptr if not found.
const XLine* FindLineWithId(const XPlane& plane, int64_t id);

XStat* FindOrAddMutableStat(const XStatMetadata& stat_metadata, XEvent* event);

void RemovePlane(XSpace* space, const XPlane* plane);
void RemovePlanes(XSpace* space, const std::vector<const XPlane*>& planes);
void RemoveLine(XPlane* plane, const XLine* line);
void RemoveEvents(XLine* line,
                  const absl::flat_hash_set<const XEvent*>& events);

void RemoveEmptyPlanes(XSpace* space);
void RemoveEmptyLines(XPlane* plane);

// Sort lines in plane with a provided comparator.
template <class Compare>
void SortXLinesBy(XPlane* plane, Compare comp) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSxplane_utilsDTh mht_1(mht_1_v, 243, "", "./tensorflow/core/profiler/utils/xplane_utils.h", "SortXLinesBy");

  std::sort(plane->mutable_lines()->pointer_begin(),
            plane->mutable_lines()->pointer_end(), comp);
}

class XLinesComparatorByName {
 public:
  bool operator()(const XLine* a, const XLine* b) const {
    auto& line_a = a->display_name().empty() ? a->name() : a->display_name();
    auto& line_b = b->display_name().empty() ? b->name() : b->display_name();
    return line_a < line_b;
  }
};

// Sorts each XLine's XEvents by offset_ps (ascending) and duration_ps
// (descending) so nested events are sorted from outer to innermost.
void SortXPlane(XPlane* plane);
// Sorts each plane of the XSpace.
void SortXSpace(XSpace* space);

// Functor that compares XEvents for sorting by timespan.
struct XEventsComparator {
  bool operator()(const XEvent* a, const XEvent* b) const;
};

// Returns a sorted vector of all XEvents in the given XPlane.
template <class Compare>
std::vector<XEvent*> GetSortedEvents(XPlane* plane, Compare comp,
                                     bool include_derived_events = false) {
  std::vector<XEvent*> events;
  for (XLine& line : *plane->mutable_lines()) {
    if (!include_derived_events && IsDerivedThreadId(line.id())) continue;
    for (XEvent& event : *line.mutable_events()) {
      events.push_back(&event);
    }
  }
  absl::c_sort(events, XEventsComparator());
  return events;
}

// Normalize timestamps by time-shifting to start_time_ns_ as origin.
void NormalizeTimestamps(XPlane* plane, uint64 start_time_ns);
void NormalizeTimestamps(XSpace* space, uint64 start_time_ns);

// Merges src_plane into dst_plane. Both plane level stats, lines, events and
// event level stats are merged. If src_plane and dst_plane both have the same
// line, which have different start timestamps, we will normalize the events
// offset timestamp correspondingly.
void MergePlanes(const XPlane& src_plane, XPlane* dst_plane);

// Merges each plane with a src_planes, into the dst_plane.
void MergePlanes(const std::vector<const XPlane*>& src_planes,
                 XPlane* dst_plane);

// Plane's start timestamp is defined as the minimum of all lines' start
// timestamps. If zero line exists, return 0;
uint64 GetStartTimestampNs(const XPlane& plane);

// Returns true if there are no XEvents.
bool IsEmpty(const XSpace& space);

// Mutate the XPlane by adding predefined XFlow. e.g. GPU kernel launches =>
// GPU kernel events.
void AddFlowsToXplane(int32_t host_id, bool is_host_plane, bool connect_traceme,
                      XPlane* plane);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_UTILS_H_
