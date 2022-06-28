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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTh() {
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


#include <functional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"

namespace tensorflow {
namespace profiler {

// Additional information of an XEvent used to separate consecutive invocations
// of the same Op on the XLine.
struct XEventInfo {
  int64_t group_id;  // group ID of this XEvent or kInvalidGroupId.
  // The set of low level events associated with this XEvent.
  // For a TF op that is compiled by XLA, these are its composing HLO op names.
  // For a TF op that is not compiled by XLA, these are its composing kernel
  // names.
  absl::flat_hash_set<std::string> low_level_event_names;
  XEventInfo(int64_t gid, absl::string_view low_level_event_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("low_level_event_name: \"" + std::string(low_level_event_name.data(), low_level_event_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTh mht_0(mht_0_v, 212, "", "./tensorflow/core/profiler/utils/derived_timeline.h", "XEventInfo");

    group_id = gid;
    if (!low_level_event_name.empty()) {
      low_level_event_names.insert(std::string(low_level_event_name));
    }
  }
};

// Helper for deriving an XLine from events in another XLine.
class DerivedXLineBuilder {
 public:
  static const int64_t kInvalidGroupId = -1;
  DerivedXLineBuilder(XPlaneBuilder* plane, int64_t line_id,
                      absl::string_view name, int64_t timestamp_ns,
                      std::vector<DerivedXLineBuilder*> dependent_lines);

  // Either merges event with the last event or creates a new event on this
  // XLine. group_id and low_level_event_name may be passed to separate
  // consecutive invocations of the same event, depending on the XEvent type:
  //   TF-op, TF name scope: both group_id and low_level_event_name are used.
  //   HLO-op, step: only group_id is used.
  //   HLO module, source: both group_id and low_level_event_name are NOT used.
  void ExpandOrAddEvent(const XEvent& event, int64_t group_id = kInvalidGroupId,
                        absl::string_view low_level_event_name = "") {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTh mht_1(mht_1_v, 238, "", "./tensorflow/core/profiler/utils/derived_timeline.h", "ExpandOrAddEvent");

    ExpandOrAddLevelEvent(event, group_id, low_level_event_name,
                          /*level=*/0);
  }

  // The multi-level version of ExpandOrAddEvent. Here, the XEvents at different
  // levels all share the same group_id and low_level_event_name.
  void ExpandOrAddEvents(const std::vector<XEvent>& event_per_level,
                         int64_t group_id = kInvalidGroupId,
                         absl::string_view low_level_event_name = "") {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTh mht_2(mht_2_v, 250, "", "./tensorflow/core/profiler/utils/derived_timeline.h", "ExpandOrAddEvents");

    size_t current_nested_level = event_per_level.size();
    for (size_t level = 0; level < current_nested_level; ++level) {
      ExpandOrAddLevelEvent(event_per_level[level], group_id,
                            low_level_event_name, level);
    }
    if (current_nested_level) ResetLastEvents(current_nested_level);
  }

  // Reset the last events lower than or equal to the given level.
  void ResetLastEvents(int level = 0);

 private:
  // If the last event of the given level has the same metadata, expands it to
  // include the time until the given event's (offset_ps + duration_ps).
  // Otherwise, adds a new event and clears last_event_by_level_ for the levels
  // below the given level and all levels of the dependent lines. Clearing
  // last_event_by_level_ prevents a nested event from growing larger than the
  // parent event(s).
  void ExpandOrAddLevelEvent(const XEvent& event, int64_t group_id,
                             absl::string_view low_level_event_name, int level);

  void ResetDependentLines() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSderived_timelineDTh mht_3(mht_3_v, 275, "", "./tensorflow/core/profiler/utils/derived_timeline.h", "ResetDependentLines");

    for (DerivedXLineBuilder* line : dependent_lines_) {
      line->ResetLastEvents();
    }
  }

  const XStatMetadata* level_stats_ = nullptr;
  XLineBuilder line_;
  absl::flat_hash_map<int, absl::optional<XEventBuilder>> last_event_by_level_;
  absl::flat_hash_map<int, absl::optional<XEventInfo>> last_eventinfo_by_level_;
  std::vector<DerivedXLineBuilder*> dependent_lines_;
};

struct Symbol {
  absl::string_view tf_op_name;
  std::string source_info;
};

using SymbolResolver = std::function<Symbol(absl::optional<uint64_t> program_id,
                                            absl::string_view hlo_module_name,
                                            absl::string_view hlo_op)>;

// Derives TF name scope and op events from the TF op's fully qualified name
// with the name of the originating low-level event.
void ProcessTfOpEvent(absl::string_view tf_op_full_name,
                      absl::string_view low_level_event_name, int64_t offset_ps,
                      int64_t duration_ps, absl::optional<int64_t> group_id,
                      XPlaneBuilder* plane_builder,
                      DerivedXLineBuilder* tf_name_scope_line_builder,
                      DerivedXLineBuilder* tf_op_line_builder);

// Derives "Step Info", "Tensorflow Ops", "XLA Ops" and "XLA Module" lines in
// an NVIDIA_GPU device trace from data passed as ScopedAnnotations and stored
// as XStats in XEvents corresponding to GPU Kernels. Consecutive annotations
// with the same value are merged into a single event except for XLA modules.
// The device_trace is both input and output.
void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 const GroupMetadataMap& group_metadata_map,
                                 XPlane* device_trace,
                                 bool step_info_only = false);

// Derives "Launch Activities Summary" line from host trace.
void DeriveEventsFromHostTrace(const XPlane* host_trace,
                               const GroupMetadataMap& group_metadata_map,
                               std::vector<XPlane*> device_traces);

// Loops through XPlanes of input XSpace, if it is "device" XPlane, generating
// derived timelines for the plane by calling DeriveEventsFromAnnotations.
void GenerateDerivedTimeLines(const GroupMetadataMap& group_metadata_map,
                              XSpace* space, bool step_info_only = false);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_DERIVED_TIMELINE_H_
