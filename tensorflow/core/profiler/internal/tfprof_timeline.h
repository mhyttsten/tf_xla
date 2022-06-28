/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TIMELINE_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TIMELINE_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh() {
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


#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/profiler/internal/tfprof_node_show.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {

typedef std::map<string, string> Event;

// Class for generating timeline json output.
class ChromeTraceFormatter {
 public:
  ChromeTraceFormatter() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_0(mht_0_v, 203, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "ChromeTraceFormatter");
}
  // The following methods creates timeline nodes. See chrome tracing format
  // document for details.
  Json::Value CreateEvent(const string& ph, const string& category,
                          const string& name, int64_t pid, int64_t tid,
                          int64_t ts);

  void EmitPID(const string& name, int64_t pid);

  void EmitRegion(int64_t ts, int64_t duration, int64_t pid, int64_t tid,
                  const string& category, const string& name, Json::Value args);

  void EmitFlowStart(const string& name, int64_t ts, int64_t pid, int64_t tid,
                     int64_t flow_id);

  void EmitFlowEnd(const string& name, int64_t ts, int64_t pid, int64_t tid,
                   int64_t flow_id);

  void EmitCounter(const string& category, const string& name, int64_t pid,
                   int64_t ts, const string& device, int64_t bytes,
                   const std::map<int64_t, std::vector<string>>& tensor_mem);

  string Format();

 private:
  // A event is a visualization unit in timeline.
  std::vector<Json::Value> events_;
  std::vector<Json::Value> metadata_;
};

// A process (time series of events) in the timeline.
class Process {
 public:
  Process(const string& device, int64_t pid) : device(device), pid(pid) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_1(mht_1_v, 240, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "Process");
}

  // Each lane is a map from start_time to end_time.
  std::vector<std::map<int64_t, int64_t>> lanes;
  // device for the time series.
  string device;
  // unique id for the time series.
  int64_t pid;
};

class TimeNode {
 public:
  TimeNode(Process* process, GraphNode* node, int64_t start_micros,
           int64_t exec_micros)
      : process(process),
        node(node),
        start_micros(start_micros),
        exec_micros(exec_micros),
        tid(-1) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_2(mht_2_v, 261, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "TimeNode");
}
  virtual ~TimeNode() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_3(mht_3_v, 265, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "~TimeNode");
}

  const string& name() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_4(mht_4_v, 270, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "name");
 return node->name(); }

  Process* process;
  GraphNode* node;
  int64_t start_micros;
  int64_t exec_micros;
  int64_t tid;
  std::vector<TimeNode*> next_tnodes;
};

// Tracking the memory based on the op input/output, temporary bytes and
// persistent bytes.
// Currently, we calculate a "predicted" memory, but do not use it for display.
// The displayed memory timeline is directly from the TensorFlow allocator,
// which is the groundtruth.
class MemoryTracker {
 public:
  class Device {
   public:
    // map from tensor name to a pair of <alloc time, bytes_in_use>.
    std::map<string, std::map<int64_t, int64_t>> tensor_allocs;
    // ground truth memory stats. time->bytes.
    std::map<int64_t, int64_t> allocations;
    // tracked allocations, might miss some bytes.
    std::map<int64_t, int64_t> tracked_allocations;
  };

  void TrackNode(int64_t step, const GraphNode* node);

  const std::map<string, Device>& devices() const { return devices_; }

 private:
  std::map<string, Device> devices_;
};

class Timeline {
 public:
  Timeline(int64_t step, const string& outfile)
      : step_(step), outfile_(outfile) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("outfile: \"" + outfile + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_5(mht_5_v, 312, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "Timeline");
}
  ~Timeline() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_6(mht_6_v, 316, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "~Timeline");
}

  int64_t step() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_7(mht_7_v, 321, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "step");
 return step_; }
  void SetStep(int64_t step) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_8(mht_8_v, 325, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "SetStep");
 step_ = step; }

  void GenerateGraphTimeline(const std::vector<GraphNode*>& gnodes);

  void GenerateScopeTimeline(const ScopeNode* node);

  void GenerateCodeTimeline(const CodeNode* node);

 private:
  void TrackNode(const GraphNode* node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_9(mht_9_v, 337, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "TrackNode");
 mem_tracker_.TrackNode(step_, node); }

  void OutputTimeline();

  template <typename Node>
  void EmitTreeNode(const Node* node, int64_t start_time, int64_t duration,
                    int64_t depth, std::set<int64_t>* visited_depth) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTh mht_10(mht_10_v, 346, "", "./tensorflow/core/profiler/internal/tfprof_timeline.h", "EmitTreeNode");

    if (visited_depth->find(depth) == visited_depth->end()) {
      chrome_formatter_.EmitPID(absl::StrCat("Scope:", depth), depth);
      visited_depth->insert(depth);
    }

    Json::Value args(Json::objectValue);
    args["name"] = Json::Value(node->name());
    args["op"] = Json::Value(node->name());
    chrome_formatter_.EmitRegion(start_time, duration, depth, 0, "Op",
                                 node->name(), args);

    int64_t total_micros = 0;
    int64_t c_start_time = start_time;
    for (const Node* child : node->show_children) {
      int64_t total_exec_micros = child->proto().total_exec_micros();
      if (total_exec_micros <= 0) {
        continue;
      }
      EmitTreeNode(child, c_start_time, total_exec_micros, depth + 1,
                   visited_depth);
      c_start_time += total_exec_micros;
      total_micros += total_exec_micros;
    }
    CHECK(total_micros <= duration) << node->name() << " parent:" << duration
                                    << " children:" << total_micros;
  }

  void AllocateTimeNodes(GraphNode* gnode);

  void AllocateLanes();

  int64_t AllocatePID();

  int64_t step_;
  const string outfile_;
  int64_t next_pid_ = 0;
  MemoryTracker mem_tracker_;
  ChromeTraceFormatter chrome_formatter_;
  std::map<string, int64_t> device_pids_;

  std::map<string, std::unique_ptr<Process>> process_;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, TimeNode*>>>
      alloc_nodes_;
  std::map<string, std::map<int64_t, std::unique_ptr<TimeNode>>> tnodes_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_TIMELINE_H_
