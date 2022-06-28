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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc() {
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

#include "tensorflow/core/profiler/internal/tfprof_timeline.h"

#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"

namespace tensorflow {
namespace tfprof {
namespace {
int kMaxDisplayedMemNode = 10;

std::string GetTimeDevName(const std::string& dev) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("dev: \"" + dev + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "GetTimeDevName");

  if (dev.find("stream") != dev.npos) {
    return absl::StrCat("Op execution threads: ", dev);
  } else {
    return absl::StrCat("Op scheduling threads: ", dev);
  }
}
std::string GetMemoryLaneName(const std::string& dev) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("dev: \"" + dev + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "GetMemoryLaneName");

  return absl::StrCat("mem usage on:", dev);
}
}  // namespace

Json::Value ChromeTraceFormatter::CreateEvent(const string& ph,
                                              const string& category,
                                              const string& name, int64_t pid,
                                              int64_t tid, int64_t ts) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("ph: \"" + ph + "\"");
   mht_2_v.push_back("category: \"" + category + "\"");
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "ChromeTraceFormatter::CreateEvent");

  Json::Value event(Json::objectValue);
  event["ph"] = Json::Value(ph);
  event["cat"] = Json::Value(category);
  event["name"] = Json::Value(name);
  event["pid"] = Json::Int64(pid);
  event["tid"] = Json::Int64(tid);
  event["ts"] = Json::Int64(ts);
  return event;
}

void ChromeTraceFormatter::EmitPID(const string& name, int64_t pid) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "ChromeTraceFormatter::EmitPID");

  Json::Value event(Json::objectValue);
  event["name"] = Json::Value("process_name");
  event["ph"] = Json::Value("M");
  event["pid"] = Json::Int64(pid);
  Json::Value args(Json::objectValue);
  args["name"] = Json::Value(name);
  event["args"] = args;
  metadata_.push_back(event);
}

void ChromeTraceFormatter::EmitRegion(int64_t ts, int64_t duration, int64_t pid,
                                      int64_t tid, const string& category,
                                      const string& name, Json::Value args) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("category: \"" + category + "\"");
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_4(mht_4_v, 258, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "ChromeTraceFormatter::EmitRegion");

  Json::Value event = CreateEvent("X", category, name, pid, tid, ts);
  event["dur"] = Json::Int64(duration);
  event["args"] = std::move(args);
  metadata_.push_back(event);
}

void ChromeTraceFormatter::EmitFlowStart(const string& name, int64_t ts,
                                         int64_t pid, int64_t tid,
                                         int64_t flow_id) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_5(mht_5_v, 271, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "ChromeTraceFormatter::EmitFlowStart");

  Json::Value event = CreateEvent("s", "DataFlow", name, pid, tid, ts);
  event["id"] = Json::Int64(flow_id);
  events_.push_back(event);
}

void ChromeTraceFormatter::EmitFlowEnd(const string& name, int64_t ts,
                                       int64_t pid, int64_t tid,
                                       int64_t flow_id) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_6(mht_6_v, 283, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "ChromeTraceFormatter::EmitFlowEnd");

  Json::Value event = CreateEvent("t", "DataFlow", name, pid, tid, ts);
  event["id"] = Json::Int64(flow_id);
  events_.push_back(event);
}

void ChromeTraceFormatter::EmitCounter(
    const string& category, const string& name, int64_t pid, int64_t ts,
    const string& device, int64_t bytes,
    const std::map<int64_t, std::vector<string>>& tensor_mem) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("category: \"" + category + "\"");
   mht_7_v.push_back("name: \"" + name + "\"");
   mht_7_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_7(mht_7_v, 298, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "ChromeTraceFormatter::EmitCounter");

  Json::Value event = CreateEvent("C", category, "Allocated Bytes", pid, 0, ts);
  Json::Value args(Json::objectValue);
  args["Allocator Bytes in Use"] = Json::Int64(bytes);
  event["args"] = args;
  events_.push_back(event);

  // TODO(xpan): chrome://tracing is not ideal visualization for memory.
  // It would be great to have a customized UI for it.
  Json::Value event2 =
      CreateEvent("C", category, "Top Allocations", pid + 1, 0, ts);
  Json::Value args2(Json::objectValue);
  // Need to reserve the same args for all locations.
  for (int i = 1; i < kMaxDisplayedMemNode; ++i) {
    args2[absl::StrFormat("Top Allocation %02d", i)] = Json::Value("N/A");
  }
  int count = 0;
  for (auto it = tensor_mem.rbegin(); it != tensor_mem.rend(); ++it) {
    for (const string& t : it->second) {
      if (bytes < it->first || count >= kMaxDisplayedMemNode) {
        break;
      }
      args2[absl::StrFormat("Top Allocation %02d", count)] =
          Json::Value(absl::StrCat(it->first / 1000000.0, " MB from ", t));
      ++count;
      bytes -= it->first;
    }
  }
  args2[std::string("Not Displayed")] =
      Json::Value(absl::StrFormat("%.2f MB", bytes / 1000000.0));
  event2["args"] = args2;
  events_.push_back(event2);
}

string ChromeTraceFormatter::Format() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_8(mht_8_v, 335, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "ChromeTraceFormatter::Format");

  Json::Value trace;
  trace["traceEvents"] = Json::Value(Json::arrayValue);
  for (const Json::Value& v : metadata_) {
    trace["traceEvents"].append(v);
  }
  for (const Json::Value& v : events_) {
    trace["traceEvents"].append(v);
  }
  Json::FastWriter writer;
  string trace_str = writer.write(trace);
  if (trace_str.length() > 200 * 1024 * 1024) {
    absl::FPrintF(stderr,
                  "Trace file is over 200MB. Chrome might not be able to "
                  "display it. Consider to use filters (e.g. -min_micros "
                  "> 1000 or -op_type .*gpu:0.* to reduce the size.\n");
  }
  return trace_str;
}

void MemoryTracker::TrackNode(int64_t step, const GraphNode* node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_9(mht_9_v, 358, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "MemoryTracker::TrackNode");

  if (!node->Trackable(step)) {
    return;
  }

  Device& dev = devices_[node->node->canonical_device()];

  std::map<int64_t, int64_t> allocs;
  for (const auto& alloc : node->node->allocations(step)) {
    allocs[alloc.alloc_micros()] += alloc.alloc_bytes();
    dev.tracked_allocations[alloc.alloc_micros()] += alloc.alloc_bytes();
  }
  dev.tracked_allocations[0] += node->node->accelerator_persistent_bytes();
  allocs[0] += node->node->accelerator_persistent_bytes();

  int64_t last = 0;
  std::map<int64_t, int64_t>& aggregate_allocs =
      dev.tensor_allocs[node->name()];
  for (auto it = allocs.begin(); it != allocs.end(); ++it) {
    last += it->second;
    aggregate_allocs[it->first] = last;
  }
  for (const auto& bytes_in_use : node->node->allocator_bytes_in_use(step)) {
    if (bytes_in_use.first <= 0) continue;
    dev.allocations[bytes_in_use.first] = bytes_in_use.second;
  }
}

void Timeline::AllocateTimeNodes(GraphNode* gnode) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_10(mht_10_v, 389, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "Timeline::AllocateTimeNodes");

  if (gnode->Trackable(step_)) {
    TrackNode(gnode);
    const TFGraphNode* node = gnode->node;
    for (const auto& kernel_execs : node->op_execs(step_)) {
      const string& device = kernel_execs.first;

      if (process_.find(device) == process_.end()) {
        int64_t pid = AllocatePID();
        process_[device].reset(new Process(device, pid));
        chrome_formatter_.EmitPID(GetTimeDevName(device), pid);
      }
      Process* p = process_[device].get();

      for (const auto& exec : kernel_execs.second) {
        int64_t start_micros = exec.first;
        int64_t exec_micros = exec.second;
        // TODO(xpan): There might be start time duplication here.
        if (tnodes_[device].find(start_micros) == tnodes_[device].end()) {
          // TODO(xpan): Give each kernel call a unique_name.
          tnodes_[device][start_micros].reset(
              new TimeNode(p, gnode, start_micros, exec_micros));
        }
      }
    }
  }
  for (GraphNode* n : gnode->show_children) {
    AllocateTimeNodes(n);
  }
}

void Timeline::GenerateGraphTimeline(const std::vector<GraphNode*>& gnodes) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_11(mht_11_v, 423, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "Timeline::GenerateGraphTimeline");

  for (GraphNode* gnode : gnodes) {
    AllocateTimeNodes(gnode);
  }
  // To save memory, we only track cross-device (canonical device) flows.
  for (auto& process : tnodes_) {
    if (!IsCanonicalDevice(process.first)) continue;
    for (auto& tn : process.second) {
      TimeNode* tnode = tn.second.get();
      for (GraphNode* inp : tnode->node->children) {
        if (!inp->account || !inp->Trackable(step_)) {
          continue;
        }
        for (const auto& execs : inp->node->cpu_execs(step_)) {
          if (!IsCanonicalDevice(execs.first)) continue;
          if (process.first == execs.first) {
            // Not interested in flow within the same device.
            continue;
          }
          for (const auto& exec : execs.second) {
            int64_t start_micros = exec.first;
            auto cprocess = tnodes_.find(execs.first);
            if (cprocess == tnodes_.end()) continue;
            auto ctn = cprocess->second.find(start_micros);
            if (ctn == cprocess->second.end()) continue;
            ctn->second->next_tnodes.push_back(tnode);
          }
        }
      }
    }
  }

  AllocateLanes();
  absl::FPrintF(stdout, "generating trace file.\n");
  int64_t flow_id = 1;
  for (const auto& process : alloc_nodes_) {
    for (const auto& lane : process.second) {
      for (const auto& node : lane.second) {
        TimeNode* tnode = node.second;

        Json::Value args(Json::objectValue);
        args["name"] = Json::Value(tnode->name());
        chrome_formatter_.EmitRegion(node.first, tnode->exec_micros,
                                     process.first, lane.first, "Op",
                                     tnode->name(), args);
        // Flow is a directed arrow pointing from src to dst.
        // TODO(xpan): Disable flow to reduce json file size for now. Need
        // to think of a better way to make flow interpretable.
        for (TimeNode* next_tnode : node.second->next_tnodes) {
          chrome_formatter_.EmitFlowStart(
              tnode->name() + "_flow", tnode->start_micros + tnode->exec_micros,
              process.first, lane.first, flow_id);
          chrome_formatter_.EmitFlowEnd(
              tnode->name() + "_flow", next_tnode->start_micros,
              next_tnode->process->pid, next_tnode->tid, flow_id);
          flow_id += 1;
        }
      }
    }
  }
  for (const auto& dev : mem_tracker_.devices()) {
    if (IsPlacedOnCPU(dev.first)) {
      // TODO(xpan): Maybe also support CPU allocator memory tracking.
      continue;
    }
    int64_t pid = AllocatePID();
    chrome_formatter_.EmitPID(GetMemoryLaneName(dev.first), pid);
    int64_t pid2 = AllocatePID();
    chrome_formatter_.EmitPID(GetMemoryLaneName(dev.first) + " allocations",
                              pid2);

    const MemoryTracker::Device& device = dev.second;

    int64_t max_bytes_in_use = 0;
    int64_t cur_bytes_in_use = 0;
    int64_t last_point = 0;
    for (const auto& alloc : device.allocations) {
      cur_bytes_in_use = alloc.second;
      max_bytes_in_use = std::max(max_bytes_in_use, cur_bytes_in_use);
      // Do not plot too dense to reduce file size.
      int64_t ts = alloc.first;
      if (ts - last_point < 100) continue;
      last_point = ts;

      std::map<int64_t, std::vector<string>> tensor_mem;
      for (const auto& tensor_alloc_it : dev.second.tensor_allocs) {
        const auto& tensor_alloc = tensor_alloc_it.second;
        auto it = tensor_alloc.lower_bound(ts);
        if (it != tensor_alloc.begin()) {
          --it;
        }
        if (it->second > 0) {
          tensor_mem[it->second].push_back(tensor_alloc_it.first);
        }
      }
      chrome_formatter_.EmitCounter("Memory", "Memory Series", pid, ts,
                                    dev.first, cur_bytes_in_use, tensor_mem);
    }
    if (IsPlacedOnAccelerator(dev.first)) {
      absl::FPrintF(stdout, "%s peak memory: %.2f MB\n", dev.first,
                    max_bytes_in_use / 1000000.0);
    }
  }
  OutputTimeline();
}

void Timeline::GenerateScopeTimeline(const ScopeNode* node) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_12(mht_12_v, 532, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "Timeline::GenerateScopeTimeline");

  std::set<int64_t> visited_depth;
  EmitTreeNode(node, 0, node->proto().total_exec_micros(), 0, &visited_depth);
  OutputTimeline();
}

void Timeline::GenerateCodeTimeline(const CodeNode* node) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_13(mht_13_v, 541, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "Timeline::GenerateCodeTimeline");

  std::set<int64_t> visited_depth;
  EmitTreeNode(node, 0, node->proto().total_exec_micros(), 0, &visited_depth);
  OutputTimeline();
}

void Timeline::OutputTimeline() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_14(mht_14_v, 550, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "Timeline::OutputTimeline");

  std::string outfile = absl::StrFormat("%s_%d", outfile_, step());
  Status s =
      WriteStringToFile(Env::Default(), outfile, chrome_formatter_.Format());
  if (!s.ok()) {
    absl::FPrintF(stderr, "Failed to write timeline file: %s\nError: %s\n",
                  outfile, s.ToString());
    return;
  }
  absl::FPrintF(stdout,
                "\n******************************************************\n");
  absl::FPrintF(stdout,
                "Timeline file is written to %s.\n"
                "Open a Chrome browser, enter URL chrome://tracing and "
                "load the timeline file.",
                outfile);
  absl::FPrintF(stdout,
                "\n******************************************************\n");
  fflush(stdout);
}

void Timeline::AllocateLanes() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_15(mht_15_v, 574, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "Timeline::AllocateLanes");

  for (auto& process : tnodes_) {
    Process* p = process_[process.first].get();
    for (auto& tnode : process.second) {
      int64_t start_time = tnode.second->start_micros;
      int64_t end_time = tnode.second->start_micros + tnode.second->exec_micros;
      int64_t l = -1;
      for (int64_t i = 0, end = p->lanes.size(); i < end; ++i) {
        const auto& lane = p->lanes[i];
        l = i;
        for (auto cur_it = lane.rbegin(); cur_it != lane.rend(); ++cur_it) {
          if (cur_it->second > start_time) {
            l = -1;
            break;
          }
          if (start_time > cur_it->second) {
            break;
          }
        }
        if (l >= 0) {
          break;
        }
      }
      if (l < 0) {
        l = p->lanes.size();
        std::map<int64_t, int64_t> nlane;
        nlane[start_time] = end_time;
        p->lanes.push_back(nlane);
      } else {
        p->lanes[l][start_time] = end_time;
      }
      tnode.second->tid = l;
      alloc_nodes_[p->pid][l][start_time] = tnode.second.get();
    }
  }
}

int64_t Timeline::AllocatePID() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_timelineDTcc mht_16(mht_16_v, 614, "", "./tensorflow/core/profiler/internal/tfprof_timeline.cc", "Timeline::AllocatePID");

  int64_t cur_pid = next_pid_;
  next_pid_ += 1;
  return cur_pid;
}

}  // namespace tfprof
}  // namespace tensorflow
