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
class MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc {
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
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <cstdio>
#include <functional>
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include "tensorflow/lite/simple_memory_arena.h"

namespace tflite {
namespace {
// Same w/ that defined in tensorflow/lite/arena_planner.cc.
constexpr int32_t kNodeNotAssigned = std::numeric_limits<int32_t>::max();

void PrintIntVector(const std::vector<int>& v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "PrintIntVector");

  if (v.empty()) {
    printf("[]");
    return;
  }

  int range_start = v[0];
  int range_end = range_start;
  std::function<void(const char*)> print_range = [&](const char* suffix) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("suffix: \"" + (suffix == nullptr ? std::string("nullptr") : std::string((char*)suffix)) + "\"");
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "lambda");

    if (range_end == range_start) {
      printf("%d%s", range_start, suffix);
    } else if (range_end == range_start + 1) {
      printf("%d,%d%s", range_start, range_end, suffix);
    } else {
      printf("%d-%d%s", range_start, range_end, suffix);
    }
  };

  printf("[");
  for (int i = 1; i < v.size(); ++i) {
    int current = v[i];
    if (current == range_end + 1) {
      range_end = current;
    } else {
      print_range(",");
      range_start = range_end = current;
    }
  }
  print_range("]");
}

struct PerLayerInfo {
  PerLayerInfo() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_2(mht_2_v, 238, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "PerLayerInfo");
}
  PerLayerInfo(int id, size_t bytes, const std::vector<int>& tensors)
      : node_id(id), total_bytes(bytes), live_tensors(tensors) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_3(mht_3_v, 243, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "PerLayerInfo");
}
  int node_id;
  size_t total_bytes = 0;
  std::vector<int> live_tensors;
};
struct PerLayerInfoGreater {
  bool operator()(const PerLayerInfo& l, const PerLayerInfo& r) {
    return l.total_bytes > r.total_bytes;
  }
};
class PerLayerMinHeap
    : public std::priority_queue<PerLayerInfo, std::vector<PerLayerInfo>,
                                 PerLayerInfoGreater> {
 public:
  // Just to expose iterators to simplify iterating over contained elements.
  std::vector<PerLayerInfo>::const_iterator begin() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_4(mht_4_v, 261, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "begin");
 return c.begin(); }
  std::vector<PerLayerInfo>::const_iterator end() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_5(mht_5_v, 265, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "end");
 return c.end(); }
};

class TopKLayers {
 public:
  TopKLayers(size_t top_k, size_t arena_size)
      : top_k_(top_k), arena_size_(arena_size) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_6(mht_6_v, 274, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "TopKLayers");
}

  void Add(int node_id, size_t total_bytes,
           const std::vector<int>& live_tensors) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_7(mht_7_v, 280, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "Add");

    if (topk_usage_.size() < top_k_) {
      topk_usage_.emplace(PerLayerInfo(node_id, total_bytes, live_tensors));
      return;
    }
    if (total_bytes < topk_usage_.top().total_bytes) return;
    topk_usage_.pop();
    topk_usage_.emplace(PerLayerInfo(node_id, total_bytes, live_tensors));
  }

  void Print() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_8(mht_8_v, 293, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "Print");

    printf("\nTop %zu memory-consuming layers:\n",
           topk_usage_.size() < top_k_ ? topk_usage_.size() : top_k_);
    // As we use a min-heap but want to print out usage in decreasing order, we
    // use a temporary vector to hold pointers to top memory-consuming layers
    // and do a sorting on it.
    std::vector<const PerLayerInfo*> tops;
    for (const auto& usage : topk_usage_) tops.push_back(&usage);
    std::sort(tops.begin(), tops.end(),
              [](const PerLayerInfo* l, const PerLayerInfo* r) {
                return l->total_bytes > r->total_bytes;
              });
    for (const auto* usage : tops) {
      printf(
          "Node %d: %zu bytes (%.3f MB), utilization rate: %.3f%%, %zu live "
          "tensors: ",
          usage->node_id, usage->total_bytes,
          static_cast<float>(usage->total_bytes) / (1 << 20),
          static_cast<float>(usage->total_bytes) / arena_size_ * 100.0,
          usage->live_tensors.size());
      PrintIntVector(usage->live_tensors);
      printf("\n");
    }
    printf("\n");
  }

 private:
  const size_t top_k_;
  const size_t arena_size_;
  PerLayerMinHeap topk_usage_;
};
}  // namespace

// Corresponding weak declaration found in lite/simple_memory_arena.cc
void DumpArenaInfo(const std::string& name,
                   const std::vector<int>& execution_plan, size_t arena_size,
                   const std::vector<ArenaAllocWithUsageInterval>& allocs) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_debug_dumpDTcc mht_9(mht_9_v, 333, "", "./tensorflow/lite/simple_memory_arena_debug_dump.cc", "DumpArenaInfo");

  if (allocs.empty() || execution_plan.empty()) return;

  const int max_node_id =
      *std::max_element(execution_plan.begin(), execution_plan.end());

  printf("=== Beginning of %s ===\n", name.c_str());
  printf("Total size is %zu bytes (%.3f MB), holding %zu tensors.\n",
         arena_size, static_cast<float>(arena_size) / (1 << 20), allocs.size());
  std::vector<int> max_size_tensors;
  size_t max_tensor_size = 0;
  for (const auto& alloc_info : allocs) {
    printf("tensor %d: life_span: node [%d, %d], size:  %zu bytes (%.3f MB).\n",
           alloc_info.tensor, alloc_info.first_node,
           alloc_info.last_node == kNodeNotAssigned ? max_node_id
                                                    : alloc_info.last_node,
           alloc_info.size, static_cast<float>(alloc_info.size) / (1 << 20));
    if (alloc_info.size > max_tensor_size) {
      max_size_tensors.clear();
      max_size_tensors.push_back(alloc_info.tensor);
      max_tensor_size = alloc_info.size;
    } else if (alloc_info.size == max_tensor_size) {
      max_size_tensors.push_back(alloc_info.tensor);
    }
  }
  std::sort(max_size_tensors.begin(), max_size_tensors.end());
  printf("%zu tensors are of same max size (%zu B (%.3f MB)): ",
         max_size_tensors.size(), max_tensor_size,
         static_cast<float>(max_tensor_size) / (1 << 20));
  PrintIntVector(max_size_tensors);

  printf("\nPer-layer-info in the order of op execution:\n");
  // A straightforward way of computing per-op memory consumption
  // in the order of O(execution_plan.size() * allocs.size().
  std::vector<size_t> per_op_mem_bytes(execution_plan.size());
  // Track top 5 layers that consume most memory.
  TopKLayers top_usage(5, arena_size);
  for (int i = 0; i < execution_plan.size(); ++i) {
    const int node_id = execution_plan[i];
    size_t total_bytes = 0;
    std::vector<int> live_tensors;
    for (const auto& alloc_info : allocs) {
      if (node_id >= alloc_info.first_node && node_id <= alloc_info.last_node) {
        total_bytes += alloc_info.size;
        live_tensors.push_back(alloc_info.tensor);
      }
    }
    per_op_mem_bytes[i] = total_bytes;
    std::sort(live_tensors.begin(), live_tensors.end());
    printf(
        "Node %d: %zu bytes (%.3f MB), utilization rate: %.3f%%, %zu live "
        "tensors: ",
        node_id, total_bytes, static_cast<float>(total_bytes) / (1 << 20),
        static_cast<float>(total_bytes) / arena_size * 100.0,
        live_tensors.size());
    PrintIntVector(live_tensors);
    printf("\n");
    top_usage.Add(node_id, total_bytes, live_tensors);
  }
  top_usage.Print();
  printf("===End of %s ===\n\n", name.c_str());
}
}  // namespace tflite
