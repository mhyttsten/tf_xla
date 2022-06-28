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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"

namespace tflite {
namespace gpu {
namespace {

// This class build flow graph and solves Minimum-cost flow problem in it.
class MinCostFlowSolver {
 public:
  // Build auxiliary flow graph, based on information about intermediate
  // tensors.
  void Build(const std::vector<TensorUsageRecord<size_t>>& usage_records) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "Build");

    usage_records_ = &usage_records;
    num_tensors_ = usage_records.size();
    source_ = 2 * num_tensors_;
    sink_ = source_ + 1;
    edges_from_.resize(sink_ + 1);
    std::vector<size_t> old_record_ids;
    std::priority_queue<QueueRecord> objects_in_use;
    for (size_t i = 0; i < usage_records.size(); i++) {
      // Pop from the queue all objects that are no longer in use at the time
      // of execution of the first_task of i-th intermediate tensor.
      while (!objects_in_use.empty() &&
             objects_in_use.top().last_task < usage_records[i].first_task) {
        old_record_ids.push_back(objects_in_use.top().object_id);
        objects_in_use.pop();
      }
      objects_in_use.push({usage_records[i].last_task, i});
      AddEdge(source_, i, 1, 0);
      AddEdge(RightPartTwin(i), sink_, 1, 0);

      // Edge from source_ to i-th vertex in the right part of flow graph
      // are added for the case of allocation of new shared object for i-th
      // tensor. Cost of these edges is equal to the size of i-th tensor.
      AddEdge(source_, RightPartTwin(i), 1, usage_records[i].tensor_size);

      // Edges from vertices of the left part of flow graph, corresponding to
      // old_record_ids, to i-th vertex in the right part of flow graph are
      // added for the case of reusing previously created shared objects for
      // i-th tensor. Cost of these edges is an approximation of the size of
      // new allocated memory.
      for (auto record_id : old_record_ids) {
        int cost = 0;
        if (usage_records[i].tensor_size >
            usage_records[record_id].tensor_size) {
          cost = usage_records[i].tensor_size -
                 usage_records[record_id].tensor_size;
        }
        AddEdge(record_id, RightPartTwin(i), 1, cost);
      }
    }
  }

  // Solve Minimum-cost flow problem with Shortest Path Faster Algorithm.
  void Solve() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_1(mht_1_v, 253, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "Solve");

    const int kInf = std::numeric_limits<int>::max();
    std::vector<size_t> prev_edge(sink_ + 1);
    while (true) {
      std::queue<size_t> cur_queue, next_queue;
      std::vector<size_t> last_it_in_queue(sink_ + 1);
      std::vector<size_t> dist(sink_ + 1, kInf);
      size_t it = 1;
      cur_queue.push(source_);
      last_it_in_queue[source_] = it;
      dist[source_] = 0;
      // Find shortest path from source_ to sink_, using only edges with
      // positive capacity.
      while (!cur_queue.empty()) {
        ++it;
        while (!cur_queue.empty()) {
          auto v = cur_queue.front();
          cur_queue.pop();
          for (const auto& edge_id : edges_from_[v]) {
            const Edge& edge = edges_[edge_id];
            if (edge.cap > 0) {
              auto u = edge.dst;
              int new_dist = dist[v] + edge.cost;
              if (new_dist < dist[u]) {
                dist[u] = new_dist;
                prev_edge[u] = edge_id;
                if (last_it_in_queue[u] != it) {
                  next_queue.push(u);
                  last_it_in_queue[u] = it;
                }
              }
            }
          }
        }
        std::swap(cur_queue, next_queue);
      }
      // If path is not found, final result is ready.
      if (dist[sink_] == kInf) break;

      // If path is found, we need to decrease the capacity of its edges, and
      // increase the capacity of its reversed edges.
      for (size_t v = sink_; v != source_;) {
        --edges_[prev_edge[v]].cap;
        Edge& rev_edge = edges_[prev_edge[v] ^ 1];
        ++rev_edge.cap;
        v = rev_edge.dst;
      }
    }
  }

  void CalculateAssignment(ObjectsAssignment<size_t>* assignment) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_2(mht_2_v, 306, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "CalculateAssignment");

    assignment->object_sizes.clear();
    assignment->object_ids.assign(num_tensors_, kNotAssigned);
    is_tensor_assigned_.resize(num_tensors_);
    for (const auto& edge_id : edges_from_[source_]) {
      const Edge& edge = edges_[edge_id];
      if (edge.cap == 0 && IsRightPartVertex(edge.dst)) {
        assignment->object_sizes.push_back(
            AssignTensorsToNewSharedObject(LeftPartTwin(edge.dst), assignment));
      }
    }
  }

 private:
  struct Edge {
    Edge(size_t dst, int cap, int cost) : dst(dst), cap(cap), cost(cost) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_3(mht_3_v, 324, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "Edge");
}

    size_t dst;
    int cap;
    int cost;
  };

  // Add edge from vertex src to vertex dst with given capacity and cost and
  // its reversed edge to the flow graph. If some edge has index idx, its
  // reversed edge has index idx^1.
  void AddEdge(size_t src, size_t dst, int cap, int cost) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_4(mht_4_v, 337, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "AddEdge");

    edges_from_[src].push_back(edges_.size());
    edges_.emplace_back(dst, cap, cost);
    edges_from_[dst].push_back(edges_.size());
    edges_.push_back({src, 0, -cost});
  }

  // Check, if vertex_id belongs to right part of the flow graph.
  bool IsRightPartVertex(size_t vertex_id) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_5(mht_5_v, 348, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "IsRightPartVertex");

    return vertex_id >= num_tensors_ && vertex_id < 2 * num_tensors_;
  }

  // Return vertex from another part of the graph, that corresponds to the
  // same intermediate tensor.
  size_t LeftPartTwin(size_t vertex_id) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_6(mht_6_v, 357, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "LeftPartTwin");

    return vertex_id - num_tensors_;
  }
  size_t RightPartTwin(size_t vertex_id) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_7(mht_7_v, 363, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "RightPartTwin");

    return vertex_id + num_tensors_;
  }

  // This function uses recursive implementation of depth-first search and
  // returns maximum size from tensor tensor_id and all tensors, that will be
  // allocated at the same place with it after all operations that use
  // tensor_id are executed. Next tensor to be allocated at the same place
  // with tensor_id is a left part twin of such vertex v, that the edge
  // tensor_id->v is saturated (has zero residual capacity).
  size_t AssignTensorsToNewSharedObject(size_t tensor_id,
                                        ObjectsAssignment<size_t>* assignment) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_8(mht_8_v, 377, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "AssignTensorsToNewSharedObject");

    size_t cost = (*usage_records_)[tensor_id].tensor_size;
    is_tensor_assigned_[tensor_id] = true;
    assignment->object_ids[tensor_id] = assignment->object_sizes.size();
    for (const auto& edge_id : edges_from_[tensor_id]) {
      const Edge& edge = edges_[edge_id];
      size_t v = edge.dst;
      size_t left_twin = LeftPartTwin(v);
      if (edge.cap == 0 && IsRightPartVertex(v) &&
          !is_tensor_assigned_[left_twin]) {
        cost = std::max(cost,
                        AssignTensorsToNewSharedObject(left_twin, assignment));
      }
    }
    return cost;
  }

  size_t source_;
  size_t sink_;
  size_t num_tensors_;
  const std::vector<TensorUsageRecord<size_t>>* usage_records_;
  std::vector<Edge> edges_;
  std::vector<std::vector<size_t>> edges_from_;
  std::vector<bool> is_tensor_assigned_;
};

}  // namespace

// Implements memory management with a Minimum-cost flow matching algorithm.
//
// The problem of memory management is NP-complete. This function creates
// auxiliary flow graph, find minimum-cost flow in it and calculates the
// assignment of shared objects to tensors, using the result of the flow
// algorithm.
absl::Status MinCostFlowAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSmemory_managementPSmin_cost_flow_assignmentDTcc mht_9(mht_9_v, 416, "", "./tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.cc", "MinCostFlowAssignment");

  MinCostFlowSolver solver;
  solver.Build(usage_records);
  solver.Solve();
  solver.CalculateAssignment(assignment);
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
