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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/costs/graph_memory.h"

#include <deque>
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

Status GraphMemory::InferStatically(
    const std::unordered_map<string, DeviceProperties>& devices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/grappler/costs/graph_memory.cc", "GraphMemory::InferStatically");

  VirtualCluster cluster(devices);
  TF_RETURN_IF_ERROR(cluster.Provision());
  TF_RETURN_IF_ERROR(cluster.Initialize(item_));
  RunMetadata metadata;
  Status s = cluster.Run(item_, &metadata);
  // The virtual cluster returns the RESOURCE_EXHAUSTED error when it detects
  // that the model would run out of memory. We still get the metadata we need
  // out of the simulation, so we just ignore this error.
  if (!s.ok() && s.code() != error::RESOURCE_EXHAUSTED) {
    return s;
  }
  InferFromTrace(metadata.step_stats());
  return Status::OK();
}

Status GraphMemory::InferDynamically(Cluster* cluster) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/grappler/costs/graph_memory.cc", "GraphMemory::InferDynamically");

  if (!cluster->DetailedStatsEnabled()) {
    return errors::Unavailable("Detailed stats collection must be enabled");
  }

  TF_RETURN_IF_ERROR(cluster->Initialize(item_));
  RunMetadata metadata;
  TF_RETURN_IF_ERROR(cluster->Run(item_, &metadata));
  InferFromTrace(metadata.step_stats());
  return Status::OK();
}

int64_t GraphMemory::GetWorstCaseMemoryUsage() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/grappler/costs/graph_memory.cc", "GraphMemory::GetWorstCaseMemoryUsage");

  int64_t worst_case = -1;
  for (const auto& peak_usage : peak_usage_) {
    worst_case = std::max(worst_case, peak_usage.second.used_memory);
  }
  return worst_case;
}

void GraphMemory::InferMemUsageForNodes(
    const std::vector<const NodeDef*>& nodes, GraphProperties* properties,
    int64_t* worst_case_memory_usage, int64_t* best_case_memory_usage) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/grappler/costs/graph_memory.cc", "GraphMemory::InferMemUsageForNodes");

  // TODO(bsteiner) refine this: we should consider the multidevice case.
  *worst_case_memory_usage = 0;
  *best_case_memory_usage = 0;
  for (const auto& node : item_.graph.node()) {
    // Estimate the memory required to store the tensors generated by the node.
    std::vector<OpInfo::TensorProperties> outputs =
        properties->GetOutputProperties(node.name());
    int64_t node_memory_usage = InferMemUsageForNeighbors(outputs);

    // Worst case memory usage corresponds to the case where all the nodes are
    // alive.
    *worst_case_memory_usage += node_memory_usage;

    // Estimate the memory required to store the input tensors needed by the
    // node.
    std::vector<OpInfo::TensorProperties> inputs =
        properties->GetInputProperties(node.name());
    node_memory_usage += InferMemUsageForNeighbors(inputs);

    *best_case_memory_usage =
        std::max(*best_case_memory_usage, node_memory_usage);
  }
}

int64_t GraphMemory::InferMemUsageForNeighbors(
    const std::vector<OpInfo::TensorProperties>& props) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc mht_4(mht_4_v, 280, "", "./tensorflow/core/grappler/costs/graph_memory.cc", "GraphMemory::InferMemUsageForNeighbors");

  int64_t neighbors_memory_usage = 0;
  for (const auto& prop : props) {
    DataType dtype = prop.dtype();
    int size = DataTypeSize(dtype);
    TensorShapeProto shape = prop.shape();
    if (shape.unknown_rank()) {
      // Can't infer the size if the rank is unknown, just skip.
      continue;
    }
    // If one of the dimensions is unknown statically, assume it's one.
    for (int i = 0; i < shape.dim_size(); ++i) {
      if (shape.dim(i).size() < 0) {
        shape.mutable_dim(i)->set_size(1);
      }
    }
    int num_elems = TensorShape(shape).num_elements();
    neighbors_memory_usage += num_elems * size;
  }
  return neighbors_memory_usage;
}

static GraphMemory::LiveTensor* FindOrCreateLiveTensor(
    const string& node_name, int output_id,
    std::unordered_map<string, GraphMemory::LiveTensor*>* live_tensors,
    std::deque<GraphMemory::LiveTensor>* device_tensors) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc mht_5(mht_5_v, 309, "", "./tensorflow/core/grappler/costs/graph_memory.cc", "FindOrCreateLiveTensor");

  string name = strings::StrCat(node_name, ":", output_id);
  GraphMemory::LiveTensor* live;
  auto it = live_tensors->find(name);
  if (it == live_tensors->end()) {
    GraphMemory::LiveTensor temp;
    temp.node = node_name;
    temp.output_id = output_id;
    temp.allocation_time = 0;
    temp.deallocation_time = 0;
    device_tensors->push_front(temp);
    live = &device_tensors->front();
    (*live_tensors)[name] = live;
  } else {
    live = it->second;
  }
  return live;
}

namespace {
struct Event {
  Event(int64_t _timestamp, bool _allocated,
        const GraphMemory::LiveTensor* _tensor)
      : timestamp(_timestamp), allocated(_allocated), tensor(_tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc mht_6(mht_6_v, 335, "", "./tensorflow/core/grappler/costs/graph_memory.cc", "Event");
}

  int64_t timestamp;
  bool allocated;
  const GraphMemory::LiveTensor* tensor;

  bool operator<(const Event& other) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc mht_7(mht_7_v, 344, "", "./tensorflow/core/grappler/costs/graph_memory.cc", "operator<");

    return timestamp < other.timestamp;
  }
};
}  // namespace

void GraphMemory::InferFromTrace(const StepStats& timeline) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_memoryDTcc mht_8(mht_8_v, 353, "", "./tensorflow/core/grappler/costs/graph_memory.cc", "GraphMemory::InferFromTrace");

  std::unordered_map<string, string> node_placement;
  for (const auto& dev_stats : timeline.dev_stats()) {
    for (const auto& node_stats : dev_stats.node_stats()) {
      node_placement[node_stats.node_name()] = dev_stats.device();
    }
  }

  std::unordered_map<string, LiveTensor*> live_tensors;
  std::unordered_map<string, std::deque<LiveTensor>> live_tensors_per_device;
  std::unordered_map<string, const NodeDef*> node_map;
  for (const NodeDef& node : item_.graph.node()) {
    node_map[node.name()] = &node;
  }
  for (const auto& dev_stats : timeline.dev_stats()) {
    const string& device_name = dev_stats.device();
    const bool is_gpu = (device_name.find("GPU:") || device_name.find("gpu:"));
    std::deque<LiveTensor>& device_tensors =
        live_tensors_per_device[dev_stats.device()];
    for (const auto& node_stats : dev_stats.node_stats()) {
      for (int i = 0; i < node_stats.output_size(); ++i) {
        const auto& output = node_stats.output(i);

        LiveTensor* live = FindOrCreateLiveTensor(
            node_stats.node_name(), i, &live_tensors, &device_tensors);
        live->memory_used = output.tensor_description()
                                .allocation_description()
                                .allocated_bytes();

        // Allocations typically take place at the very beginning of the op
        // execution.
        live->allocation_time =
            Costs::MicroSeconds(node_stats.all_start_micros());
        // Add one nanosecond to the completion time of the ops to account for
        // TF overhead that slightly delays deallocations.
        live->deallocation_time = std::max<Costs::Duration>(
            live->deallocation_time,
            Costs::NanoSeconds(1) +
                Costs::MicroSeconds(node_stats.all_start_micros() +
                                    node_stats.op_end_rel_micros()));
      }

      auto it = node_map.find(node_stats.node_name());
      if (it == node_map.end()) {
        // Skip nodes inserted by TF since they don't exist in the original
        // graph (e.g _Send/_Recv nodes).
        continue;
      }
      const NodeDef* node = it->second;
      std::unordered_set<int> swapped_inputs;
      if (is_gpu) {
        auto it = node->attr().find("_swap_to_host");
        if (it != node->attr().end()) {
          const AttrValue& val = it->second;
          for (int port_id : val.list().i()) {
            swapped_inputs.insert(port_id);
          }
        }
      }
      for (int i = 0; i < node->input_size(); ++i) {
        if (swapped_inputs.find(i) != swapped_inputs.end()) {
          // The memory of swapped inputs will be released as early as possible:
          // therefore ignore this input when determining the deallocation time
          // of the tensor.
          continue;
        }
        const string& input = node->input(i);
        int position;
        string input_node = ParseNodeName(input, &position);
        if (position < 0) {
          // Skip control dependencies
          continue;
        }
        LiveTensor* live = FindOrCreateLiveTensor(
            input_node, position, &live_tensors,
            &live_tensors_per_device[node_placement[input_node]]);
        live->deallocation_time = std::max<Costs::Duration>(
            live->deallocation_time,
            Costs::NanoSeconds(1) +
                Costs::MicroSeconds(node_stats.all_start_micros() +
                                    node_stats.op_end_rel_micros()));
      }
    }
  }

  for (const auto& live_per_device : live_tensors_per_device) {
    std::vector<Event> events;
    events.reserve(2 * live_per_device.second.size());
    for (const auto& live : live_per_device.second) {
      events.emplace_back(static_cast<int64_t>(live.allocation_time.count()),
                          true, &live);
      events.emplace_back(static_cast<int64_t>(live.deallocation_time.count()),
                          false, &live);
    }
    std::stable_sort(events.begin(), events.end());
    size_t peak = 0;
    std::unordered_set<const LiveTensor*> live_at_peak;
    size_t current = 0;
    std::unordered_set<const LiveTensor*> currently_live;
    int events_size = events.size();
    for (int i = 0; i < events_size; ++i) {
      const auto& event = events[i];

      if (event.allocated) {
        VLOG(1) << "At time " << event.timestamp << " allocated "
                << event.tensor->memory_used << " for tensor "
                << event.tensor->node << ":" << event.tensor->output_id;
        current += event.tensor->memory_used;
        currently_live.insert(event.tensor);
      } else {
        VLOG(1) << "At time " << event.timestamp << " deallocated "
                << event.tensor->memory_used << " for tensor "
                << event.tensor->node << ":" << event.tensor->output_id;
        current -= event.tensor->memory_used;
        currently_live.erase(event.tensor);
      }
      if (i + 1 == events_size || event.timestamp != events[i + 1].timestamp) {
        if (current > peak) {
          peak = current;
          live_at_peak = currently_live;
        }
      }
    }
    MemoryUsage& peak_mem_usage = peak_usage_[live_per_device.first];
    peak_mem_usage.used_memory = peak;
    peak_mem_usage.live_tensors.clear();
    peak_mem_usage.live_tensors.reserve(live_at_peak.size());
    for (const auto& live : live_at_peak) {
      peak_mem_usage.live_tensors.push_back(*live);
    }
  }
}

}  // end namespace grappler
}  // end namespace tensorflow
