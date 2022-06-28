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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc() {
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
#include "tensorflow/core/common_runtime/partitioning_utils.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_partition.h"

namespace tensorflow {

namespace {

// A helper to partiton a `graph` given a `device_set` and a `graph`.
// `partitions` maps device names to the graphdef assigned to that device.
Status PartitionFunctionGraph(
    const DeviceSet& device_set, Graph* graph,
    std::unordered_map<string, GraphDef>* partitions,
    std::function<string(const Node*)> node_to_loc,
    std::function<string(const Edge*)> get_tensor_name_attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/common_runtime/partitioning_utils.cc", "PartitionFunctionGraph");

  PartitionOptions partition_options;
  if (node_to_loc != nullptr) {
    partition_options.node_to_loc = node_to_loc;
  } else {
    partition_options.node_to_loc = [](const Node* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/common_runtime/partitioning_utils.cc", "lambda");

      // TODO(iga): To support the distributed case, first split the graph by
      // worker (e.g,. using the master session's `SplitByWorker` policy), and
      // then recursively partition the per-worker shards at the remote
      // worker(s). Currently, we simply split the graph at device boundaries.
      return node->assigned_device_name();
    };
  }
  int64_t edge_name_counter = 0;
  partition_options.new_name = [&edge_name_counter](const string& prefix) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/common_runtime/partitioning_utils.cc", "lambda");

    return strings::StrCat(prefix, "/_", ++edge_name_counter);
  };
  partition_options.get_incarnation =
      [&device_set](const string& name) -> int64 {
    const Device* d = device_set.FindDeviceByName(name);
    if (d == nullptr) {
      return PartitionOptions::kIllegalIncarnation;
    } else {
      return d->attributes().incarnation();
    }
  };
  partition_options.control_flow_added = false;
  partition_options.get_tensor_name_attr = get_tensor_name_attr;

  return Partition(partition_options, graph, partitions);
}

}  // namespace

Status PartitionFunctionGraph(
    const DeviceSet& device_set, std::unique_ptr<Graph> graph,
    std::unordered_map<string, std::unique_ptr<Graph>>* subgraphs,
    std::function<string(const Edge*)> get_tensor_name_attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/common_runtime/partitioning_utils.cc", "PartitionFunctionGraph");

  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(
      PartitionFunctionGraph(device_set, graph.get(), &partitions,
                             /*node_to_loc=*/nullptr, get_tensor_name_attr));

  for (auto& partition : partitions) {
    const string& device = partition.first;
    GraphDef& graph_def = partition.second;
    // Each partition gets a new graph.
    std::unique_ptr<Graph> subgraph(
        new Graph(graph->flib_def().default_registry()));
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(
        ConvertGraphDefToGraph(opts, std::move(graph_def), subgraph.get()));
    subgraphs->emplace(device, std::move(subgraph));
  }

  return Status::OK();
}

StatusOr<std::unique_ptr<Graph>> InsertTransferOps(
    const DeviceSet& device_set, std::unique_ptr<Graph> graph) {
  // Skip transfer op insertion if the graph nodes are not assigned to multiple
  // devices.
  auto node_to_loc = [](const Node* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc mht_4(mht_4_v, 287, "", "./tensorflow/core/common_runtime/partitioning_utils.cc", "lambda");

    return node->assigned_device_name();
  };
  bool has_multiple_devices = false;
  absl::optional<std::string> location;
  for (const Node* node : graph->op_nodes()) {
    if (location) {
      if (*location != node_to_loc(node)) {
        has_multiple_devices = true;
        break;
      }
    } else {
      location = node_to_loc(node);
    }
  }
  if (!has_multiple_devices) {
    return graph;
  }

  // Transfer ops are needed as there are multiple devices, so proceed with the
  // partitioning.
  auto new_graph = std::make_unique<Graph>(graph->flib_def());

  std::unordered_map<string, GraphDef> partitions;
  TF_RETURN_IF_ERROR(PartitionFunctionGraph(device_set, graph.get(),
                                            &partitions, node_to_loc,
                                            /*get_tensor_name_attr=*/nullptr));

  GraphDef merged_graph_def;
  if (!partitions.empty()) {
    auto iter = partitions.begin();
    merged_graph_def = std::move(iter->second);
    while (++iter != partitions.end()) {
      // TODO(b/220440252): MergeFrom() does memory copies when merging repeated
      // fields. Ideally, we can merge repeated fields by 'moving' data.
      // Consider using `proto2::util::MoveToEnd()` or so, once it is open
      // sourced.
      merged_graph_def.MergeFrom(iter->second);
    }
  }

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, std::move(merged_graph_def),
                                            new_graph.get()));
  return std::move(new_graph);
}

Status UpdateArgAndRetvalMetadata(
    Graph* graph, std::vector<FunctionArgIndex>* arg_indices,
    std::vector<int>* ret_indices,
    std::vector<AllocatorAttributes>* arg_alloc_attrs,
    std::vector<AllocatorAttributes>* ret_alloc_attrs, bool ints_on_device) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc mht_5(mht_5_v, 343, "", "./tensorflow/core/common_runtime/partitioning_utils.cc", "UpdateArgAndRetvalMetadata");

  std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes;
  std::vector<std::pair<Node*, int>> ret_nodes;
  const AttrValue* attr_value;

  // Find the Arg and Retval nodes, along with their corresponding indices
  // in the original function.
  for (Node* node : graph->op_nodes()) {
    if (node->IsArg()) {
      TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
      int index = static_cast<int>(attr_value->i());
      int sub_index = -1;
      if (node->attrs().Find("sub_index", &attr_value).ok()) {
        sub_index = static_cast<int>(attr_value->i());
      }
      arg_nodes.emplace_back(node, FunctionArgIndex(index, sub_index));
    } else if (node->IsRetval()) {
      TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
      int index = static_cast<int>(attr_value->i());
      ret_nodes.emplace_back(node, index);
    }
  }

  // Sort the nodes by index so that the order is stable.
  //
  // In particular, this enables calling a single-partition function with
  // the same signature as the original unpartitioned function.
  auto arg_comparator = [](std::pair<Node*, FunctionArgIndex> a,
                           std::pair<Node*, FunctionArgIndex> b) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc mht_6(mht_6_v, 374, "", "./tensorflow/core/common_runtime/partitioning_utils.cc", "lambda");

    return std::tie(a.second.index, a.second.sub_index) <
           std::tie(b.second.index, b.second.sub_index);
  };
  std::sort(arg_nodes.begin(), arg_nodes.end(), arg_comparator);
  auto ret_comparator = [](std::pair<Node*, int> a, std::pair<Node*, int> b) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc mht_7(mht_7_v, 382, "", "./tensorflow/core/common_runtime/partitioning_utils.cc", "lambda");

    return a.second < b.second;
  };
  std::sort(ret_nodes.begin(), ret_nodes.end(), ret_comparator);

  arg_indices->reserve(arg_nodes.size());
  for (const auto& pair : arg_nodes) arg_indices->push_back(pair.second);
  ret_indices->reserve(ret_nodes.size());
  for (const auto& pair : ret_nodes) ret_indices->push_back(pair.second);

  for (int i = 0; i < arg_nodes.size(); ++i) {
    Node* arg = arg_nodes[i].first;
    arg->AddAttr("index", i);
    TF_RETURN_IF_ERROR(arg->attrs().Find("T", &attr_value));
    if (arg_alloc_attrs != nullptr) {
      AllocatorAttributes alloc_attr;
      DataType type = attr_value->type();
      MemoryType mtype = ints_on_device ? MTypeFromDTypeIntsOnDevice(type)
                                        : MTypeFromDType(type);
      if (mtype == HOST_MEMORY) {
        alloc_attr.set_on_host(true);
      }
      arg_alloc_attrs->push_back(alloc_attr);
    }
  }
  for (int i = 0; i < ret_nodes.size(); ++i) {
    Node* ret = ret_nodes[i].first;
    ret->AddAttr("index", i);
    TF_RETURN_IF_ERROR(ret->attrs().Find("T", &attr_value));
    if (ret_alloc_attrs) {
      AllocatorAttributes alloc_attr;
      DataType type = attr_value->type();
      MemoryType mtype = ints_on_device ? MTypeFromDTypeIntsOnDevice(type)
                                        : MTypeFromDType(type);
      if (mtype == HOST_MEMORY) {
        alloc_attr.set_on_host(true);
      }
      ret_alloc_attrs->push_back(alloc_attr);
    }
  }

  return Status::OK();
}

string FunctionNameGenerator::GetName() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTcc mht_8(mht_8_v, 429, "", "./tensorflow/core/common_runtime/partitioning_utils.cc", "FunctionNameGenerator::GetName");

  while (true) {
    const string candidate = strings::StrCat(name_, "_", counter_++);
    if (flib_def_->Find(candidate) == nullptr) {
      return candidate;
    }
  }
}

}  // namespace tensorflow
