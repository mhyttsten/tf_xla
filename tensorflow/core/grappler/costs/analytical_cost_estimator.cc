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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimatorDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimatorDTcc() {
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

#include "tensorflow/core/grappler/costs/analytical_cost_estimator.h"

#include <limits>
#include <unordered_map>

#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/costs/virtual_scheduler.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/util/overflow.h"

namespace tensorflow {
namespace grappler {

namespace {

// Helper function in PredictCosts() to add cost node to cost_graph.
Status AddCostNode(ReadyNodeManager* node_manager, const OpContext& op_context,
                   int node_id, const Costs& node_costs,
                   gtl::FlatMap<string, CostGraphDef::Node*>* name_to_cost_node,
                   gtl::FlatMap<string, int>* name_to_id,
                   CostGraphDef* cost_graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimatorDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/grappler/costs/analytical_cost_estimator.cc", "AddCostNode");

  const string& op_name = op_context.name;
  auto it = name_to_cost_node->find(op_name);
  CostGraphDef::Node* node;
  if (it != name_to_cost_node->end()) {
    node = it->second;
    node->clear_input_info();
    node->clear_output_info();
  } else {
    node = cost_graph->add_node();
    (*name_to_cost_node)[op_name] = node;
    node->set_name(op_name);
    node->set_id(node_id);
    (*name_to_id)[node->name()] = node->id();
  }
  // For nodes we have seen before (e.g. Merge nodes are executed twice by
  // VirtualScheduler), the following fields will be overwritten/updated.
  node->set_device(op_context.device_name);
  node->set_compute_cost(node_costs.execution_time.asMicroSeconds().count());
  node->set_compute_time(node_costs.compute_time.asMicroSeconds().count());
  node->set_memory_time(node_costs.memory_time.asMicroSeconds().count());
  node->set_temporary_memory_size(node_costs.temporary_memory);
  node->set_persistent_memory_size(node_costs.persistent_memory);
  node->set_inaccurate(node_costs.inaccurate);

  for (const string& input : node_manager->GetCurrNode()->input()) {
    int input_port;
    string input_name = ParseNodeName(input, &input_port);

    // All inputs should have been seen already unless this is a Merge node.
    if (name_to_id->find(input_name) == name_to_id->end()) {
      if (!IsMerge(*node_manager->GetCurrNode()))
        VLOG(1) << "input: " << input
                << " not found for non-Merge node: " << op_name;

      // For Merge node, some of inputs may not be seen before
      // For example, for a typical while loop in tensorflow, Merge node
      // will be executed twice by VirtualScheduler (one for Enter, the
      // other for NextIteration), so eventually both inputs will be added.
      continue;
    }

    if (IsControlInput(input)) {
      node->add_control_input(name_to_id->at(input_name));
    } else {
      auto* input_info = node->add_input_info();
      input_info->set_preceding_node(name_to_id->at(input_name));
      input_info->set_preceding_port(input_port);
    }
  }

  for (const auto& output : op_context.op_info.outputs()) {
    auto output_info = node->add_output_info();
    output_info->set_alias_input_port(-1);
    output_info->set_dtype(output.dtype());
    *output_info->mutable_shape() = output.shape();

    int64_t size = DataTypeSize(output.dtype());
    for (const auto& dim : output.shape().dim()) {
      size = MultiplyWithoutOverflow(size, std::max<int64_t>(1, dim.size()));
      if (size < 0) {
        return errors::InvalidArgument(
            "Integer overflow encountered in dimension size.");
      }
    }
    output_info->set_size(size);
  }
  return Status::OK();
}

}  // namespace

AnalyticalCostEstimator::AnalyticalCostEstimator(
    Cluster* cluster, bool use_static_shapes,
    bool use_aggressive_shape_inference)
    : AnalyticalCostEstimator(
          cluster, absl::make_unique<OpLevelCostEstimator>(),
          ReadyNodeManagerFactory("FirstReady"), use_static_shapes,
          use_aggressive_shape_inference) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimatorDTcc mht_1(mht_1_v, 294, "", "./tensorflow/core/grappler/costs/analytical_cost_estimator.cc", "AnalyticalCostEstimator::AnalyticalCostEstimator");
}

AnalyticalCostEstimator::AnalyticalCostEstimator(
    Cluster* cluster, std::unique_ptr<OpLevelCostEstimator> node_estimator,
    std::unique_ptr<ReadyNodeManager> node_manager, bool use_static_shapes,
    bool use_aggressive_shape_inference)
    : node_estimator_(std::move(node_estimator)),
      node_manager_(std::move(node_manager)),
      use_static_shapes_(use_static_shapes),
      use_aggressive_shape_inference_(use_aggressive_shape_inference) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimatorDTcc mht_2(mht_2_v, 306, "", "./tensorflow/core/grappler/costs/analytical_cost_estimator.cc", "AnalyticalCostEstimator::AnalyticalCostEstimator");

  scheduler_ = absl::make_unique<VirtualScheduler>(
      use_static_shapes_, use_aggressive_shape_inference_, cluster,
      node_manager_.get(),
      absl::make_unique<VirtualPlacer>(cluster->GetDevices()));
}

AnalyticalCostEstimator::AnalyticalCostEstimator(
    Cluster* cluster, std::unique_ptr<OpLevelCostEstimator> node_estimator,
    std::unique_ptr<ReadyNodeManager> node_manager,
    std::unique_ptr<VirtualPlacer> placer, bool use_static_shapes,
    bool use_aggressive_shape_inference)
    : node_estimator_(std::move(node_estimator)),
      node_manager_(std::move(node_manager)),
      use_static_shapes_(use_static_shapes),
      use_aggressive_shape_inference_(use_aggressive_shape_inference) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimatorDTcc mht_3(mht_3_v, 324, "", "./tensorflow/core/grappler/costs/analytical_cost_estimator.cc", "AnalyticalCostEstimator::AnalyticalCostEstimator");

  scheduler_ = absl::make_unique<VirtualScheduler>(
      use_static_shapes_, use_aggressive_shape_inference_, cluster,
      node_manager_.get(), std::move(placer));
}

Status AnalyticalCostEstimator::Initialize(const GrapplerItem& item) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimatorDTcc mht_4(mht_4_v, 333, "", "./tensorflow/core/grappler/costs/analytical_cost_estimator.cc", "AnalyticalCostEstimator::Initialize");

  item_ = &item;
  return Status::OK();
}

Status AnalyticalCostEstimator::PredictCosts(const GraphDef& optimized_graph,
                                             RunMetadata* run_metadata,
                                             Costs* costs) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSanalytical_cost_estimatorDTcc mht_5(mht_5_v, 343, "", "./tensorflow/core/grappler/costs/analytical_cost_estimator.cc", "AnalyticalCostEstimator::PredictCosts");

  std::unique_ptr<GrapplerItem> item_storage;
  const GrapplerItem* item;
  // Many callers to PredictCosts() pass the same optimized_graph as was used
  // to initialize the estimator.
  if (&optimized_graph == &item_->graph) {
    item = item_;
  } else {
    GraphDef graph_copy = optimized_graph;
    item_storage = absl::make_unique<GrapplerItem>(
        item_->WithGraph(std::move(graph_copy)));
    item = item_storage.get();
  }

  auto status = scheduler_->Init(item);
  if (!status.ok()) {
    if (costs) {
      costs->execution_time = Costs::Duration::max();
    }
    return status;
  }

  gtl::FlatMap<string, CostGraphDef::Node*> name_to_cost_node;
  CostGraphDef* cost_graph = nullptr;
  if (run_metadata) {
    cost_graph = run_metadata->mutable_cost_graph();
    // TODO(pcma): Clear nodes in cost_graph after we make sure we always pass
    // in an empty cost_graph (a non-empty but incomplete cost_graph will cause
    // problems, e.g., no node_id in cost_graph).
    for (auto& node : *cost_graph->mutable_node()) {
      name_to_cost_node[node.name()] = &node;
    }
  }
  std::vector<string> inaccurate_nodes;
  int nodes_executed = 0;
  int node_id = 0;
  gtl::FlatMap<string, int> name_to_id;

  Costs node_costs;
  do {
    ++nodes_executed;
    OpContext op_context = scheduler_->GetCurrNode();
    node_costs = node_estimator_->PredictCosts(op_context);

    if (node_costs.inaccurate) {
      inaccurate_nodes.push_back(op_context.name);
      if (node_costs.num_ops_with_unknown_shapes > 0)
        VLOG(4) << op_context.name << " has "
                << node_costs.num_ops_with_unknown_shapes << " unknown shapes";
    }

    // TODO(pcma): Add unit tests for generating CostGraphDef.
    if (cost_graph) {
      Status s =
          AddCostNode(node_manager_.get(), op_context, node_id++, node_costs,
                      &name_to_cost_node, &name_to_id, cost_graph);
      if (!s.ok()) {
        return s;
      }
    }
  } while (scheduler_->MarkCurrNodeExecuted(node_costs));

  VLOG(1) << inaccurate_nodes.size() << " out of " << nodes_executed
          << " nodes have inaccurate time estimation";
  if (VLOG_IS_ON(3)) {
    for (const auto& node : inaccurate_nodes) {
      VLOG(4) << "Node with inaccurate time estimation: " << node;
    }
  }

  // run_metadata gets step_stats and partition_graphs from Summary.
  if (costs) {
    *costs = scheduler_->Summary(run_metadata);
  } else if (run_metadata) {
    scheduler_->GenerateRunMetadata(run_metadata);
  }

  if (VLOG_IS_ON(1)) {
    bool verbose = VLOG_IS_ON(2);
    if (run_metadata) {
      VLOG(1) << GetStatsStringFromRunMetadata(*run_metadata, verbose);
    } else {
      RunMetadata run_metadata;
      scheduler_->GenerateRunMetadata(&run_metadata);
      VLOG(1) << GetStatsStringFromRunMetadata(run_metadata, verbose);
    }
  }

  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
