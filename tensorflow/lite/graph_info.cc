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
class MHTracer_DTPStensorflowPSlitePSgraph_infoDTcc {
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
   MHTracer_DTPStensorflowPSlitePSgraph_infoDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSgraph_infoDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/graph_info.h"

#include <algorithm>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"

namespace tflite {
namespace {

// Helper class that actually performs partitioning by node sub set.
// Outputs to a provided `NodeSubset` structure.
//
// Example usage:
// PartitionGraphIntoIndependentNodeSubsetsImpl partitioner(
//     info, nodes_to_part, node_subsets);
// partitioner.Partition();
//
// NOTE: Changing the partitioning logic would require a change to
// FP16GraphPartitionHelper.
// LINT.IfChange
class PartitionGraphIntoIndependentNodeSubsetsImpl {
 public:
  PartitionGraphIntoIndependentNodeSubsetsImpl(
      const GraphInfo* info, const TfLiteIntArray* nodes_to_partition,
      std::vector<NodeSubset>* node_subsets)
      : info_(info),
        node_subsets_(node_subsets),
        node_type_(info_->num_total_nodes(), NodeSubset::kTfNonPartition) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSgraph_infoDTcc mht_0(mht_0_v, 213, "", "./tensorflow/lite/graph_info.cc", "PartitionGraphIntoIndependentNodeSubsetsImpl");

    // Populate the node_type_ map.
    for (auto node_index : TfLiteIntArrayView(nodes_to_partition)) {
      node_type_[node_index] = NodeSubset::kTfPartition;
    }
  }

  // Actually partition the graph.
  void Partition() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSgraph_infoDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/graph_info.cc", "Partition");

    // Initialize here to make Partition() re-entrant.
    node_subsets_->clear();
    tensor_epochs_.clear();
    tensor_epochs_.resize(info_->num_tensors(), kEpochAlwaysReady);
    node_epochs_.clear();
    node_epochs_.resize(info_->num_execution_nodes(), kEpochNotReady);
    control_deps_.clear();
    control_deps_.resize(info_->num_execution_nodes());
    // Add control dependency between stateful ops.
    // TODO(b/149099381): Revisit better way for adding control dependency.
    int last_op_with_side_effect = -1;
    for (int i = 0; i < info_->num_execution_nodes(); ++i) {
      const auto& node = info_->node(i);
      // Set default value.
      control_deps_[i] = -1;
      if (node.might_have_side_effect) {
        if (last_op_with_side_effect != -1) {
          control_deps_[i] = last_op_with_side_effect;
        }
        last_op_with_side_effect = i;
      }
    }
    // Set computed tensors to be kEpochNotReady (initializer set everything to
    // AlwaysReady).
    for (int node_index = 0; node_index < info_->num_execution_nodes();
         node_index++) {
      const TfLiteNode& node = info_->node(node_index);
      for (int output_tensor_index : TfLiteIntArrayView(node.outputs)) {
        tensor_epochs_[output_tensor_index] = kEpochNotReady;
      }
    }

    // Do a graph traversal where each iteration in the loop is an epoch
    // that corresponds to a node sub set that only contains nodes that are of
    // the same node_type_.
    while (true) {
      BuildNodeSubset();
      if (node_subsets_->back().nodes.empty()) {
        node_subsets_->pop_back();
        break;
      }
    }

    // Mark model outputs as node sub set outputs. All the rest have already
    // been identified.
    for (int output_index : info_->outputs()) {
      int output_epoch = tensor_epochs_[output_index];
      if (output_epoch == kEpochAlwaysReady) {
        // This happens when an input of subgraph is also an output of subgraph.
        continue;
      }
      NodeSubset& output_subset = (*node_subsets_)[output_epoch];
      output_subset.output_tensors.push_back(output_index);
    }
    // Make sure every node sub set's inputs and outputs are unique. Since the
    // list of inputs and outputs is generated in a way that produces
    // duplicates.
    for (NodeSubset& node_subset : *node_subsets_) {
      // Sort and uniquefy using standard library algorithms.
      auto uniquefy = [](std::vector<int>* items) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSgraph_infoDTcc mht_2(mht_2_v, 287, "", "./tensorflow/lite/graph_info.cc", "lambda");

        std::sort(items->begin(), items->end());
        auto last = std::unique(items->begin(), items->end());
        items->erase(last, items->end());
      };
      uniquefy(&node_subset.input_tensors);
      uniquefy(&node_subset.output_tensors);
    }
  }

 private:
  // Special integer values needed for tensor_epochs_ and node_epochs_.
  enum {
    // The node or tensor is not ready to be assigned an epoch. e.g. a node's
    // inputs have not all been assigned epochs.
    kEpochNotReady = -1,
    // Used for tensor_epochs_. This means that the tensor is always ready.
    // e.g. an input to the whole model or a constant that has no dependencies.
    kEpochAlwaysReady = -2
  };

  // Updates the node at `node_index` in the execution plan and returns true if
  // it is assigned to an epoch. False is returned if the node is already set to
  // an epoch, its inputs are not all assigned to epochs, or if it cannot be
  // assigned to the current epoch since the epoch's node_type doesn't match.
  bool UpdateNode(int node_index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSgraph_infoDTcc mht_3(mht_3_v, 315, "", "./tensorflow/lite/graph_info.cc", "UpdateNode");

    const TfLiteNode& node = info_->node(node_index);
    NodeSubset& current_subset = node_subsets_->back();
    int current_epoch = node_subsets_->size() - 1;
    // Check if node is already done.
    if (node_epochs_[node_index] != kEpochNotReady) {
      return false;
    }
    // See if all dependencies of this node are already assigned to a
    // node sub set.
    for (int input_tensor_index : TfLiteIntArrayView(node.inputs)) {
      if (input_tensor_index != kTfLiteOptionalTensor &&
          tensor_epochs_[input_tensor_index] == kEpochNotReady) {
        return false;
      }
    }
    // If any of the nodes that current node depend on is not assigned
    // any epochs then don't process this node.
    if (control_deps_[node_index] != -1 &&
        node_epochs_[control_deps_[node_index]] == kEpochNotReady) {
      return false;
    }

    int original_node_idx = info_->node_index(node_index);
    // When we are starting a new epoch, the first ready node defines
    // the type of that epoch.
    if (current_subset.type == NodeSubset::kTfUnexplored) {
      current_subset.type = node_type_[original_node_idx];
    }
    // The node gets assigned to this epoch if it is the same type as
    // the epoch's assigned type. Note, if this is the current ready
    // node encountered during this epoch, this condition will be
    // automatically true.
    if (current_subset.type == node_type_[original_node_idx]) {
      node_epochs_[node_index] = current_epoch;
      current_subset.nodes.push_back(original_node_idx);
      // All outputs of this node now are assigned to this epoch as
      // well.
      for (int output_tensor_index : TfLiteIntArrayView(node.outputs)) {
        tensor_epochs_[output_tensor_index] = current_epoch;
      }
      // Look at our inputs one more time to update that tensor's
      // epochs' outputs
      for (int input_tensor_index : TfLiteIntArrayView(node.inputs)) {
        if (input_tensor_index == kTfLiteOptionalTensor) {
          continue;
        }
        int input_epoch = tensor_epochs_[input_tensor_index];
        int node_epoch = current_epoch;
        if (input_epoch != node_epoch) {
          current_subset.input_tensors.push_back(input_tensor_index);
          // Set inputs to be outputs of the node sub set where they reside.
          // the if condition makes sure inputs to the whole computation
          // are not included (i.e. those initialized to -2 above).
          if (input_epoch >= 0) {
            NodeSubset& input_subset = (*node_subsets_)[input_epoch];
            input_subset.output_tensors.push_back(input_tensor_index);
          }
        }
      }
      return true;
    } else {
      return false;
    }
  }

  // Completely populates the current node_subset by doing graph traversal
  void BuildNodeSubset() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSgraph_infoDTcc mht_4(mht_4_v, 385, "", "./tensorflow/lite/graph_info.cc", "BuildNodeSubset");

    node_subsets_->emplace_back(NodeSubset());
    // loop until no more nodes can be updated.
    while (true) {
      bool did_something = false;
      for (int node_index = 0; node_index < info_->num_execution_nodes();
           node_index++) {
        if (UpdateNode(node_index)) {
          did_something = true;
        }
      }
      if (!did_something) return;
    }
  }

  // Temporary data needed for partitioning.
  const GraphInfo* info_;
  // List of node_subsets to populate
  std::vector<NodeSubset>* node_subsets_;
  // NOTE: This vector contains a place-holder for *all* nodes in the graph, not
  // just ones in the execution plan. This is because nodes_to_partition is
  // passed in as a list of original node indices & not execution plan indices.
  std::vector<NodeSubset::Type> node_type_;
  // Maps from tensor index to the epoch in which it is assigned. Also special
  // negative values of kEpochNotReady if not assigned, kEpochAlwaysReady if it
  // is an input to the whole model or a constant that has no dependencies.
  std::vector<int> tensor_epochs_;
  // Maps from tensor index to the epoch in which it is assigned. Also special
  // negative values of kEpochNotReady if not assigned.
  std::vector<int> node_epochs_;
  // For each node the node id that this op depends on.
  // TODO(b/149099381): This should be a list, but we are now chaining
  // dependency between previous ops.
  std::vector<int> control_deps_;
};
// LINT.ThenChange(//tensorflow/lite/delegates/utils.h)

}  // namespace

TfLiteStatus PartitionGraphIntoIndependentNodeSubsets(
    const GraphInfo* info, const TfLiteIntArray* nodes_to_partition,
    std::vector<NodeSubset>* node_subsets) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSgraph_infoDTcc mht_5(mht_5_v, 429, "", "./tensorflow/lite/graph_info.cc", "PartitionGraphIntoIndependentNodeSubsets");

  PartitionGraphIntoIndependentNodeSubsetsImpl(info, nodes_to_partition,
                                               node_subsets)
      .Partition();
  return kTfLiteOk;
}

}  // namespace tflite
