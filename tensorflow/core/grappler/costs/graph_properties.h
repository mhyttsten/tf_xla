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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_propertiesDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_propertiesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_propertiesDTh() {
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


#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {

namespace grappler {

// Optional attributes that tell about node output information.
// We use these side information, if provided, for static shape inference
// and VirtualScheduler scheduling.

// Switch op attribute as a vector of int that tells which branch the
// Switch output is taken on every round of execution.
// Used for scheduling ops after Switch correctly (e.g., While loop).
ABSL_CONST_INIT const char kOutputSlots[] = "_output_slot_vector";

// Example:
// Assume a node has two outputs and iterated for three times. Then it has:
// _execution_count = 3
// _output_sizes_vector = [2, 2, 2]
// _output_dtype_vector.size = 6
// _output_shape_vector.size = 6

// If all the iterations have same output shapes, then
// _execution_count = 3
// _same_output_for_iterations = true
// _output_sizes_vector = [2]
// _output_dtype_vector.size = 2
// _output_shape_vector.size = 2

// How many times this node has been executed.
ABSL_CONST_INIT const char kExecutionCount[] = "_execution_count";

// Records the output sizes for each round of execution.
ABSL_CONST_INIT const char kOutputSizes[] = "_output_sizes_vector";

// The node has been scheduled multiple times with outputs that have the same
// shape.
ABSL_CONST_INIT const char kOutputSame[] = "_same_output_for_iterations";

// Outputs DataType vector.
ABSL_CONST_INIT const char kOutputTypes[] = "_output_dtype_vector";

// Outputs TensorShapeProto vector.
ABSL_CONST_INIT const char kOutputShapes[] = "_output_shape_vector";

class SymbolicShapeRefiner;
class TopoQueue;

// Infer OpInfo::TensorProperties for graph nodes inputs/outputs.
//
// Typical use case, is to infer tensor properties from a graph, before doing
// optimization pass. Nodes modified during optimization pass have to be
// invalidated, to prevent further incorrect optimizations based on wrong shape
// and data type properties.
class GraphProperties {
 public:
  // The item must outlive the properties
  explicit GraphProperties(const GrapplerItem& item) : item_(item) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_propertiesDTh mht_0(mht_0_v, 253, "", "./tensorflow/core/grappler/costs/graph_properties.h", "GraphProperties");
}

  // Infer the shapes through abstract interpretation. Feed information can be
  // incorrect so it should be discarded to ensure correctness of the analysis.
  // However, it can help infer shapes in the fanout of fed nodes (even though
  // the correctness of these shapes can't be guaranteed), so in some cases
  // (such as simulation or scheduling) it makes sense of keep these shapes.
  // aggressive_shape_inference option executes nodes on the host to identify
  // output values when possible and does other aggressive strategies.
  // Similar to assuming_valid_feeds, this may cause incorrectness in graph
  // analyses, but is useful for simulation or scheduling.
  // If include_input_tensor_values is true, the values of constant tensors
  // will included in the input properties.
  // If include_output_tensor_values is true, the values of constant tensors
  // will be included in the output properties.
  Status InferStatically(bool assume_valid_feeds,
                         bool aggressive_shape_inference,
                         bool include_input_tensor_values,
                         bool include_output_tensor_values);
  Status InferStatically(bool assume_valid_feeds,
                         bool aggressive_shape_inference,
                         bool include_tensor_values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_propertiesDTh mht_1(mht_1_v, 277, "", "./tensorflow/core/grappler/costs/graph_properties.h", "InferStatically");

    return InferStatically(
        assume_valid_feeds,
        /*aggressive_shape_inference=*/aggressive_shape_inference,
        /*include_input_tensor_values=*/include_tensor_values,
        /*include_output_tensor_values=*/include_tensor_values);
  }
  Status InferStatically(bool assume_valid_feeds) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_propertiesDTh mht_2(mht_2_v, 287, "", "./tensorflow/core/grappler/costs/graph_properties.h", "InferStatically");

    return InferStatically(assume_valid_feeds,
                           /*aggressive_shape_inference=*/false,
                           /*include_tensor_values=*/true);
  }
  // Infer the shape by running the graph on the specified cluster and recording
  // the shapes of the processed tensors.
  Status InferDynamically(Cluster* cluster);
  // Extract the properties from a cost graph. For testing only since there is
  // no way to ensure that the cost graph match the item.
  Status InferFromCostGraph(const CostGraphDef& cost_graph);

  // Stores `item_.graph` with the inferred output shapes to `output_graph_def`.
  Status AnnotateOutputShapes(GraphDef* output_graph_def) const;

  // Return the properties of node inputs/outputs, including data types and
  // shapes. Note that the dimensions in the shapes can be negative. We use the
  // -1 value to denote that we don't know anything about a dimension. We use
  // values strictly less than -1 to encode symbolic dimensions: although we
  // don't know the actual value of the symbolic dimension, we know that all the
  // dimensions denoted by the same negative value are the equal.
  bool HasInputProperties(const string& node_name) const;
  bool HasOutputProperties(const string& node_name) const;
  const std::vector<OpInfo::TensorProperties>& GetInputProperties(
      const string& node_name) const;
  const std::vector<OpInfo::TensorProperties>& GetOutputProperties(
      const string& node_name) const;

  // Invalidate input/output properties for nodes modified during graph
  // optimization pass, to prevent potential optimizations, based on incorrect
  // shape information.
  void ClearInputProperties(const string& node_name);
  void ClearOutputProperties(const string& node_name);
  // Returns true if we have *any* properties.
  bool has_properties() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_propertiesDTh mht_3(mht_3_v, 324, "", "./tensorflow/core/grappler/costs/graph_properties.h", "has_properties");

    return !input_properties_.empty() || !output_properties_.empty();
  }

  bool CheckShapeIncompatible(const string& node_name) const {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_propertiesDTh mht_4(mht_4_v, 332, "", "./tensorflow/core/grappler/costs/graph_properties.h", "CheckShapeIncompatible");

    return incompatible_shape_nodes_.find(node_name) !=
           incompatible_shape_nodes_.end();
  }

  // Clear all infered properties.
  void Clear() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSgraph_propertiesDTh mht_5(mht_5_v, 341, "", "./tensorflow/core/grappler/costs/graph_properties.h", "Clear");

    input_properties_.clear();
    output_properties_.clear();
  }

 private:
  // Relaxes shapes <shapes_and_types>, determined from an EnqueueV2 node, into
  // <*queue_shapes_and_types>.
  static Status RelaxEnqueueShapesAndMergeTypes(
      SymbolicShapeRefiner* shape_refiner, const NodeDef* qnode,
      const std::vector<shape_inference::ShapeAndType>& shapes_and_types,
      std::vector<shape_inference::ShapeAndType>* queue_shapes_and_types);

  // Update the shapes of the enqueue node, port them over to the corresponding
  // queue, and schedule the reprocessing of the queue if needed.
  static Status UpdateEnqueue(
      const NodeDef* enqueue_node,
      const absl::flat_hash_map<const NodeDef*, const NodeDef*>&
          resource_handles,
      SymbolicShapeRefiner* shape_refiner, bool* new_shapes);

  // Update the shapes and types of the Queue node, if not set by Enqueue node.
  static Status UpdateQueue(const NodeDef* queue_node,
                            SymbolicShapeRefiner* shape_refiner,
                            bool* new_shapes);

  // Update the output shapes of a Merge node, and enqueue its fanout in
  // new_shapes if needed.
  Status UpdateMerge(SymbolicShapeRefiner* shape_refiner, const NodeDef* node,
                     bool* new_shapes) const;
  // Process the Enter node, and enqueue its fanout in new_shapes if needed.
  static Status UpdateEnter(SymbolicShapeRefiner* shape_refiner,
                            const NodeDef* node, bool* new_shapes);
  // Update the shapes for node 'n'. If output shapes for n have changed,
  // enqueue its fanout in 'new_shapes'.
  Status UpdateShapes(SymbolicShapeRefiner* shape_refiner,
                      const absl::flat_hash_map<const NodeDef*, const NodeDef*>&
                          resource_handles,
                      const NodeDef* n, bool* new_shapes) const;
  // Propagate the shapes for the nodes enqueued in new_shapes and their
  // transitive fanout until a fixed point is reached.
  Status PropagateShapes(
      SymbolicShapeRefiner* shape_refiner, TopoQueue* new_shapes,
      const absl::flat_hash_map<const NodeDef*, const NodeDef*>&
          resource_handles,
      int num_loops) const;

  // Data members
  const GrapplerItem& item_;
  absl::flat_hash_map<string, std::vector<OpInfo::TensorProperties>>
      input_properties_;
  absl::flat_hash_map<string, std::vector<OpInfo::TensorProperties>>
      output_properties_;
  const std::vector<OpInfo::TensorProperties> missing_properties_;

  // Nodes with output shape incompatible between shape inference and
  // annotation.
  std::unordered_set<string> incompatible_shape_nodes_;
};

// Helper function for GraphProperties.
bool IsShapeFullyDefinedIntegerVectorOrScalar(
    shape_inference::InferenceContext* ic,
    const shape_inference::ShapeHandle& shape,
    const shape_inference::ShapeHandle& tensor_as_shape, const DataType& dtype);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_GRAPH_PROPERTIES_H_
