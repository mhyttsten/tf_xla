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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTh() {
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


#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {

// Returns the index of the first element in collection that fulfills predicate.
// If no such element exists, returns -1.
template <typename Predicate, typename Collection>
int GetFirstElementIndexWithPredicate(const Predicate& predicate,
                                      const Collection& collection) {
  unsigned idx = 0;
  for (auto&& element : collection) {
    if (predicate(element)) {
      return idx;
    }
    idx++;
  }
  return -1;
}

// Adds a node to the graph.
NodeDef* AddNode(StringPiece name, StringPiece op,
                 const std::vector<string>& inputs,
                 const std::vector<std::pair<string, AttrValue>>& attributes,
                 MutableGraphView* graph);

// Adds Placeholder node for given type.
NodeDef* AddScalarPlaceholder(DataType dtype, MutableGraphView* graph);

// Adds a Const node with the given value to the graph.
template <typename T>
NodeDef* AddScalarConstNode(T v, MutableGraphView* graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTh mht_0(mht_0_v, 231, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.h", "AddScalarConstNode");

  // is_same is an idiomatic hack for making it compile if not instantiated.
  // Replacing with false will result in a compile-time error.
  static_assert(!std::is_same<T, T>::value,
                "Invalid specialization of this method for type T.");
  return {};
}

template <>
NodeDef* AddScalarConstNode(bool v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(double v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(float v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(int v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(int64_t v, MutableGraphView* graph);
template <>
NodeDef* AddScalarConstNode(StringPiece v, MutableGraphView* graph);

// Retrieves the value of a const node. Returns an error
// if the node is not const, or its value is of a different type.
template <typename T>
Status GetScalarConstNodeValue(const NodeDef& node, T* value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_utilsDTh mht_1(mht_1_v, 258, "", "./tensorflow/core/grappler/optimizers/data/graph_utils.h", "GetScalarConstNodeValue");

  // is_same is an idiomatic hack for making it compile if not instantiated.
  // Replacing with false will result in a compile-time error.
  static_assert(!std::is_same<T, T>::value,
                "Invalid specialization of this method fo rtype T.");
}

template <>
Status GetScalarConstNodeValue(const NodeDef& node, int64_t* value);
template <>
Status GetScalarConstNodeValue(const NodeDef& node, bool* value);

// Checks whether the two graphs are the same.
bool Compare(const GraphDef& g1, const GraphDef& g2);

// Checks whether the graph contains a node with the given name.
bool ContainsGraphNodeWithName(StringPiece name, const GraphDef& graph);

// Checks whether the library contains a function with the given name.
bool ContainsGraphFunctionWithName(StringPiece name,
                                   const FunctionDefLibrary& library);

// Checks whether the graph contains a node with the given op.
bool ContainsNodeWithOp(StringPiece op, const GraphDef& graph);

// Returns the index of the node with the given name or -1 if the node does
// not exist.
int FindGraphNodeWithName(StringPiece name, const GraphDef& graph);

// Returns the index of the function with the given name or -1 if the function
// does not exist.
int FindGraphFunctionWithName(StringPiece name,
                              const FunctionDefLibrary& library);

// Returns the index of the first node with the given op or -1 if no such  node
// exists.
int FindGraphNodeWithOp(StringPiece op, const GraphDef& graph);

// Gets the 0th input to a node in the graph.
NodeDef* GetInputNode(const NodeDef& node, const MutableGraphView& graph);

// Gets the ith input to a node in the graph.
NodeDef* GetInputNode(const NodeDef& node, const MutableGraphView& graph,
                      int64_t i);

// Gets the attr corresponding to a dataset node's output types, if it exists.
Status GetDatasetOutputTypesAttr(const NodeDef& node,
                                 DataTypeVector* output_types);

// Returns the list of indices of all nodes with the given op or empty list if
// no such node exists.
std::vector<int> FindAllGraphNodesWithOp(const string& op,
                                         const GraphDef& graph);

// Sets the node name using `prefix` as a prefix while guaranteeing the name
// is unique across the graph.
void SetUniqueGraphNodeName(StringPiece prefix, GraphDef* graph, NodeDef* node);

// Sets the function name using the `prefix` name as a prefix while guaranteeing
// the name is unique across the function library.
void SetUniqueGraphFunctionName(StringPiece prefix,
                                const FunctionDefLibrary* library,
                                FunctionDef* function);

// Copies attribute having name `attribute_name` from node `from` to node
// `to_node`.
void CopyAttribute(const string& attribute_name, const NodeDef& from,
                   NodeDef* to_node);

// Concatenates list attribute having name `attribute_name` from `first` and
// `second` node, setting it to `to_node`.
void ConcatAttributeList(const string& attribute_name, const NodeDef& first,
                         const NodeDef& second, NodeDef* to_node);

// Checks that all nodes in the graphs have unique names, and sets their names
// to be unique if they are not already.  This is necessary as Graph does not
// have the provisions to deduplicate names, and name deduplication elsewhere
// in tensorflow happens in other layers (for example, in the Scope class of the
// C++ API). Note that the nodes in the graph are identified by their id,
// and renaming nodes does not mutate any edges.
Status EnsureNodeNamesUnique(Graph* g);

// Returns the item's fetch node, if there is exactly one. Otherwise, returns an
// error.
Status GetFetchNode(const MutableGraphView& graph, const GrapplerItem& item,
                    NodeDef** fetch_node);

// Returns true if `item` is derived from a `FunctionDef`, false otherwise.
// Currently, we determine this heuristically: If we don't have any fetch nodes
// or all fetch nodes are `Retval` ops, then we consider this item as derived
// from a `FunctionDef`.
bool IsItemDerivedFromFunctionDef(const GrapplerItem& item,
                                  const MutableGraphView& graph_view);

// If both input nodes have the "metadata" attribute set, it populates the
// "metadata" attribute for the fused node.
void MaybeSetFusedMetadata(const NodeDef& node1, const NodeDef& node2,
                           NodeDef* fused_node);

// Copies the attributes `output_shapes`, `output_types` from node `from` to
// node `to_node` if they exist. The method will return `true` if attributes
// copied successfully, otherwise it will return `false`.
//
// Some tf.data transformations set `Toutput_types` instead of `output_types`
// when the attribute describes type of tensor inputs (e.g. TensorDataset,
// TensorSliceDataset, and PaddedBatchDataset). In this case the method copies
// the attribute `Toutput_types` of node `from` to the attribute `output_types`
// of node `to_node`.
bool CopyShapesAndTypesAttrs(const NodeDef& from, NodeDef* to_node);

// Checks whether the op has a "sloppy" attribute.
bool HasSloppyAttr(const string& op);

// Checks whether the op has a "deterministic" attribute.
bool HasDeterministicAttr(const string& op);

// Sets the `name` as the metadata name of the `node`. It returns an error if
// the `node` already has a metadata name.
Status SetMetadataName(const std::string& name, NodeDef* node);

}  // namespace graph_utils
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_UTILS_H_
