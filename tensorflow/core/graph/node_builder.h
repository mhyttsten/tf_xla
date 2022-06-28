/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPH_NODE_BUILDER_H_
#define TENSORFLOW_CORE_GRAPH_NODE_BUILDER_H_
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
class MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTh {
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
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTh() {
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


#include <vector>
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// This is a helper for creating a Node and adding it to a Graph.
// Internally, it uses a NodeDefBuilder to automatically set attrs
// that can be inferred from the inputs, and use default values
// (where they exist) for unspecified attrs.  Example usage:
//
//  Node* node;
//  Status status = NodeBuilder(node_name, op_name)
//                           .Input(...)
//                           .Attr(...)
//                           .Finalize(&graph, &node);
//  if (!status.ok()) return status;
//  // Use node here.
class NodeBuilder {
 public:
  // For specifying the output of a Node to provide to one of the Input()
  // functions below.  It supports both regular inputs (where you are
  // connecting to an existing Node*), and inputs from outside the graph
  // (or haven't been added to the graph yet, like back edges, where
  // you don't have a Node*). Both types can be mixed, e.g. in an
  // ArraySlice.
  struct NodeOut {
    // For referencing an existing Node.
    NodeOut(Node* n, int32_t i = 0);
    NodeOut(OutputTensor t);

    // For referencing Nodes not in the graph being built. It is
    // useful when preparing a graph for ExtendSession or creating a
    // back edge to a node that hasn't been added to the graph yet,
    // but will be.
    NodeOut(StringPiece name, int32_t i, DataType t);

    // Default constructor for std::vector<NodeOut>.
    NodeOut();

    Node* node;
    // error is set to true if:
    // * the NodeOut was default constructed and never overwritten,
    // * a nullptr Node* was passed to the NodeOut constructor, or
    // * an out-of-range index was passed to the NodeOut constructor.
    bool error;
    string name;
    int32 index;
    DataType dt;
  };

  // Specify the name and the Op (either via an OpDef or the name of
  // the Op plus a registry) for the Node.  Other fields are
  // specified by calling the methods below.
  // REQUIRES: The OpDef must satisfy ValidateOpDef().
  NodeBuilder(StringPiece name, StringPiece op_name,
              const OpRegistryInterface* op_registry = OpRegistry::Global(),
              const NodeDebugInfo* debug = nullptr);
  NodeBuilder(StringPiece name, const OpDef* op_def);

  // Create a NodeBuilder from an existing NodeDefBuilder.
  NodeBuilder(const NodeDefBuilder& def_builder);

  // You must call one Input() function per input_arg in the Op,
  // *and in the same order as the input_args appear in the OpDef.*

  // For inputs that take a single tensor.
  NodeBuilder& Input(Node* src_node, int src_index = 0);
  NodeBuilder& Input(NodeOut src);

  // For inputs that take a list of tensors.
  NodeBuilder& Input(gtl::ArraySlice<NodeOut> src_list);

  // Require that this node run after src_node(s).
  NodeBuilder& ControlInput(Node* src_node);
  NodeBuilder& ControlInputs(gtl::ArraySlice<Node*> src_nodes);

  // Sets the "requested device spec" in the NodeDef (not the
  // "assigned device" in the Node).
  NodeBuilder& Device(StringPiece device_spec);

  // Sets the device name in the "assigned device" field in tensorflow::Node.
  NodeBuilder& AssignedDevice(StringPiece device);

  // Sets the _XlaCluster attribute in created node to `xla_cluster`.
  NodeBuilder& XlaCluster(StringPiece xla_cluster);

  // Set the value of an attr.  attr_name must match the name of one of
  // attrs defined by the Op, and value must have the corresponding type
  // (see SetAttrValue() in ../framework/attr_value_util.h for legal
  // types for value).  Note that attrs will be set automatically if
  // they can be determined by the inputs.
  template <class T>
  NodeBuilder& Attr(StringPiece attr_name, T&& value);
  template <class T>
  NodeBuilder& Attr(StringPiece attr_name, std::initializer_list<T> value);

  // Validates the described node and adds it to *graph, adding edges
  // for all (non-back) inputs.  If created_node is not nullptr,
  // *created_node will be set to the new node (or nullptr on error).
  // If `consume` is true, the builder state will be moved into `node_def`,
  // and the builder will be left in an undefined state.
  Status Finalize(Graph* graph, Node** created_node, bool consume = false);

  // Same as `Finalize` above, but using StatusOr to return value. Preferred
  // form.
  StatusOr<Node*> Finalize(Graph* graph, bool consume = false);

  // Accessors for the values set in the constructor.
  const string& node_name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTh mht_0(mht_0_v, 302, "", "./tensorflow/core/graph/node_builder.h", "node_name");
 return def_builder_.node_name(); }
  const OpDef& op_def() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTh mht_1(mht_1_v, 306, "", "./tensorflow/core/graph/node_builder.h", "op_def");
 return def_builder_.op_def(); }

 private:
  static DataType SafeGetOutput(const Node* node, int i, bool* error) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTh mht_2(mht_2_v, 312, "", "./tensorflow/core/graph/node_builder.h", "SafeGetOutput");

    if (node != nullptr && i >= 0 && i < node->num_outputs()) {
      *error = false;
      return node->output_type(i);
    } else {
      *error = true;
      return DT_FLOAT;
    }
  }

  // If SafeGetOutput indicates a range error, add it to errors_.
  void AddIndexError(const Node* node, int i);

  // Set *dt and returns true if i is in range. Combines
  // SafeGetOutput() and AddIndexError().
  bool GetOutputType(const Node* node, int i, DataType* dt);

  NodeDefBuilder def_builder_;
  const OpRegistryInterface* op_registry_;
  std::vector<NodeOut> inputs_;
  std::vector<Node*> control_inputs_;
  std::vector<string> errors_;
  string assigned_device_;
};

// IMPLEMENTATION -------------------------------------------------------------

template <class T>
NodeBuilder& NodeBuilder::Attr(StringPiece attr_name, T&& value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTh mht_3(mht_3_v, 343, "", "./tensorflow/core/graph/node_builder.h", "NodeBuilder::Attr");

  def_builder_.Attr(attr_name, std::forward<T>(value));
  return *this;
}

template <class T>
NodeBuilder& NodeBuilder::Attr(StringPiece attr_name,
                               std::initializer_list<T> value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTh mht_4(mht_4_v, 353, "", "./tensorflow/core/graph/node_builder.h", "NodeBuilder::Attr");

  def_builder_.Attr(attr_name, value);
  return *this;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_NODE_BUILDER_H_
