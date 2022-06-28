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

#ifndef TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_UTILS_H_
#define TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_UTILS_H_
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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTh {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTh() {
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


#include <set>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace graph_transforms {

// Used to quickly look up nodes in the graph def from a name.
void MapNamesToNodes(const GraphDef& graph_def,
                     std::map<string, const NodeDef*>* result);

// For every node in the graph create a list of the nodes that use it as an
// input.
void MapNodesToOutputs(const GraphDef& graph_def,
                       std::map<string, std::vector<const NodeDef*>>* result);

// NodeDef input strings can contain other information besides the name of an
// input node. These include:
//  - Optional '^' prefix, indicating this is a control edge.
//  - The required name of the input node.
//  - Optional ':<number>' suffix, showing which output of the node to use.
// This function takes a raw string, and breaks it into those component parts.
// The rules for inputs in function libraries are a bit more complex, and
// aren't handled by this routine.
void NodeNamePartsFromInput(const string& input_name, string* prefix,
                            string* node_name, string* suffix);

// Adds a ':0' port to any inputs with no suffix, to make comparisons easier.
string CanonicalInputName(const string& input_name);

// Convenience function to strip the optional prefix and suffix components from
// a string pulled from a NodeDef input, and return the plain node name.
string NodeNameFromInput(const string& input_name);

// Returns a stable hash for the contents of the NodeDef, so that equivalent
// nodes should have equal hashes.
uint64 HashNodeDef(const NodeDef& node);

// Adds the given node name to the end of the node's inputs.
void AddNodeInput(const string& input_name, NodeDef* node);

// Copies an attribute from one NodeDef to another.
void CopyNodeAttr(const NodeDef& source, const string& source_key,
                  const string& dest_key, NodeDef* dest);

// Inserts a value into a NodeDef's map of attributes.
// This is a bit different than AddNodeAttr in node_def_util.h because it
// overwrites any existing attributes with the same key.
template <class T>
inline void SetNodeAttr(const string& key, const T& value, NodeDef* node) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTh mht_0(mht_0_v, 246, "", "./tensorflow/tools/graph_transforms/transform_utils.h", "SetNodeAttr");

  AttrValue attr_value;
  SetAttrValue(value, &attr_value);
  auto* attr_map = node->mutable_attr();
  (*attr_map)[key] = attr_value;
}

template <class T>
inline void SetNodeTensorAttr(const string& key, const Tensor& tensor,
                              NodeDef* node) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTh mht_1(mht_1_v, 259, "", "./tensorflow/tools/graph_transforms/transform_utils.h", "SetNodeTensorAttr");

  TensorProto tensor_proto;
  tensor.AsProtoTensorContent(&tensor_proto);
  SetNodeAttr(key, tensor_proto, node);
}

// Inserts a Tensor into the specified attribute of a NodeDef.
template <class T>
inline void SetNodeTensorAttr(const string& key, const TensorShape& shape,
                              const std::vector<T>& values, NodeDef* node) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTh mht_2(mht_2_v, 272, "", "./tensorflow/tools/graph_transforms/transform_utils.h", "SetNodeTensorAttr");

  const DataType dtype = DataTypeToEnum<T>::v();
  CHECK_EQ(shape.num_elements(), values.size());
  Tensor tensor(dtype, shape);
  T* dest_data = tensor.flat<T>().data();
  std::copy_n(values.data(), values.size(), dest_data);
  SetNodeTensorAttr<T>(key, tensor, node);
}

// Retrieves a tensor value from a NodeDef attribute.
Tensor GetNodeTensorAttr(const NodeDef& node, const string& key);

// Creates a copy of the input GraphDef, but only containing the nodes where the
// supplied selector function returned true.
void FilterGraphDef(const GraphDef& input_graph_def,
                    std::function<bool(const NodeDef&)> selector,
                    GraphDef* output_graph_def);

// Creates a copy of the input graph, with all occurrences of the attributes
// with the names in the argument removed from the node defs.
void RemoveAttributes(const GraphDef& input_graph_def,
                      const std::vector<string>& attributes,
                      GraphDef* output_graph_def);

// For a lot of replacement and matching operations it's useful to have the
// nodes processed in a controlled order, so this does a topological sort to
// ensure that nodes always appear in the GraphDef.node list after their inputs.
Status SortByExecutionOrder(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def);

// Finds inputs that refer to nodes that are not in the graph.
void FindInvalidInputs(const GraphDef& graph_def,
                       std::vector<std::pair<string, string>>* invalid_inputs);

// Returns a descriptive error status if there are problems spotted with the
// graph.
Status IsGraphValid(const GraphDef& graph_def);

// Returns input and output types for a particular NodeDef.
Status GetInOutTypes(const NodeDef& node_def, DataTypeVector* inputs,
                     DataTypeVector* outputs);

// Takes a comma-separated string of numbers and parses them into a shape.
Status TensorShapeFromString(const string& shape_string, TensorShape* result);

// This is used to spot particular subgraphs in a larger model. To use it,
// create a pattern like:
// OpTypePattern pattern({"Conv2D", {{"ResizeBilinear", {{"MirrorPad"}}}}});
// This defines a subgraph where a Conv2D has a ResizeBilinear input, which
// pulls from a MirrorPad op.
// Regular expressions aren't supported for the op names, but you can use "*" to
// match any op. You can also use | as a separator to match multiple op names,
// like "Reshape|Concat|Conv2D".
struct OpTypePattern {
  string op;
  std::vector<OpTypePattern> inputs;
  string DebugString() const;
};

// Returns a sub-graph of nodes that match a pattern.
struct NodeMatch {
  NodeMatch() : node() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTh mht_3(mht_3_v, 336, "", "./tensorflow/tools/graph_transforms/transform_utils.h", "NodeMatch");
}
  NodeDef node;
  std::vector<NodeMatch> inputs;
  string DebugString() const;
};

// Utility class to spot subgraphs matching particular patterns.
class GraphMatcher {
 public:
  GraphMatcher(const GraphDef& graph_def);

  // Sorts the input nodes into execution order, and then skips any previously
  // matches so that no node appears in more than one match. The NodeDef
  // pointers contained in the results are owned by the GraphMatcher object, and
  // so will be invalid after its lifetime.
  Status GetOpTypeMatches(const OpTypePattern& pattern,
                          std::vector<NodeMatch>* matches);

 private:
  bool DoesOpTypeMatch(const NodeDef& node, const OpTypePattern& pattern,
                       const std::set<string>& previously_matched_nodes,
                       NodeMatch* match);

  GraphDef graph_def_;
  std::map<string, const NodeDef*> node_map_;
};

struct ReplaceMatchingOpTypesOptions {
  // Whether to raise an error if the graph is left with dangling inputs. If you
  // enable this option, you must fix inconsistencies in a later pass.
  bool allow_inconsistencies;
};

// Replaces all of the matching sub-graphs with new ops. This calls into the
// given function, and expects to receive a set of new nodes to replace each
// matched sub-graph. It has some logic to protect the integrity of the
// resulting graph, for example making sure that nodes needed by other nodes
// outside the sub-graph aren't removed. These are passed in as the set of
// outputs, and nodes with the same names must be added to the new nodes
// produced by the replacement function. Many of these checks can be disabled
// by setting allow_inconsistencies to true in the options, but then it's the
// caller's responsibility to patch up any problems before passing on the graph
// to others. There's more comprehensive usage documentation in the README.
Status ReplaceMatchingOpTypes(
    const GraphDef& input_graph_def, const OpTypePattern& pattern,
    const std::function<Status(const NodeMatch&, const std::set<string>&,
                               const std::set<string>&, std::vector<NodeDef>*)>&
        node_generator,
    const ReplaceMatchingOpTypesOptions& options, GraphDef* output_graph_def);

// Returns a list of the unique nodes found in this match.
void MatchedNodesAsArray(const NodeMatch& match, std::vector<NodeDef>* result);

// Changes all input references to a particular node name. Any nodes with names
// listed in nodes_to_ignore will not have their inputs rewritten.
Status RenameNodeInputs(const GraphDef& input_graph_def,
                        const std::map<string, string>& inputs_to_rename,
                        const std::unordered_set<string>& nodes_to_ignore,
                        GraphDef* output_graph_def);

// Utility function that copies all the nodes found in a match into the
// new_nodes list. This is useful in replacement functions when you decide to
// leave the original matched subgraph untouched and make no changes.
void CopyOriginalMatch(const NodeMatch& match, std::vector<NodeDef>* new_nodes);

// Holds information that's needed for transform functions.
typedef std::map<string, std::vector<string>> TransformFuncParameters;
struct TransformFuncContext {
  std::vector<string> input_names;
  std::vector<string> output_names;
  TransformFuncParameters params;

  // Returns how many occurrences of the given parameter are present.
  int CountParameters(const string& name) const;

  // Gets a single instance of a parameter, using a default if it's not present.
  Status GetOneStringParameter(const string& name, const string& default_value,
                               string* result) const;

  // Gets a single occurrence of a parameter as a 32-bit integer, falling back
  // to a default if it isn't present and returning an error if it isn't
  // convertible to a number.
  Status GetOneInt32Parameter(const string& name, int32_t default_value,
                              int32* result) const;

  // Gets a single occurrence of a parameter as a 64-bit integer, falling back
  // to a default if it isn't present and returning an error if it isn't
  // convertible to a number.
  Status GetOneInt64Parameter(const string& name, int64_t default_value,
                              int64_t* result) const;

  // Gets a single occurrence of a parameter as a floating point number, falling
  // back to a default if it isn't present and returning an error if it isn't
  // convertible to a number.
  Status GetOneFloatParameter(const string& name, float default_value,
                              float* result) const;

  // Gets a single occurrence of a parameter as a boolean, falling back to a
  // default if it isn't present and returning an error if it's not one of
  // "true", "1", "false", or "0".
  Status GetOneBoolParameter(const string& name, bool default_value,
                             bool* result) const;
};

// This is the function API for all graph transformations, taking an input
// GraphDef and other arguments, and returning a transformed GraphDef.
typedef std::function<Status(const GraphDef&,
                             const TransformFuncContext& context, GraphDef*)>
    TransformFunc;

// To add a new graph transform function, call the macro:
// REGISTER_GRAPH_TRANSFORM("fold_constants", FoldConstants);
// Under the hood this adds the function to the list of known transforms, so you
// just need to link in the .cc file with your registration call to have access
// to it through the command line tool.
// The rest of the machinery below is to enable that automagical registration.
typedef std::map<string, TransformFunc> TransformRegistry;
TransformRegistry* GetTransformRegistry();
class TransformRegistrar {
 public:
  TransformRegistrar(const string& name, TransformFunc transform_func) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTh mht_4(mht_4_v, 460, "", "./tensorflow/tools/graph_transforms/transform_utils.h", "TransformRegistrar");

    TransformRegistry* transform_registry = GetTransformRegistry();
    (*transform_registry)[name] = transform_func;
  }
};
#define REGISTER_GRAPH_TRANSFORM(name, func) \
  REGISTER_GRAPH_TRANSFORM_UNIQ_HELPER(__COUNTER__, name, func)
#define REGISTER_GRAPH_TRANSFORM_UNIQ_HELPER(ctr, name, func) \
  REGISTER_GRAPH_TRANSFORM_UNIQ(ctr, name, func)
#define REGISTER_GRAPH_TRANSFORM_UNIQ(ctr, name, func)    \
  static tensorflow::graph_transforms::TransformRegistrar \
      registrar__body__##ctr##__object(name, func);

}  // namespace graph_transforms
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_UTILS_H_
