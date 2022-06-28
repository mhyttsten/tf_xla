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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc() {
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

#include "tensorflow/core/common_runtime/graph_constructor.h"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {

// We remove duplicate control inputs before adding edges to the Graph, so we
// can skip expensive duplicates check in 'AddControlEdge'.
static constexpr const bool kDoNotCheckDuplicates = true;

inline bool IsMerge(const NodeDef& node_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_0(mht_0_v, 230, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "IsMerge");

  return node_def.op() == "Merge" || node_def.op() == "RefMerge" ||
         node_def.op() == "_XlaMerge";
}

inline bool IsNextIteration(const NodeDef& node_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "IsNextIteration");

  return node_def.op() == "NextIteration" ||
         node_def.op() == "RefNextIteration";
}

bool IsValidNodeName(StringPiece s, bool allow_internal_ops) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "IsValidNodeName");

  using ::tensorflow::strings::Scanner;
  Scanner scanner(s);
  scanner
      .One(allow_internal_ops ? Scanner::LETTER_DIGIT_DOT_UNDERSCORE
                              : Scanner::LETTER_DIGIT_DOT)
      .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);

  while (true) {
    if (!scanner.GetResult())  // Some error in previous iteration.
      return false;
    if (scanner.empty())  // No error, but nothing left, good.
      return true;

    // Absorb another piece, starting with a '>'
    scanner.One(Scanner::RANGLE)
        .One(Scanner::LETTER_DIGIT_DOT)
        .Any(Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE);
  }
}

class GraphConstructor {
 public:
  struct Options {
    Options(const GraphConstructorOptions& in)  // NOLINT(runtime/explicit)
        : allow_internal_ops(in.allow_internal_ops),
          expect_device_spec(in.expect_device_spec),
          importing(false),
          validate_nodes(in.validate_nodes),
          validate_colocation_constraints(false),
          add_default_attributes(in.add_default_attributes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_3(mht_3_v, 279, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "Options");
}
    Options(const ImportGraphDefOptions& in)  // NOLINT(runtime/explicit)
        : allow_internal_ops(false),
          expect_device_spec(false),
          prefix(in.prefix.empty() || str_util::EndsWith(in.prefix, "/")
                     ? in.prefix
                     : in.prefix + "/"),
          uniquify_names(in.uniquify_names),
          uniquify_prefix(in.uniquify_prefix),
          input_map(in.input_map.begin(), in.input_map.end()),
          skip_mapped_nodes(in.skip_mapped_nodes),
          control_dependencies(in.control_dependencies),
          return_tensors(in.return_tensors.begin(), in.return_tensors.end()),
          return_nodes(in.return_nodes),
          importing(true),
          validate_nodes(true),
          validate_colocation_constraints(in.validate_colocation_constraints),
          validate_shape(in.validate_shape),
          default_device(in.default_device) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_4(mht_4_v, 300, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "Options");
}

    bool allow_internal_ops;
    bool expect_device_spec;

    string prefix;
    bool uniquify_names;
    bool uniquify_prefix;
    std::map<TensorId, TensorId> input_map;
    bool skip_mapped_nodes;
    std::vector<string> control_dependencies;
    std::vector<TensorId> return_tensors;
    std::vector<string> return_nodes;

    // TODO(ashankar): This bool exists to separate out functionality required
    // to make ImportGraphDef a close equivalent of Python's import_graph_def
    // without affecting the behavior of ConvertGraphDefToGraph at the time
    // ImportGraphDef was added.
    //
    // That said, the functionality here (shape and op validation) seems
    // applicable to ConvertGraphDefToGraph as well, so make an attempt to
    // remove this.
    bool importing;
    // If true, validates that nodes being converted have all expected attrs
    // set and no unknown attrs set by calling ValidateNodeDef().
    // `validate_nodes` is always true when `importing` is set.
    bool validate_nodes;
    bool validate_colocation_constraints;
    bool validate_shape = true;

    // If true, GraphConstructor will add attributes with their default
    // value to the Node when they are missing from the NodeDef.
    bool add_default_attributes = true;

    string default_device;
  };

  typedef gtl::ArraySlice<const NodeDef*> NodeDefSlice;

  // versions and library may be nullptr
  static Status Construct(
      const Options& opts, NodeDefSlice node_defs, const VersionDef* versions,
      const FunctionDefLibrary* library, Graph* g, ShapeRefiner* refiner,
      std::vector<std::pair<Node*, int>>* return_tensors,
      std::vector<Node*>* return_nodes,
      std::vector<SafeTensorId>* missing_unused_input_map_keys);

  static Status Construct(
      const Options& opts, GraphDef&& graph_def, Graph* g,
      ShapeRefiner* refiner, std::vector<std::pair<Node*, int>>* return_tensors,
      std::vector<Node*>* return_nodes,
      std::vector<SafeTensorId>* missing_unused_input_map_keys);

 protected:
  GraphConstructor(const Options& opts, Graph* g, ShapeRefiner* refiner,
                   std::vector<std::pair<Node*, int>>* return_tensors,
                   std::vector<Node*>* return_nodes,
                   std::vector<SafeTensorId>* missing_unused_input_map_keys)
      : opts_(opts),
        g_(g),
        original_versions_(g->versions()),
        prefix_(opts.prefix),
        refiner_(refiner),
        return_tensors_(return_tensors),
        return_nodes_(return_nodes),
        missing_unused_input_map_keys_(missing_unused_input_map_keys) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_5(mht_5_v, 368, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor");
}

  virtual ~GraphConstructor() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_6(mht_6_v, 373, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "~GraphConstructor");
}

  Status TryImport() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_7(mht_7_v, 378, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "TryImport");

    TF_RETURN_IF_ERROR(EnsureNoNameCollisions());
    TF_RETURN_IF_ERROR(ValidateInputMapAndControlDependencies());
    TF_RETURN_IF_ERROR(BuildNodeIndex());
    TF_RETURN_IF_ERROR(InitFromEdges());

    // NOTE: Convert() invokes `consume_node_def()` on each node in the input
    // graph, so `get_node_def()` is no longer usable once it is called.
    TF_RETURN_IF_ERROR(Convert());

    TF_RETURN_IF_ERROR(AddBackEdges());
    TF_RETURN_IF_ERROR(UpdateVersionDef());
    TF_RETURN_IF_ERROR(PopulateReturnTensors());
    TF_RETURN_IF_ERROR(PopulateReturnNodes());
    TF_RETURN_IF_ERROR(PopulateMissingUnusedInputMapKeys());
    UpdateUniquifiedColocationNames();
    FixupSourceAndSinkEdges(g_);
    return Status::OK();
  }

 private:
  Status EnsureNoNameCollisions();
  Status ValidateInputMapAndControlDependencies();
  Status BuildNodeIndex();
  Status InitFromEdges();
  Status Convert();
  Status AddBackEdges();
  Status UpdateVersionDef();
  Status PopulateReturnTensors();
  Status PopulateReturnNodes();
  Status PopulateMissingUnusedInputMapKeys();

  void Undo();

  // Prints cycles in the graph.
  void PrintCycles();
  // Performs DFS starting at `cur_node` and prints any cycles found.
  void DFS(int cur_node, std::vector<int>* cur_branch,
           std::vector<bool>* is_on_cur_branch,
           absl::flat_hash_set<int>* unvisited,
           const std::vector<absl::string_view>& node_names);
  Status IsNodeFullyMapped(const NodeDef& node_def, bool* is_node_mapped);
  Status ValidateColocationConstraints(const NodeDef& node_def);
  Status MakeNode(NodeDef&& node_def, Node** node);
  Status MakeEdge(Node* src, int output_index, Node* dst, int input_index);
  Status ValidateShape(Node* node);
  Status ModifyNodeDefForImport(NodeDef* node_def);
  // Modifies node_def's inputs according to opts_.input_map.
  // input_already_exists is a pre-initialized vector of length
  // node_def->input_size(). This function will mark inputs that are remapped to
  // true.
  void RemapNodeDefInputs(NodeDef* node_def,
                          std::vector<bool>* input_already_exists);
  // input_already_exists is a pre-initialized vector of length
  // node_def->input_size(). This function will add and mark control inputs as
  // true.
  void AddControlDependencies(NodeDef* node_def,
                              std::vector<bool>* input_already_exists);
  void AddPrefixToNodeDef(const std::vector<bool>& input_already_exists,
                          NodeDef* node_def);

  // Modifies `node_def` if its name isn't unique, or if any of its inputs'
  // names have been uniquified. This must be called in topological order on all
  // nodes.
  void UniquifyNames(const std::vector<bool>& input_already_exists,
                     NodeDef* node_def);

  // Updates any constructed nodes' colocation group names if the name has been
  // updated by UniquifyNames. This is called after all the nodes have been
  // constructed so all the names have been uniquified if necessary.
  void UpdateUniquifiedColocationNames();

  // Returns true if `name` already exists in `g_` (either as a node name or
  // prefix).
  bool NameExistsInGraph(StringPiece name);

  // Returns true if `name` already exists in the GraphDef being imported
  // (either as a node name or prefix).
  bool NameExistsInGraphDef(StringPiece name);

  // Returns a unique version of `original_name`, or `original_name` if it's
  // already unique in the graph.
  string FindUniqueName(StringPiece original_name);

  // Decrement pending count for users of `processed` and add the ones that now
  // have all of their pending inputs satisfied to `ready_`.
  void UpdatePendingCountAndReady(int processed, bool is_next_iteration);

  // Subclasses override the following virtual methods to provide efficient
  // access to the original protocol buffer-based graph.

  // Returns the number of nodes in the graph.
  virtual size_t node_def_count() const = 0;
  // Returns the i^th node in the graph. Must not be called after
  // consume_node_def(i).
  virtual const NodeDef& get_node_def(int i) const = 0;
  // Destructively reads the i^th node in the graph, avoiding a copy if
  // possible. After calling this method, the result of get_node_def(i) is
  // undefined.
  virtual NodeDef consume_node_def(int i) = 0;
  // Returns the version information for the graph, or nullptr if none is
  // available.
  virtual const VersionDef* versions() const = 0;
  // Returns the function information for the graph, or nullptr if none is
  // available.
  virtual const FunctionDefLibrary* library() const = 0;

  // From constructor
  const Options opts_;
  Graph* g_;
  const VersionDef original_versions_;

  // A copy of opts_.prefix, possibly uniquified.
  string prefix_;

  ShapeRefiner* refiner_;

  // May be null. Not owned.
  std::vector<std::pair<Node*, int>>* return_tensors_;

  // May be null. Not owned.
  std::vector<Node*>* return_nodes_;

  // May be null. Not owned.
  std::vector<SafeTensorId>* missing_unused_input_map_keys_;

  // Intermediate datastructure used to populate
  // `missing_unused_input_map_keys_`.
  std::set<TensorId> used_input_map_keys_;

  // Intermediate datastructure used to track the destinations of back edges.
  absl::flat_hash_set<int> merge_node_indices_;

  // Mapping from node name to the index within node_defs_.
  struct NodeInfo {
    explicit NodeInfo(int i) : gdef_index(i), node(nullptr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_8(mht_8_v, 516, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "NodeInfo");
}
    // Containers require that we have a default constructor.
    NodeInfo() : NodeInfo(-1) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_9(mht_9_v, 521, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "NodeInfo");
}
    int gdef_index;
    Node* node;  // nullptr until the NodeDef is converted to a Node.
  };
  absl::flat_hash_map<std::string, NodeInfo> gdef_nodes_;

  // Prefixes already used in the GraphDef being imported.
  absl::flat_hash_set<StringPiece> gdef_prefixes_;

  // Mapping from node name to the existing node in g_.
  absl::flat_hash_map<StringPiece, Node*> existing_nodes_;

  // Prefixes already used in the graph.
  absl::flat_hash_set<StringPiece> existing_prefixes_;

  // Imported node names that have been uniquified. The key is the original
  // name, the value is the new unique name.
  gtl::FlatMap<string, string> uniquified_names_;

  // Index of NodeDefs in node_defs_ with all inputs already converted. We use a
  // (sorted) set so nodes are created in the order defined in the GraphDef.
  std::set<int> ready_;

  // Mapping between index within node_defs_ and the number of inputs that
  // still need to be converted.
  std::vector<int> pending_count_;

  // Mapping between index within node_defs_ and the index within node_defs_ of
  // all nodes it outputs to.
  std::vector<gtl::InlinedVector<int, 4>> outputs_;

  // Used in the conversion from node_defs_ to g_ to represent the ith input
  // of a node.
  struct InputInfo {
    explicit InputInfo(const string& node_name, Node* n, int i)
        : name(node_name), node(n), index(i) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_10(mht_10_v, 560, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "InputInfo");
}
    // Use string instead of StringPiece so we don't have to manage lifetime
    string name;
    Node* node;
    int index;

    static bool IsControlInput(const InputInfo& input) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_11(mht_11_v, 569, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "IsControlInput");

      return input.index == Graph::kControlSlot;
    }
    static int CompareName(const InputInfo& lhs, const InputInfo& rhs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_12(mht_12_v, 575, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "CompareName");

      return lhs.name < rhs.name;
    }
    static bool IsSameName(const InputInfo& lhs, const InputInfo& rhs) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_13(mht_13_v, 581, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "IsSameName");

      return lhs.name == rhs.name;
    }
  };

  // Used in the conversion from node_defs_ to g_ to represent an edge from
  // the node named 'name' to node 'n'.
  struct EdgeInfo {
    explicit EdgeInfo(const string& name, int i1, Node* n, int i2)
        : src_name(name), src_index(i1), dst_node(n), dst_index(i2) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_14(mht_14_v, 594, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "EdgeInfo");
}
    // Use string instead of StringPiece so we don't have to manage lifetime
    string src_name;
    int src_index;
    Node* dst_node;
    int dst_index;
  };
  std::vector<EdgeInfo> back_edges_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphConstructor);
};

// Implementation of GraphConstructor that does not take ownership of the
// input NodeDef messages and thus copies the nodes into the constructed Graph*.
//
// NOTE(mrry): Whenever possible, use NodeDefMovingGraphConstructor, which
// avoids copying each NodeDef into the constructed Graph*.
class NodeDefCopyingGraphConstructor : public GraphConstructor {
 public:
  NodeDefCopyingGraphConstructor(
      const Options& opts, NodeDefSlice node_defs, const VersionDef* versions,
      const FunctionDefLibrary* library, Graph* g, ShapeRefiner* refiner,
      std::vector<std::pair<Node*, int>>* return_tensors,
      std::vector<Node*>* return_nodes,
      std::vector<SafeTensorId>* missing_unused_input_map_keys)
      : GraphConstructor(opts, g, refiner, return_tensors, return_nodes,
                         missing_unused_input_map_keys),
        node_defs_(node_defs),
        versions_(versions),
        library_(library) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_15(mht_15_v, 626, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "NodeDefCopyingGraphConstructor");
}

 private:
  size_t node_def_count() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_16(mht_16_v, 632, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "node_def_count");
 return node_defs_.size(); }
  const NodeDef& get_node_def(int i) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_17(mht_17_v, 636, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "get_node_def");
 return *node_defs_[i]; }
  NodeDef consume_node_def(int i) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_18(mht_18_v, 640, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "consume_node_def");
 return *node_defs_[i]; }
  const VersionDef* versions() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_19(mht_19_v, 644, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "versions");
 return versions_; }
  const FunctionDefLibrary* library() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_20(mht_20_v, 648, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "library");
 return library_; }

  const NodeDefSlice node_defs_;
  const VersionDef* const versions_;
  const FunctionDefLibrary* const library_;
};

// Implementation of GraphConstructor that takes ownership of the input
// GraphDef, and can perform destructive reads.
class NodeDefMovingGraphConstructor : public GraphConstructor {
 public:
  NodeDefMovingGraphConstructor(
      const Options& opts, GraphDef&& graph_def, Graph* g,
      ShapeRefiner* refiner, std::vector<std::pair<Node*, int>>* return_tensors,
      std::vector<Node*>* return_nodes,
      std::vector<SafeTensorId>* missing_unused_input_map_keys)
      : GraphConstructor(opts, g, refiner, return_tensors, return_nodes,
                         missing_unused_input_map_keys),
        graph_def_(std::move(graph_def)),
        is_consumed_(graph_def_.node_size(), false) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_21(mht_21_v, 670, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "NodeDefMovingGraphConstructor");
}

 private:
  size_t node_def_count() const override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_22(mht_22_v, 676, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "node_def_count");
 return graph_def_.node().size(); }
  const NodeDef& get_node_def(int i) const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_23(mht_23_v, 680, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "get_node_def");

    CHECK(!is_consumed_[i])
        << "NodeDef " << i << " accessed after it was consumed.";
    return graph_def_.node(i);
  }
  NodeDef consume_node_def(int i) override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_24(mht_24_v, 688, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "consume_node_def");

    CHECK(!is_consumed_[i]) << "NodeDef " << i << " consumed twice.";
    is_consumed_[i] = true;
    return std::move(*graph_def_.mutable_node(i));
  }
  const VersionDef* versions() const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_25(mht_25_v, 696, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "versions");
 return &graph_def_.versions(); }
  const FunctionDefLibrary* library() const override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_26(mht_26_v, 700, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "library");

    return &graph_def_.library();
  }

  GraphDef graph_def_;
  std::vector<bool> is_consumed_;
};

bool ForwardCompatibilityWindowPassed(const VersionDef& versions) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_27(mht_27_v, 711, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "ForwardCompatibilityWindowPassed");

  // TF_GRAPH_DEF_VERSION is incremented daily.
  // TF has a 3 week forward compatibility guarantee.
  return (versions.producer() - TF_GRAPH_DEF_VERSION) > 21;
}

Status MaybeAppendVersionWarning(const VersionDef* versions,
                                 const Status& import_status) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_28(mht_28_v, 721, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "MaybeAppendVersionWarning");

  if (versions && ForwardCompatibilityWindowPassed(*versions)) {
    return Status(
        import_status.code(),
        absl::StrCat(
            "Converting GraphDef to Graph has failed with an error: '",
            import_status.error_message(),
            "' The binary trying to import the GraphDef was built when "
            "GraphDef version was ",
            TF_GRAPH_DEF_VERSION,
            ". The GraphDef was produced by a binary built when GraphDef "
            "version was ",
            versions->producer(),
            ". The difference between these versions is larger than "
            "TensorFlow's forward compatibility guarantee, and might be the "
            "root cause for failing to import the GraphDef."));
  }
  return import_status;
}

/* static */ Status GraphConstructor::Construct(
    const Options& opts, NodeDefSlice node_defs, const VersionDef* versions,
    const FunctionDefLibrary* library, Graph* g, ShapeRefiner* refiner,
    std::vector<std::pair<Node*, int>>* return_tensors,
    std::vector<Node*>* return_nodes,
    std::vector<SafeTensorId>* missing_unused_input_map_keys) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_29(mht_29_v, 749, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::Construct");

  if (versions) {
    TF_RETURN_IF_ERROR(CheckVersions(*versions, TF_GRAPH_DEF_VERSION,
                                     TF_GRAPH_DEF_VERSION_MIN_PRODUCER,
                                     "GraphDef", "graph"));
  }
  NodeDefCopyingGraphConstructor c(opts, node_defs, versions, library, g,
                                   refiner, return_tensors, return_nodes,
                                   missing_unused_input_map_keys);
  Status s = c.TryImport();
  if (!s.ok()) {
    c.Undo();
    s = MaybeAppendVersionWarning(versions, s);
  }
  return s;
}

/* static */ Status GraphConstructor::Construct(
    const Options& opts, GraphDef&& graph_def, Graph* g, ShapeRefiner* refiner,
    std::vector<std::pair<Node*, int>>* return_tensors,
    std::vector<Node*>* return_nodes,
    std::vector<SafeTensorId>* missing_unused_input_map_keys) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_30(mht_30_v, 773, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::Construct");

  TF_RETURN_IF_ERROR(CheckVersions(graph_def.versions(), TF_GRAPH_DEF_VERSION,
                                   TF_GRAPH_DEF_VERSION_MIN_PRODUCER,
                                   "GraphDef", "graph"));
  VersionDef version_def = graph_def.versions();
  NodeDefMovingGraphConstructor c(opts, std::move(graph_def), g, refiner,
                                  return_tensors, return_nodes,
                                  missing_unused_input_map_keys);
  Status s = c.TryImport();
  if (!s.ok()) {
    c.Undo();
    s = MaybeAppendVersionWarning(&version_def, s);
  }
  return s;
}

void GraphConstructor::UpdatePendingCountAndReady(int processed,
                                                  bool is_next_iteration) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_31(mht_31_v, 793, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::UpdatePendingCountAndReady");

  for (size_t i = 0; i < outputs_[processed].size(); ++i) {
    const int output = outputs_[processed][i];
    // We didn't consider NextIteration->Merge edges when computing
    // pending_counts_ so we should not have to consider it here either.
    bool is_next_iteration_to_merge_edge =
        is_next_iteration && merge_node_indices_.count(output) == 1;
    if (!is_next_iteration_to_merge_edge) {
      int* current_pending_count = &pending_count_[output];
      CHECK_GT(*current_pending_count, 0);
      (*current_pending_count)--;
      if (*current_pending_count == 0) {
        ready_.insert(output);
      }
    }
  }
}

// This could be expensive but we don't expect to call it often, if at all (only
// if there are multiple nodes in g_ with the same name)
bool NodeNameInValues(const std::map<TensorId, TensorId>& input_map,
                      const StringPiece& node_name) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_32(mht_32_v, 817, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "NodeNameInValues");

  for (auto iter = input_map.begin(); iter != input_map.end(); ++iter) {
    if (iter->second.first == node_name) return true;
  }
  return false;
}

bool NodeNameInValues(const std::vector<string>& control_dependencies,
                      const StringPiece& node_name) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_33(mht_33_v, 828, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "NodeNameInValues");

  return std::find(control_dependencies.begin(), control_dependencies.end(),
                   node_name) != control_dependencies.end();
}

// Adds any prefixes of `node_name` (not including the full name itself) to
// `prefixes`.
void AddPrefixes(StringPiece node_name,
                 absl::flat_hash_set<StringPiece>* prefixes) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_34(mht_34_v, 839, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "AddPrefixes");

  size_t idx = -1;
  while ((idx = node_name.find('/', idx + 1)) != StringPiece::npos) {
    prefixes->insert(node_name.substr(0, idx));
  }
}

Status GraphConstructor::EnsureNoNameCollisions() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_35(mht_35_v, 849, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::EnsureNoNameCollisions");

  existing_nodes_.reserve(g_->num_nodes());
  // Populate existing_nodes_ and existing_prefixes_.
  for (Node* n : g_->nodes()) {
    bool already_exists = !existing_nodes_.insert({n->name(), n}).second;
    if (already_exists) {
      if (NodeNameInValues(opts_.input_map, n->name())) {
        return errors::InvalidArgument(
            "cannot resolve input_map because multiple nodes exist with name '",
            n->name(), "'");
      }
      if (NodeNameInValues(opts_.control_dependencies, n->name())) {
        return errors::InvalidArgument(
            "cannot resolve control_dependencies because multiple nodes exist "
            "with name '",
            n->name(), "'");
      }
    }
    AddPrefixes(n->name(), &existing_prefixes_);
  }
  if (prefix_.empty() && opts_.importing && !opts_.uniquify_names) {
    for (size_t i = 0; i < node_def_count(); ++i) {
      const string& name = get_node_def(i).name();
      if (NameExistsInGraph(name)) {
        return errors::InvalidArgument("Node name '", name,
                                       "' already exists in the Graph");
      }
    }
  } else if (!prefix_.empty()) {
    StringPiece prefix_no_slash(prefix_);
    prefix_no_slash.remove_suffix(1);
    if (!IsValidNodeName(prefix_no_slash, false)) {
      return errors::InvalidArgument("Imported node name prefix '", prefix_,
                                     "' would lead to invalid node names");
    }
    if (NameExistsInGraph(prefix_no_slash) && opts_.uniquify_prefix) {
      prefix_ = strings::StrCat(FindUniqueName(prefix_no_slash), "/");
    }
  }
  return Status::OK();
}

Status GraphConstructor::ValidateInputMapAndControlDependencies() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_36(mht_36_v, 894, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::ValidateInputMapAndControlDependencies");

  for (const auto& mapping : opts_.input_map) {
    TensorId src = mapping.first;
    TensorId dst = mapping.second;
    if (existing_nodes_.count(dst.first) == 0) {
      return errors::InvalidArgument(
          "node '", dst.first, "' in input_map does not exist in graph ",
          "(input_map entry: ", src.ToString(), "->", dst.ToString(), ")");
    }
    if ((src.second == Graph::kControlSlot) !=
        (dst.second == Graph::kControlSlot)) {
      return errors::InvalidArgument("input_map entry ", src.ToString(), "->",
                                     dst.ToString(), " between ",
                                     "control edge and non-control edge");
    }
  }
  for (const string& node : opts_.control_dependencies) {
    if (existing_nodes_.count(node) == 0) {
      return errors::InvalidArgument(
          "node '", node,
          "' in control_dependencies does not exist in "
          "graph");
    }
  }
  return Status::OK();
}

Status GraphConstructor::BuildNodeIndex() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_37(mht_37_v, 924, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::BuildNodeIndex");

  // Validate the node names and add them to gdef_nodes_ and gdef_prefixes_.
  for (int n = 0; n < node_def_count(); ++n) {
    const NodeDef& node_def = get_node_def(n);
    if (!IsValidNodeName(node_def.name(), opts_.allow_internal_ops)) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "': Node name contains invalid characters");
    }
    if (!gdef_nodes_.insert(std::make_pair(node_def.name(), NodeInfo(n)))
             .second) {
      return errors::InvalidArgument("Node '", node_def.name(),
                                     "' is not unique");
    }
    // Validate the operation's type.
    if (node_def.op().empty()) {
      return errors::InvalidArgument("Node '", node_def.name(),
                                     "' does not specify an operation");
    }
    if (opts_.expect_device_spec && node_def.device().empty()) {
      return errors::InvalidArgument("Node '", node_def.name(),
                                     "' is missing a device specification");
    }
    if (IsMerge(node_def)) {
      merge_node_indices_.insert(n);
    }
    // Validate control edges at end
    bool in_control_dependence = false;
    for (int i = 0; i < node_def.input_size(); ++i) {
      StringPiece input_name = node_def.input(i);
      if (!input_name.empty() && absl::StartsWith(input_name, "^")) {
        in_control_dependence = true;
      } else if (in_control_dependence) {
        return errors::InvalidArgument(
            "Node '", node_def.name(),
            "': Control dependencies must come after regular dependencies");
      }
    }
    // Update gdef_prefixes_.
    AddPrefixes(node_def.name(), &gdef_prefixes_);
  }
  return Status::OK();
}

Status GraphConstructor::InitFromEdges() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_38(mht_38_v, 971, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::InitFromEdges");

  const int num_nodes = node_def_count();
  pending_count_.reserve(num_nodes);
  outputs_.resize(num_nodes);
  gtl::FlatSet<string> next_iteration_nodes;
  for (int n = 0; n < node_def_count(); ++n) {
    const NodeDef& node_def = get_node_def(n);
    if (IsNextIteration(node_def)) {
      next_iteration_nodes.insert(node_def.name());
    }
  }

  // Parse the inputs for each node.
  for (int n = 0; n < num_nodes; ++n) {
    const NodeDef& node_def = get_node_def(n);
    int pending_count = node_def.input_size();
    if (IsMerge(node_def)) {
      // Cycles in the graph are only allowed for while loops. A while loop is
      // identified by an edge from a NextIteration node to a Merge node. For
      // such Merge nodes, only wait for one non-control input before
      // considering the node ready to process in Convert().
      int32_t num_control_edges = 0;
      bool has_loop_back_edge = false;
      for (int i = 0; i < node_def.input_size(); ++i) {
        StringPiece input_name(node_def.input(i));
        if (absl::StartsWith(input_name, "^")) {
          num_control_edges++;
        } else {
          TensorId id(ParseTensorName(input_name));
          if (next_iteration_nodes.find(string(id.first)) !=
              next_iteration_nodes.end()) {
            has_loop_back_edge = true;
          }
        }
      }
      if (has_loop_back_edge) {
        pending_count = num_control_edges + 1;
      }
    }
    for (int i = 0; i < node_def.input_size(); ++i) {
      StringPiece input_name = node_def.input(i);
      TensorId id(ParseTensorName(input_name));
      if (opts_.input_map.count(id) == 0) {
        // If an input is not mapped, then the input should appear in the graph
        // being imported.
        auto iter = gdef_nodes_.find(id.first);
        if (iter == gdef_nodes_.end()) {
          return errors::InvalidArgument("Node '", node_def.name(),
                                         "': Unknown input node '",
                                         node_def.input(i), "'");
        }
        outputs_[iter->second.gdef_index].push_back(n);
      } else {
        // This input is mapped to an existing edge. Therefore this input is
        // as good as being already processed.
        --pending_count;
        DCHECK_GE(pending_count, 0);
      }
    }
    if (pending_count == 0) {
      ready_.insert(n);
    }
    pending_count_.push_back(pending_count);
  }
  return Status::OK();
}

Status GraphConstructor::ValidateColocationConstraints(
    const NodeDef& node_def) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_39(mht_39_v, 1042, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::ValidateColocationConstraints");

  if (!opts_.validate_colocation_constraints || !opts_.importing)
    return Status::OK();
  const auto iter = node_def.attr().find(kColocationAttrName);
  if (iter == node_def.attr().end()) return Status::OK();
  for (const string& c : iter->second.list().s()) {
    StringPiece s(c);
    if (absl::ConsumePrefix(&s, kColocationGroupPrefix) &&
        gdef_nodes_.find(s) == gdef_nodes_.end()) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "' expects to be colocated with unknown node '", s, "'");
    }
  }
  return Status::OK();
}

Status GraphConstructor::MakeNode(NodeDef&& node_def, Node** node) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_40(mht_40_v, 1062, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::MakeNode");

  // Add the node to the graph.
  Status status;
  *node = g_->AddNode(std::move(node_def), &status);
  if (!status.ok()) return status;
  if (opts_.expect_device_spec) {
    (*node)->set_assigned_device_name((*node)->def().device());
  }
  return Status::OK();
}

Status GraphConstructor::ValidateShape(Node* node) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_41(mht_41_v, 1076, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::ValidateShape");

  if (!opts_.importing || !opts_.validate_shape) return Status::OK();
  TF_RETURN_IF_ERROR(refiner_->AddNode(node));
  // For nodes with the _output_shapes attribute, override the shape.
  std::vector<const TensorShapeProto*> shape_attrs;
  const char* kAttrName = "_output_shapes";
  if (!TryGetNodeAttr(node->attrs(), kAttrName, &shape_attrs)) {
    // No _output_shapes attribute, the AddNode call above was sufficient.
    return Status::OK();
  }
  auto* ic = refiner_->GetContext(node);
  DCHECK(ic != nullptr)
      << "ShapeRefiner::AddNode() should have created the InferenceContext";
  if (shape_attrs.size() < node->num_outputs()) {
    return errors::InvalidArgument(
        "Node '", node->name(), "' has ", node->num_outputs(),
        " outputs but the ", kAttrName, " attribute specifies shapes for ",
        shape_attrs.size(), " outputs");
  }
  // NOTE(skyewm): we don't raise an error here because some users depend on
  // this behavior, even though it's unsafe.
  // TODO(b/74619486): raise an error.
  if (shape_attrs.size() > node->num_outputs()) {
    LOG(WARNING) << "Node '" << node->name() << "' has " << node->num_outputs()
                 << " outputs but the " << kAttrName
                 << " attribute specifies shapes for " << shape_attrs.size()
                 << " outputs. Output shapes may be inaccurate.";
  }
  for (int i = 0; i < node->num_outputs(); ++i) {
    const TensorShapeProto& p = *shape_attrs[i];
    shape_inference::ShapeHandle h;
    Status s = ic->MakeShapeFromShapeProto(p, &h);
    if (!s.ok()) {
      return errors::InvalidArgument("Node '", node->name(), " has an invalid ",
                                     kAttrName, " attribute (shape #", i,
                                     " error:'", s.error_message(), "'");
    }
    s = refiner_->SetShape(node, i, h);
    if (!s.ok()) {
      return errors::InvalidArgument(
          "Node '", node->name(), "' has an ", kAttrName,
          " attribute inconsistent with the GraphDef for output #", i, ": ",
          s.error_message());
    }
  }
  node->ClearAttr(kAttrName);
  return Status::OK();
}

Status GraphConstructor::ModifyNodeDefForImport(NodeDef* node_def) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_42(mht_42_v, 1128, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::ModifyNodeDefForImport");

  const OpDef* op_def;
  TF_RETURN_IF_ERROR(g_->op_registry()->LookUpOpDef(node_def->op(), &op_def));
  AddDefaultsToNodeDef(*op_def, node_def);
  TF_RETURN_IF_ERROR(ValidateNodeDef(*node_def, *op_def));
  if (versions()) {
    TF_RETURN_IF_ERROR(CheckOpDeprecation(*op_def, versions()->producer()));
  }
  return Status::OK();
}

void RemoveInputs(const std::vector<int>& inputs_to_remove, NodeDef* node_def,
                  std::vector<bool>* input_already_exists) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_43(mht_43_v, 1143, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "RemoveInputs");

  // Remove 'inputs_to_remove' from 'node_def'
  NodeDef copy;
  copy.mutable_input()->Reserve(node_def->input_size() -
                                inputs_to_remove.size());
  for (int i = 0, j = 0; i < node_def->input_size(); ++i) {
    if (j < inputs_to_remove.size() && i == inputs_to_remove[j]) {
      ++j;
    } else {
      copy.add_input()->swap(*node_def->mutable_input(i));
    }
  }
  node_def->mutable_input()->Swap(copy.mutable_input());
  // Remove 'inputs_to_remove' from 'input_already_exists'
  for (int idx : inputs_to_remove) {
    input_already_exists->erase(input_already_exists->begin() + idx);
  }
  DCHECK_EQ(input_already_exists->size(), node_def->input_size());
}

void GraphConstructor::RemapNodeDefInputs(
    NodeDef* node_def, std::vector<bool>* input_already_exists) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_44(mht_44_v, 1167, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::RemapNodeDefInputs");

  DCHECK_EQ(input_already_exists->size(), node_def->input_size());
  std::set<TensorId> control_inputs;
  std::vector<int> inputs_to_remove;

  for (int i = 0; i < node_def->input_size(); ++i) {
    auto iter = opts_.input_map.find(ParseTensorName(node_def->input(i)));
    if (iter == opts_.input_map.end()) continue;
    used_input_map_keys_.insert(iter->first);

    TensorId new_input = iter->second;
    if (new_input.second == Graph::kControlSlot) {
      // Check if we've already remapped a different input to new_input, and if
      // so remove this input.
      if (control_inputs.count(new_input) > 0) {
        inputs_to_remove.push_back(i);
        continue;
      }
      control_inputs.insert(new_input);
    }
    node_def->set_input(i, new_input.ToString());
    (*input_already_exists)[i] = true;
  }
  if (!inputs_to_remove.empty()) {
    RemoveInputs(inputs_to_remove, node_def, input_already_exists);
  }
}

void GraphConstructor::AddControlDependencies(
    NodeDef* node_def, std::vector<bool>* input_already_exists) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_45(mht_45_v, 1199, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::AddControlDependencies");

  // To avoid adding redundant control dependencies to every imported node, skip
  // nodes that will inherit the dependencies from another imported node.
  bool inherits_deps = false;
  for (int i = 0; i < node_def->input_size(); ++i) {
    // Assume we won't inherit dependencies from remapped inputs that already
    // exist in the graph. Even if we're wrong, we'll only add redundant
    // dependencies.
    if ((*input_already_exists)[i]) continue;

    // If this input is a backedge, assume we won't inherit the dependencies.
    // TODO(skyewm): we have many redundant ParseTensorName calls. It could be
    // worth optimizing these.
    TensorId id(ParseTensorName(node_def->input(i)));
    auto iter = gdef_nodes_.find(id.first);
    DCHECK(iter != gdef_nodes_.end()) << id.first;
    if (iter->second.node == nullptr) {
      // Input hasn't been created yet, indicating it's a backedge.
      continue;
    }
    inherits_deps = true;
  }
  if (inherits_deps) return;

  // node_def either has no inputs or all remapped inputs, add the control
  // dependencies
  for (const string& control_dep : opts_.control_dependencies) {
    string input = TensorId(control_dep, Graph::kControlSlot).ToString();
    bool found = false;
    for (int i = node_def->input_size() - 1; i >= 0; --i) {
      const string& node_input = node_def->input(i);
      if (node_input[0] != '^') {
        // Control inputs are at the end. Break when we reach the non-control
        // inputs.
        break;
      }
      if (node_input == input) {
        // Control dependency already exists
        found = true;
        break;
      }
    }
    if (found) {
      continue;
    }
    node_def->add_input(input);
    input_already_exists->push_back(true);
  }
}

void GraphConstructor::AddPrefixToNodeDef(
    const std::vector<bool>& input_already_exists, NodeDef* node_def) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_46(mht_46_v, 1253, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::AddPrefixToNodeDef");

  if (prefix_.empty()) return;
  node_def->set_name(strings::StrCat(prefix_, node_def->name()));
  // Update names of input nodes
  for (int i = 0; i < node_def->input_size(); ++i) {
    // Skip remapped inputs (which already exist in g_ and are not being
    // imported).
    if (input_already_exists[i]) continue;
    StringPiece input(node_def->input(i));
    if (absl::ConsumePrefix(&input, "^")) {
      node_def->set_input(i, strings::StrCat("^", prefix_, input));
    } else {
      node_def->set_input(i, strings::StrCat(prefix_, input));
    }
  }
  // Update names of colocation groups
  if (node_def->attr().find(kColocationAttrName) != node_def->attr().end()) {
    auto* list =
        node_def->mutable_attr()->at(kColocationAttrName).mutable_list();
    for (int i = 0; i < list->s_size(); ++i) {
      StringPiece v(list->s(i));
      if (absl::ConsumePrefix(&v, kColocationGroupPrefix)) {
        list->set_s(i, strings::StrCat(kColocationGroupPrefix, prefix_, v));
      }
    }
  }
}

void GraphConstructor::UniquifyNames(
    const std::vector<bool>& input_already_exists, NodeDef* node_def) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_47(mht_47_v, 1285, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::UniquifyNames");

  if (NameExistsInGraph(node_def->name())) {
    string old_name = node_def->name();
    node_def->set_name(FindUniqueName(node_def->name()));
    uniquified_names_[old_name] = node_def->name();
    // Note that we don't have to update gdef_nodes_ or gdef_prefixes_ with
    // `name` because we guarantee the original NodeDef names are unique,
    // meaning we won't generate this name again.
  }
  for (int i = 0; i < node_def->input_size(); ++i) {
    // Skip remapped inputs (which already exist in g_ and are not being
    // imported).
    if (input_already_exists[i]) continue;
    TensorId id = ParseTensorName(node_def->input(i));
    // We require that UniquifyNames() is called on all NodeDefs in topological
    // order. This guarantees that node_def's inputs will already be uniquified
    // if necessary.
    auto iter = uniquified_names_.find(string(id.first));
    if (iter == uniquified_names_.end()) continue;
    id.first = iter->second;
    node_def->set_input(i, id.ToString());
  }
}

void GraphConstructor::UpdateUniquifiedColocationNames() {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_48(mht_48_v, 1312, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::UpdateUniquifiedColocationNames");

  for (const auto& pair : gdef_nodes_) {
    Node* node = pair.second.node;
    if (node == nullptr) continue;
    std::vector<string> coloc_values;
    if (!TryGetNodeAttr(node->attrs(), kColocationAttrName, &coloc_values))
      continue;
    bool updated = false;
    for (size_t i = 0; i < coloc_values.size(); ++i) {
      StringPiece val(coloc_values[i]);
      if (absl::ConsumePrefix(&val, kColocationGroupPrefix)) {
        auto name_pair = uniquified_names_.find(string(val));
        if (name_pair == uniquified_names_.end()) continue;
        updated = true;
        coloc_values[i] =
            strings::StrCat(kColocationGroupPrefix, name_pair->second);
      }
    }
    if (updated) {
      node->AddAttr(kColocationAttrName, std::move(coloc_values));
    }
  }
}

bool GraphConstructor::NameExistsInGraph(StringPiece name) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_49(mht_49_v, 1339, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::NameExistsInGraph");

  if (existing_nodes_.find(name) != existing_nodes_.end()) return true;
  if (existing_prefixes_.find(name) != existing_prefixes_.end()) return true;
  return false;
}

bool GraphConstructor::NameExistsInGraphDef(StringPiece name) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_50(mht_50_v, 1348, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::NameExistsInGraphDef");

  if (gdef_nodes_.find(name) != gdef_nodes_.end()) return true;
  if (gdef_prefixes_.find(name) != gdef_prefixes_.end()) return true;
  return false;
}

string GraphConstructor::FindUniqueName(StringPiece original_name) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_51(mht_51_v, 1357, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::FindUniqueName");

  string name(original_name);
  int count = 0;
  // Check that any generated names don't collide with imported NodeDefs (as
  // well as nodes in g_).
  while (NameExistsInGraph(name) || (count > 0 && NameExistsInGraphDef(name))) {
    name = strings::StrCat(original_name, "_", ++count);
  }
  return name;
}

Status GraphConstructor::IsNodeFullyMapped(const NodeDef& node_def,
                                           bool* is_node_mapped) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_52(mht_52_v, 1372, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::IsNodeFullyMapped");

  const OpDef* op_def;
  TF_RETURN_IF_ERROR(g_->op_registry()->LookUpOpDef(node_def.op(), &op_def));
  for (int i = 0; i < op_def->output_arg_size(); ++i) {
    if (opts_.input_map.find({node_def.name(), i}) == opts_.input_map.end()) {
      *is_node_mapped = false;
      return Status::OK();
    }
  }
  *is_node_mapped = true;
  return Status::OK();
}

void GraphConstructor::DFS(int cur_node, std::vector<int>* cur_branch,
                           std::vector<bool>* is_on_cur_branch,
                           absl::flat_hash_set<int>* unvisited,
                           const std::vector<absl::string_view>& node_names) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_53(mht_53_v, 1391, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::DFS");

  cur_branch->push_back(cur_node);
  is_on_cur_branch->at(cur_node) = true;
  for (auto next_node : outputs_[cur_node]) {
    if (unvisited->find(next_node) != unvisited->end()) {
      if (is_on_cur_branch->at(next_node)) {
        auto iter =
            std::find(cur_branch->begin(), cur_branch->end(), next_node);
        LOG(WARNING) << "Cycle detected:";
        while (iter != cur_branch->end()) {
          const absl::string_view name = node_names[*iter];
          DCHECK(!name.empty());
          LOG(WARNING) << "node id=" << *iter << ", name=" << name;
          ++iter;
        }
        LOG(WARNING) << "End of cycle";
      } else {
        DFS(next_node, cur_branch, is_on_cur_branch, unvisited, node_names);
      }
    }
  }
  cur_branch->pop_back();
  is_on_cur_branch->at(cur_node) = false;
  unvisited->erase(cur_node);
}

void GraphConstructor::PrintCycles() {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_54(mht_54_v, 1420, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::PrintCycles");

  int num_nodes = outputs_.size();

  std::vector<absl::string_view> node_names;
  node_names.resize(num_nodes);
  for (const auto& named_node : gdef_nodes_) {
    DCHECK_GE(named_node.second.gdef_index, 0);
    DCHECK_LT(named_node.second.gdef_index, num_nodes);
    node_names[named_node.second.gdef_index] = named_node.first;
  }

  absl::flat_hash_set<int> unvisited;
  for (int i = 0; i < num_nodes; i++) {
    unvisited.insert(i);
  }

  while (!unvisited.empty()) {
    int cur_node = *unvisited.begin();
    // Nodes on the current branch of DFS in traversal order. This is used for
    // printing the nodes in the cycle.
    std::vector<int> cur_branch;
    // This is just to make lookups O(1).
    // is_on_cur_branch[i] ==
    //   (std::find(cur_branch.start(),
    //              cur_branch.end(), i) != cur_branch.end())
    std::vector<bool> is_on_cur_branch(num_nodes, false);
    DFS(cur_node, &cur_branch, &is_on_cur_branch, &unvisited, node_names);
  }
}

Status GraphConstructor::Convert() {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_55(mht_55_v, 1453, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::Convert");

  // Import functions before adding nodes, since imported nodes may refer to
  // functions
  if (library()) {
    // TODO(b/135705010): Add rvalue overloads into the function library, to
    // avoid unnecessarily copying `*library()` here.
    TF_RETURN_IF_ERROR(g_->AddFunctionLibrary(*library()));
  }

  std::vector<InputInfo> inputs;
  int processed = 0;

  std::vector<bool> input_already_exists;

  // Process the NodeDefs in topological order.
  // (InitFromEdges() sets this up by filling in ready_ with nodes that have no
  // inputs, pending_counts_ with the number of inputs for each node and
  // outputs_ with the outputs of each node).
  while (!ready_.empty()) {
    int o = *ready_.begin();
    ready_.erase(ready_.begin());
    ++processed;
    inputs.clear();
    bool has_data_back_edge = false;

    NodeDef node_def = consume_node_def(o);

    // input_already_exists[i] is true iff the i-th input of the node we're
    // importing refers to a preexisting node in g_ (i.e. input[i] existed prior
    // to importing node_defs_).  Conversely, input_already_exists[i] is false
    // iff the input refers to a node in node_defs_.
    input_already_exists.clear();
    input_already_exists.resize(node_def.input_size(), false);

    std::string node_name = node_def.name();

    if (opts_.importing) {
      if (opts_.skip_mapped_nodes) {
        bool is_node_mapped = false;
        TF_RETURN_IF_ERROR(IsNodeFullyMapped(node_def, &is_node_mapped));
        if (is_node_mapped) {
          // Skip this node after updating pending_count_ for outputs
          UpdatePendingCountAndReady(o, IsNextIteration(node_def));
          continue;
        }
      }

      if (!opts_.input_map.empty()) {
        // Note that input_already_exists can shrink here
        RemapNodeDefInputs(&node_def, &input_already_exists);
      }
      if (!opts_.control_dependencies.empty()) {
        // Note that input_already_exists can grow here
        AddControlDependencies(&node_def, &input_already_exists);
      }
      if (!opts_.default_device.empty() && node_def.device().empty()) {
        node_def.set_device(opts_.default_device);
      }
    }

    DCHECK_EQ(node_def.input_size(), input_already_exists.size());
    TF_RETURN_IF_ERROR(ValidateColocationConstraints(node_def));
    for (int i = 0; i < node_def.input_size(); ++i) {
      TensorId tensor_id = ParseTensorName(node_def.input(i));
      Node* src_node;
      int src_index;

      if (!input_already_exists[i]) {
        // Locate input in newly-imported nodes
        auto iter = gdef_nodes_.find(tensor_id.node());
        DCHECK(iter != gdef_nodes_.end()) << tensor_id.node();
        src_node = iter->second.node;
        src_index = tensor_id.index();
        if (src_node == nullptr) has_data_back_edge = true;
      } else {
        // Input refers to preexistng node in graph
        auto iter = existing_nodes_.find(tensor_id.node());
        DCHECK(iter != existing_nodes_.end()) << tensor_id.node();
        src_node = iter->second;
        src_index = tensor_id.index();
      }

      if (src_node != nullptr && src_index >= src_node->num_outputs()) {
        std::ostringstream out;
        out << "Node '" << node_def.name() << "': Connecting to invalid output "
            << tensor_id.index() << " of source node " << tensor_id.node()
            << " which has " << src_node->num_outputs() << " outputs.";

        if (src_node->type_string() == "If" ||
            src_node->type_string() == "StatelessIf" ||
            src_node->type_string() == "While" ||
            src_node->type_string() == "StatelessWhile") {
          out << " Try using "
              << "tf.compat.v1.experimental.output_all_intermediates(True).";
        }
        return errors::InvalidArgument(out.str());
      }

      inputs.emplace_back(string(tensor_id.node()), src_node, src_index);
    }

    if (has_data_back_edge && !IsMerge(node_def)) {
      return errors::InvalidArgument(
          "Node '", node_def.name(),
          "' had a back edge, but only Merge nodes can have back edges.");
    }

    Node* node;
    if (opts_.importing) {
      if (!prefix_.empty()) {
        AddPrefixToNodeDef(input_already_exists, &node_def);
      }
      // Note: no need to uniquify names if the prefix already guarantees
      // uniqueness
      if (opts_.uniquify_names && (prefix_.empty() || !opts_.uniquify_prefix)) {
        UniquifyNames(input_already_exists, &node_def);
      }
    }

    if (opts_.importing) {
      TF_RETURN_IF_ERROR(ModifyNodeDefForImport(&node_def));
    } else {
      const OpDef* op_def;
      TF_RETURN_IF_ERROR(
          g_->op_registry()->LookUpOpDef(node_def.op(), &op_def));
      if (opts_.add_default_attributes) {
        AddDefaultsToNodeDef(*op_def, &node_def);
      }
      if (opts_.validate_nodes) {
        TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *op_def));
      }
    }

    TF_RETURN_IF_ERROR(MakeNode(std::move(node_def), &node));

    gdef_nodes_[node_name].node = node;

    // Remove duplicate control inputs before adding edges to the graph. It
    // will allow us to skip expensive duplicates check in 'AddControlEdge'.
    auto first_control = absl::c_find_if(inputs, &InputInfo::IsControlInput);
    auto first_control_copy = first_control;
    std::sort(first_control, inputs.end(), &InputInfo::CompareName);
    inputs.erase(
        std::unique(first_control_copy, inputs.end(), &InputInfo::IsSameName),
        inputs.end());

    // Add edges from inputs to *node to the graph.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].node == nullptr) {
        // Record this back edge, which will be added after all nodes
        // are created.
        back_edges_.emplace_back(inputs[i].name, inputs[i].index, node, i);
      } else if (inputs[i].index == Graph::kControlSlot) {
        g_->AddControlEdge(inputs[i].node, node, kDoNotCheckDuplicates);
      } else {
        TF_RETURN_IF_ERROR(MakeEdge(inputs[i].node, inputs[i].index, node, i));
      }
    }

    TF_RETURN_IF_ERROR(ValidateShape(node));

    // Update pending_count_ for outputs.
    UpdatePendingCountAndReady(o, node->IsNextIteration());
  }

  if (processed < node_def_count()) {
    LOG(WARNING) << "IN " << __func__ << " " << (node_def_count() - processed)
                 << " NODES IN A CYCLE";
    for (int64_t i = 0; i < node_def_count(); i++) {
      if (pending_count_[i] != 0) {
        LOG(WARNING) << "PENDING: " << SummarizeNodeDef(get_node_def(i))
                     << " WITH PENDING COUNT = " << pending_count_[i];
      }
    }
    PrintCycles();
    return errors::InvalidArgument(node_def_count() - processed,
                                   " nodes in a cycle");
  }

  return Status::OK();
}

Status GraphConstructor::AddBackEdges() {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_56(mht_56_v, 1638, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::AddBackEdges");

  // Add the back edges after all nodes are created.
  for (const auto& e : back_edges_) {
    Node* src_node = gdef_nodes_[e.src_name].node;
    if (e.src_index == Graph::kControlSlot) {
      g_->AddControlEdge(src_node, e.dst_node, kDoNotCheckDuplicates);
    } else {
      TF_RETURN_IF_ERROR(
          MakeEdge(src_node, e.src_index, e.dst_node, e.dst_index));
    }

    VLOG(2) << "Add back edge: " << src_node->name() << " -> "
            << e.dst_node->name();
  }
  return Status::OK();
}

Status GraphConstructor::UpdateVersionDef() {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_57(mht_57_v, 1658, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::UpdateVersionDef");

  if (versions() == nullptr) return Status::OK();

  if (!opts_.importing) {
    g_->set_versions(*versions());
    return Status::OK();
  }
  VersionDef g_versions = g_->versions();
  g_versions.set_producer(
      std::min(g_versions.producer(), versions()->producer()));
  g_versions.set_min_consumer(
      std::max(g_versions.min_consumer(), versions()->min_consumer()));
  if (versions()->bad_consumers_size() > 0) {
    std::set<int> bad(g_versions.bad_consumers().begin(),
                      g_versions.bad_consumers().end());
    bad.insert(versions()->bad_consumers().begin(),
               versions()->bad_consumers().end());
    g_versions.clear_bad_consumers();
    for (int v : bad) {
      g_versions.add_bad_consumers(v);
    }
  }
  g_->set_versions(g_versions);
  return Status::OK();
}

Status GraphConstructor::PopulateReturnTensors() {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_58(mht_58_v, 1687, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::PopulateReturnTensors");

  if (opts_.return_tensors.empty()) return Status::OK();
  for (const TensorId& id : opts_.return_tensors) {
    auto iter = opts_.input_map.find(id);
    if (iter == opts_.input_map.end()) {
      // Locate id in imported nodes
      auto iter = gdef_nodes_.find(id.first);
      if (iter == gdef_nodes_.end()) {
        return errors::InvalidArgument("Requested return tensor '",
                                       id.ToString(),
                                       "' not found in graph def");
      }
      int num_outputs = iter->second.node->num_outputs();
      if ((id.second < 0 || id.second >= num_outputs) &&
          id.second != Graph::kControlSlot) {
        return errors::InvalidArgument("Invalid return output ", id.second,
                                       " of node '", id.first, "', which has ",
                                       num_outputs, " output(s)");
      }
      return_tensors_->push_back({iter->second.node, id.second});
    } else {
      // id was remapped to existing node
      TensorId remapped_id = iter->second;
      DCHECK_GT(existing_nodes_.count(remapped_id.first), 0);
      Node* node = existing_nodes_[remapped_id.first];
      return_tensors_->push_back({node, remapped_id.second});
    }
  }
  return Status::OK();
}

Status GraphConstructor::PopulateReturnNodes() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_59(mht_59_v, 1721, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::PopulateReturnNodes");

  if (opts_.return_nodes.empty()) return Status::OK();
  for (StringPiece name : opts_.return_nodes) {
    auto iter = gdef_nodes_.find(name);
    if (iter == gdef_nodes_.end()) {
      return errors::InvalidArgument("Requested return node '", name,
                                     "' not found in graph def");
    }
    return_nodes_->push_back(iter->second.node);
  }
  return Status::OK();
}

Status GraphConstructor::PopulateMissingUnusedInputMapKeys() {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_60(mht_60_v, 1737, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::PopulateMissingUnusedInputMapKeys");

  if (missing_unused_input_map_keys_ == nullptr) return Status::OK();
  for (const auto& input_map_pair : opts_.input_map) {
    TensorId key = input_map_pair.first;
    if (used_input_map_keys_.count(key) > 0) continue;

    auto pair = gdef_nodes_.find(key.first);
    if (pair == gdef_nodes_.end()) {
      // key's node doesn't exist in GraphDef
      missing_unused_input_map_keys_->push_back(key);
      continue;
    }

    // Check that key's index is in bounds. Get the number of outputs from the
    // NodeDef, rather than the imported Node, since the Node may not exist if
    // opts_.skip_mapped_nodes is true.
    const NodeDef& node_def = get_node_def(pair->second.gdef_index);
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(g_->op_registry()->LookUpOpDef(node_def.op(), &op_def));
    int num_outputs;
    TF_RETURN_IF_ERROR(NumOutputsForNode(node_def, *op_def, &num_outputs));
    if (key.second >= num_outputs) {
      // key's index out of bounds
      missing_unused_input_map_keys_->push_back(key);
    }
  }
  return Status::OK();
}

void GraphConstructor::Undo() {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_61(mht_61_v, 1769, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::Undo");

  for (const auto& iter : gdef_nodes_) {
    if (iter.second.node != nullptr) {
      g_->RemoveNode(iter.second.node);
    }
  }
  g_->set_versions(original_versions_);
}

Status GraphConstructor::MakeEdge(Node* src, int output_index, Node* dst,
                                  int input_index) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_62(mht_62_v, 1782, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "GraphConstructor::MakeEdge");

  if (output_index >= src->num_outputs()) {
    return errors::InvalidArgument(
        "Output ", output_index, " of node ", src->name(),
        " does not exist. Node only has ", src->num_outputs(), " outputs.");
  }
  if (input_index >= dst->num_inputs()) {
    return errors::InvalidArgument(
        "Input ", input_index, " of node ", dst->name(),
        " does not exist. Node only has ", dst->num_inputs(), " inputs.");
  }

  DataType src_out = src->output_type(output_index);
  DataType dst_in = dst->input_type(input_index);
  if (!TypesCompatible(dst_in, src_out)) {
    return errors::InvalidArgument(
        "Input ", input_index, " of node ", dst->name(), " was passed ",
        DataTypeString(src_out), " from ", src->name(), ":", output_index,
        " incompatible with expected ", DataTypeString(dst_in), ".");
  }
  g_->AddEdge(src, output_index, dst, input_index);
  return Status::OK();
}

}  // namespace

Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                              const GraphDef& gdef, Graph* g) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_63(mht_63_v, 1812, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "ConvertGraphDefToGraph");

  ShapeRefiner refiner(gdef.versions().producer(), g->op_registry());
  return GraphConstructor::Construct(
      opts, gdef.node(), &gdef.versions(), &gdef.library(), g, &refiner,
      /*return_tensors=*/nullptr, /*return_nodes=*/nullptr,
      /*missing_unused_input_map_keys=*/nullptr);
}

Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                              GraphDef&& gdef, Graph* g) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_64(mht_64_v, 1824, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "ConvertGraphDefToGraph");

  ShapeRefiner refiner(gdef.versions().producer(), g->op_registry());
  return GraphConstructor::Construct(opts, std::move(gdef), g, &refiner,
                                     /*return_tensors=*/nullptr,
                                     /*return_nodes=*/nullptr,
                                     /*missing_unused_input_map_keys=*/nullptr);
}

Status ConvertNodeDefsToGraph(const GraphConstructorOptions& opts,
                              gtl::ArraySlice<NodeDef> nodes, Graph* g) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_65(mht_65_v, 1836, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "ConvertNodeDefsToGraph");

  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, g->op_registry());
  // TODO(irving): Copy will go away once NodeInfo exists
  std::vector<const NodeDef*> node_defs;
  node_defs.reserve(nodes.size());
  for (const auto& n : nodes) {
    node_defs.push_back(&n);
  }
  return GraphConstructor::Construct(opts, node_defs, nullptr, nullptr, g,
                                     &refiner, /*return_tensors=*/nullptr,
                                     /*return_nodes=*/nullptr,
                                     /*missing_unused_input_map_keys=*/nullptr);
}

Status ImportGraphDef(const ImportGraphDefOptions& opts, const GraphDef& gdef,
                      Graph* g, ShapeRefiner* refiner,
                      ImportGraphDefResults* results) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_66(mht_66_v, 1855, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "ImportGraphDef");

  if (!opts.return_tensors.empty()) {
    if (results == nullptr) {
      return errors::InvalidArgument(
          "results argument to ImportGraphDef() must be non-null if "
          "opts.return_tensors is non-empty");
    }
  }

  if (!opts.return_nodes.empty()) {
    if (opts.skip_mapped_nodes) {
      return errors::InvalidArgument(
          "Requesting return_nodes with skip_mapped_nodes set is not currently "
          "supported");
    }
    if (results == nullptr) {
      return errors::InvalidArgument(
          "results argument to ImportGraphDef() must be non-null if "
          "opts.return_nodes is non-empty");
    }
  }

  if (results != nullptr) {
    if (!results->return_tensors.empty() || !results->return_nodes.empty() ||
        !results->missing_unused_input_map_keys.empty()) {
      return errors::InvalidArgument(
          "All fields in results argument to ImportGraphDef() must be empty.");
    }
  }

  ShapeRefiner default_refiner(gdef.versions().producer(), g->op_registry());
  if (refiner == nullptr) {
    refiner = &default_refiner;
  } else {
    // Log a warning if we are importing a GraphDef at an older
    // producer version after already having added non-source/sink
    // nodes to the graph in the past.
    if (gdef.versions().producer() > 0 &&
        gdef.versions().producer() < refiner->graph_def_version() &&
        g->num_nodes() > 2) {
      LOG(WARNING) << "Importing a graph with a lower producer version "
                   << gdef.versions().producer()
                   << " into an existing graph with producer version "
                   << refiner->graph_def_version() << ". Shape inference will "
                   << "have run different parts of the graph with different "
                   << "producer versions.";
    }
  }

  // Set the graph def version of the refiner as the min of the
  // current value and the version from the graph we are about to
  // import.
  //
  // Note: to match Run() semantics, we should re-run shape inference
  // on the entire graph if the producer version has changed.  For now
  // we log the warning above.
  refiner->set_graph_def_version(
      std::min(refiner->graph_def_version(), gdef.versions().producer()));

  if (results == nullptr) {
    return GraphConstructor::Construct(opts, gdef.node(), &gdef.versions(),
                                       &gdef.library(), g, refiner, nullptr,
                                       nullptr, nullptr);
  } else {
    return GraphConstructor::Construct(
        opts, gdef.node(), &gdef.versions(), &gdef.library(), g, refiner,
        &results->return_tensors, &results->return_nodes,
        &results->missing_unused_input_map_keys);
  }
}

void CopyGraph(const Graph& src, Graph* dest) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_constructorDTcc mht_67(mht_67_v, 1929, "", "./tensorflow/core/common_runtime/graph_constructor.cc", "CopyGraph");
 dest->Copy(src); }

}  // namespace tensorflow
