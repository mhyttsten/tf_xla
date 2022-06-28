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

#ifndef TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh() {
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

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// ValidateConfig returns OK iff config is valid.
Status ValidateConfig(const tf2xla::Config& config);

// Modifies <graph_def> to include placeholders for each fed tensor, and
// update references to the fed tensors to refer to the placeholders.
// The existing nodes referenced by the feeds are not removed or modified
// (except where their input edges are modified by the replacement of other
// feeds).
Status AddPlaceholdersForFeeds(
    const tf2xla::Config& config, const OpRegistryInterface* op_registry,
    std::unordered_map<string, string>* feed_remapping, GraphDef* graph_def);

// Returns in <out> a copy of <in>, pruned to only include fetches from
// <config>.
Status PruneGraphDefInto(const tf2xla::Config& config, const GraphDef& in,
                         GraphDef* out);

// Returns node:port for the given <id>.
string TensorIdToString(const tf2xla::TensorId& id);

// Updates the sharding of <n> based on the sharding of its neighbors.
// If <out_edges> is true, outgoing edges from <n> are considered; else incoming
// edges are considered.
Status SetNodeShardingFromNeighbors(Node* n, bool out_edges);

// Add an allowed data type to the AttrConstraint with the given name.
void AddDtypeToKernelDefConstraint(absl::string_view name, DataType dtype,
                                   KernelDef* kdef);

// Returns the next random seed to use for seeding xla rng.
uint32 GetXLARandomSeed();

// Indicates how a FunctionDef is associated with a graph node (e.g. the node is
// a function call, or the node has function attrs).
class AssociatedFunctionInfo {
 public:
  enum AssociatedFunctionType {
    kFunctionAttr = 0,
    kFunctionCallNode = 1,
    kSymbolicGradient = 2,
  };

  // The function is an attr of the node.
  static AssociatedFunctionInfo FunctionAttr(const string& func_name,
                                             const AttrValueMap& attrs,
                                             const string& attr_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("func_name: \"" + func_name + "\"");
   mht_0_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_0(mht_0_v, 249, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "FunctionAttr");

    return AssociatedFunctionInfo(kFunctionAttr, func_name, attrs, attr_name);
  }

  // The node is a function call.
  static AssociatedFunctionInfo FunctionCall(const string& func_name,
                                             const AttrValueMap& attrs) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_1(mht_1_v, 259, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "FunctionCall");

    // attr_name will not be used in this case.
    return AssociatedFunctionInfo(kFunctionCallNode, func_name, attrs,
                                  /*attr_name=*/"");
  }

  // The node is a SymbolicGradient op.
  static AssociatedFunctionInfo SymbolicGradient(const string& func_name,
                                                 const AttrValueMap& attrs) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_2(mht_2_v, 271, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "SymbolicGradient");

    // attr_name will not be used in this case.
    return AssociatedFunctionInfo(kSymbolicGradient, func_name, attrs,
                                  /*attr_name=*/"");
  }

  AssociatedFunctionType type() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_3(mht_3_v, 280, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "type");
 return type_; }

  const string& func_name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_4(mht_4_v, 285, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "func_name");
 return func_name_; }

  const string& attr_name() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_5(mht_5_v, 290, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "attr_name");
 return attr_name_; }

  const AttrValueMap& attrs() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_6(mht_6_v, 295, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "attrs");
 return attrs_; }

 private:
  AssociatedFunctionInfo(AssociatedFunctionType type, const string& func_name,
                         const AttrValueMap& attrs, const string& attr_name)
      : type_(type),
        func_name_(func_name),
        attrs_(attrs),
        attr_name_(attr_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("func_name: \"" + func_name + "\"");
   mht_7_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_7(mht_7_v, 308, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "AssociatedFunctionInfo");
}

  // Available for all instances.
  AssociatedFunctionType type_;
  string func_name_;
  AttrValueMap attrs_;

  // Only available if the function is defined in an attr.
  string attr_name_;
};

// Returns if the NodeDef has associated function.
bool HasAssociatedFunction(const NodeDef& node_def,
                           const FunctionLibraryDefinition* fld);

// Gets functions associated with the node. Current cases:
// 1. For function call node, its function name;
// 2. For SymbolicGradient op, returned func_name will be "SymbolicGradient",
//    and returned attrs will be this node's attributes;
// 3. For nodes like XlaWhile/XlaIf, all their function attributes.
std::vector<AssociatedFunctionInfo> GetAssociatedFunctions(
    const Node& node, const FunctionLibraryDefinition* fld);

// Changes associated functions for the node. Current cases:
// 1. For function call node, creates a new node with the new function name and
//    remove the old node;
// 2. For SymbolicGradient op, add or replace GradientDef in
//    FunctionLibraryDefinition;
// 3. For nodes like XlaWhile/XlaIf, modify their function attributes.
Status RewriteAssociatedFunction(
    Graph* graph, Node* node, FunctionLibraryDefinition* fld,
    const AssociatedFunctionInfo& associated_function,
    const string& rewritten_function_name);

// Attribute to mark nodes to be executed on host.
extern const char kXlaOutsideCompilationAttrName[];

// Class to act as cache for FunctionLibraryRuntime::Handle objects.
class CachedFunctionHandles {
 public:
  CachedFunctionHandles(FunctionLibraryRuntime* flr) : flr_(flr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_8(mht_8_v, 351, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "CachedFunctionHandles");
}

  // Populates `handle` for requested function and attributes. If we have
  // instantiated the function with the same attributes before, `handle` will be
  // cached handle; otherwise instantiate the function and populate `handle`.
  Status GetOrInstantiate(const string& func_name, AttrSlice attrs,
                          FunctionLibraryRuntime::Handle* handle);

  // Releases all handles in the cache. Returns first non-OK status if any;
  // returns OK otherwise.
  Status ReleaseAllHandles();

  ~CachedFunctionHandles() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_9(mht_9_v, 366, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "~CachedFunctionHandles");
 ReleaseAllHandles().IgnoreError(); }

 private:
  FunctionLibraryRuntime* flr_;
  std::map<string, FunctionLibraryRuntime::Handle> handles_;

  TF_DISALLOW_COPY_AND_ASSIGN(CachedFunctionHandles);
};

// Struct for node's output edge info.
struct OutEdgeInfo {
  Node* dst;
  int src_output, dst_input;
};

// Replaces node `n` with a new node whose NodeDef is `node_def`.
StatusOr<Node*> ReplaceNode(Graph* g, Node* n, const NodeDef& node_def);

// Helper function that builds an Identity node.
StatusOr<Node*> BuildIdentityNode(Graph* graph, const string& node_name,
                                  DataType dtype, const Node* input,
                                  absl::optional<string> requested_device);

// For "If"/"While" nodes, if some of their inputs are Const nodes, rewrite
// body functions to use the Const nodes instead of original _Arg nodes.
//
// For example, say we have the following computation:
//     shape = constant_op.constant([1])
//     return tf.cond(pred, lambda: tf.ones(shape), lambda: tf.zeros(shape))
// If we do not rewrite then/else function, they will use _Arg node as shape
// input for tf.ones/tf.zeros. But XLA requires that shape input to be compile
// time constant, so XLA compilation will fail. This rewriting process will
// change the shape input to Const node.
Status PropagateConstIntoFunctionalNodes(
    Graph* g, const FunctionLibraryDefinition* lookup_fld,
    FunctionLibraryDefinition* fld);

// Prunes unreachable FunctionDefs from FunctionLibraryDefinition.
Status PruneUnreachableFunctionsFromGraph(const Graph& g,
                                          FunctionLibraryDefinition* fld);

// Finds the following pattern in the graph:
// 1) EmptyTensorList -> forward While op -> backward While op,
// 2) in forward While op, a Const node is pushed,
// 3) in backward While op, data is popped from the tensor list.
// And rewrites backward While op to use Const node instead of TensorListPopBack
// result.
// TODO(b/128633174) remove the TensorList and related TensorList ops.
Status RewriteTensorListWithConstElement(Graph* g,
                                         FunctionLibraryDefinition* fld);

extern const char kTpuReplicateAttrName[];

inline bool IsConstTraversableOpType(const Node* node) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_utilDTh mht_10(mht_10_v, 422, "", "./tensorflow/compiler/tf2xla/tf2xla_util.h", "IsConstTraversableOpType");

  return node->type_string() == "Identity" ||
         node->type_string() == "IdentityN" || node->IsWhileNode();
}

// Determines whether a loop body is invariant for the given argument index.
StatusOr<bool> IsLoopInvariant(const FunctionBody* loop_body, int index,
                               const FunctionLibraryDefinition* lookup_fld);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_
