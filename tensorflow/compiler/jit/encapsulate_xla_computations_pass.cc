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
class MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc() {
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

#include "tensorflow/compiler/jit/encapsulate_xla_computations_pass.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

const char* const kXlaClusterOutput = "XlaClusterOutput";

bool IsCpuGpuCompile(const Graph* graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "IsCpuGpuCompile");

  for (Node* n : graph->nodes()) {
    string name;
    // Only consider nodes being compiled.
    if (!TryGetNodeAttr(n->attrs(), kXlaClusterIdAttr, &name)) continue;
    // Early return for any node with a device that is not a CPU or GPU.
    DeviceNameUtils::ParsedName parsed;
    if (DeviceNameUtils::ParseFullName(n->requested_device(), &parsed)) {
      if (parsed.type != DEVICE_CPU && parsed.type != DEVICE_GPU) {
        return false;
      }
    }
  }
  return true;
}

// Checks if a graph node is marked to be a guaranteed constant.
bool is_guaranteed_constant(const Node& n) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "is_guaranteed_constant");

  bool guaranteed_constant = false;
  if (!TryGetNodeAttr(n.attrs(), "_is_guaranteed_constant",
                      &guaranteed_constant)) {
    return false;
  }
  return guaranteed_constant;
}

// Finds the `index` of an _Arg or _Retval node.
Status GetIndexAttr(const Node& n, int num_args, int* index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_2(mht_2_v, 245, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "GetIndexAttr");

  TF_RETURN_IF_ERROR(GetNodeAttr(n.attrs(), "index", index));
  if (*index < 0 || *index >= num_args) {
    return errors::InvalidArgument("Invalid ", n.type_string(), " number ",
                                   *index);
  }
  return Status::OK();
}

// Returns the data type of the destination of an edge.
DataType EdgeType(const Edge* edge) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_3(mht_3_v, 258, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "EdgeType");

  return edge->dst()->input_type(edge->dst_input());
}

// Adds the control inputs of `node` to `*deps`.
void AddControlInputs(const Node& node, absl::flat_hash_set<Node*>* deps) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_4(mht_4_v, 266, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "AddControlInputs");

  for (const Edge* edge : node.in_edges()) {
    if (edge->IsControlEdge()) {
      deps->insert(edge->src());
    }
  }
}

// Adds the control outputs of `node` to `*deps`.
void AddControlOutputs(const Node& node, absl::flat_hash_set<Node*>* deps) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_5(mht_5_v, 278, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "AddControlOutputs");

  for (const Edge* edge : node.out_edges()) {
    if (edge->IsControlEdge()) {
      deps->insert(edge->dst());
    }
  }
}

// Rewrite function to be passed to EncapsulateSubgraphsInFunctions that sorts
// the arguments into the order expected by XlaLaunch computations:
// 1) arguments
// 2) resource variable arguments
// See the documentation of EncapsulateSubgraphsInFunctions for the meaning
// of the arguments.
//
// TODO(b/113166435): Ordering constraints on XlaLaunch op can be relaxed.
Status RewriteSubgraph(const std::vector<OutputTensor>& arg_source_tensors,
                       std::unique_ptr<Graph>* graph_ptr,
                       std::vector<int>* input_permutation,
                       std::vector<int>* output_permutation,
                       NodeDef* call_def) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_6(mht_6_v, 301, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "RewriteSubgraph");

  Graph* graph = graph_ptr->get();
  const int num_args = input_permutation->size();
  const int num_retvals = output_permutation->size();

  std::vector<Node*> args;
  std::vector<Node*> retvals;
  args.reserve(num_args);
  retvals.reserve(num_retvals);
  for (Node* n : graph->nodes()) {
    if (n->type_string() == "_Arg") {
      // Check if this is a guaranteed constant.
      if (is_guaranteed_constant(*n)) {
        return errors::InvalidArgument(
            "Guaranteed constants are not supported (", n->name(), ")");
      }
      args.push_back(n);
    } else if (n->type_string() == "_Retval") {
      retvals.push_back(n);
    }
  }

  if (std::find(args.begin(), args.end(), nullptr) != args.end()) {
    return errors::InvalidArgument("Missing or non-consecutive arguments");
  }

  // Reorders the arguments.
  std::sort(args.begin(), args.end(), [&](Node* a, Node* b) {
    // Non-resources appear before resources
    bool a_is_resource = (a->output_type(0) == DT_RESOURCE);
    bool b_is_resource = (b->output_type(0) == DT_RESOURCE);
    // Uses the name as a tiebreaker so the output is deterministic.
    StringPiece a_name(a->name());
    StringPiece b_name(b->name());
    return std::tie(a_is_resource, a_name) < std::tie(b_is_resource, b_name);
  });

  // Sorts the retvals by name so the order is deterministic.
  std::sort(retvals.begin(), retvals.end(),
            [](Node* a, Node* b) { return a->name() < b->name(); });

  // Computes the permutation to produce the correct argument order, and update
  // the argument indices.
  int variable_start_index = num_args;
  for (int i = 0; i < num_args; ++i) {
    int index;
    TF_RETURN_IF_ERROR(GetIndexAttr(*args[i], num_args, &index));
    if (args[i]->output_type(0) == DT_RESOURCE &&
        variable_start_index == num_args) {
      variable_start_index = i;
    }
    (*input_permutation)[index] = i;
    args[i]->AddAttr("index", i);
  }
  VLOG(4) << "variable_start_index: " << variable_start_index;

  // Computes the permutation to produce the correct retval order, and update
  // the argument indices.
  for (int i = 0; i < num_retvals; ++i) {
    int index;
    TF_RETURN_IF_ERROR(GetIndexAttr(*retvals[i], num_retvals, &index));
    (*output_permutation)[index] = i;
    retvals[i]->AddAttr("index", i);
  }

  AddNodeAttr(kXlaClusterIdAttr, call_def->name(), call_def);
  AddNodeAttr("_variable_start_index", variable_start_index, call_def);

  // Uniquify the function name by computing a fingerprint of the function.
  // Nondeterminism in serialization would not lead to incorrect results, but
  // may cause spurious cache misses.
  TF_ASSIGN_OR_RETURN(uint64 fingerprint, FingerprintGraph(*graph));
  VLOG(1) << "Subgraph fingerprint:" << fingerprint;
  call_def->set_op(absl::StrCat(call_def->op(), "_", fingerprint));
  return Status::OK();
}

}  // namespace

/*static*/ Status EncapsulateXlaComputationsPass::Encapsulate(
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_7(mht_7_v, 384, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "EncapsulateXlaComputationsPass::Encapsulate");

  // Check for undeclared outputs before Encapsulation, so we can give a better
  // error message.
  // TODO(phawkins): merge this with the encapsulation code to avoid the extra
  // O(n) pass over the edges.
  for (const Edge* e : (*graph)->edges()) {
    if (!e->IsControlEdge() &&
        e->src()->attrs().Find(kXlaClusterIdAttr) != nullptr &&
        e->dst()->attrs().Find(kXlaClusterIdAttr) == nullptr &&
        e->dst()->type_string() != kXlaClusterOutput) {
      return errors::InvalidArgument(
          "Undeclared output of XLA computation. Some common causes of this "
          "error are: 1) variable initializers that depend on the XLA "
          "computation; 2) gradient computations that depend on the XLA "
          "computation, which can be mitigated by moving gradient computations "
          "inside XLA computation. Offending edge: ",
          e->src()->name(), ":", e->src_output(), " -> ", e->dst()->name(), ":",
          e->dst_input());
    }
  }

  auto output = absl::make_unique<Graph>((*graph)->op_registry());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      EncapsulateSubgraphsInFunctions(
          kXlaClusterIdAttr, **graph, RewriteSubgraph,
          /*reuse_existing_functions=*/true, &output, flib_def),
      "EncapsulateXlaComputationsPass failed");
  graph->swap(output);
  return Status::OK();
}

/*static*/ Status EncapsulateXlaComputationsPass::BuildXlaLaunchOps(
    Graph* graph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_8(mht_8_v, 419, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "EncapsulateXlaComputationsPass::BuildXlaLaunchOps");

  // Finds all of the XlaLaunch function calls, to avoid mutating the graph
  // while iterating.
  std::vector<Node*> launch_nodes;
  for (Node* n : graph->nodes()) {
    const string& name = GetNodeAttrString(n->attrs(), kXlaClusterIdAttr);
    if (!name.empty()) {
      launch_nodes.push_back(n);
    }
  }

  // Replaces each launch function call together with its neighboring
  // XlaClusterOutput nodes with a XlaLaunch node.
  for (Node* launch : launch_nodes) {
    int variable_start_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(launch->attrs(), "_variable_start_index",
                                   &variable_start_index));

    std::vector<const Edge*> in_edges;
    TF_RETURN_IF_ERROR(launch->input_edges(&in_edges));

    const int num_inputs = in_edges.size();
    const int num_variables = num_inputs - variable_start_index;
    const int num_args = variable_start_index;

    VLOG(4) << "Launch node '" << launch->name() << "'"
            << " input edges: " << in_edges.size() << " num_args: " << num_args
            << " num_variables: " << num_variables;

    std::vector<Node*> nodes_to_remove = {launch};

    // Data and control inputs to the new XlaLaunch node.
    std::vector<std::pair<Node*, int>> data_inputs(num_inputs);
    absl::flat_hash_set<Node*> control_inputs;
    DataTypeVector arg_types(num_args);

    AddControlInputs(*launch, &control_inputs);

    for (int i = 0; i < num_args; ++i) {
      const Edge* edge = in_edges[i];
      data_inputs[i] = {edge->src(), edge->src_output()};
      arg_types[i] = EdgeType(edge);
    }

    // Appends the variable inputs.
    for (int i = 0; i < num_variables; ++i) {
      int pos = variable_start_index + i;
      const Edge* edge = in_edges[pos];
      data_inputs[pos] = {edge->src(), edge->src_output()};
    }

    // Outputs.
    const int num_outputs = launch->output_types().size();
    absl::flat_hash_set<Node*> control_outputs;
    std::vector<std::vector<std::pair<Node*, int>>> data_outputs(num_outputs);
    DataTypeVector output_types(num_outputs);

    for (const Edge* le : launch->out_edges()) {
      if (le->IsControlEdge()) {
        control_outputs.insert(le->dst());
      } else {
        TF_RET_CHECK(le->src_output() < num_outputs);
        Node* output_node = le->dst();

        TF_RET_CHECK(output_node->type_string() == kXlaClusterOutput)
            << le->DebugString();
        nodes_to_remove.push_back(output_node);

        for (const Edge* oe : output_node->out_edges()) {
          TF_RET_CHECK(!oe->IsControlEdge());
          data_outputs[le->src_output()].push_back(
              {oe->dst(), oe->dst_input()});
        }
        output_types[le->src_output()] = output_node->input_type(0);

        AddControlOutputs(*output_node, &control_outputs);
      }
    }

    NodeDef def;
    def.set_name(launch->name());
    MergeDebugInfo(NodeDebugInfo(launch->def()), &def);

    // Target the XLA CPU/GPU backends.
    VLOG(2) << "Replacing with XlaLaunch";
    VLOG(2) << "Device is " << launch->requested_device();
    def.set_op("XlaLaunch");
    def.set_device(launch->requested_device());
    AddNodeAttr("Tconstants", DataTypeVector{}, &def);
    AddNodeAttr("Targs", arg_types, &def);
    AddNodeAttr("Nresources", num_variables, &def);
    AddNodeAttr("Tresults", output_types, &def);
    NameAttrList function;
    function.set_name(launch->type_string());
    AddNodeAttr("function", function, &def);

    for (Node* node : nodes_to_remove) {
      VLOG(2) << "Deleting node " << node->DebugString();
      // Ensure that we do not attempt to add control edges to nodes that are
      // deleted.
      control_inputs.erase(node);
      control_outputs.erase(node);
      graph->RemoveNode(node);
    }

    TF_ASSIGN_OR_RETURN(Node * xla_launch, graph->AddNode(def));
    for (int i = 0, end = data_inputs.size(); i < end; ++i) {
      graph->AddEdge(data_inputs[i].first, data_inputs[i].second, xla_launch,
                     i);
    }
    for (Node* n : control_inputs) {
      graph->AddControlEdge(n, xla_launch);
    }
    for (int i = 0, end = data_outputs.size(); i < end; ++i) {
      for (const auto& successor : data_outputs[i]) {
        graph->AddEdge(xla_launch, i, successor.first, successor.second);
      }
    }
    for (Node* n : control_outputs) {
      graph->AddControlEdge(xla_launch, n);
    }
  }
  return Status::OK();
}

Status EncapsulateXlaComputationsPass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSencapsulate_xla_computations_passDTcc mht_9(mht_9_v, 548, "", "./tensorflow/compiler/jit/encapsulate_xla_computations_pass.cc", "EncapsulateXlaComputationsPass::Run");

  VLOG(1) << "EncapsulateXlaComputations(): "
          << DumpGraphToFile("encapsulate_xla_computations_before",
                             **options.graph, options.flib_def);

  const char* additional_help =
      IsCpuGpuCompile(options.graph->get())
          ? xla::status_macros::kPossibleAutoJitAlternative
          : "";

  TF_RETURN_WITH_CONTEXT_IF_ERROR(Encapsulate(options.graph, options.flib_def),
                                  additional_help);
  VLOG(1) << "EncapsulateXlaComputations() half-way: "
          << DumpGraphToFile("encapsulate_xla_computations_halfway",
                             **options.graph, options.flib_def);

  TF_RETURN_WITH_CONTEXT_IF_ERROR(BuildXlaLaunchOps(options.graph->get()),
                                  additional_help);
  VLOG(1) << "EncapsulateXlaComputations() finished: "
          << DumpGraphToFile("encapsulate_xla_computations_after",
                             **options.graph, options.flib_def);
  return Status::OK();
}

}  // namespace tensorflow
