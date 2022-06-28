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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSaccumulate_n_optimizerDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSaccumulate_n_optimizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSaccumulate_n_optimizerDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace {

static constexpr const char* const kParallelIterationsAttrName =
    "parallel_iterations";

Tensor make_zeros(const DataType& dtype, const TensorShapeProto& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSaccumulate_n_optimizerDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/common_runtime/accumulate_n_optimizer.cc", "make_zeros");

  Tensor tensor(dtype, TensorShape(shape));

  // Conveniently, all numeric data types have 0x0 == zero.  Otherwise we would
  // need a giant switch statement here.
  memset(const_cast<char*>(tensor.tensor_data().data()), 0,
         tensor.tensor_data().size());

  return tensor;
}

// Replaces occurrences of the "AccumulateNV2" stub operator with a graph of
// lower-level ops. The graph is equivalent (modulo certain corner cases)
// to the semantics of the original accumulate_n() Python op in math_ops.py.
// Implementing the op with a rewrite allows this new variant of accumulate_n
// to be differentiable.
//
// The binary code that generates AccumulateNV2 stub ops is located in a
// dynamic library built out of tensorflow/contrib/framework. Ideally, this
// class would also be in contrib, but calls to REGISTER_OPTIMIZATION() from
// third-party libraries aren't currently supported.
class AccumulateNV2RemovePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSaccumulate_n_optimizerDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/common_runtime/accumulate_n_optimizer.cc", "Run");

    // TODO(freiss.oss@gmail.com): Substantial shared code with
    // ParallelConcatRemovePass::Run(). Consider refactoring if someone makes
    // a third similar rewrite.
    if (options.graph == nullptr) {
      // TODO(apassos) returning OK feels weird here as we can't do anything
      // without a graph, but some tests require this.
      return Status::OK();
    }

    Graph* g = options.graph->get();
    if (g == nullptr) {
      return errors::Internal(
          "AccumulateNV2 removal should happen before partitioning and a "
          "graph should be available.");
    }

    // Build up a todo list of ops to replace, *then* modify the graph
    gtl::InlinedVector<Node*, 2> matches;
    for (Node* n : g->op_nodes()) {
      if (n->type_string() == "AccumulateNV2") {
        matches.push_back(n);
      }
    }
    if (matches.empty()) return Status::OK();

    std::vector<ControlFlowInfo> control_flow_info;
    TF_RETURN_IF_ERROR(BuildControlFlowInfo(g, &control_flow_info));

    for (Node* n : matches) {
      // Temporary variables do not work inside while loops with parallel
      // iterations. If the `AccumulateNV2` node is executed inside a loop, we
      // rewrite it into 'AddN' node.
      const Node* frame = control_flow_info[n->id()].frame;
      bool is_in_while_loop = frame->id() != Graph::kSourceId;

      // With `parallel_iterations == 1` it's safe to use TemporaryVariable.
      if (is_in_while_loop) {
        int parallel_iterations;
        bool found = TryGetNodeAttr(frame->attrs(), kParallelIterationsAttrName,
                                    &parallel_iterations);
        if (found && parallel_iterations == 1) {
          is_in_while_loop = false;
        }
      }

      if (is_in_while_loop) {
        TF_RETURN_IF_ERROR(RewriteIntoAddN(n, g));
      } else {
        TF_RETURN_IF_ERROR(RewriteIntoTempVariable(n, g));
      }
    }
    return Status::OK();
  }

  Status RewriteIntoTempVariable(Node* n, Graph* g) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSaccumulate_n_optimizerDTcc mht_2(mht_2_v, 279, "", "./tensorflow/core/common_runtime/accumulate_n_optimizer.cc", "RewriteIntoTempVariable");

    VLOG(3) << "Rewrite AccumulateNV2 into TemporaryVariable and Assign: "
            << SummarizeNode(*n);

    AttrSlice n_attrs = n->attrs();
    auto base_make_node = [n, &n_attrs](const string& op, const string& name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + op + "\"");
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSaccumulate_n_optimizerDTcc mht_3(mht_3_v, 289, "", "./tensorflow/core/common_runtime/accumulate_n_optimizer.cc", "lambda");

      NodeDebugInfo debug_info(*n);
      NodeBuilder node_builder(name, op, OpRegistry::Global(), &debug_info);

      // The pieces of AccumulateNV2 should all be on the same node.
      node_builder.Device(n->requested_device());
      const string& colo = GetNodeAttrString(n_attrs, kColocationAttrName);
      if (!colo.empty()) {
        node_builder.Attr(kColocationAttrName, colo);
      }
      return node_builder;
    };
    auto make_node = [n, g, &base_make_node](string op) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSaccumulate_n_optimizerDTcc mht_4(mht_4_v, 305, "", "./tensorflow/core/common_runtime/accumulate_n_optimizer.cc", "lambda");

      return base_make_node(
          op, g->NewName(strings::StrCat(n->name(), "/Internal")));
    };

    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "T", &dtype));
    TensorShapeProto shape;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "shape", &shape));

    std::vector<const Edge*> data_edges, control_edges;
    for (const Edge* input_edge : n->in_edges()) {
      if (input_edge->IsControlEdge()) {
        control_edges.push_back(input_edge);
      } else {
        data_edges.push_back(input_edge);
      }
    }

    // Create the following ops to replace the AccumulateNV2 placeholder:
    Node* create_accumulator = nullptr;            // TemporaryVariable op
    Node* initial_val = nullptr;                   // Const op
    Node* initialize_accumulator = nullptr;        // Assign op
    std::vector<Node*> add_values_to_accumulator;  // AssignAdd ops
    Node* clean_up_accumulator = nullptr;          // DestroyTemporaryVariable

    const string accumulator_name =
        strings::StrCat(n->name(), "/Internal/Accumulator");
    TensorShapeProto variable_shape;
    variable_shape.add_dim()->set_size(0);
    TF_RETURN_IF_ERROR(make_node("TemporaryVariable")
                           .Attr("shape", variable_shape)
                           .Attr("dtype", dtype)
                           .Attr("var_name", accumulator_name)
                           .Finalize(g, &create_accumulator));
    PartialTensorShape partial_shape(shape);
    // Make a Fill operation to make a zero tensor with the shape of the first
    // input.
    Node* shape_node;
    TF_RETURN_IF_ERROR(
        make_node("Shape")
            .Input(data_edges[0]->src(), data_edges[0]->src_output())
            .Finalize(g, &shape_node));
    Node* zero;
    TF_RETURN_IF_ERROR(make_node("Const")
                           .Attr("value", make_zeros(dtype, TensorShapeProto()))
                           .Attr("dtype", dtype)
                           .Finalize(g, &zero));
    TF_RETURN_IF_ERROR(make_node("Fill")
                           .Input(shape_node)
                           .Input(zero)
                           .Finalize(g, &initial_val));
    TF_RETURN_IF_ERROR(make_node("Assign")
                           .Attr("T", dtype)
                           .Input(create_accumulator)  // ref: Ref(T)
                           .Input(initial_val)         // value: T
                           .Attr("validate_shape", false)
                           .Finalize(g, &initialize_accumulator));
    for (int i = 0; i < data_edges.size(); ++i) {
      Node* assignAdd;
      TF_RETURN_IF_ERROR(make_node("AssignAdd")
                             .Attr("T", dtype)
                             .Attr("use_locking", true)
                             .Input(initialize_accumulator)  // ref: Ref(T)
                             .Input(data_edges[i]->src(),
                                    data_edges[i]->src_output())  // value: T
                             .Finalize(g, &assignAdd));

      add_values_to_accumulator.push_back(assignAdd);
    }

    // Note that we use the original placeholder op's name here
    TF_RETURN_IF_ERROR(base_make_node("DestroyTemporaryVariable", n->name())
                           .Attr("T", dtype)
                           .Attr("var_name", accumulator_name)
                           .Input(initialize_accumulator)
                           .Finalize(g, &clean_up_accumulator));

    // Add edges to the graph to ensure that operations occur in the right
    // order:
    // 1. Do anything that had a control edge to the AccumulateNV2 placeholder
    // 2. Initialize accumulator
    // 3. Add input values to accumulator (already handled by data edges
    //    added above)
    // 4. Reclaim the buffer that held the accumulator
    // 5. Do anything that depended on the AccumulateNV2 placeholder
    for (const Edge* control_edge : control_edges) {
      g->AddControlEdge(control_edge->src(), initialize_accumulator);
    }

    for (Node* assign_add : add_values_to_accumulator) {
      g->AddControlEdge(assign_add, clean_up_accumulator);
    }

    for (const Edge* out_edge : n->out_edges()) {
      if (out_edge->IsControlEdge()) {
        g->AddControlEdge(clean_up_accumulator, out_edge->dst());
      } else {
        g->AddEdge(clean_up_accumulator, 0, out_edge->dst(),
                   out_edge->dst_input());
      }
    }

    // Remove the original AccumulateNV2 placeholder op.
    // This removal modifies the op and must happen after we have finished
    // using its incoming/outgoing edge sets.
    g->RemoveNode(n);

    return Status::OK();
  }

  Status RewriteIntoAddN(Node* n, Graph* g) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSaccumulate_n_optimizerDTcc mht_5(mht_5_v, 419, "", "./tensorflow/core/common_runtime/accumulate_n_optimizer.cc", "RewriteIntoAddN");

    VLOG(3) << "Rewrite AccumulateNV2 into AddN: " << SummarizeNode(*n);

    AttrSlice n_attrs = n->attrs();
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "T", &dtype));
    int num_inputs;
    TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "N", &num_inputs));

    Node* add_n_node = nullptr;

    std::vector<NodeBuilder::NodeOut> data_inputs;
    std::vector<Node*> control_inputs;
    data_inputs.reserve(n->num_inputs());
    control_inputs.reserve(n->in_edges().size() - n->num_inputs());
    for (const Edge* in_edge : n->in_edges()) {
      if (in_edge->IsControlEdge()) {
        control_inputs.push_back(in_edge->src());
      } else {
        data_inputs.emplace_back(in_edge->src(), in_edge->src_output());
      }
    }

    // Rewrite `AccumulateNV2` node into `AddN` node.
    NodeDebugInfo debug_info(*n);
    NodeBuilder builder =
        NodeBuilder(n->name(), "AddN", OpRegistry::Global(), &debug_info)
            .Device(n->requested_device())
            .Attr("N", num_inputs)
            .Attr("T", dtype)
            .Input(data_inputs)
            .ControlInputs(control_inputs);
    const string& colo = GetNodeAttrString(n_attrs, kColocationAttrName);
    if (!colo.empty()) {
      builder.Attr(kColocationAttrName, colo);
    }
    TF_RETURN_IF_ERROR(builder.Finalize(g, &add_n_node));

    // Forward all consumers to the new node.
    for (const Edge* out_edge : n->out_edges()) {
      if (out_edge->IsControlEdge()) {
        g->AddControlEdge(add_n_node, out_edge->dst());
      } else {
        g->AddEdge(add_n_node, 0, out_edge->dst(), out_edge->dst_input());
      }
    }

    // Remove the original AccumulateNV2 placeholder op.
    // This removal modifies the op and must happen after we have finished
    // using its incoming/outgoing edge sets.
    g->RemoveNode(n);

    return Status::OK();
  }
};
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 10,
                      AccumulateNV2RemovePass);

}  // namespace
}  // namespace tensorflow
