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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/lower_case_op.h"

#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {

using NodeOut = NodeBuilder::NodeOut;

constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
    LowerFunctionalOpsConstants::kLowerAsMultiDeviceFunctionAttr;

// Convenience builder to make it easy to construct a case with a single
// function call in each branch. This first converts the Case node
// into switches (for inputs) and merges (for outputs) around a function call
// per branch.
class CaseBuilder {
 public:
  // Create a CaseBuilder to create the lowered form of `case` with branch
  // functions identified by `branch_fn_names` in the `graph`.
  CaseBuilder(Node* case_op, const std::vector<string>& branch_fn_names,
              bool keep_node_fetchable, Graph* graph);

  // Constructs the basic conditional control flow using switch and merge nodes.
  Status CreatePivotNodes();

  // Adds the inputs from the if node to the merge nodes of the lowered if.
  Status AddInputs();

  // Adds the outputs from the if node to the merge nodes of the lowered if.
  // Note: no inputs can be added once outputs are added as the then and else
  // nodes are finalized while adding outputs.
  Status AddOutputs();

  // Builds an identity node with the same outputs as Case.
  Status BuildLoweredCaseOutput();

 private:
  // Returns unique name containing the name of the Case op being rewritten
  // (name_), infix and a suffix to ensure it is unique within the graph.
  string NewName(const string& infix);

  // Adds input to both the then and else nodes from src:src_output.
  Status AddInput(Node* src, int src_output);

  // The merged outputs of the then and else nodes.
  std::vector<NodeOut> outputs_;

  // The node that dominates all execution of the then and else body nodes.
  Node* control_predecessor_;
  // The original Case op.
  Node* case_op_;
  // The node with the same name as the original Case op:
  //   (a) IdentityN node with same outputs if 'keep_node_fetchable_ == true'
  //       and if the original Case op had non-zero data outputs.
  //   (b) NoOp node with control edge from 'branch_executed_node_' otherwise.
  Node* lowered_case_output_;
  // The branch selector of the case.
  OutputTensor branch_index_;
  int num_branches_;
  // Nodes corresponding to pivot branch of branch_index _SwitchN, which is
  // the pivot node that dominates all nodes in the i'th branch.
  std::vector<Node*> pivots_;
  std::vector<Node*> call_nodes_;
  // Merge node that has inputs from each of pivots_ and control edges from
  // [^call_node for call_node in call_nodes_]. This node will guarantee that
  // even when branch functions do not have outputs, they still will be executed
  // for the side effects.
  Node* branch_executed_node_;
  Graph* graph_;
  string name_;
  bool keep_node_fetchable_;

  NodeDebugInfo debug_info_;
  std::vector<NodeBuilder> branch_call_builders_;
};

CaseBuilder::CaseBuilder(Node* case_op,
                         const std::vector<string>& branch_fn_names,
                         bool keep_node_fetchable, Graph* graph)
    : case_op_(case_op),
      num_branches_(branch_fn_names.size()),
      graph_(graph),
      name_(case_op->name()),
      keep_node_fetchable_(keep_node_fetchable),
      debug_info_(*case_op_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc mht_0(mht_0_v, 275, "", "./tensorflow/core/common_runtime/lower_case_op.cc", "CaseBuilder::CaseBuilder");

  branch_call_builders_.reserve(num_branches_);
  for (int b = 0; b < num_branches_; b++) {
    branch_call_builders_.emplace_back(NewName(strings::StrCat("branch", b)),
                                       branch_fn_names[b], graph->op_registry(),
                                       &debug_info_);
    branch_call_builders_[b].Device(case_op_->requested_device());
    branch_call_builders_[b].Attr(kLowerAsMultiDeviceFunctionAttr, true);
  }
  TF_CHECK_OK(case_op_->input_tensor(0, &branch_index_));
}

Status CaseBuilder::CreatePivotNodes() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc mht_1(mht_1_v, 290, "", "./tensorflow/core/common_runtime/lower_case_op.cc", "CaseBuilder::CreatePivotNodes");

  // Construct the basic case body (consisting of feeding in the val to
  // create pivot nodes).
  Node* branch_index;
  TF_RETURN_IF_ERROR(NodeBuilder(NewName("branch_index"), "_SwitchN",
                                 graph_->op_registry(), &debug_info_)
                         .Input(NodeOut(branch_index_))
                         .Input(NodeOut(branch_index_))
                         .Attr("num_outs", num_branches_)
                         .Device(case_op_->requested_device())
                         .Finalize(graph_, &branch_index));
  control_predecessor_ = branch_index;
  pivots_.resize(num_branches_, nullptr);
  for (int b = 0; b < num_branches_; b++) {
    TF_RETURN_IF_ERROR(NodeBuilder(NewName(strings::StrCat("pivot_", b)),
                                   "Identity", graph_->op_registry(),
                                   &debug_info_)
                           .Input(branch_index, b)
                           .Device(case_op_->requested_device())
                           .Finalize(graph_, &pivots_[b]));
  }
  return Status::OK();
}

string CaseBuilder::NewName(const string& infix) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("infix: \"" + infix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc mht_2(mht_2_v, 318, "", "./tensorflow/core/common_runtime/lower_case_op.cc", "CaseBuilder::NewName");

  return graph_->NewName(strings::StrCat(name_, "/", infix));
}

Status CaseBuilder::AddInput(Node* src, int src_output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc mht_3(mht_3_v, 325, "", "./tensorflow/core/common_runtime/lower_case_op.cc", "CaseBuilder::AddInput");

  Node* input;
  NodeDebugInfo debug_info(*src);
  // Colocate the Switch node with the `src` node.
  //
  // This is to avoid unnecessary Host<->Device copies between src and the
  // _SwitchN node. This aligns with the implementation of legacy tf.cond in
  // control_flow_ops.py. The legacy impl colocates the Switch with the
  // input tensor which resets the device stack and forces the Switch to have
  // the same device as the input node (if set) and sets the colocation _class
  // attr. It also ignores the existing colocation constraints on the input node
  // using colocate_with(ignore_existing=True).
  TF_RETURN_IF_ERROR(NodeBuilder(NewName(src->name()), "_SwitchN",
                                 graph_->op_registry(), &debug_info)
                         .Input(src, src_output)
                         .Input(branch_index_)
                         .Device(src->requested_device())
                         .Attr("_class", {src->name()})
                         .Attr("num_outs", num_branches_)
                         .Finalize(graph_, &input));
  for (int b = 0; b < num_branches_; b++) {
    branch_call_builders_[b].Input(input, b);
  }
  return Status::OK();
}

Status CaseBuilder::AddInputs() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc mht_4(mht_4_v, 354, "", "./tensorflow/core/common_runtime/lower_case_op.cc", "CaseBuilder::AddInputs");

  // Add input data edges.
  std::vector<const Edge*> edges;
  TF_RETURN_IF_ERROR(case_op_->input_edges(&edges));
  // Start at index 1 as the first input is the branch index.
  for (int i = 1; i < edges.size(); ++i) {
    const Edge* e = edges[i];
    TF_RETURN_IF_ERROR(AddInput(e->src(), e->src_output()));
  }
  // Add input control edges.
  for (const Edge* e : case_op_->in_edges()) {
    if (e->IsControlEdge()) {
      graph_->AddControlEdge(e->src(), control_predecessor_);
    }
  }
  return Status::OK();
}

Status CaseBuilder::AddOutputs() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc mht_5(mht_5_v, 375, "", "./tensorflow/core/common_runtime/lower_case_op.cc", "CaseBuilder::AddOutputs");

  // Construct the call nodes for each branch.
  call_nodes_.resize(num_branches_, nullptr);
  for (int b = 0; b < num_branches_; b++) {
    TF_RETURN_IF_ERROR(
        branch_call_builders_[b].Finalize(graph_, &call_nodes_[b]));
    graph_->AddControlEdge(pivots_[b], call_nodes_[b]);
  }

  // Merge the outputs from the N branches (all branches have matching outputs).
  const int num_outputs = call_nodes_[0]->num_outputs();
  std::vector<Node*> merges(num_outputs);
  outputs_.resize(merges.size());
  for (int i = 0; i < num_outputs; ++i) {
    std::vector<NodeOut> merge_input;
    merge_input.reserve(num_branches_);
    for (int j = 0; j < num_branches_; j++) {
      merge_input.emplace_back(call_nodes_[j], i);
    }
    TF_RETURN_IF_ERROR(NodeBuilder(NewName("merge"), "Merge",
                                   graph_->op_registry(), &debug_info_)
                           .Input(merge_input)
                           .Device(case_op_->requested_device())
                           .Finalize(graph_, &merges[i]));
    outputs_[i] = NodeOut(merges[i], 0);
  }

  // Add a Merge node that will be used as a control dependency source for the
  // lowered output node. This Merge node will guarantee that lowered else/then
  // function calls will be executed even if they do not have data outputs.
  //
  // Furthermore it will guarantee that all function side effects will be
  // executed, if the function will be inlined into the graph. Having data
  // outputs is not enough, because they might become unused after inlining.
  //
  // We will use this node to rewrite outgoing control edges from lowered 'Case'
  // node. All data edges will read tensors directly from Merge nodes.
  std::vector<NodeOut> pivots(num_branches_);
  for (int j = 0; j < num_branches_; j++) {
    pivots[j] = NodeOut(pivots_[j]);
  }
  TF_RETURN_IF_ERROR(NodeBuilder(NewName("branch_executed"), "Merge",
                                 graph_->op_registry(), &debug_info_)
                         .Input(pivots)
                         .ControlInputs(call_nodes_)
                         .Device(case_op_->requested_device())
                         .Finalize(graph_, &branch_executed_node_));

  TF_RETURN_IF_ERROR(BuildLoweredCaseOutput());

  // Add outputs.
  for (const Edge* e : case_op_->out_edges()) {
    if (e->IsControlEdge()) {
      graph_->AddControlEdge(branch_executed_node_, e->dst());
    } else {
      // Feed the outputs directly from the merge nodes so that downstream ops
      // can start before all the outputs have been computed.
      graph_->AddEdge(merges[e->src_output()], 0, e->dst(), e->dst_input());
    }
  }
  return Status::OK();
}

Status CaseBuilder::BuildLoweredCaseOutput() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc mht_6(mht_6_v, 441, "", "./tensorflow/core/common_runtime/lower_case_op.cc", "CaseBuilder::BuildLoweredCaseOutput");

  // If outputs are empty, it means that we might have only output control
  // edges (already connected to the `branch_executed_node`). Furthermore it's
  // illegal to have an IdentityN with empty inputs.
  //
  // We still must keep lowered Case node as a valid source of control edges,
  // because it might be a part of function control output set.
  NodeBuilder builder = keep_node_fetchable_ && !outputs_.empty()
                            ? NodeBuilder(name_, "IdentityN").Input(outputs_)
                            : NodeBuilder(name_, "NoOp");
  return builder.Device(case_op_->requested_device())
      .ControlInput(branch_executed_node_)
      .Finalize(graph_, &lowered_case_output_);
}

}  // namespace

Status RewriteCaseNode(Node* n, Graph* g, bool keep_node_fetchable) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlower_case_opDTcc mht_7(mht_7_v, 461, "", "./tensorflow/core/common_runtime/lower_case_op.cc", "RewriteCaseNode");

  VLOG(2) << "Lower Case node (keep_node_fetchable=" << keep_node_fetchable
          << "): " << SummarizeNode(*n);
  const AttrValue* branches_attr = n->attrs().Find("branches");
  if (branches_attr == nullptr) {
    return errors::InvalidArgument("branch functions missing");
  }

  int num_branches = branches_attr->list().func_size();
  std::vector<string> branch_fn_names;
  branch_fn_names.reserve(num_branches);
  for (int b = 0; b < num_branches; b++) {
    branch_fn_names.emplace_back(branches_attr->list().func(b).name());
  }
  CaseBuilder cb(n, branch_fn_names, keep_node_fetchable, g);
  TF_RETURN_IF_ERROR(cb.CreatePivotNodes());
  TF_RETURN_IF_ERROR(cb.AddInputs());
  TF_RETURN_IF_ERROR(cb.AddOutputs());
  g->RemoveNode(n);

  return Status::OK();
}

}  // namespace tensorflow
