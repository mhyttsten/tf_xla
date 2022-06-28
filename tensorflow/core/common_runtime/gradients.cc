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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc() {
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

#include "tensorflow/core/common_runtime/gradients.h"

#include <deque>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// TODO(andydavis) Remove some of the code duplicated between this module
// and that in 'common_runtime/function.cc'.
// A few string constant used throughout this module.
static const char* const kGradientOp = "SymbolicGradient";
static const char* const kNodeLabel = "Func";

string NodeOut::name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/common_runtime/gradients.cc", "NodeOut::name");

  if (index == 0) {
    return node->name();
  } else {
    return strings::StrCat(node->name(), ":", index);
  }
}

DataType NodeOut::dtype() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/common_runtime/gradients.cc", "NodeOut::dtype");
 return node->output_type(index); }

struct NodeOutHash {
  uint64 operator()(const NodeOut& x) const {
    return Hash64(reinterpret_cast<const char*>(&x.node), sizeof(Node*),
                  x.index);
  }
};

struct NodeOutEq {
  bool operator()(const NodeOut& x, const NodeOut& y) const {
    return (x.node == y.node) && (x.index == y.index);
  }
};

static Node* AddZerosLike(Graph* g, NodeOut input) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/common_runtime/gradients.cc", "AddZerosLike");

  DCHECK_LT(0, input.dtype());
  DCHECK_LT(input.dtype(), DT_FLOAT_REF);
  if (input.dtype() == DT_RESOURCE) {
    NodeDef read_def;
    read_def.set_name(g->NewName("Read"));
    read_def.set_op("ReadVariableOp");
    read_def.add_input(input.name());
    AddNodeAttr("dtype", DT_FLOAT, &read_def);
    Status s;
    Node* read = g->AddNode(read_def, &s);
    TF_CHECK_OK(s);
    g->AddEdge(input.node, input.index, read, 0);
    NodeDef ndef;
    ndef.set_name(g->NewName(kNodeLabel));
    ndef.set_op("ZerosLike");
    ndef.add_input(read_def.name());
    AddNodeAttr("T", DT_FLOAT, &ndef);
    Node* ret = g->AddNode(ndef, &s);
    TF_CHECK_OK(s);
    g->AddEdge(read, 0, ret, 0);
    return ret;
  } else {
    NodeDef ndef;
    ndef.set_name(g->NewName(kNodeLabel));
    ndef.set_op("ZerosLike");
    ndef.add_input(input.name());
    AddNodeAttr("T", input.dtype(), &ndef);
    Status s;
    Node* ret = g->AddNode(ndef, &s);
    TF_CHECK_OK(s);
    g->AddEdge(input.node, input.index, ret, 0);
    return ret;
  }
}

static Node* AddSymGrad(Graph* g, Node* n, gtl::ArraySlice<NodeOut> grads) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_3(mht_3_v, 280, "", "./tensorflow/core/common_runtime/gradients.cc", "AddSymGrad");

  const int num_x = n->num_inputs();
  const int num_y = n->num_outputs();
  CHECK_EQ(num_y, grads.size());

  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op(kGradientOp);

  // The gradient node should have num_x + num_y inputs.
  std::vector<NodeOut> n_inputs(num_x);
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) continue;
    n_inputs[e->dst_input()] = {e->src(), e->src_output()};
  }
  DataTypeVector in_types;
  for (const NodeOut& nout : n_inputs) {
    ndef.add_input(nout.name());
    in_types.push_back(nout.dtype());
  }
  for (const NodeOut& nout : grads) {
    ndef.add_input(nout.name());
    in_types.push_back(nout.dtype());
  }
  CHECK_EQ(ndef.input_size(), num_x + num_y);

  AddNodeAttr("Tin", in_types, &ndef);

  // The gradient node's outputs have the same types as the node 'n's
  // inputs, except for resources.
  DataTypeVector out_types = n->input_types();
  for (int i = 0, end = out_types.size(); i < end; ++i) {
    if (out_types[i] == DT_RESOURCE) {
      // TODO(apassos): figure out how to get the right dtype
      out_types[i] = DT_FLOAT;
    }
  }
  AddNodeAttr("Tout", out_types, &ndef);
  NameAttrList func;
  func.set_name(n->type_string());
  for (const auto& attr : n->attrs()) {
    (*func.mutable_attr())[attr.first] = attr.second;
  }
  AddNodeAttr("f", func, &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  return ret;
}

class SymbolicGradientBuilder {
 public:
  SymbolicGradientBuilder(gtl::ArraySlice<NodeOut> y_node_outputs,
                          gtl::ArraySlice<NodeOut> x_node_outputs,
                          gtl::ArraySlice<NodeOut> y_grad_node_outputs,
                          std::vector<NodeOut>* x_grad_node_outputs,
                          Graph* graph);

  Status Compute();

 private:
  gtl::ArraySlice<NodeOut> y_node_outputs_;
  gtl::ArraySlice<NodeOut> x_node_outputs_;
  gtl::ArraySlice<NodeOut> y_grad_node_outputs_;
  std::vector<NodeOut>* x_grad_node_outputs_;
  Graph* graph_;  // Not owned.

  // A vector of output endpoints which represents backpropagated
  // gradients
  typedef std::vector<NodeOut> BackproppedGradients;

  // backprops_ is a map from a node output to its accumulated
  // gradients.  When a node output has accumulated all its
  // gradients, we add a node which sums them up.
  std::unordered_map<NodeOut, BackproppedGradients, NodeOutHash, NodeOutEq>
      backprops_;

  // pending[i] is count-down counter for i-th node's expected
  // backprops.  When pending[i] becomes zero, we collected all
  // backprop gradients for all outputs of the ith-node.
  std::vector<int> pending_;

  // 'ready' keeps track of nodes that have been completely
  // backpropped. Initially, for every output y of the function f, we
  // add dy as an input of the gradient function.
  std::deque<Node*> ready_;

  // The set of node ids at which to stop backprop.
  std::unordered_set<int> stop_nodes_;

  // Initialize pending_ and ready_.
  void InitBackprop();

  // In the original function body, there is a forward edge from 'src'
  // to 'dst', when the backprop algorithm constructs the node
  // 'dst_grad' which computes the gradient, we need to propagate it
  // to 'src'.
  void BackpropAlongEdge(const NodeOut& dst_grad, const NodeOut& src);
  void BackpropZerosAlongEdge(const NodeOut& src);

  // Returns a node representing the sum of any backpropped gradients for 'src'.
  // This will be an AddN node if there is more than one accumulated gradient.
  // Returns zeros if there are no gradients, or the dtype is DT_BOOL.
  NodeOut SumGradients(const NodeOut& src);

  TF_DISALLOW_COPY_AND_ASSIGN(SymbolicGradientBuilder);
};

SymbolicGradientBuilder::SymbolicGradientBuilder(
    gtl::ArraySlice<NodeOut> y_node_outputs,
    gtl::ArraySlice<NodeOut> x_node_outputs,
    gtl::ArraySlice<NodeOut> y_grad_node_outputs,
    std::vector<NodeOut>* x_grad_node_outputs, Graph* graph)
    : y_node_outputs_(y_node_outputs),
      x_node_outputs_(x_node_outputs),
      y_grad_node_outputs_(y_grad_node_outputs),
      x_grad_node_outputs_(x_grad_node_outputs),
      graph_(graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_4(mht_4_v, 400, "", "./tensorflow/core/common_runtime/gradients.cc", "SymbolicGradientBuilder::SymbolicGradientBuilder");

  CHECK_EQ(y_node_outputs_.size(), y_grad_node_outputs.size());
  x_grad_node_outputs_->clear();
  x_grad_node_outputs_->resize(x_node_outputs_.size());
  stop_nodes_.reserve(x_node_outputs_.size());
  for (int i = 0, end = x_node_outputs_.size(); i < end; ++i) {
    stop_nodes_.insert(x_node_outputs_[i].node->id());
  }
}

void SymbolicGradientBuilder::BackpropAlongEdge(const NodeOut& dst_grad,
                                                const NodeOut& src) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_5(mht_5_v, 414, "", "./tensorflow/core/common_runtime/gradients.cc", "SymbolicGradientBuilder::BackpropAlongEdge");

  CHECK_NOTNULL(src.node);
  auto iter = backprops_.find(src);
  if (iter != backprops_.end()) {
    auto* grads = &iter->second;
    grads->push_back(dst_grad);
    if (--pending_[src.node->id()] == 0) {
      ready_.push_back(src.node);
    }
  }
}

void SymbolicGradientBuilder::BackpropZerosAlongEdge(const NodeOut& src) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_6(mht_6_v, 429, "", "./tensorflow/core/common_runtime/gradients.cc", "SymbolicGradientBuilder::BackpropZerosAlongEdge");

  CHECK_NOTNULL(src.node);
  auto iter = backprops_.find(src);
  if (iter != backprops_.end()) {
    if (--pending_[src.node->id()] == 0) {
      ready_.push_back(src.node);
    }
  }
}

void SymbolicGradientBuilder::InitBackprop() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_7(mht_7_v, 442, "", "./tensorflow/core/common_runtime/gradients.cc", "SymbolicGradientBuilder::InitBackprop");

  pending_.resize(graph_->num_node_ids(), 0);
  {
    backprops_.clear();
    std::unordered_set<Node*> visited;
    std::deque<Node*> queue;
    for (const NodeOut& nout : y_node_outputs_) {
      queue.push_back(nout.node);
      visited.insert(nout.node);
    }

    // Going forward to figure out which endpoints need backprop-ed.
    // A node's endpoints need to be backprop-ed only if one of the
    // return nodes can reach backwards to the node via data edges.
    while (!queue.empty()) {
      Node* n = queue.front();
      queue.pop_front();
      for (int i = 0; i < n->num_outputs(); ++i) {
        backprops_[{n, i}].clear();
      }
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) continue;
        pending_[e->src()->id()]++;
        if (visited.find(e->src()) == visited.end()) {
          queue.push_back(e->src());
          visited.insert(e->src());
        }
      }
    }

    // Create entries in backprops_ for all x_node_outputs_, because they will
    // not be added in above loop if they are not reverse reachable from
    // y_node_outputs_.
    for (const NodeOut& nout : x_node_outputs_) {
      backprops_[{nout.node, nout.index}].clear();
    }
  }

  {
    const int num_y = y_grad_node_outputs_.size();
    for (int i = 0; i < num_y; ++i) {
      Node* y = y_node_outputs_[i].node;
      for (const Edge* e : y->in_edges()) {
        if (e->IsControlEdge()) continue;
        BackpropAlongEdge(y_grad_node_outputs_[i], {e->src(), e->src_output()});
      }
    }
  }
  CHECK(!ready_.empty());
}

NodeOut SymbolicGradientBuilder::SumGradients(const NodeOut& src) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_8(mht_8_v, 496, "", "./tensorflow/core/common_runtime/gradients.cc", "SymbolicGradientBuilder::SumGradients");

  const DataType dtype = src.dtype();
  auto iter = backprops_.find(src);
  CHECK(iter != backprops_.end());
  const auto& grads = iter->second;
  if (grads.empty() || dtype == DT_BOOL) {
    // Nothing propagated back. The best we can come up is zeros.
    Node* zero_like = AddZerosLike(graph_, src);
    return {zero_like, 0};
  }
  if (grads.size() == 1) {
    // Just one backprop edge.
    return grads[0];
  }
  // Otherwise, adds backprop-ed gradients.
  NodeDef ndef;
  ndef.set_name(graph_->NewName(kNodeLabel));
  ndef.set_op("AddN");  // N-way Add
  for (const NodeOut& nout : grads) {
    ndef.add_input(nout.name());
  }
  AddNodeAttr("N", static_cast<int64_t>(grads.size()), &ndef);
  AddNodeAttr("T", dtype, &ndef);
  Status s;
  Node* add = graph_->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  for (size_t i = 0; i < grads.size(); ++i) {
    const NodeOut& nout = grads[i];
    graph_->AddEdge(nout.node, nout.index, add, i);
  }
  return {add, 0};
}

static bool IsPrimitiveOpWithNoGrad(const string& func) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_9(mht_9_v, 533, "", "./tensorflow/core/common_runtime/gradients.cc", "IsPrimitiveOpWithNoGrad");

  gradient::Creator creator;
  Status s = gradient::GetOpGradientCreator(func, &creator);
  return s.ok() && (creator == nullptr);
}

Status SymbolicGradientBuilder::Compute() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_10(mht_10_v, 542, "", "./tensorflow/core/common_runtime/gradients.cc", "SymbolicGradientBuilder::Compute");

  // Initialize backprops.
  InitBackprop();

  // Backward propagation.
  gtl::InlinedVector<NodeOut, 8> dy;
  while (!ready_.empty()) {
    // n has collected all gradients.
    Node* n = ready_.front();
    ready_.pop_front();

    // "n" has num_x inputs and num_y outputs.
    const int num_x = n->num_inputs();
    const int num_y = n->num_outputs();

    auto iter = stop_nodes_.find(n->id());
    if (iter != stop_nodes_.end()) {
      // Stop backprop.
      // TODO(andydavis) Support stop nodes with more than one output.
      CHECK_EQ(1, num_y);
      continue;
    }

    // dy[i] is the sum of i-th output's backpropped gradients.
    dy.clear();
    dy.resize(num_y, {nullptr, 0});
    for (int i = 0; i < num_y; ++i) {
      dy[i] = SumGradients({n, i});
    }

    if (IsPrimitiveOpWithNoGrad(n->type_string())) {
      // No grad defined for this op: Backprop zeros along the in edges.
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) continue;
        BackpropZerosAlongEdge({e->src(), e->src_output()});
      }
      continue;
    }

    // Adds a gradient node with num_x + num_y inputs and num_x
    // outputs.
    // TODO(andydavis) Support primitive gradient ops.
    Node* grad = AddSymGrad(graph_, n, dy);
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      graph_->AddEdge(e->src(), e->src_output(), grad, e->dst_input());
    }
    for (int i = 0; i < num_y; ++i) {
      graph_->AddEdge(dy[i].node, dy[i].index, grad, num_x + i);
    }

    // Backprops along the in edges.
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      BackpropAlongEdge({grad, e->dst_input()}, {e->src(), e->src_output()});
    }
  }

  for (int i = 0, end = x_node_outputs_.size(); i < end; ++i) {
    (*x_grad_node_outputs_)[i] = SumGradients(x_node_outputs_[i]);
  }

  return Status::OK();
}

Status AddSymbolicGradients(gtl::ArraySlice<NodeOut> y_node_outputs,
                            gtl::ArraySlice<NodeOut> x_node_outputs,
                            gtl::ArraySlice<NodeOut> y_grad_node_outputs,
                            std::vector<NodeOut>* x_grad_node_outputs,
                            Graph* graph) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgradientsDTcc mht_11(mht_11_v, 614, "", "./tensorflow/core/common_runtime/gradients.cc", "AddSymbolicGradients");

  SymbolicGradientBuilder builder(y_node_outputs, x_node_outputs,
                                  y_grad_node_outputs, x_grad_node_outputs,
                                  graph);
  return builder.Compute();
}

}  // end namespace tensorflow
