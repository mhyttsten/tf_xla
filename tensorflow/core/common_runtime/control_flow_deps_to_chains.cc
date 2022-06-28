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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScontrol_flow_deps_to_chainsDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScontrol_flow_deps_to_chainsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScontrol_flow_deps_to_chainsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/control_flow_deps_to_chains.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

// TODO(mdan): Move this into Grappler - cleaner interface.
Status ControlFlowDepsToChainsPass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScontrol_flow_deps_to_chainsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/common_runtime/control_flow_deps_to_chains.cc", "ControlFlowDepsToChainsPass::Run");

  VLOG(1) << "ControlFlowDepsToChainsPass::Run";

  if (options.graph == nullptr) {
    VLOG(1) << "ControlFlowDepsToChainsPass::Run Aborted";
    return Status::OK();
  }

  Graph* g = options.graph->get();
  DCHECK(g != nullptr);
  FunctionLibraryDefinition* flib_def = options.flib_def;
  DCHECK(flib_def != nullptr);

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("control_flow_deps_to_chains_before", *g, flib_def);
  }

  for (Node* n : g->nodes()) {
    if (n == nullptr) {
      continue;
    }
    if (!n->IsWhileNode()) {
      continue;
    }

    // TODO(mdan): This breaks encapsulation of Node/Graph. Is there any needed?
    // TODO(mdan): Consolidate this with AddWhileInputHack.
    NodeDef* while_node = n->mutable_def();
    const auto& attrs = while_node->attr();
    auto* mattrs = while_node->mutable_attr();

    string body_name = attrs.at("body").func().name();
    auto* body_graph = flib_def->Find(body_name);
    DCHECK(body_graph != nullptr);

    // Look for required annotations.

    if (attrs.find("_stateful_parallelism") == attrs.end()) {
      continue;
    }
    if (!attrs.at("_stateful_parallelism").b()) {
      continue;
    }
    if (attrs.find("parallel_iterations") != attrs.end()) {
      if (attrs.at("parallel_iterations").i() < 2) {
        continue;  // Loops which are already sequential are more efficient
                   // without chains.
      }
    }
    // TODO(mdan): We don't really need this attribute.
    if (attrs.find("_num_original_outputs") == attrs.end()) {
      continue;
    }
    int body_barrier_loc = -1;
    std::map<string, int> node_index;
    for (int i = 0, s = body_graph->node_def_size(); i < s; i++) {
      node_index.emplace(body_graph->node_def(i).name(), i);
      if (body_barrier_loc < 0) {
        const auto& node_attr = body_graph->node_def(i).attr();
        if (node_attr.find("_acd_function_control_output") != node_attr.end()) {
          body_barrier_loc = i;
        }
      }
    }
    if (body_barrier_loc < 0) {
      continue;
    }
    bool ok_for_lowering = true;
    for (int i = 0; i < body_graph->control_ret_size(); i++) {
      const auto& control_node = body_graph->node_def(
          node_index[body_graph->signature().control_output(i)]);
      const auto& control_attr = control_node.attr();
      if (control_attr.find("_res_first_used_by") == control_attr.end()) {
        ok_for_lowering = false;
        break;
      }
    }
    if (!ok_for_lowering) {
      continue;
    }

    int num_loop_vars = body_graph->signature().input_arg_size();
    int num_new_chains = body_graph->control_ret_size();
    int num_node_inputs = while_node->input_size();

    if (!num_new_chains) {
      continue;  // Nothing to do for stateless loops.
    }

    // Add extra loop vars to the while node.

    // TODO(mdan): If the loop vars contains the resource, we should reuse it.
    // Note that stateful ops of resource inputs cause their resources to be
    // captured into the loop vars (through the body/cond captures). We could
    // effectively use those as chains.

    // TODO(mdan): Is there a more efficient way to do this?
    // Insert the new While node inputs: at the end of the loop vars, but before
    // any non-loop var inputs (like control dependencies). Once the initial
    // chain values are created below, they will be added to these inputs.
    for (int i = 0; i < num_new_chains; i++) {
      while_node->add_input();
    }
    for (int i = num_node_inputs - 1; i >= num_loop_vars; i--) {
      while_node->set_input(i + num_new_chains, while_node->input(i));
    }

    std::vector<Node*> new_inputs;
    std::vector<int> new_input_locations;
    // Set their name to a gensym, type to float and shape to scalar.
    for (int i = 0; i < num_new_chains; i++) {
      string c_name = g->NewName("acd__chain");

      // The initial value for the i'th chain loop var.
      NodeDef new_in;
      new_in.set_name(c_name);
      new_in.set_op("Const");
      AttrValue att_dtype;
      att_dtype.set_type(DT_FLOAT);
      new_in.mutable_attr()->insert({"dtype", att_dtype});
      AttrValue att_value;
      att_value.mutable_tensor()->set_dtype(DT_FLOAT);
      att_value.mutable_tensor()->mutable_tensor_shape();
      att_value.mutable_tensor()->add_int_val(0);
      new_in.mutable_attr()->insert({"value", att_value});
      Status status;
      new_inputs.push_back(g->AddNode(new_in, &status));
      TF_RETURN_WITH_CONTEXT_IF_ERROR(status, "while creating chain", c_name);

      int loc = num_loop_vars + i;
      new_input_locations.push_back(loc);
      while_node->set_input(loc, c_name);
      mattrs->at("T").mutable_list()->add_type(DT_FLOAT);
      mattrs->at("output_shapes").mutable_list()->add_shape();
    }

    // TODO(mdan): This should not be necessary to update. Delete?
    mattrs->at("_num_original_outputs").set_i(num_loop_vars + num_new_chains);
    n->UpdateProperties();
    for (int i = 0; i < num_new_chains; i++) {
      g->AddEdge(new_inputs[i], 0, n, new_input_locations[i]);
    }

    // TODO(mdan): This is wasteful. Can we just mutate the original proto?
    FunctionDef modified_body = *body_graph;

    // Disable the global end-of-body barrier from the body function.
    // Because removing a node is too inefficient (would have to walk all the
    // inputs of all graph nodes), we instead clear its control dependencies.
    modified_body.mutable_node_def(body_barrier_loc)->clear_input();

    // Add extra loop vars to the body function.

    for (int i = 0; i < num_new_chains; i++) {
      // Input loop vars.
      // TODO(mdan): Double check that this doesn't clash with names in body.
      string c_name = g->NewName("acd__chainv");
      std::replace(c_name.begin(), c_name.end(), '/', '_');
      auto* new_arg = modified_body.mutable_signature()->add_input_arg();
      new_arg->set_name(c_name);
      new_arg->set_type(DT_FLOAT);

      // Output ops. These are copies of the inputs conditioned on the actual
      // control outputs.
      string c_out_name = g->NewName("acd__outchain");
      auto* new_out = modified_body.add_node_def();
      new_out->set_name(c_out_name);
      new_out->set_op("Identity");
      new_out->add_input(c_name);
      new_out->add_input(
          strings::StrCat("^", body_graph->signature().control_output(i)));
      AttrValue attr;
      attr.set_type(DT_FLOAT);
      new_out->mutable_attr()->insert({"T", attr});

      // Output loop var declarations.
      string c_ret_name = c_out_name;
      std::replace(c_ret_name.begin(), c_ret_name.end(), '/', '_');
      auto* new_out_arg = modified_body.mutable_signature()->add_output_arg();
      new_out_arg->set_name(c_ret_name);
      new_out_arg->set_type(DT_FLOAT);

      // Actual output loop vars.
      modified_body.mutable_ret()->insert(
          {c_ret_name, strings::StrCat(c_out_name, ":output:0")});
      AttrValue attr_val;
      attr_val.mutable_list()->add_shape();
      FunctionDef_ArgAttrs arg_attrs;
      arg_attrs.mutable_attr()->insert({"_output_shapes", attr_val});
      modified_body.mutable_arg_attr()->insert(
          {static_cast<uint32_t>(i + num_loop_vars), arg_attrs});
    }

    // Wire chain loop vars to the ops they need to condition.

    node_index.clear();
    for (int i = 0; i < modified_body.node_def_size(); i++) {
      node_index.emplace(modified_body.node_def(i).name(), i);
    }
    auto& modified_sig = modified_body.signature();
    for (int i = 0; i < num_new_chains; i++) {
      const auto& control_node =
          modified_body.node_def(node_index[modified_sig.control_output(i)]);
      for (const auto& r :
           control_node.attr().at("_res_first_used_by").list().s()) {
        NodeDef* first_node = modified_body.mutable_node_def(node_index[r]);
        // This control dependency ensures proper sequencing of stateful ops
        // upon entry into the loop body, so that they run after the ops
        // which affected the same resource in the previous iteration.
        first_node->add_input(strings::StrCat(
            "^", modified_sig.input_arg(i + num_loop_vars).name()));
      }
    }

    // Clear body function's control returns.
    modified_body.mutable_control_ret()->clear();

    // Add extra loop vars to the cond function.

    // TODO(mdan): This is wasteful. Can't we just mutate the original proto?
    string cond_name = attrs.at("cond").func().name();
    auto* cond_graph = flib_def->Find(cond_name);
    DCHECK(cond_graph != nullptr);
    FunctionDef modified_cond = *cond_graph;

    int cond_barrier_loc = -1;
    for (int i = 0, s = cond_graph->node_def_size(); i < s; i++) {
      if (cond_barrier_loc < 0) {
        const auto& node_attr = cond_graph->node_def(i).attr();
        if (node_attr.find("_acd_function_control_output") != node_attr.end()) {
          cond_barrier_loc = i;
        }
      }
    }
    if (cond_barrier_loc > 0) {
      // Disable the global end-of-body barrier from the cond function.
      // Because removing a node is too inefficient (would have to walk all the
      // inputs of all graph nodes), we instead clear its control dependencies.
      modified_cond.mutable_node_def(cond_barrier_loc)->clear_input();
    }

    for (int i = 0; i < num_new_chains; i++) {
      // Input loop vars.
      // TODO(mdan): These should gate the stateful ops in the cond.
      // Until ACD supplies the necessary information, these are dummies in this
      // function.
      string c_name = g->NewName("acd__chain");
      auto* new_arg = modified_cond.mutable_signature()->add_input_arg();
      new_arg->set_name(c_name);
      new_arg->set_type(DT_FLOAT);

      // TODO(mdan): Return values on the cond function? Most likely a bug.
      AttrValue attr_val;
      attr_val.mutable_list()->add_shape();
      FunctionDef_ArgAttrs arg_attrs;
      arg_attrs.mutable_attr()->insert({"_output_shapes", attr_val});
      modified_cond.mutable_arg_attr()->insert(
          {static_cast<uint32_t>(i + num_loop_vars), arg_attrs});
    }

    // Wire the new cond/body functions to the While node.

    string new_cond_name = g->NewName("acd__while_cond");
    modified_cond.mutable_signature()->set_name(new_cond_name);
    mattrs->at("cond").mutable_func()->set_name(new_cond_name);

    string new_body_name = g->NewName("acd__while_body");
    modified_body.mutable_signature()->set_name(new_body_name);
    mattrs->at("body").mutable_func()->set_name(new_body_name);

    // Commit the new functions.

    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        flib_def->AddFunctionDef(modified_body,
                                 flib_def->GetStackTraces(body_name)),
        "while attaching ", new_body_name, " to flib_def");
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        flib_def->AddFunctionDef(modified_cond,
                                 flib_def->GetStackTraces(cond_name)),
        "while attaching ", new_cond_name, " to flib_def");

    // TODO(b/183666205): This should not be necessary.
    // It's unclear why adding the functions here is also required.
    // Moreover, it's unclear when graph_lib's parent is flib_def itself.
    auto* graph_lib = g->mutable_flib_def();
    if (graph_lib->default_registry() != flib_def) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          graph_lib->AddFunctionDef(modified_body,
                                    graph_lib->GetStackTraces(body_name)),
          "while attaching ", new_body_name, " to graph");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          graph_lib->AddFunctionDef(modified_cond,
                                    graph_lib->GetStackTraces(cond_name)),
          "while attaching ", new_cond_name, " to graph");
    }
  }

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("control_flow_deps_to_chains_after", *g, flib_def);
  }

  return Status::OK();
}

// Note: This needs to run before functional control flow lowering, which is 10.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 9,
                      ControlFlowDepsToChainsPass);

}  // namespace tensorflow
