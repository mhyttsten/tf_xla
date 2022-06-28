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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc() {
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

#include "tensorflow/core/grappler/optimizers/implementation_selector.h"

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/function_api_info.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

constexpr char kConstOp[] = "Const";
constexpr char kCaseOp[] = "Case";
constexpr char kStatelessCaseOp[] = "StatelessCase";
constexpr char kDeviceIndexOp[] = "DeviceIndex";

// TODO(b/157615690): clean up function implementation swap code.
// The overall idea for the function swap is like below:
//          -----------                            -----------
//  inp_1 ->|  P_C    | -> out_1         g_inp_1 ->|  P_C    | -> g_out_1
//  inp_2 ->| forward | -> out_2         g_inp_2 ->| backward| -> g_out_2
//          | FUNC_1  | -> out_3         g_inp_3 ->| FUNC_1  |
//          -----------                            -----------
//           |  |  |                                 ^  ^  ^
//           v  v  v                                 |  |  |
//           s1 s2 s3                                s1 s2 s3
//           |                                       ^
//           |                                       |
//           |             --------------            |
//           |-----------> | Identity_1 | ---------->|
//                         --------------
// P_C: op Partitioned_call or stateful_partitioned_call
// FUNC1 (forward): TF function generated for the forward path.
// FUNC1 (backward): TF function generated for the backward path.
// inp_x: input tensors for the forward path.
// out_x: output tensors for the forward path.
// g_inp_x: gradient input tensors for the backward path.
// g_out_x: gradient output tensors for the backward path.
// s_x: intermediate result generated by forward tf function, which will be
//      consumed by backward function for gradient calculation.
//
// In the example above, the FUNC_1 takes 2 inputs, and return 3 outputs, in the
// meantime, generate 3 intermediate results for gradient calculation.
// The backward function will take 6 inputs, 3 for the gradient value for out_x,
// and 3 for the intermediate results s1/2/3. It returns 2 outputs for gradient
// value wrt inp_x.
//
// Given the graph, especially after the device placement is done, we could
// check if there is an alternative FUNC_2 that is better for the assigned
// device type. Note that FUNC_2 (both forward and backward) should have same
// amount of input output tensor with same dtype. However, it can generate
// different intermediate state tensor, both number wise and type wise, since it
// depends on the implementation detail.
//
// Also note that there might be some Identity op being added to the output of
// the forward function by IsolatePlacerInspectionRequiredOps for device
// placement. When the output DTYPE changes when switching from FUNC_1 to
// FUNC_2, the Identity node down the stream also need to be updated with new
// DTYPE.
//
// Based on this, the rewrite need to happen for following items:
//
// 1. P_C forward/backward need to use FUNC_2 instead of FUNC_1.
// 2. The T_IN for P_C backward need to be updated since the s_x can be
//    different between FUNC_1 and FUNC_2.
// 3. The T_OUT for P_C forward need to be updated since the s_x can be
//    different between FUNC_1 and FUNC_2.
// 4. The input edge for P_C backward need to be updated since the amount of
//    intermediate result can be different between FUNC_1 and FUNC_2.
// 5. DTYPE of the Identity node after s_1/2/3 need to be updated if they exist.

string FindForwardNode(utils::MutableNodeView* backward_node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_0(mht_0_v, 271, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "FindForwardNode");

  // For the tf function, Identity op node might be added by
  // placer_inspection_required_ops_utils for device placement. Those ops might
  // be removed by model_pruner, or stay there if the Identity op is cross
  // device. Given the partitioned_call node for backward function, we want to
  // find the partitioned_call node for the forward function, so that we can
  // add/remove/updated input tensors for backward function, which is the step
  // 4 as described above.

  // Find the last input
  const int last_input_index = backward_node->NumRegularFanins() - 1;
  const utils::MutableFanoutView& input =
      backward_node->GetRegularFanin(last_input_index);
  // For the input node, it should either be the partitioned call, which is
  // the forward node we need, or a Identity op which just pass through the
  // output of the partitioned call.
  if (IsIdentity(*input.node_view()->node())) {
    // Find the only input to this op, which should be the original forward node
    return input.node_view()->node()->input(0);
  } else if (IsPartitionedCall(*input.node_view()->node()) ||
             IsStatefulPartitionedCall(*input.node_view()->node())) {
    // Found the forward node.
    return backward_node->node()->input(last_input_index);
  } else {
    // Unhandled situation.
    return "";
  }
}

void UpdateForwardIdentityNodeDtype(utils::MutableNodeView* forward_node,
                                    const DataTypeVector& dtypes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_1(mht_1_v, 304, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "UpdateForwardIdentityNodeDtype");

  const auto& fanouts_vector = forward_node->GetRegularFanouts();
  for (int pos = 0, pos_limit = fanouts_vector.size(); pos < pos_limit; ++pos) {
    const auto& fanouts_at_pos = fanouts_vector[pos];
    for (const auto& fanout : fanouts_at_pos) {
      if ("Identity" == fanout.node_view()->GetOp()) {
        (*fanout.node_view()->node()->mutable_attr())["T"].set_type(
            dtypes[pos]);
        VLOG(3) << "Updated DTYPE for Identity node: "
                << fanout.node_view()->node()->DebugString();
      }
    }
  }
}

Status UpdateNodeDef(utils::MutableNodeView* node_view, const string& funcName,
                     const FunctionApiInfo& apiInfo) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("funcName: \"" + funcName + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_2(mht_2_v, 324, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "UpdateNodeDef");

  NodeDef* node_def = node_view->node();

  VLOG(3) << "Node def before swap is: " << node_def->DebugString();

  // For step 1 above.
  node_def->mutable_attr()->find("f")->second.mutable_func()->set_name(
      funcName);

  // For step 2 above.
  auto tin = node_def->mutable_attr()->find("Tin");
  tin->second.mutable_list()->clear_type();
  for (const auto& tin_dtype : apiInfo.input_arg_dtypes()) {
    tin->second.mutable_list()->add_type(tin_dtype);
  }

  // For step 3 above.
  auto tout = node_def->mutable_attr()->find("Tout");
  tout->second.mutable_list()->clear_type();
  for (const auto& tout_dtype : apiInfo.output_arg_dtypes()) {
    tout->second.mutable_list()->add_type(tout_dtype);
  }

  if (apiInfo.function_type() == FunctionApiInfo::BACKWARD) {
    // Strip node control dependencies. We'll add them back after updating
    // all the data inputs.
    std::vector<std::string> control_deps;
    for (int i = node_def->input_size() - 1; i >= 0; --i) {
      if (!IsControlInput(node_def->input(i))) break;
      control_deps.push_back(node_def->input(i));
      node_def->mutable_input()->RemoveLast();
    }

    // For step 4 above.
    const int prev_input_size = node_def->input_size();
    const int diff = prev_input_size - apiInfo.input_arg_dtypes().size();
    if (diff >= 0) {
      for (int i = 0; i < diff; ++i) node_def->mutable_input()->RemoveLast();
    } else {
      // Adding new inputs for internal states, the name of the internal states
      // should be in format "{forward_node_name}:{index}", where the newly
      // added index should start from last index of the state.
      // Eg:
      // {
      //   input: "gradients/unified_lstm/strided_slice_1_grad/StridedSliceGrad"
      //   input: "gradients/zeros_like_1"
      //   input: "gradients/zeros_like_2"
      //   input: "unified_lstm/StatefulPartitionedCall:3"
      //   input: "unified_lstm/StatefulPartitionedCall:4"
      //   # New input should be "unified_lstm/StatefulPartitionedCall:5"
      // }
      const string last_input = FindForwardNode(node_view);
      const std::vector<string> name_index = ::absl::StrSplit(last_input, ':');
      if (name_index.size() != 2) {
        return errors::InvalidArgument(
            "Invalid format of input node name: ", last_input,
            " Expected: {forward_node_name}:{index}");
      }
      const absl::string_view node_name = name_index[0];
      int last_index;
      if (!::absl::SimpleAtoi(name_index[1], &last_index)) {
        return errors::InvalidArgument(
            "The index of input node is expected to be number, got: ",
            name_index[1]);
      }
      for (int i = 1; i <= -diff; ++i)
        node_def->add_input(strings::StrCat(node_name, ":", i + last_index));
    }

    // Add control dependencies back.
    for (std::string& control : control_deps)
      node_def->add_input(std::move(control));

  } else if (apiInfo.function_type() == FunctionApiInfo::FORWARD) {
    // For forward function, since the DTYPE of the intermediate state might
    // have been changed, we want to update the down stream Identity node if
    // any. This is the step 5 in the commend above.
    UpdateForwardIdentityNodeDtype(node_view, apiInfo.output_arg_dtypes());
  }

  VLOG(3) << "Node def after swap is: " << node_def->DebugString();
  return Status::OK();
}

Status ImplementationSelector::LoadFunctions(const GraphDef& graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_3(mht_3_v, 411, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "ImplementationSelector::LoadFunctions");

  lib_info_ = absl::make_unique<FunctionLibraryApiInfo>();
  TF_RETURN_IF_ERROR(lib_info_->Init(graph.library()));
  return Status::OK();
}

Status ImplementationSelector::MaybeOptimizeFunctionCall(
    utils::MutableNodeView* node_view) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_4(mht_4_v, 421, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "ImplementationSelector::MaybeOptimizeFunctionCall");

  // There are two ways of calling functions:
  //  1. By specifying an op name as a function name, or
  //  2. Via the @defun functional interface, where the real function call
  //     happens with partitionedcall op, and the function name appear as the
  //     attribute with name "f" and type func. In this use case, there are more
  //     attributes need to be taken care, like Tin and Tout which take care of
  //     the DTYPE of input/output.
  NodeDef* node_def = node_view->node();

  std::vector<string> function_attribute_names;
  for (const auto& attr : node_def->attr()) {
    if (attr.second.has_func() &&
        lib_info_->GetApiInfo(attr.second.func().name()) != nullptr) {
      function_attribute_names.emplace_back(attr.first);
    }
  }

  if (function_attribute_names.empty() &&
      lib_info_->GetApiInfo(node_def->op()) == nullptr) {
    // A regular op, or a function which has no interface.
    return Status::OK();
  }

  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(node_def->device(), &parsed_name) ||
      !parsed_name.has_type) {
    return errors::Internal("Could not parse device name:", node_def->device());
  }
  VLOG(2) << "Op " << node_def->name() << " runs on " << node_def->device()
          << " = (" << parsed_name.type << ")";

  for (const auto& attr_name : function_attribute_names) {
    string function_name = node_def->attr().at(attr_name).func().name();
    // Skip the function if its already optimized by function optimizer.
    if (::absl::StrContains(function_name, "_specialized_for_")) continue;
    std::vector<string> equiv_func_names;
    TF_RETURN_IF_ERROR(lib_info_->GetEquivalentImplementations(
        function_name, &equiv_func_names));
    for (const auto& func_name : equiv_func_names) {
      const auto& func_api_info = lib_info_->GetApiInfo(func_name);
      if (func_api_info->preferred_device() == parsed_name.type) {
        VLOG(2) << "Swapping: " << function_name << " TO: " << func_name;
        TF_RETURN_IF_ERROR(UpdateNodeDef(node_view, func_name, *func_api_info));
        break;
      }
    }
  }

  if (lib_info_->GetApiInfo(node_def->op()) != nullptr &&
      !::absl::StrContains(node_def->op(), "_specialized_for_")) {
    std::vector<string> equiv_func_names;
    TF_RETURN_IF_ERROR(lib_info_->GetEquivalentImplementations(
        node_def->op(), &equiv_func_names));
    for (const string& func_name : equiv_func_names) {
      const auto func_api_info = lib_info_->GetApiInfo(func_name);
      if (func_api_info->preferred_device() == parsed_name.type) {
        node_def->set_op(func_name);
        break;
      }
    }
  }
  return Status::OK();
}

// Finds the index of the device from the device name list.
Status FindDeviceIndex(const utils::MutableNodeView* device_index_node,
                       const string& device, int* index) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_5(mht_5_v, 492, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "FindDeviceIndex");

  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(device, &parsed_name) ||
      !parsed_name.has_type) {
    return errors::Internal("Could not parse device name:", device);
  }
  const auto& device_list =
      device_index_node->GetAttr("device_names")->list().s();
  auto it = absl::c_find(device_list, parsed_name.type);
  if (it != device_list.end()) {
    *index = it - device_list.begin();
  } else {
    // Sets *index to device_list.size() because the default_fn is guaranteed to
    // be the final item in the case op branching list.
    *index = device_list.size();
  }
  return Status::OK();
}

// Rewrites the device_index op to a const op with value of the index.
void RewriteDeviceIndexOp(utils::MutableNodeView* device_index_node,
                          int index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_6(mht_6_v, 516, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "RewriteDeviceIndexOp");

  // Modifies the DeviceIndex node to be an Const op with correct device index.
  auto node = device_index_node->node();
  node->set_op(kConstOp);
  EraseRegularNodeAttributes(node);
  (*node->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*node->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);
  tensor->add_int_val(index);
  VLOG(2) << "Node after rewriting:" << node->DebugString();
}

Status ImplementationSelector::SelectDeviceIndex(GraphDef* graph) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_7(mht_7_v, 531, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "ImplementationSelector::SelectDeviceIndex");

  Status status;
  VLOG(2) << "graph before rewriting device index:" << graph->DebugString();
  utils::MutableGraphView graph_view(graph, &status);
  TF_RETURN_IF_ERROR(status);
  const int num_nodes = graph_view.NumNodes();
  for (int k = 0; k < num_nodes; ++k) {
    auto* node_view = graph_view.GetNode(k);
    if (node_view->GetOp() != kDeviceIndexOp) {
      continue;
    }
    VLOG(2) << "Found a node to rewrite the device index";

    // Find the case node with device index node as input, rewrite the
    // DeviceIndex node to have the value of the index of device type of the
    // case node.
    for (const auto& fanouts : node_view->GetRegularFanouts()) {
      for (const auto& fanout : fanouts) {
        if (fanout.node_view()->GetOp() != kCaseOp &&
            fanout.node_view()->GetOp() != kStatelessCaseOp)
          continue;
        int index;
        // If any error is thrown out during device parsing, we simply skip
        // and do not modify the DeviceIndexNode.
        Status status =
            FindDeviceIndex(node_view, fanout.node_view()->GetDevice(), &index);
        if (status.ok()) {
          RewriteDeviceIndexOp(node_view, index);
        }
      }
    }
  }
  return Status::OK();
}

Status ImplementationSelector::SelectImplementation(GraphDef* graph) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_8(mht_8_v, 569, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "ImplementationSelector::SelectImplementation");

  if (!graph->has_library()) {
    VLOG(2) << "Skipping graph since it does not have function def";
    return Status::OK();
  }
  if (lib_info_->empty()) {
    VLOG(2) << "Skipping optimization since lib_info is empty";
    return Status::OK();
  }

  Status status;
  utils::MutableGraphView graph_view(graph, &status);
  TF_RETURN_IF_ERROR(status);

  const int num_nodes = graph_view.NumNodes();
  for (int k = 0; k < num_nodes; ++k) {
    TF_RETURN_IF_ERROR(MaybeOptimizeFunctionCall(graph_view.GetNode(k)));
  }

  return Status::OK();
}

Status ImplementationSelector::Optimize(Cluster* cluster,
                                        const GrapplerItem& item,
                                        GraphDef* optimized_graph) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSimplementation_selectorDTcc mht_9(mht_9_v, 596, "", "./tensorflow/core/grappler/optimizers/implementation_selector.cc", "ImplementationSelector::Optimize");

  auto status = LoadFunctions(item.graph);
  // Eat up the error from function loading, since this optimizer might run
  // several times, and might try to run against functions generated by
  // function_optimizer from previous runs, which will fail due to function
  // signature mismatch.
  if (!status.ok()) {
    VLOG(2) << "Skipping optimization due to error while loading function "
            << "libraries: " << status;
    return errors::Aborted("Skipped Optimization");
  }

  *optimized_graph = item.graph;
  status = SelectDeviceIndex(optimized_graph);
  if (!status.ok()) {
    *optimized_graph = item.graph;
    VLOG(2) << "Could not rewrite device index due to error:" << status;
  }
  return SelectImplementation(optimized_graph);
}

}  // end namespace grappler
}  // end namespace tensorflow
