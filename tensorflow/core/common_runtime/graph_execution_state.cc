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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc() {
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

#include "tensorflow/core/common_runtime/graph_execution_state.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/collective_order.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/util.h"

#ifndef IS_MOBILE_PLATFORM
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

namespace {
bool IsCollectiveV2(const string& op) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "IsCollectiveV2");

  return op == "CollectiveReduceV2" || op == "CollectiveGatherV2" ||
         op == "CollectiveBcastRecvV2" || op == "CollectiveBcastSendV2";
}
}  // namespace

GraphExecutionState::GraphExecutionState(
    std::unique_ptr<GraphDef>&& graph_def,
    std::unique_ptr<FunctionLibraryDefinition>&& flib_def,
    const GraphExecutionStateOptions& options)
    : stateful_placements_(options.stateful_placements),
      original_graph_def_(std::move(graph_def)),
      device_set_(options.device_set),
      session_options_(options.session_options),
      session_handle_(options.session_handle),
      flib_def_(std::move(flib_def)),
      graph_(nullptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_1(mht_1_v, 256, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::GraphExecutionState");
}

GraphExecutionState::~GraphExecutionState() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::~GraphExecutionState");

  node_name_to_cost_id_map_.clear();
  delete graph_;
}

/* static */ Status GraphExecutionState::MakeForBaseGraph(
    GraphDef&& graph_def, const GraphExecutionStateOptions& options,
    std::unique_ptr<GraphExecutionState>* out_state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_3(mht_3_v, 271, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::MakeForBaseGraph");

#ifndef __ANDROID__
  VLOG(4) << "Graph proto is \n" << graph_def.DebugString();
#endif  // __ANDROID__

  auto flib_def = absl::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), graph_def.library());

  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&graph_def, *flib_def, 0));

  if (options.session_options->config.graph_options().place_pruned_graph() ||
      !options.session_options->config.experimental()
           .optimize_for_static_graph()) {
    auto ret = absl::WrapUnique(new GraphExecutionState(
        absl::make_unique<GraphDef>(std::move(graph_def)), std::move(flib_def),
        options));

    // When place_pruned_graph is true, a different Graph* will be initialized
    // each time we prune the original graph, so there is no need to
    // construct a Graph* in this case.
    if (!options.session_options->config.graph_options().place_pruned_graph()) {
      auto base_graph = absl::make_unique<Graph>(OpRegistry::Global());
      TF_RETURN_IF_ERROR(ConvertGraphDefToGraph({}, *ret->original_graph_def_,
                                                base_graph.get()));
      TF_RETURN_IF_ERROR(ret->InitBaseGraph(std::move(base_graph)));
    }
    *out_state = std::move(ret);
  } else {
    auto ret = absl::WrapUnique(
        new GraphExecutionState(nullptr, std::move(flib_def), options));
    auto base_graph = absl::make_unique<Graph>(OpRegistry::Global());
    TF_RETURN_IF_ERROR(
        ConvertGraphDefToGraph({}, std::move(graph_def), base_graph.get()));
    TF_RETURN_IF_ERROR(ret->InitBaseGraph(std::move(base_graph)));
    *out_state = std::move(ret);
  }
  return Status::OK();
}

/* static */ Status GraphExecutionState::MakeForPrunedGraph(
    const GraphExecutionState& base_execution_state,
    const GraphExecutionStateOptions& options,
    const BuildGraphOptions& subgraph_options,
    std::unique_ptr<GraphExecutionState>* out_state,
    std::unique_ptr<ClientGraph>* out_client_graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_4(mht_4_v, 318, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::MakeForPrunedGraph");

  if (!(base_execution_state.session_options_->config.graph_options()
            .place_pruned_graph() &&
        options.session_options->config.graph_options().place_pruned_graph())) {
    return errors::Internal(
        "MakeForPrunedGraph is only supported when the `place_pruned_graph` "
        "option is true.");
  }
  if (!base_execution_state.original_graph_def_) {
    // NOTE(mrry): By adding this restriction, which matches the only current
    // usage of this (fairly obscure) method, we do not need to store a
    // redundant copy of the original graph in `*out_state`.
    return errors::Internal(
        "MakeForPrunedGraph is only supported when `base_execution_state` is "
        "the Session-level `GraphExecutionState`.");
  }

  // NOTE(mrry): This makes a copy of `graph_def`, which is
  // regrettable. We could make `GraphDef` objects shareable between
  // execution states to optimize pruned graph execution, but since
  // this case is primarily used for interactive sessions, we make the
  // bet that graph construction is not performance-critical. (Note
  // also that the previous version used `Extend()`, which is strictly
  // more expensive than copying a `GraphDef`.)
  GraphDef temp(*base_execution_state.original_graph_def_);
  auto flib_def = absl::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), temp.library());
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&temp, *flib_def, 0));
  auto ret = absl::WrapUnique(
      new GraphExecutionState(nullptr, std::move(flib_def), options));

  auto base_graph = absl::make_unique<Graph>(OpRegistry::Global());
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph({}, std::move(temp), base_graph.get()));

  // Rewrite the graph before placement.
  ret->rewrite_metadata_.reset(new subgraph::RewriteGraphMetadata);
  TF_RETURN_IF_ERROR(ret->PruneGraph(subgraph_options, base_graph.get(),
                                     ret->rewrite_metadata_.get()));
  TF_RETURN_IF_ERROR(ret->InitBaseGraph(std::move(base_graph)));
  TF_RETURN_IF_ERROR(ret->BuildGraph(subgraph_options, out_client_graph));
  *out_state = std::move(ret);
  return Status::OK();
}

Status GraphExecutionState::Extend(
    const GraphDef& extension_def,
    std::unique_ptr<GraphExecutionState>* out) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_5(mht_5_v, 368, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::Extend");

  if (session_options_->config.experimental().optimize_for_static_graph()) {
    return errors::FailedPrecondition(
        "Extending the graph is not supported when "
        "`optimize_for_static_graph` is true.");
  }

  GraphDef gdef;

  // 1. Copy the function library.
  TF_RETURN_IF_ERROR(flib_def_->AddLibrary(extension_def.library()));
  *gdef.mutable_library() = flib_def_->ToProto();

  // 2. Build an index of the new node names.
  std::unordered_set<string> new_names;
  for (const NodeDef& node : extension_def.node()) {
    new_names.insert(node.name());
  }

  // 3. Add the non-duplicates from the old graph to the new graph.
  //    Return an error if the same node name appears in both the
  //    old graph and the extension.
  for (const NodeDef& node : original_graph_def_->node()) {
    if (new_names.count(node.name()) == 0) {
      *gdef.add_node() = node;
    } else {
      return errors::InvalidArgument(
          "GraphDef argument to Extend includes node '", node.name(),
          "', which was created by a previous call to Create or Extend in this "
          "session.");
    }
  }

  // 4. Merge the versions field.
  int old_node_size = gdef.node_size();
  gdef.mutable_node()->MergeFrom(extension_def.node());
  TF_RETURN_IF_ERROR(
      AddDefaultAttrsToGraphDef(&gdef, *flib_def_, old_node_size));
  // Merge versions
  if (gdef.has_versions()) {
    if (gdef.versions().producer() != extension_def.versions().producer()) {
      return errors::InvalidArgument(
          "Can't extend GraphDef at version ", gdef.versions().producer(),
          " with graph at version ", extension_def.versions().producer());
    }
    VersionDef* versions = gdef.mutable_versions();
    versions->set_min_consumer(std::max(
        versions->min_consumer(), extension_def.versions().min_consumer()));
    if (extension_def.versions().bad_consumers_size()) {
      // Add new bad_consumers that aren't already marked bad.
      //
      // Note: This implementation is quadratic time if there are many calls to
      // ExtendLocked with many bad consumers.  Since this is unlikely, and
      // fixing it would require data structures outside of this routine,
      // quadratic time it is.
      auto* bad_consumers = versions->mutable_bad_consumers();
      const std::unordered_set<int> existing(bad_consumers->begin(),
                                             bad_consumers->end());
      for (const int v : extension_def.versions().bad_consumers()) {
        if (existing.find(v) == existing.end()) {
          bad_consumers->Add(v);
        }
      }
    }

  } else {
    gdef.mutable_versions()->CopyFrom(extension_def.versions());
  }

  // 5. Validate that the final graphdef is valid.
  if (gdef.versions().producer() >= 5) {
    // Validate the graph: we assume that merging two valid graphs
    // should maintain graph validity.
    TF_RETURN_IF_ERROR(graph::ValidateGraphDef(gdef, *flib_def_));
  }

  // 6. Add the extension.
  GraphExecutionStateOptions combined_options;
  combined_options.device_set = device_set_;
  combined_options.session_options = session_options_;
  combined_options.session_handle = session_handle_;
  combined_options.stateful_placements = stateful_placements_;

  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&gdef, *flib_def_, 0));
  auto flib_def = absl::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), gdef.library());
  auto new_execution_state = absl::WrapUnique(
      new GraphExecutionState(absl::make_unique<GraphDef>(std::move(gdef)),
                              std::move(flib_def), combined_options));

  if (!session_options_->config.graph_options().place_pruned_graph()) {
    auto base_graph = absl::make_unique<Graph>(OpRegistry::Global());
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        {}, *new_execution_state->original_graph_def_, base_graph.get()));
    TF_RETURN_IF_ERROR(
        new_execution_state->InitBaseGraph(std::move(base_graph)));
  }
  *out = std::move(new_execution_state);

  // NOTE(mrry): Extend() is likely to be used for non-throughput-sensitive
  // interactive workloads, but in future we may want to transfer other
  // parts of the placement and/or cost model.
  return Status::OK();
}

void GraphExecutionState::SaveStatefulNodes(Graph* graph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_6(mht_6_v, 476, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::SaveStatefulNodes");

  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      VLOG(2) << "Saving " << n->DebugString();
      stateful_placements_[n->name()] = n->assigned_device_name();
    }
  }
}

void GraphExecutionState::RestoreStatefulNodes(Graph* graph) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_7(mht_7_v, 488, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::RestoreStatefulNodes");

  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      auto iter = stateful_placements_.find(n->name());
      if (iter != stateful_placements_.end()) {
        n->set_assigned_device_name(iter->second);
        VLOG(2) << "Restored " << n->DebugString();
      }
    }
  }
}

namespace {

class TensorConnectionPruneRewrite : public subgraph::PruneRewrite {
 public:
  TensorConnectionPruneRewrite(const string* endpoint_name,
                               NodeBuilder::NodeOut from_tensor)
      : subgraph::PruneRewrite(endpoint_name, nullptr /* device_info */),
        from_tensor_(std::move(from_tensor)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_8(mht_8_v, 510, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "TensorConnectionPruneRewrite");
}

  Status AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                 Node** out_node) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_9(mht_9_v, 516, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "AddNode");

    Status s;
    auto check_no_cycle_fn = [this, feed_tensor, &s](Node* n) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_10(mht_10_v, 521, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "lambda");

      if (n == feed_tensor.node) {
        s.Update(errors::InvalidArgument(
            "Requested Tensor connection between nodes \"",
            feed_tensor.node->name(), "\" and \"", from_tensor_.node->name(),
            "\" would create a cycle."));
      }
    };
    ReverseDFSFrom(*g, {from_tensor_.node}, std::move(check_no_cycle_fn),
                   nullptr);
    TF_RETURN_IF_ERROR(s);

    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat("_identity_", feed_tensor.node->name(), "_",
                                    feed_tensor.index),
                    "Identity")
            .Input(from_tensor_)
            .Attr("T",
                  BaseType(from_tensor_.node->output_type(from_tensor_.index)))
            .Finalize(g, out_node));

    (*out_node)->set_assigned_device_name(
        feed_tensor.node->assigned_device_name());
    return Status::OK();
  }

 private:
  NodeBuilder::NodeOut from_tensor_;
};

template <class Map>
Status LookupDevice(const DeviceSet& device_set, const string& tensor_name,
                    const Map& tensor2device,
                    const tensorflow::DeviceAttributes** out_device_attrs) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("tensor_name: \"" + tensor_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_11(mht_11_v, 558, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "LookupDevice");

  *out_device_attrs = nullptr;
  if (tensor2device.empty()) {
    *out_device_attrs = &device_set.client_device()->attributes();
    return Status::OK();
  }
  const auto it = tensor2device.find(tensor_name);
  if (it == tensor2device.end()) {
    *out_device_attrs = &device_set.client_device()->attributes();
    return Status::OK();
  }
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(it->second, &parsed_name)) {
    return errors::InvalidArgument("Invalid device name ('", it->second,
                                   "') provided for the tensor '", tensor_name,
                                   "' in CallableOptions");
  }
  Device* device = device_set.FindDeviceByName(
      DeviceNameUtils::ParsedNameToString(parsed_name));
  if (device == nullptr) {
    return errors::InvalidArgument("Device '", it->second,
                                   "' specified for tensor '", tensor_name,
                                   "' in CallableOptions does not exist");
  }
  *out_device_attrs = &device->attributes();
  return Status::OK();
}

struct TensorAndDevice {
  // WARNING: backing memory for the 'tensor' field is NOT owend.
  const TensorId tensor;
  // WARNING: device pointer is not owned, so must outlive TensorAndDevice.
  const DeviceAttributes* device;
};

// Tensors of some DataTypes cannot placed in device memory as feeds or
// fetches. Validate against a allowlist of those known to work.
bool IsFeedAndFetchSupported(DataType dtype, const string& device_type) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("device_type: \"" + device_type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_12(mht_12_v, 599, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "IsFeedAndFetchSupported");

  // The mechanism for supporting feeds of device-backed Tensors requires
  // the _Arg kernel to be registered for the corresponding type (and that
  // the input to the kernel be in device and not host memory).
  //
  // The mechanism for supporting fetches of device-backed Tensors requires
  // the _Retval kernel to be registered for the corresponding type (and
  // that the output is produced in device and not host memory).
  //
  // For now, we return true iff there are _Arg AND _Retval kernels for dtype on
  // the device. False negatives are okay, false positives would be bad.
  //
  // TODO(ashankar): Instead of a allowlist here, perhaps we could query
  // the kernel registry for _Arg and _Retval kernels instead.
  if (device_type == DEVICE_CPU) return true;
  if (device_type != DEVICE_GPU &&
      !DeviceFactory::IsPluggableDevice(device_type))
    return false;
  switch (dtype) {
    case DT_BFLOAT16:
    case DT_BOOL:
    case DT_COMPLEX128:
    case DT_COMPLEX64:
    case DT_DOUBLE:
    case DT_FLOAT:
    case DT_HALF:
    case DT_INT16:
    case DT_INT64:
    case DT_INT8:
    case DT_UINT16:
    case DT_UINT8:
      return true;
    default:
      return false;
  }
}

Status ValidateFeedAndFetchDevices(
    const Graph& graph,
    const std::vector<TensorAndDevice>& tensors_and_devices) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_13(mht_13_v, 641, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "ValidateFeedAndFetchDevices");

  if (tensors_and_devices.empty()) return Status::OK();
  std::vector<bool> found(tensors_and_devices.size(), false);
  for (const Node* node : graph.nodes()) {
    // Linearly looping through all nodes and then all feed+fetch tensors isn't
    // quite efficient. At the time of this writing, the expectation was that
    // tensors_and_devices.size() is really small in practice, so this won't be
    // problematic.
    // Revist and make a more efficient lookup possible if needed (e.g., perhaps
    // Graph can maintain a map from node name to Node*).
    for (int i = 0; i < tensors_and_devices.size(); ++i) {
      const TensorAndDevice& td = tensors_and_devices[i];
      if (td.tensor.first != node->name()) continue;
      found[i] = true;
      TF_RETURN_IF_ERROR(graph.IsValidOutputTensor(node, td.tensor.second));
      const DataType dtype = node->output_type(td.tensor.second);
      if (!IsFeedAndFetchSupported(dtype, td.device->device_type())) {
        return errors::Unimplemented(
            "Cannot feed or fetch tensor '", td.tensor.ToString(),
            "' from device ", td.device->name(), " as feeding/fetching from ",
            td.device->device_type(), " devices is not yet supported for ",
            DataTypeString(dtype), " tensors");
      }
    }
  }
  for (int i = 0; i < found.size(); ++i) {
    if (!found[i]) {
      return errors::InvalidArgument(
          "Tensor ", tensors_and_devices[i].tensor.ToString(),
          ", specified in either feed_devices or fetch_devices was not found "
          "in the Graph");
    }
  }
  return Status::OK();
}

Status GetFeedShapeAndTypeFromAttribute(const NodeDef& node,
                                        PartialTensorShape* shape,
                                        DataType* type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_14(mht_14_v, 682, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GetFeedShapeAndTypeFromAttribute");

  static const gtl::FlatSet<string>* const kHasExplicitShapeAttribute =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "Placeholder", "PlaceholderV2", "PlaceholderWithDefault",
          "ParallelConcat", "ImmutableConst", "_ParallelConcatStart",
          "InfeedDequeue", "OutfeedDequeue", "CollectiveBcastSend",
          "CollectiveBcastRecv", "AccumulateNV2", "VariableV2", "Variable",
          "TemporaryVariable", "NcclBroadcast", "_ScopedAllocator",
          "_ScopedAllocatorConcat"}));

  // All the node types handled here have their output datatype set in
  // either attribute 'dtype' or 'T'.
  if (!TryGetNodeAttr(node, "dtype", type) &&
      !TryGetNodeAttr(node, "T", type)) {
    return errors::InvalidArgument(
        "Could not determine output type for feed node: ", node.name(),
        " of type ", node.op());
  }

  // First handle the case of feeding a const node.
  if (node.op() == "Const" && HasNodeAttr(node, "value")) {
    *shape =
        PartialTensorShape(node.attr().at("value").tensor().tensor_shape());
  } else if (kHasExplicitShapeAttribute->find(node.op()) !=
             kHasExplicitShapeAttribute->end()) {
    TF_RETURN_IF_ERROR(GetNodeAttr(node, "shape", shape));
  } else {
    return errors::InvalidArgument("Could not determine shape for feed node: ",
                                   node.name(), " of type ", node.op());
  }
  return Status::OK();
}

}  // namespace

Status GraphExecutionState::PruneGraph(
    const BuildGraphOptions& options, Graph* graph,
    subgraph::RewriteGraphMetadata* out_rewrite_metadata) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_15(mht_15_v, 722, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::PruneGraph");

  std::vector<std::unique_ptr<subgraph::PruneRewrite>> feed_rewrites;
  feed_rewrites.reserve(options.callable_options.feed_size());
  std::vector<std::unique_ptr<subgraph::PruneRewrite>> fetch_rewrites;
  fetch_rewrites.reserve(options.callable_options.fetch_size());
  if (options.use_function_convention) {
    std::vector<TensorAndDevice> tensors_and_devices;
    for (int i = 0; i < options.callable_options.feed_size(); ++i) {
      // WARNING: feed MUST be a reference, since ArgFeedRewrite and
      // tensors_and_devices holds on to its address.
      const string& feed = options.callable_options.feed(i);
      const DeviceAttributes* device_info;
      TF_RETURN_IF_ERROR(LookupDevice(*device_set_, feed,
                                      options.callable_options.feed_devices(),
                                      &device_info));
      feed_rewrites.emplace_back(
          new subgraph::ArgFeedRewrite(&feed, device_info, i));
      tensors_and_devices.push_back({ParseTensorName(feed), device_info});
    }
    if (!options.callable_options.fetch_devices().empty() &&
        !options.callable_options.fetch_skip_sync()) {
      return errors::Unimplemented(
          "CallableOptions.fetch_skip_sync = false is not yet implemented. You "
          "can set it to true instead, but MUST ensure that Device::Sync() is "
          "invoked on the Device corresponding to the fetched tensor before "
          "dereferencing the Tensor's memory.");
    }
    for (int i = 0; i < options.callable_options.fetch_size(); ++i) {
      // WARNING: fetch MUST be a reference, since RetvalFetchRewrite and
      // tensors_and_devices holds on to its address.
      const string& fetch = options.callable_options.fetch(i);
      const DeviceAttributes* device_info;
      TF_RETURN_IF_ERROR(LookupDevice(*device_set_, fetch,
                                      options.callable_options.fetch_devices(),
                                      &device_info));
      fetch_rewrites.emplace_back(
          new subgraph::RetvalFetchRewrite(&fetch, device_info, i));
      tensors_and_devices.push_back({ParseTensorName(fetch), device_info});
    }
    TF_RETURN_IF_ERROR(
        ValidateFeedAndFetchDevices(*graph, tensors_and_devices));
  } else {
    if (!options.callable_options.feed_devices().empty() ||
        !options.callable_options.fetch_devices().empty()) {
      return errors::Unimplemented(
          "CallableOptions::feed_devices and CallableOptions::fetch_devices "
          "to configure feeding/fetching tensors to/from device memory is not "
          "yet supported when using a remote session.");
    }
    const DeviceAttributes* device_info =
        &device_set_->client_device()->attributes();
    for (const string& feed : options.callable_options.feed()) {
      feed_rewrites.emplace_back(
          new subgraph::RecvFeedRewrite(&feed, device_info));
    }
    for (const string& fetch : options.callable_options.fetch()) {
      fetch_rewrites.emplace_back(
          new subgraph::SendFetchRewrite(&fetch, device_info));
    }
  }

  for (const TensorConnection& tensor_connection :
       options.callable_options.tensor_connection()) {
    Node* from_node = nullptr;
    TensorId from_id(ParseTensorName(tensor_connection.from_tensor()));

    for (Node* n : graph->nodes()) {
      if (n->name() == from_id.first) {
        from_node = n;
        break;
      }
    }
    if (from_node == nullptr) {
      return errors::InvalidArgument(
          "Requested tensor connection from unknown node: \"",
          tensor_connection.to_tensor(), "\".");
    }
    if (from_id.second >= from_node->num_outputs()) {
      return errors::InvalidArgument(
          "Requested tensor connection from unknown edge: \"",
          tensor_connection.to_tensor(),
          "\" (actual number of outputs = ", from_node->num_outputs(), ").");
    }

    feed_rewrites.emplace_back(new TensorConnectionPruneRewrite(
        &tensor_connection.to_tensor(), {from_node, from_id.second}));
  }

  std::vector<string> target_node_names(
      options.callable_options.target().begin(),
      options.callable_options.target().end());
  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      graph, feed_rewrites, fetch_rewrites, target_node_names,
      out_rewrite_metadata));

  CHECK_EQ(out_rewrite_metadata->feed_types.size(),
           options.callable_options.feed_size() +
               options.callable_options.tensor_connection_size());
  for (int i = 0; i < options.callable_options.tensor_connection_size(); ++i) {
    out_rewrite_metadata->feed_types.pop_back();
  }
  return Status::OK();
}

Status GraphExecutionState::InitBaseGraph(std::unique_ptr<Graph>&& new_graph) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_16(mht_16_v, 829, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::InitBaseGraph");

  // Save stateful placements before placing.
  RestoreStatefulNodes(new_graph.get());

  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_handle = session_handle_;
  optimization_options.session_options = session_options_;
  optimization_options.graph = &new_graph;
  optimization_options.flib_def = flib_def_.get();
  optimization_options.device_set = device_set_;

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));

  Placer placer(new_graph.get(), "", flib_def_.get(), device_set_,
                /* default_local_device= */ nullptr,
                session_options_ == nullptr ||
                    session_options_->config.allow_soft_placement(),
                session_options_ != nullptr &&
                    session_options_->config.log_device_placement());
  // TODO(mrry): Consider making the Placer cancellable.
  TF_RETURN_IF_ERROR(placer.Run());

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_PLACEMENT, optimization_options));

  for (const Node* n : new_graph->nodes()) {
    VLOG(2) << "Mapping " << n->name() << " to " << n->cost_id();
    node_name_to_cost_id_map_[n->name()] = n->cost_id();
  }

  SaveStatefulNodes(new_graph.get());
  graph_ = new_graph.release();
  return Status::OK();
}

Status GraphExecutionState::OptimizeGraph(
    const BuildGraphOptions& options, const Graph& graph,
    const FunctionLibraryDefinition* flib_def,
    std::unique_ptr<Graph>* optimized_graph,
    std::unique_ptr<FunctionLibraryDefinition>* optimized_flib) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_17(mht_17_v, 872, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::OptimizeGraph");

#ifdef IS_MOBILE_PLATFORM
  return errors::InvalidArgument("Mobile platforms not supported");
#else
  if (session_options_->config.graph_options().place_pruned_graph()) {
    return errors::InvalidArgument("Can't optimize a pruned graph");
  }

  if (grappler::MetaOptimizerEnabled(session_options_->config)) {
    // Here we build the GrapplerItem before calling the optimizer.
    grappler::GrapplerItem item;
    item.id = "tf_graph";

    // Add devices to the GrapplerItem
    // It's ok to skip invalid device annotations in Grappler.
    for (const Device* d : device_set_->devices()) {
      Status added_device = item.AddDevice(d->name());
      if (!added_device.ok()) VLOG(3) << added_device.error_message();
    }
    VLOG(3) << "Grappler available devices: "
            << absl::StrJoin(item.devices(), ", ");

    // Add fetches to the GrapplerItem.
    item.fetch.insert(item.fetch.end(),
                      options.callable_options.fetch().begin(),
                      options.callable_options.fetch().end());
    item.fetch.insert(item.fetch.end(),
                      options.callable_options.target().begin(),
                      options.callable_options.target().end());

    for (const TensorConnection& tensor_connection :
         options.callable_options.tensor_connection()) {
      item.fetch.push_back(tensor_connection.from_tensor());
    }

    // Add feeds to the GrapplerItem if we know them.
    absl::flat_hash_set<absl::string_view> node_names;
    if (!(options.callable_options.feed().empty() &&
          options.callable_options.tensor_connection().empty())) {
      std::vector<SafeTensorId> feeds;

      for (const string& feed : options.callable_options.feed()) {
        feeds.emplace_back(ParseTensorName(feed));
      }
      for (const TensorConnection& tensor_connection :
           options.callable_options.tensor_connection()) {
        feeds.emplace_back(ParseTensorName(tensor_connection.to_tensor()));
      }

      // For feeds with tensor index 0 we try to find the corresponding node in
      // the graph to infer feed data type and shape.
      absl::flat_hash_set<absl::string_view> feed_nodes;

      // For feeds with tensor index larger than 0, we can't infer data type or
      // shape from the graph. Currently we only support type and shape
      // inference from a small set of node types: Placeholder, Const, etc...
      for (const SafeTensorId& feed : feeds) {
        if (feed.index() > 0) {
          VLOG(3) << "Add undefined feed for: " << feed.ToString();
          Tensor fake_input(DT_INVALID, {0});
          item.feed.emplace_back(feed.ToString(), fake_input);
        } else {
          VLOG(3) << "Add node for feed inference: " << feed.ToString();
          feed_nodes.insert(feed.node());
          continue;
        }
      }

      // For feeds with tensor index == 0 we try to infer data type and tensor
      // shape from the graph, by looking at the fed node attributes.
      node_names.reserve(graph.num_nodes());
      for (const Node* node : graph.nodes()) {
        node_names.insert(node->name());
        if (feed_nodes.find(node->name()) == feed_nodes.end()) continue;

        // Try to get the type and shape of the feed node.
        PartialTensorShape partial_shape;
        DataType type;
        Status st = GetFeedShapeAndTypeFromAttribute(node->def(),
                                                     &partial_shape, &type);

        // Failed to get type and shape of the feed node.
        if (!st.ok()) {
          VLOG(3) << "Failed to infer feed node type and shape."
                  << " Add undefined feed for: " << node->name();
          Tensor fake_input(DT_INVALID, {0});
          item.feed.emplace_back(node->name(), fake_input);
          continue;
        }

        // If the shape of the placeholder is only partially known, we are free
        // to set unknown dimensions of its shape to any value we desire. We
        // choose 0 to minimize the memory impact. Note that this only matters
        // if an optimizer chooses to run the graph.
        TensorShape shape;
        if (partial_shape.unknown_rank()) {
          shape = TensorShape({0});
        } else {
          for (int i = 0; i < partial_shape.dims(); ++i) {
            if (partial_shape.dim_size(i) < 0) {
              partial_shape.set_dim(i, 0);
            }
          }
          if (!partial_shape.AsTensorShape(&shape)) {
            return errors::InvalidArgument(
                "Could not derive shape for feed node: ",
                node->def().DebugString());
          }
        }

        VLOG(3) << "Add feed for: " << node->name() << "; type: " << type
                << "; shape: " << shape;
        Tensor fake_input(type, shape);
        item.feed.emplace_back(node->name(), fake_input);
      }
    }

    // Validate that the feeds and fetches are valid.
    if (node_names.empty()) {
      // Collect all node names in the graph if we didn't already.
      node_names.reserve(graph.num_nodes());
      for (const Node* node : graph.nodes()) {
        node_names.insert(node->name());
      }
    }
    for (const auto& feed : item.feed) {
      SafeTensorId tensor_id = ParseTensorName(feed.first);
      if (node_names.find(tensor_id.node()) == node_names.end()) {
        return errors::InvalidArgument("Invalid feed, no such node in graph: ",
                                       feed.first);
      }
    }
    for (const auto& fetch : item.fetch) {
      SafeTensorId tensor_id = ParseTensorName(fetch);
      if (node_names.find(tensor_id.node()) == node_names.end()) {
        return errors::InvalidArgument("Invalid fetch, no such node in graph: ",
                                       fetch);
      }
    }

    // Convert Graph to GraphDef and add it to the GrapplerItem.
    graph.ToGraphDef(&item.graph);
    // TODO(b/114748242): Add a unit test to test this bug fix.
    if (flib_def) {
      *item.graph.mutable_library() = flib_def->ToProto();
    }

    // Construct a virtual cluster and find the cpu_device, which the
    // ConstantFolding optimizer will use for partial evaluation of the graph.
    grappler::VirtualCluster cluster(device_set_);
    Device* cpu_device = nullptr;
    for (const auto& device : device_set_->devices()) {
      if (device->parsed_name().id == 0 &&
          StringPiece(device->parsed_name().type) == "CPU" &&
          device->GetAllocator(AllocatorAttributes()) != nullptr) {
        cpu_device = device;
      }
    }

    // Now we can run the MetaOptimizer on the constructed GrapplerItem.
    GraphDef new_graph;
    TF_RETURN_IF_ERROR(
        grappler::RunMetaOptimizer(std::move(item), session_options_->config,
                                   cpu_device, &cluster, &new_graph));

    // Merge optimized graph function library with an original library.
    // Optimized graph might have new functions specialized for it's
    // instantiation context (see Grappler function optimizer), and modified
    // function body for the existing functions.
    optimized_flib->reset(new FunctionLibraryDefinition(*flib_def));

    for (const FunctionDef& fdef : new_graph.library().function()) {
      const string& func_name = fdef.signature().name();

      if ((*optimized_flib)->Contains(func_name)) {
        VLOG(3) << "Replace function: name=" << func_name;
        TF_RETURN_IF_ERROR((*optimized_flib)->ReplaceFunction(func_name, fdef));
      } else {
        VLOG(3) << "Add new function: name=" << func_name;
        TF_RETURN_IF_ERROR((*optimized_flib)->AddFunctionDef(fdef));
      }
    }
    optimized_graph->reset(new Graph(OpRegistry::Global()));

    // Convert the optimized GraphDef back to a Graph.
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, std::move(new_graph),
                                              optimized_graph->get()));
    // The graph conversion sets the requested device names but not the
    // assigned device names. However, since at this point the graph is placed
    // TF expects an assigned device name for every node. Therefore we copy
    // the requested device into the assigned device field.
    for (Node* node : optimized_graph->get()->nodes()) {
      node->set_assigned_device_name(node->requested_device());
    }
    return Status::OK();
  } else {
    return errors::InvalidArgument("Meta Optimizer disabled");
  }
#endif  // IS_MOBILE_PLATFORM
}

Status GraphExecutionState::BuildGraph(const BuildGraphOptions& options,
                                       std::unique_ptr<ClientGraph>* out) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgraph_execution_stateDTcc mht_18(mht_18_v, 1079, "", "./tensorflow/core/common_runtime/graph_execution_state.cc", "GraphExecutionState::BuildGraph");

  VLOG(1) << "BuildGraph";
  const uint64 start_time_usecs = Env::Default()->NowMicros();
  if (!graph_) {
    // It is only valid to call this method directly when the original graph
    // was created with the option `place_pruned_graph == false`.
    return errors::Internal(
        "Attempted to prune a graph that has not been fully initialized.");
  }

  // Grappler optimization might change the structure of a graph itself, and
  // also it can add/prune functions to/from the library.
  std::unique_ptr<Graph> optimized_graph;
  std::unique_ptr<FunctionLibraryDefinition> optimized_flib;

  Status s = OptimizeGraph(options, *graph_, flib_def_.get(), &optimized_graph,
                           &optimized_flib);
  if (!s.ok()) {
    VLOG(2) << "Grappler optimization failed. Error: " << s.error_message();
    // Simply copy the original graph and the function library if we couldn't
    // optimize it.
    optimized_graph.reset(new Graph(flib_def_.get()));
    CopyGraph(*graph_, optimized_graph.get());
    optimized_flib.reset(new FunctionLibraryDefinition(*flib_def_));
  }

  subgraph::RewriteGraphMetadata rewrite_metadata;
  if (session_options_ == nullptr ||
      !session_options_->config.graph_options().place_pruned_graph()) {
    TF_RETURN_IF_ERROR(
        PruneGraph(options, optimized_graph.get(), &rewrite_metadata));
  } else {
    // This GraphExecutionState represents a graph that was
    // pruned when this was constructed, so we copy the metadata from
    // a member variable.
    CHECK(rewrite_metadata_);
    rewrite_metadata = *rewrite_metadata_;
  }

  CHECK_EQ(options.callable_options.feed_size(),
           rewrite_metadata.feed_types.size());
  CHECK_EQ(options.callable_options.fetch_size(),
           rewrite_metadata.fetch_types.size());

  // TODO(andydavis): Clarify optimization pass requirements around CostModel.
  GraphOptimizationPassOptions optimization_options;
  optimization_options.session_options = session_options_;
  optimization_options.graph = &optimized_graph;
  optimization_options.flib_def = optimized_flib.get();
  optimization_options.device_set = device_set_;

  TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, optimization_options));

  int64_t collective_graph_key = options.collective_graph_key;
  if (collective_graph_key == BuildGraphOptions::kNoCollectiveGraphKey) {
    // BuildGraphOptions does not specify a collective_graph_key.  Check all
    // nodes in the Graph and FunctionLibraryDefinition for collective ops and
    // if found, initialize a collective_graph_key as a hash of the ordered set
    // of instance keys.
    std::set<int32> instance_key_set;
    bool has_collective_v2 = false;
    for (Node* node : optimized_graph->nodes()) {
      if (node->IsCollective()) {
        int32_t instance_key;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(node->attrs(), "instance_key", &instance_key));
        instance_key_set.emplace(instance_key);
      } else if (IsCollectiveV2(node->type_string())) {
        has_collective_v2 = true;
      } else {
        const FunctionDef* fdef = optimized_flib->Find(node->def().op());
        if (fdef != nullptr) {
          for (const NodeDef& ndef : fdef->node_def()) {
            if (ndef.op() == "CollectiveReduce" ||
                ndef.op() == "CollectiveBcastSend" ||
                ndef.op() == "CollectiveBcastRecv" ||
                ndef.op() == "CollectiveGather") {
              int32_t instance_key;
              TF_RETURN_IF_ERROR(
                  GetNodeAttr(ndef, "instance_key", &instance_key));
              instance_key_set.emplace(instance_key);
            } else if (IsCollectiveV2(ndef.op())) {
              has_collective_v2 = true;
            }
          }
        }
      }
    }
    if (!instance_key_set.empty()) {
      uint64 hash = 0x8774aa605c729c72ULL;
      for (int32_t instance_key : instance_key_set) {
        hash = Hash64Combine(instance_key, hash);
      }
      collective_graph_key = hash;
    } else if (has_collective_v2) {
      collective_graph_key = 0x8774aa605c729c72ULL;
    }
  }

  // Make collective execution order deterministic if needed.
  if (options.collective_order != GraphCollectiveOrder::kNone) {
    TF_RETURN_IF_ERROR(
        OrderCollectives(optimized_graph.get(), options.collective_order));
  }

  // Copy the extracted graph in order to make its node ids dense,
  // since the local CostModel used to record its stats is sized by
  // the largest node id.
  std::unique_ptr<ClientGraph> dense_copy(
      new ClientGraph(std::move(optimized_flib), rewrite_metadata.feed_types,
                      rewrite_metadata.fetch_types, collective_graph_key));
  CopyGraph(*optimized_graph, &dense_copy->graph);

  // TODO(vrv): We should check invariants of the graph here.
  metrics::UpdateGraphBuildTime(Env::Default()->NowMicros() - start_time_usecs);
  *out = std::move(dense_copy);
  return Status::OK();
}

}  // namespace tensorflow
