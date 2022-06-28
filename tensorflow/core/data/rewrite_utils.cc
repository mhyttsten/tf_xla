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
class MHTracer_DTPStensorflowPScorePSdataPSrewrite_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSrewrite_utilsDTcc() {
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
#include "tensorflow/core/data/rewrite_utils.h"

#include "tensorflow/core/platform/refcount.h"

// On mobile we do not provide this functionality because not all of its
// dependencies are available there.
#if !defined(IS_MOBILE_PLATFORM)

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/hash_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kOptimizerName[] = "tf_data_meta_optimizer";
constexpr char kOptimizers[] = "optimizers";
constexpr char kOptimizerConfigs[] = "optimizer_configs";

void AddFakeSinks(FunctionDef* function_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utilsDTcc mht_0(mht_0_v, 249, "", "./tensorflow/core/data/rewrite_utils.cc", "AddFakeSinks");

  int counter = 0;
  for (const auto& output : function_def->signature().output_arg()) {
    NodeDef* node = function_def->add_node_def();
    tensorflow::grappler::function_utils::SetUniqueFunctionNodeName(
        strings::StrCat("FakeSink", counter++), function_def, node);
    node->set_op("Identity");
    node->add_input(function_def->ret().at(output.name()));
    (*node->mutable_attr())["T"].set_type(output.type());

    (*function_def->mutable_ret())[output.name()] =
        strings::StrCat(node->name(), ":output:0");
  }
}

void RemoveFakeSinks(FunctionDef* function_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utilsDTcc mht_1(mht_1_v, 267, "", "./tensorflow/core/data/rewrite_utils.cc", "RemoveFakeSinks");

  // Map from identity node names to their input tensor strings
  std::map<std::string, std::string> identity_map;
  for (const auto& node : function_def->node_def()) {
    if (node.op() == "Identity" && node.input_size() == 1) {
      identity_map[node.name()] = node.input(0);
    }
  }
  for (const auto& output_arg : function_def->signature().output_arg()) {
    const std::string& tensor = function_def->ret().at(output_arg.name());
    const std::string& output_node = tensor.substr(0, tensor.find(':'));
    if (identity_map.find(output_node) != identity_map.end()) {
      (*function_def->mutable_ret())[output_arg.name()] =
          identity_map.at(output_node);
    }
  }
}

Status ApplyRewrites(OpKernelContext* ctx,
                     const std::function<RewriterConfig(void)> config_factory,
                     GraphDef* graph_def, string* dataset_node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utilsDTcc mht_2(mht_2_v, 290, "", "./tensorflow/core/data/rewrite_utils.cc", "ApplyRewrites");

  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      GetGrapplerItem(graph_def, dataset_node, /*add_fake_sinks=*/true);
  std::unordered_map<std::string, tensorflow::DeviceProperties> device_map;
  tensorflow::grappler::VirtualCluster cluster(device_map);

  // Run data optimizer using grappler's meta optimizer.
  tensorflow::ConfigProto config;
  *config.mutable_graph_options()->mutable_rewrite_options() = config_factory();
  TF_RETURN_IF_ERROR(tensorflow::grappler::RunMetaOptimizer(
      std::move(*grappler_item), config, ctx->device(), &cluster, graph_def));

  // Remove fake sinks after optimizations are done.
  //
  // TODO(b/118820916): When MetaOptimizer adds provisions for function retvals
  // to be optimizable, we will no longer need this.
  for (auto& function_def : *graph_def->mutable_library()->mutable_function()) {
    RemoveFakeSinks(&function_def);
  }

  return Status::OK();
}
}  // anonymous namespace

RewriterConfig CreateRewriterConfig(
    const absl::flat_hash_set<tstring>& optimizations,
    const absl::flat_hash_set<tstring>& optimizations_configs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utilsDTcc mht_3(mht_3_v, 319, "", "./tensorflow/core/data/rewrite_utils.cc", "CreateRewriterConfig");

  RewriterConfig rewriter_config;
  rewriter_config.add_optimizers(kOptimizerName);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
  rewriter_config.set_fail_on_optimizer_errors(true);
  auto custom_optimizer = rewriter_config.add_custom_optimizers();
  custom_optimizer->set_name(kOptimizerName);
  auto* custom_optimizations_list =
      (*custom_optimizer->mutable_parameter_map())[kOptimizers].mutable_list();
  const auto& registered_optimizers =
      grappler::CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  for (const auto& optimization : optimizations) {
    if (std::find(registered_optimizers.begin(), registered_optimizers.end(),
                  optimization) != registered_optimizers.end()) {
      custom_optimizations_list->add_s(optimization.data(),
                                       optimization.size());
    } else {
      VLOG(1) << "Optimization " << optimization << " is not registered.";
    }
  }
  auto* config_list =
      (*custom_optimizer->mutable_parameter_map())[kOptimizerConfigs]
          .mutable_list();
  for (const auto& config : optimizations_configs) {
    config_list->add_s(config.data(), config.size());
  }
  return rewriter_config;
}

Status RewriteDataset(OpKernelContext* ctx, const DatasetBase* input,
                      std::function<RewriterConfig(void)> config_factory,
                      bool record_fingerprint,
                      core::RefCountPtr<DatasetBase>* rewritten_input) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utilsDTcc mht_4(mht_4_v, 354, "", "./tensorflow/core/data/rewrite_utils.cc", "RewriteDataset");

  std::vector<std::pair<string, Tensor>> input_list;
  GraphDef graph_def;
  string output_node;
  TF_RETURN_IF_ERROR(
      AsGraphDefForRewrite(ctx, input, &input_list, &graph_def, &output_node));

  VLOG(3) << "Before graph rewrites: " << graph_def.DebugString();
  TF_RETURN_IF_ERROR(
      ApplyRewrites(ctx, config_factory, &graph_def, &output_node));
  VLOG(3) << "After graph rewrites: " << graph_def.DebugString();

  // Instantiate the optimized input pipeline by running the optimized graph
  // using the optimized function library.
  FunctionLibraryRuntime* flr = nullptr;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr = nullptr;
  std::unique_ptr<FunctionLibraryDefinition> lib_def = nullptr;
  TF_RETURN_IF_ERROR(
      ctx->function_library()->Clone(&lib_def, &pflr, &flr, true));

  // Some functions may have been modified without having their names changed
  // (for example, nested dataset graphs from FlatMap or Interleave).
  TF_RETURN_IF_ERROR(AddToFunctionLibrary(lib_def.get(), graph_def.library()));

  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));
  std::vector<Tensor> outputs;
  GraphRunner graph_runner(flr->device());

  TF_RETURN_IF_ERROR(
      graph_runner.Run(&graph, flr, input_list, {output_node}, &outputs));
  DatasetBase* rewritten_dataset;
  TF_RETURN_IF_ERROR(
      GetDatasetFromVariantTensor(outputs[0], &rewritten_dataset));
  rewritten_dataset->Ref();
  rewritten_input->reset(rewritten_dataset);

  if (record_fingerprint) {
    (*ctx->runner())([graph_def = std::move(graph_def),
                      lib_def = lib_def.release(),
                      input_list = std::move(input_list),
                      output_node = std::move(output_node)]() {
      std::unique_ptr<FunctionLibraryDefinition> lib_def_owner(lib_def);
      const NodeDef* node_def = nullptr;
      for (const auto& node : graph_def.node()) {
        if (node.name() == output_node) {
          node_def = &node;
          break;
        }
      }
      if (node_def == nullptr) {
        VLOG(3) << "Failed to find node: " << output_node;
        return;
      }
      uint64 hash = 0;
      Status s = HashNode(graph_def, *node_def, *lib_def, &hash);
      if (!s.ok()) {
        VLOG(3) << "Failed to hash graph: " << s.ToString();
        return;
      }
      for (const auto& pair : input_list) {
        hash = Hash64CombineUnordered(hash, Hash64(pair.first));
        uint64 tensor_hash = 0;
        Status s = HashTensor(pair.second, &tensor_hash);
        if (s.ok()) {
          hash = Hash64CombineUnordered(hash, tensor_hash);
        } else {
          VLOG(3) << "Failed to hash tensor: " << s.ToString();
        }
      }
      string graph_hash =
          strings::StrCat(strings::Hex(hash, strings::kZeroPad16));
      metrics::RecordTFDataFingerprint(graph_hash);
    });
  }

  return Status::OK();
}

std::unique_ptr<tensorflow::grappler::GrapplerItem> GetGrapplerItem(
    GraphDef* graph_def, std::string* dataset_node, bool add_fake_sinks) {
  // Add an identity node as the fetch node, otherwise we might get 'placeholder
  // is both fed and fetched' errors in some cases when using input list with
  // placeholder dataset nodes.
  NodeDef* node = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            node);
  node->set_op("Identity");
  node->add_input(*dataset_node);
  (*node->mutable_attr())["T"].set_type(DT_VARIANT);
  *dataset_node = node->name();

  if (add_fake_sinks) {
    // Add fake sink node to graph and functions to allow rewriting the actual
    // sink nodes.
    //
    // TODO(b/118820916): When MetaOptimizer adds provisions for function
    // retvals to be optimizable, we will no longer need this.
    for (auto& function_def :
         *graph_def->mutable_library()->mutable_function()) {
      AddFakeSinks(&function_def);
    }
  }

  // Create metagraph.
  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_graph_def()) = *graph_def;

  // Grappler determines fetch ops from collection 'train_op'.
  CollectionDef collection_def;
  auto node_list = collection_def.mutable_node_list();
  node_list->add_value(*dataset_node);
  (*meta_graph_def.mutable_collection_def())["train_op"] = collection_def;

  // Create Grappler item.
  tensorflow::grappler::ItemConfig item_config;
  item_config.apply_optimizations = true;
  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      tensorflow::grappler::GrapplerItemFromMetaGraphDef(
          "graph", meta_graph_def, item_config);
  // Grappler should not optimize function library of tf.data graphs. The
  // tf.data meta optimizer takes care of optimizing tf.data functions.
  grappler_item->optimization_options().optimize_function_library = false;
  return grappler_item;
}

absl::flat_hash_set<tstring> SelectOptimizations(
    const absl::flat_hash_set<string>& experiments,
    const absl::flat_hash_set<tstring>& optimizations_enabled,
    const absl::flat_hash_set<tstring>& optimizations_disabled,
    const absl::flat_hash_set<tstring>& optimizations_default) {
  absl::flat_hash_set<tstring> optimizations;

  // Add the enabled optimizations.
  optimizations.insert(optimizations_enabled.begin(),
                       optimizations_enabled.end());

  // Add all default optimization that are not disabled.
  for (const auto& optimization : optimizations_default) {
    if (!optimizations_disabled.contains(optimization)) {
      optimizations.insert(optimization);
    }
  }

  // Add experiments that correspond to an optimization unless the optimization
  // is disabled.
  const auto& registered_optimizers =
      grappler::CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  for (const auto& experiment : experiments) {
    if (std::find(registered_optimizers.begin(), registered_optimizers.end(),
                  experiment) != registered_optimizers.end() &&
        !optimizations_disabled.contains(experiment)) {
      optimizations.insert(experiment);
    }
  }

  return optimizations;
}

StatusOr<std::string> GetDatasetNode(const GraphDef& graph_def) {
  // Symbolic `_Retval` node indicates which node corresponds to the dataset.
  for (const auto& node : graph_def.node()) {
    if (node.op() == "_Retval") {
      return node.input(0);
    }
  }
  return errors::NotFound(
      absl::Substitute("Dataset node for graph is not found:\n$0",
                       graph_def.ShortDebugString()));
}

StatusOr<NodeDef> GetDatasetNodeDef(const GraphDef& graph_def) {
  TF_ASSIGN_OR_RETURN(std::string dataset_node_name, GetDatasetNode(graph_def));
  for (const auto& node : graph_def.node()) {
    if (node.name() == dataset_node_name) {
      return node;
    }
  }
  return errors::NotFound(
      absl::Substitute("Dataset node for graph is not found:\n$0",
                       graph_def.ShortDebugString()));
}

}  // namespace data
}  // namespace tensorflow
#endif  // !IS_MOBILE_PLATFORM
