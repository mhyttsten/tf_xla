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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_fusionDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_fusionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_fusionDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/map_fusion.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/fusion_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

// Sets basic function parameters and copies attributes from parent and map
// node.
NodeDef MakeFusedNode(const NodeDef& parent_map_node, const NodeDef& map_node,
                      const FunctionDef& fused_function,
                      MutableGraphView* graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_fusionDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/grappler/optimizers/data/map_fusion.cc", "MakeFusedNode");

  NodeDef fused_node;
  graph_utils::SetUniqueGraphNodeName("fused_map", graph->graph(), &fused_node);
  fused_node.set_op("MapDataset");
  fused_node.add_input(parent_map_node.input(0));

  auto attr = parent_map_node.attr().at("f");
  *attr.mutable_func()->mutable_name() = fused_function.signature().name();
  (*fused_node.mutable_attr())["f"] = std::move(attr);

  graph_utils::CopyAttribute("Targuments", parent_map_node, &fused_node);
  graph_utils::CopyShapesAndTypesAttrs(map_node, &fused_node);

  auto value_or_false = [](const AttrValue* attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_fusionDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/grappler/optimizers/data/map_fusion.cc", "lambda");

    if (!attr) return false;
    return attr->b();
  };

  const auto* first_parallelism =
      gtl::FindOrNull(parent_map_node.attr(), "use_inter_op_parallelism");
  const auto* second_parallelism =
      gtl::FindOrNull(map_node.attr(), "use_inter_op_parallelism");
  // Some graphs cannot execute with use_inter_op_parallelism=False, so we need
  // to set it to true if one of the ops have it set to true.
  (*fused_node.mutable_attr())["use_inter_op_parallelism"].set_b(
      value_or_false(first_parallelism) || value_or_false(second_parallelism));

  const auto* first_cardinality =
      gtl::FindOrNull(parent_map_node.attr(), "preserve_cardinality");
  const auto* second_cardinality =
      gtl::FindOrNull(map_node.attr(), "preserve_cardinality");
  (*fused_node.mutable_attr())["preserve_cardinality"].set_b(
      value_or_false(first_cardinality) && value_or_false(second_cardinality));

  graph_utils::MaybeSetFusedMetadata(parent_map_node, map_node, &fused_node);

  return fused_node;
}

}  // namespace

Status MapFusion::OptimizeAndCollectStats(Cluster* cluster,
                                          const GrapplerItem& item,
                                          GraphDef* output,
                                          OptimizationStats* stats) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_fusionDTcc mht_2(mht_2_v, 260, "", "./tensorflow/core/grappler/optimizers/data/map_fusion.cc", "MapFusion::OptimizeAndCollectStats");

  GraphDef sorted_old_graph = item.graph;
  TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
  *output = sorted_old_graph;

  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());

  auto get_map_node = [](const NodeDef& node) -> const NodeDef* {
    // TODO(b/148614504): Support ParallelMapDataset and MapAndBatchDataset.
    // TODO(b/148614315): Support captured inputs.
    if (node.op() == "MapDataset" && node.input_size() == 1) return &node;
    return nullptr;
  };

  auto make_fused_function = [&function_library, &output](
                                 const NodeDef* parent_map_node,
                                 const NodeDef* map_node) -> FunctionDef* {
    const auto& parent_fun = parent_map_node->attr().at("f");
    const FunctionDef* parent_func =
        function_library.Find(parent_fun.func().name());
    const auto& fun = map_node->attr().at("f");
    const FunctionDef* func = function_library.Find(fun.func().name());

    if (!fusion_utils::CanCompose(parent_func->signature(),
                                  func->signature())) {
      VLOG(1) << "Can't fuse two maps because the output signature of the "
                 "first map function does not match the input signature of the "
                 "second function\n";
      return nullptr;
    }
    return fusion_utils::FuseFunctions(
        *parent_func, *func, "fused_map", fusion_utils::ComposeSignature,
        fusion_utils::ComposeInput, fusion_utils::ComposeOutput,
        fusion_utils::MergeNodes, output->mutable_library());
  };

  for (const NodeDef& node : sorted_old_graph.node()) {
    const NodeDef* map_node = get_map_node(node);
    if (!map_node) continue;

    const NodeDef* parent_map_node =
        get_map_node(*graph_utils::GetInputNode(*map_node, graph));
    if (!parent_map_node) continue;

    const auto* fused_function = make_fused_function(parent_map_node, map_node);
    if (fused_function == nullptr) continue;

    const auto* fused_maps_node = graph.AddNode(
        MakeFusedNode(*parent_map_node, *map_node, *fused_function, &graph));

    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(map_node->name(), fused_maps_node->name()));

    TF_RETURN_IF_ERROR(function_library.AddFunctionDef(*fused_function));

    nodes_to_delete.insert(parent_map_node->name());
    nodes_to_delete.insert(map_node->name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(MapFusion, "map_fusion");

}  // namespace grappler
}  // namespace tensorflow
