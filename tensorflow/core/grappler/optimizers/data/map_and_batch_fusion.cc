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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_and_batch_fusionDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_and_batch_fusionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_and_batch_fusionDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/map_and_batch_fusion.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kFusedOpName[] = "MapAndBatchDataset";
constexpr char kParallelMap[] = "ParallelMapDataset";
constexpr char kParallelMapV2[] = "ParallelMapDatasetV2";

bool IsParallelMap(const NodeDef& node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_and_batch_fusionDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/grappler/optimizers/data/map_and_batch_fusion.cc", "IsParallelMap");

  return node.op() == kParallelMap || node.op() == kParallelMapV2;
}

NodeDef MakeMapAndBatchNode(const NodeDef& map_node, const NodeDef& batch_node,
                            MutableGraphView* graph) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_and_batch_fusionDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/grappler/optimizers/data/map_and_batch_fusion.cc", "MakeMapAndBatchNode");

  NodeDef new_node;
  new_node.set_op(kFusedOpName);
  graph_utils::SetUniqueGraphNodeName(kFusedOpName, graph->graph(), &new_node);

  // Set the `input` input argument.
  new_node.add_input(map_node.input(0));

  // Set the `other_arguments` input arguments.
  int num_other_args;
  if (IsParallelMap(map_node)) {
    num_other_args = map_node.input_size() - 2;
  } else {
    num_other_args = map_node.input_size() - 1;
  }
  for (int i = 0; i < num_other_args; i++) {
    new_node.add_input(map_node.input(i + 1));
  }

  // Set the `batch_size` input argument.
  new_node.add_input(batch_node.input(1));

  // Set the `num_parallel_calls` input argument.
  if (map_node.op() == kParallelMap) {
    // The type of the `num_parallel_calls` argument in ParallelMapDataset
    // and MapAndBatchDataset is different (int32 and int64 respectively)
    // so we cannot reuse the same Const node and thus create a new one.
    NodeDef* v = graph->GetNode(map_node.input(map_node.input_size() - 1));
    NodeDef* tmp = graph_utils::AddScalarConstNode<int64_t>(
        v->attr().at("value").tensor().int_val(0), graph);
    new_node.add_input(tmp->name());
  } else if (map_node.op() == kParallelMapV2) {
    new_node.add_input(map_node.input(map_node.input_size() - 1));
  } else {
    NodeDef* tmp = graph_utils::AddScalarConstNode<int64_t>(1, graph);
    new_node.add_input(tmp->name());
  }

  // Set the `drop_remainder` input argument.
  if (batch_node.op() == "BatchDatasetV2") {
    new_node.add_input(batch_node.input(2));
  } else {
    NodeDef* tmp = graph_utils::AddScalarConstNode<bool>(false, graph);
    new_node.add_input(tmp->name());
  }

  // Required attributes.
  for (auto key : {"f", "Targuments"}) {
    graph_utils::CopyAttribute(key, map_node, &new_node);
  }
  graph_utils::CopyShapesAndTypesAttrs(batch_node, &new_node);

  // Optional attributes.
  // TODO(jsimsa): Support `use_inter_op_parallelism` and `sloppy`.
  for (auto key : {"preserve_cardinality"}) {
    if (gtl::FindOrNull(map_node.attr(), key)) {
      graph_utils::CopyAttribute(key, map_node, &new_node);
    }
  }
  graph_utils::MaybeSetFusedMetadata(map_node, batch_node, &new_node);
  return new_node;
}

}  // namespace

Status MapAndBatchFusion::OptimizeAndCollectStats(Cluster* cluster,
                                                  const GrapplerItem& item,
                                                  GraphDef* output,
                                                  OptimizationStats* stats) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSmap_and_batch_fusionDTcc mht_2(mht_2_v, 286, "", "./tensorflow/core/grappler/optimizers/data/map_and_batch_fusion.cc", "MapAndBatchFusion::OptimizeAndCollectStats");

  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  for (const NodeDef& node : item.graph.node()) {
    if (node.op() != "BatchDataset" && node.op() != "BatchDatasetV2") {
      continue;
    }

    // Use a more descriptive variable name now that we know the node type.
    const NodeDef& batch_node = node;
    NodeDef* node2 = graph_utils::GetInputNode(batch_node, graph);

    if (node2->op() != "MapDataset" && !IsParallelMap(*node2)) {
      continue;
    }
    // Use a more descriptive variable name now that we know the node type.
    NodeDef* map_node = node2;

    auto* new_node =
        graph.AddNode(MakeMapAndBatchNode(*map_node, batch_node, &graph));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(batch_node.name(), new_node->name()));

    // Mark the `Map` and `Batch` nodes for removal.
    nodes_to_delete.insert(map_node->name());
    nodes_to_delete.insert(batch_node.name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(MapAndBatchFusion, "map_and_batch_fusion");

}  // namespace grappler
}  // namespace tensorflow
