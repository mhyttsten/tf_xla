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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSshuffle_and_repeat_fusionDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSshuffle_and_repeat_fusionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSshuffle_and_repeat_fusionDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/shuffle_and_repeat_fusion.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kShuffleDataset[] = "ShuffleDataset";
constexpr char kShuffleDatasetV2[] = "ShuffleDatasetV2";
constexpr char kShuffleDatasetV3[] = "ShuffleDatasetV3";
constexpr char kRepeatDataset[] = "RepeatDataset";
constexpr char kShuffleAndRepeatDataset[] = "ShuffleAndRepeatDataset";
constexpr char kShuffleAndRepeatDatasetV2[] = "ShuffleAndRepeatDatasetV2";

constexpr char kReshuffleEachIteration[] = "reshuffle_each_iteration";

Status FuseShuffleV1AndRepeat(const NodeDef& shuffle_node,
                              const NodeDef& repeat_node,
                              MutableGraphView* graph, GraphDef* output,
                              NodeDef* fused_node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSshuffle_and_repeat_fusionDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/grappler/optimizers/data/shuffle_and_repeat_fusion.cc", "FuseShuffleV1AndRepeat");

  fused_node->set_op(kShuffleAndRepeatDataset);
  graph_utils::SetUniqueGraphNodeName(kShuffleAndRepeatDataset, output,
                                      fused_node);

  // Set the `input` input argument.
  fused_node->add_input(shuffle_node.input(0));

  // Set the `buffer_size` input argument.
  fused_node->add_input(shuffle_node.input(1));

  // Set the `seed` input argument.
  fused_node->add_input(shuffle_node.input(2));

  // Set the `seed2` input argument.
  fused_node->add_input(shuffle_node.input(3));

  // Set the `count` input argument.
  fused_node->add_input(repeat_node.input(1));

  // Set `output_types`, `output_shapes`, and `reshuffle_each_iteration`
  // attributes.
  graph_utils::CopyShapesAndTypesAttrs(shuffle_node, fused_node);
  graph_utils::CopyAttribute(kReshuffleEachIteration, shuffle_node, fused_node);

  // Optionally set the `metadata` attribute.
  graph_utils::MaybeSetFusedMetadata(shuffle_node, repeat_node, fused_node);

  return Status::OK();
}

Status FuseShuffleV2AndRepeat(const NodeDef& shuffle_node,
                              const NodeDef& repeat_node,
                              MutableGraphView* graph, GraphDef* output,
                              NodeDef* fused_node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSshuffle_and_repeat_fusionDTcc mht_1(mht_1_v, 253, "", "./tensorflow/core/grappler/optimizers/data/shuffle_and_repeat_fusion.cc", "FuseShuffleV2AndRepeat");

  fused_node->set_op(kShuffleAndRepeatDatasetV2);
  graph_utils::SetUniqueGraphNodeName(kShuffleAndRepeatDatasetV2, output,
                                      fused_node);

  NodeDef zero_node = *graph_utils::AddScalarConstNode<int64_t>(0, graph);

  // Set the `input` input argument.
  fused_node->add_input(shuffle_node.input(0));

  // Set the `buffer_size` input argument.
  fused_node->add_input(shuffle_node.input(1));

  // Default the `seed` input argument to 0.
  fused_node->add_input(zero_node.name());

  // Default the `seed2` input argument to 0.
  fused_node->add_input(zero_node.name());

  // Set the `count` input argument.
  fused_node->add_input(repeat_node.input(1));

  // Set the `seed_generator` input argument.
  fused_node->add_input(shuffle_node.input(2));

  // Set `output_types` and `output_shapes` attributes.
  graph_utils::CopyShapesAndTypesAttrs(shuffle_node, fused_node);

  // Default the `reshuffle_each_iteration` attribute to true.
  (*fused_node->mutable_attr())[kReshuffleEachIteration].set_b(true);

  // Optionally set the `metadata` attribute.
  graph_utils::MaybeSetFusedMetadata(shuffle_node, repeat_node, fused_node);

  return Status::OK();
}

Status FuseShuffleV3AndRepeat(const NodeDef& shuffle_node,
                              const NodeDef& repeat_node,
                              MutableGraphView* graph, GraphDef* output,
                              NodeDef* fused_node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSshuffle_and_repeat_fusionDTcc mht_2(mht_2_v, 296, "", "./tensorflow/core/grappler/optimizers/data/shuffle_and_repeat_fusion.cc", "FuseShuffleV3AndRepeat");

  fused_node->set_op(kShuffleAndRepeatDatasetV2);
  graph_utils::SetUniqueGraphNodeName(kShuffleAndRepeatDataset, output,
                                      fused_node);

  // Set the `input` input argument.
  fused_node->add_input(shuffle_node.input(0));

  // Set the `buffer_size` input argument.
  fused_node->add_input(shuffle_node.input(1));

  // Set the `seed` input argument.
  fused_node->add_input(shuffle_node.input(2));

  // Set the `seed2` input argument.
  fused_node->add_input(shuffle_node.input(3));

  // Set the `count` input argument.
  fused_node->add_input(repeat_node.input(1));

  // Set the `seed_generator` input argument.
  fused_node->add_input(shuffle_node.input(4));

  // Set `output_types`, `output_shapes`, and `reshuffle_each_iteration`
  // attributes.
  graph_utils::CopyShapesAndTypesAttrs(shuffle_node, fused_node);
  graph_utils::CopyAttribute(kReshuffleEachIteration, shuffle_node, fused_node);

  // Optionally set the `metadata` attribute.
  graph_utils::MaybeSetFusedMetadata(shuffle_node, repeat_node, fused_node);

  return Status::OK();
}

}  // namespace

Status ShuffleAndRepeatFusion::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSshuffle_and_repeat_fusionDTcc mht_3(mht_3_v, 337, "", "./tensorflow/core/grappler/optimizers/data/shuffle_and_repeat_fusion.cc", "ShuffleAndRepeatFusion::OptimizeAndCollectStats");

  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;

  for (const NodeDef& repeat_node : item.graph.node()) {
    if (repeat_node.op() != kRepeatDataset) {
      continue;
    }

    const NodeDef& shuffle_node =
        *graph_utils::GetInputNode(repeat_node, graph);

    NodeDef fused_node;
    if (shuffle_node.op() == kShuffleDataset) {
      TF_RETURN_IF_ERROR(FuseShuffleV1AndRepeat(shuffle_node, repeat_node,
                                                &graph, output, &fused_node));
    } else if (shuffle_node.op() == kShuffleDatasetV2) {
      TF_RETURN_IF_ERROR(FuseShuffleV2AndRepeat(shuffle_node, repeat_node,
                                                &graph, output, &fused_node));

    } else if (shuffle_node.op() == kShuffleDatasetV3) {
      TF_RETURN_IF_ERROR(FuseShuffleV3AndRepeat(shuffle_node, repeat_node,
                                                &graph, output, &fused_node));
    } else {
      continue;
    }

    NodeDef& shuffle_and_repeat_node = *graph.AddNode(std::move(fused_node));
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(repeat_node.name(),
                                           shuffle_and_repeat_node.name()));
    // Update shuffle node fanouts to shuffle_and_repeat fanouts to take care of
    // control dependencies.
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(shuffle_node.name(),
                                           shuffle_and_repeat_node.name()));

    // Mark the `Shuffle` and `Repeat` nodes for removal (as long as neither of
    // them needs to be preserved).
    const auto nodes_to_preserve = item.NodesToPreserve();
    if (nodes_to_preserve.find(shuffle_node.name()) ==
            nodes_to_preserve.end() &&
        nodes_to_preserve.find(repeat_node.name()) == nodes_to_preserve.end()) {
      nodes_to_delete.insert(shuffle_node.name());
      nodes_to_delete.insert(repeat_node.name());
    }
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(ShuffleAndRepeatFusion,
                            "shuffle_and_repeat_fusion");

}  // namespace grappler
}  // namespace tensorflow
