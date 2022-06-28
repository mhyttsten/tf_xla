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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSautotune_buffer_sizesDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSautotune_buffer_sizesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSautotune_buffer_sizesDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/autotune_buffer_sizes.h"

#include "tensorflow/core/framework/model.h"
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

constexpr char kBufferSizeMin[] = "buffer_size_min";
constexpr char kPrefetchDataset[] = "PrefetchDataset";

constexpr std::array<const char*, 8> kAsyncDatasetOps = {
    "ExperimentalMapAndBatchDataset",
    "MapAndBatchDataset",
    "ParallelBatchDataset",
    "ParallelInterleaveDatasetV2",
    "ParallelInterleaveDatasetV3",
    "ParallelInterleaveDatasetV4",
    "ParallelMapDataset",
    "ParallelMapDatasetV2",
};

}  // namespace

Status AutotuneBufferSizes::OptimizeAndCollectStats(Cluster* cluster,
                                                    const GrapplerItem& item,
                                                    GraphDef* output,
                                                    OptimizationStats* stats) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSautotune_buffer_sizesDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/grappler/optimizers/data/autotune_buffer_sizes.cc", "AutotuneBufferSizes::OptimizeAndCollectStats");

  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization autotune_buffer_sizes is not applied if "
               "autotune is off.";
    return Status::OK();
  }
  MutableGraphView graph(output);

  // Add a const node with value kAutotune.
  NodeDef* autotune_value =
      graph_utils::AddScalarConstNode(data::model::kAutotune, &graph);

  absl::flat_hash_set<string> already_prefetched;

  // 1) Collect about all existing `PrefetchDataset` nodes, replacing
  // `prefetch(N)` with `prefetch(AUTOTUNE, buffer_size_min=N)` for all N !=-1.
  for (NodeDef& node : *output->mutable_node()) {
    if (node.op() == kPrefetchDataset) {
      NodeDef* buffer_size_node = graph.GetNode(node.input(1));
      // We only consider to rewrite if `buffer_size` is constant.
      if (buffer_size_node->op() == "Const") {
        int64_t initial_buffer_size =
            buffer_size_node->attr().at("value").tensor().int64_val(0);
        if (initial_buffer_size != data::model::kAutotune) {
          TF_RETURN_IF_ERROR(graph.UpdateFanin(node.name(),
                                               {buffer_size_node->name(), 0},
                                               {autotune_value->name(), 0}));
          node.mutable_attr()->at(kBufferSizeMin).set_i(initial_buffer_size);
          stats->num_changes++;
        }
      } else {
        return errors::FailedPrecondition(
            "The autotune_buffer_sizes rewrite does not currently support "
            "non-constant buffer_size input.");
      }
      NodeDef* prefetched_node = graph_utils::GetInputNode(node, graph);
      if (prefetched_node) {
        already_prefetched.insert(prefetched_node->name());
      }
    }
  }

  std::vector<const NodeDef*> async_datasets;
  // 2) Insert `prefetch(AUTOTUNE)` after all asynchronous transformations that
  // are not followed by a `prefetch` yet.
  for (const NodeDef& node : item.graph.node()) {
    if (already_prefetched.find(node.name()) != already_prefetched.end()) {
      continue;
    }
    for (const auto& async_dataset_op : kAsyncDatasetOps) {
      if (node.op() == async_dataset_op) {
        async_datasets.push_back(&node);
        stats->num_changes++;
        break;
      }
    }
  }

  if (async_datasets.empty()) return Status::OK();

  for (const NodeDef* async_dataset_node : async_datasets) {
    NodeDef prefetch_node;
    graph_utils::SetUniqueGraphNodeName(
        strings::StrCat("inject/prefetch_", async_dataset_node->name()),
        graph.graph(), &prefetch_node);
    prefetch_node.set_op(kPrefetchDataset);
    // `input_dataset` input
    *prefetch_node.mutable_input()->Add() = async_dataset_node->name();
    // `buffer_size` input
    *prefetch_node.mutable_input()->Add() = autotune_value->name();

    graph_utils::CopyShapesAndTypesAttrs(*async_dataset_node, &prefetch_node);

    auto* added_node = graph.AddNode(std::move(prefetch_node));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(async_dataset_node->name(), added_node->name()));
  }

  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(AutotuneBufferSizes, "autotune_buffer_sizes");

}  // namespace grappler
}  // namespace tensorflow
