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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSinject_prefetchDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSinject_prefetchDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSinject_prefetchDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/inject_prefetch.h"

#include "tensorflow/core/data/dataset_utils.h"
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

constexpr char kPrefetchDataset[] = "PrefetchDataset";
constexpr std::array<const char*, 5> kAsyncTransforms = {
    "MapAndBatchDataset", "ParallelBatchDataset", "ParallelInterleaveDataset",
    "ParallelMapDataset", "PrefetchDataset"};
constexpr std::array<const char*, 7> kDatasetsToSkip = {
    "AssertNextDataset",        "ExperimentalAssertNextDataset",
    "OptionsDataset",           "ModelDataset",
    "OptimizeDataset",          "MaxIntraOpParallelismDataset",
    "PrivateThreadPoolDataset",
};

// This function returns false if the last dataset after skipping all the
// non-user defined datasets, such as OptionsDataset, is a PrefetchDataset; true
// otherwise.
bool ShouldInjectPrefetch(const NodeDef* last_node,
                          const MutableGraphView& graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSinject_prefetchDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/grappler/optimizers/data/inject_prefetch.cc", "ShouldInjectPrefetch");

  // Skip all datasets that could be chained by tf.data to the user defined
  // pipeline because of optimization, etc.
  while (last_node != nullptr &&
         absl::c_any_of(kDatasetsToSkip, [last_node](const char* dataset) {
           return data::MatchesAnyVersion(dataset, last_node->op());
         })) {
    last_node = graph_utils::GetInputNode(*last_node, graph);
  }
  if (last_node == nullptr) {
    VLOG(1) << "The optimization inject_prefetch is not applied because graph "
               "rewrite failed to find a dataset node.";
    return false;
  }
  if (absl::c_any_of(kAsyncTransforms, [last_node](const char* dataset) {
        return data::MatchesAnyVersion(dataset, last_node->op());
      })) {
    VLOG(1) << "The optimization inject_prefetch is not applied because the "
               "last transformation of the input pipeline is an asynchronous "
               "transformation: "
            << last_node->op();
    return false;
  }
  return true;
}

}  // namespace

Status InjectPrefetch::OptimizeAndCollectStats(Cluster* cluster,
                                               const GrapplerItem& item,
                                               GraphDef* output,
                                               OptimizationStats* stats) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSinject_prefetchDTcc mht_1(mht_1_v, 252, "", "./tensorflow/core/grappler/optimizers/data/inject_prefetch.cc", "InjectPrefetch::OptimizeAndCollectStats");

  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization inject_prefetch is not applied if autotune is "
               "off.";
    return Status::OK();
  }
  MutableGraphView graph(output);

  // If the GrapplerItem is derived from a FunctionDef, we don't optimize it.
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph)) {
    return Status::OK();
  }

  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }

  NodeDef* sink_node = graph.GetNode(item.fetch.at(0));
  NodeDef* last_node = graph_utils::GetInputNode(*sink_node, graph);
  if (!ShouldInjectPrefetch(last_node, graph)) {
    return Status::OK();
  }

  // Insert `prefetch(AUTOTUNE)` after the last node.
  NodeDef prefetch_node;
  graph_utils::SetUniqueGraphNodeName(
      strings::StrCat("inject/prefetch_", last_node->name()), graph.graph(),
      &prefetch_node);
  prefetch_node.set_op(kPrefetchDataset);
  // `input_dataset` input
  *prefetch_node.mutable_input()->Add() = last_node->name();
  // `buffer_size` input
  NodeDef* autotune_value =
      graph_utils::AddScalarConstNode(data::model::kAutotune, &graph);
  *prefetch_node.mutable_input()->Add() = autotune_value->name();

  // Set `output_types` and `output_shapes` attributes by copying the relevant
  // attrs from the input node. If we fail to set the attributes, we abort the
  // rewrite.
  if (!graph_utils::CopyShapesAndTypesAttrs(*last_node, &prefetch_node))
    return Status::OK();

  TF_RETURN_IF_ERROR(
      graph_utils::SetMetadataName(prefetch_node.name(), &prefetch_node));

  auto* added_node = graph.AddNode(std::move(prefetch_node));
  TF_RETURN_IF_ERROR(
      graph.UpdateFanouts(last_node->name(), added_node->name()));

  stats->num_changes++;
  return Status::OK();
}

Status InjectPrefetchEligible::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSinject_prefetchDTcc mht_2(mht_2_v, 313, "", "./tensorflow/core/grappler/optimizers/data/inject_prefetch.cc", "InjectPrefetchEligible::OptimizeAndCollectStats");

  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization inject_prefetch_eligible is not applied if "
               "autotune is off.";
    return Status::OK();
  }
  MutableGraphView graph(output);

  // If the GrapplerItem is derived from a FunctionDef, we don't optimize it.
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph)) {
    return Status::OK();
  }

  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }

  NodeDef* sink_node = graph.GetNode(item.fetch.at(0));
  NodeDef* last_node = graph_utils::GetInputNode(*sink_node, graph);
  if (!ShouldInjectPrefetch(last_node, graph)) {
    return Status::OK();
  }

  stats->num_changes++;
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(InjectPrefetch, "inject_prefetch");
REGISTER_GRAPH_OPTIMIZER_AS(InjectPrefetchEligible, "inject_prefetch_eligible");

}  // namespace grappler
}  // namespace tensorflow
