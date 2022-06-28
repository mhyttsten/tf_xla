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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslackDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslackDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslackDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/slack.h"

#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/attr_value.pb.h"
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

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kPrefetchDatasetOp[] = "PrefetchDataset";

template <std::size_t SIZE>
bool IsDatasetNodeOfType(const NodeDef& node,
                         const std::array<const char*, SIZE>& arr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslackDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/grappler/optimizers/data/slack.cc", "IsDatasetNodeOfType");

  for (const auto& dataset_op_name : arr) {
    if (node.op() == dataset_op_name) return true;
  }
  return false;
}

// We don't pass through "Batch*" ops and nested dataset ops (FlatMap, etc)
// because the correct slack_period cannot be determined directly in those
// cases.
constexpr std::array<const char*, 2> kMultipleInputsDatasetOps = {
    "ZipDataset", "ConcatenateDataset"};

constexpr std::array<const char*, 22> kPassThroughOps = {
    "CacheDataset",
    "CacheDatasetV2",
    "ExperimentalMaxIntraOpParallelismDataset",
    "ExperimentalPrivateThreadPoolDataset",
    "FilterDataset",
    "Identity",
    "MapDataset",
    "MaxIntraOpParallelismDataset",
    "ModelDataset",
    "OptimizeDataset",
    "ParallelMapDataset",
    "PrivateThreadPoolDataset",
    "ReduceDataset",
    "RepeatDataset",
    "ShardDataset",
    "ShuffleAndRepeatDataset",
    "ShuffleDataset",
    "ShuffleDatasetV2",
    "ShuffleDatasetV3",
    "SkipDataset",
    "TakeDataset",
    "WindowDataset",
};

}  // namespace

Status Slack::RecursivelyHandleOp(const MutableGraphView& graph,
                                  NodeDef* dataset_node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslackDTcc mht_1(mht_1_v, 253, "", "./tensorflow/core/grappler/optimizers/data/slack.cc", "Slack::RecursivelyHandleOp");

  if (dataset_node->op() == kPrefetchDatasetOp) {
    if (HasNodeAttr(*dataset_node, "slack_period")) {
      (*dataset_node->mutable_attr())["slack_period"].set_i(slack_period_);
    } else {
      AddNodeAttr("slack_period", slack_period_, dataset_node);
    }
    return Status::OK();
  }
  if (IsDatasetNodeOfType(*dataset_node, kPassThroughOps)) {
    NodeDef* input_node = graph_utils::GetInputNode(*dataset_node, graph, 0);
    return RecursivelyHandleOp(graph, input_node);
  }
  if (IsDatasetNodeOfType(*dataset_node, kMultipleInputsDatasetOps)) {
    // For all multiple input datasets, all inputs are datasets themselves
    for (int i = 0; i < dataset_node->input_size(); ++i) {
      NodeDef* input_node = graph_utils::GetInputNode(*dataset_node, graph, i);
      TF_RETURN_IF_ERROR(RecursivelyHandleOp(graph, input_node));
    }
    return Status::OK();
  }

  LOG(WARNING) << "Could not find a final `prefetch` in the input pipeline to "
                  "which to introduce slack.";
  return Status::OK();
}

Status Slack::OptimizeAndCollectStats(Cluster* cluster,
                                      const GrapplerItem& item,
                                      GraphDef* output,
                                      OptimizationStats* stats) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslackDTcc mht_2(mht_2_v, 286, "", "./tensorflow/core/grappler/optimizers/data/slack.cc", "Slack::OptimizeAndCollectStats");

  if (slack_period_ < 1)
    return errors::InvalidArgument("Invalid `slack_period` parameter: ",
                                   slack_period_);

  *output = item.graph;
  MutableGraphView graph(output);

  // If the GrapplerItem is derived from a FunctionDef, we don't optimize it,
  // because we only want to add slack to the prefetch on the main dataset
  // pipeline.
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph))
    return Status::OK();

  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }
  // Walks the input pipeline backwards from the fetch node to find the last
  // PrefetchDataset node in the pipeline.
  NodeDef* dataset_node = graph.GetNode(item.fetch.at(0));
  return RecursivelyHandleOp(graph, dataset_node);
}

REGISTER_GRAPH_OPTIMIZER_AS(Slack, "slack");

}  // namespace grappler
}  // namespace tensorflow
