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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/noop_elimination.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kIdentity[] = "Identity";

bool IsTakeAll(const NodeDef& take_node, const MutableGraphView& graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "IsTakeAll");

  if (take_node.op() != "TakeDataset") return false;

  const auto& count_node = *graph.GetNode(take_node.input(1));
  if (count_node.op() != "Const") return false;
  // We are looking only for 'take' with negative count.
  const auto& tensor = count_node.attr().at("value").tensor();
  if (tensor.int64_val_size()) return tensor.int64_val(0) < 0;
  return false;
}

bool IsConstNodeWithValue(const NodeDef& node, int value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "IsConstNodeWithValue");

  if (node.op() != "Const") return false;
  const auto& tensor = node.attr().at("value").tensor();
  if (tensor.int64_val_size()) return tensor.int64_val(0) == value;
  return value == 0;
}

bool IsSkipNone(const NodeDef& skip_node, const MutableGraphView& graph) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "IsSkipNone");

  if (skip_node.op() != "SkipDataset") return false;
  // We are looking only for skip(0) nodes.
  return IsConstNodeWithValue(*graph.GetNode(skip_node.input(1)), 0);
}

bool IsRepeatOne(const NodeDef& repeat_node, const MutableGraphView& graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "IsRepeatOne");

  if (repeat_node.op() != "RepeatDataset") return false;
  // We are looking only for repeat(1) nodes.
  return IsConstNodeWithValue(*graph.GetNode(repeat_node.input(1)), 1);
}

bool IsPrefetchZero(const NodeDef& prefetch_node,
                    const MutableGraphView& graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_4(mht_4_v, 249, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "IsPrefetchZero");

  if (prefetch_node.op() != "PrefetchDataset") return false;
  // We are looking only for prefetch(0) nodes.
  return IsConstNodeWithValue(*graph.GetNode(prefetch_node.input(1)), 0);
}

bool IsShardOne(const NodeDef& shard_node, const MutableGraphView& graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_5(mht_5_v, 258, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "IsShardOne");

  if (shard_node.op() != "ShardDataset") return false;
  // We are looking only for shard(0) nodes.
  return IsConstNodeWithValue(*graph.GetNode(shard_node.input(1)), 1);
}

bool IsOutputIdentityOfInput(const FunctionDef& fdef, const string& output_arg,
                             const string& input_arg) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("output_arg: \"" + output_arg + "\"");
   mht_6_v.push_back("input_arg: \"" + input_arg + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_6(mht_6_v, 270, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "IsOutputIdentityOfInput");

  if (!fdef.ret().contains(output_arg)) {
    LOG(WARNING)
        << "Malformed FunctionDef: ret dict does not contain output arg key.";
    return false;
  }

  const auto& ret_val = fdef.ret().at(output_arg);
  auto input = function_utils::FunctionDefTensorDesc(ret_val);

  // Walk from output to input. If any node along the path is not an
  // Identity node, return false.
  while (function_utils::ContainsFunctionNodeWithName(input.node_name, fdef)) {
    int idx = function_utils::FindFunctionNodeWithName(input.node_name, fdef);

    const NodeDef& node = fdef.node_def(idx);
    if (node.op() != kIdentity) {
      return false;
    }

    input = function_utils::FunctionDefTensorDesc(node.input(0));
  }

  // If we get here, input is not a node. Check that it matches the correct
  // input arg name.
  return input.node_name == input_arg;
}

bool IsMapIdentity(const NodeDef& map_node, const MutableGraphView& graph) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_7(mht_7_v, 301, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "IsMapIdentity");

  if (map_node.op() != "MapDataset" && map_node.op() != "ParallelMapDataset" &&
      map_node.op() != "ParallelMapDatasetV2") {
    return false;
  }

  // We are looking only for map(lambda *x: x) nodes.

  // Don't eliminate map nodes with captured arguments.
  if (map_node.attr().at("Targuments").list().type_size() != 0) return false;

  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             graph.graph()->library());
  const FunctionDef* fdef =
      function_library.Find(map_node.attr().at("f").func().name());

  // Don't eliminate map nodes with stateful functions.
  if (function_utils::IsFunctionStateful(function_library, *fdef)) return false;

  const auto& sig = fdef->signature();
  if (sig.input_arg_size() != sig.output_arg_size()) return false;

  // For each output, check that it maps to input i
  for (int i = 0; i < sig.input_arg_size(); ++i) {
    if (!IsOutputIdentityOfInput(*fdef, sig.output_arg(i).name(),
                                 sig.input_arg(i).name())) {
      return false;
    }
  }
  return true;
}

bool IsNoOp(const NodeDef& node, const MutableGraphView& graph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_8(mht_8_v, 336, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "IsNoOp");

  return IsTakeAll(node, graph) || IsSkipNone(node, graph) ||
         IsRepeatOne(node, graph) || IsPrefetchZero(node, graph) ||
         IsShardOne(node, graph) || IsMapIdentity(node, graph);
}

}  // namespace

Status NoOpElimination::OptimizeAndCollectStats(Cluster* cluster,
                                                const GrapplerItem& item,
                                                GraphDef* output,
                                                OptimizationStats* stats) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSnoop_eliminationDTcc mht_9(mht_9_v, 350, "", "./tensorflow/core/grappler/optimizers/data/noop_elimination.cc", "NoOpElimination::OptimizeAndCollectStats");

  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  for (const NodeDef& node : item.graph.node()) {
    if (!IsNoOp(node, graph)) continue;

    NodeDef* const parent = graph_utils::GetInputNode(node, graph);
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(node.name(), parent->name()));

    nodes_to_delete.insert(node.name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(NoOpElimination, "noop_elimination");

}  // namespace grappler
}  // namespace tensorflow
