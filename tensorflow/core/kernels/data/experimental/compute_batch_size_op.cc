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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompute_batch_size_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompute_batch_size_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompute_batch_size_opDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

using grappler::graph_utils::GetScalarConstNodeValue;

constexpr char kMapAndBatchOp[] = "MapAndBatchDataset";
constexpr char kExperimentalMapAndBatchOp[] = "ExperimentalMapAndBatchDataset";

constexpr std::array<const char*, 4> kBatchDatasetOps = {
    "BatchDataset",
    "PaddedBatchDataset",
    kMapAndBatchOp,
    kExperimentalMapAndBatchOp,
};

constexpr std::array<const char*, 2> kMultipleInputDatasetOps = {
    "ConcatenateDataset",
    "ZipDataset",
};

constexpr std::array<const char*, 16> kPassThroughOps = {
    "AssertCardinalityDataset",
    "CacheDataset",
    "FilterDataset",
    "FinalizeDataset",
    "Identity",
    "ModelDataset",
    "OptimizeDataset",
    "OptionsDataset",
    "ParseExampleDataset",
    "PrefetchDataset",
    "RepeatDataset",
    "ShardDataset",
    "ShuffleAndRepeatDataset",
    "ShuffleDataset",
    "SkipDataset",
    "TakeDataset",
};

template <std::size_t SIZE>
bool IsDatasetNodeOfType(const NodeDef& node,
                         const std::array<const char*, SIZE>& arr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompute_batch_size_opDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/kernels/data/experimental/compute_batch_size_op.cc", "IsDatasetNodeOfType");

  for (const auto& dataset_op : arr) {
    if (MatchesAnyVersion(dataset_op, node.op())) return true;
  }
  return false;
}

const NodeDef* GetInputNode(const NodeDef& node,
                            const grappler::GraphView& graph,
                            int64_t input_index) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompute_batch_size_opDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/kernels/data/experimental/compute_batch_size_op.cc", "GetInputNode");

  if (node.input_size() == 0) return nullptr;
  grappler::GraphView::InputPort input_port =
      graph.GetInputPort(node.name(), input_index);
  return graph.GetRegularFanin(input_port).node;
}

// TODO(rachelim): This op traverses the dataset graph using a allowlist-based
// approach. As an alternative, we could instead rewrite all batching datasets'
// drop_remainder parameter to True, then rerun the dataset graph to derive
// new output shapes using C++ shape inference. This is more robust in cases
// where datasets have shape inference implemented in C++. If this allowlist-
// based approach proves hard to maintain, consider doing the alternative.
class ComputeBatchSizeOp : public OpKernel {
 public:
  explicit ComputeBatchSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompute_batch_size_opDTcc mht_2(mht_2_v, 267, "", "./tensorflow/core/kernels/data/experimental/compute_batch_size_op.cc", "ComputeBatchSizeOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompute_batch_size_opDTcc mht_3(mht_3_v, 272, "", "./tensorflow/core/kernels/data/experimental/compute_batch_size_op.cc", "Compute");

    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));

    std::vector<std::pair<string, Tensor>> input_list;
    GraphDef graph_def;
    string dataset_node_name;
    OP_REQUIRES_OK(ctx, AsGraphDefForRewrite(ctx, dataset, &input_list,
                                             &graph_def, &dataset_node_name));

    // Create GraphView for easier traversal of graph.
    grappler::GraphView graph_view(&graph_def);

    const NodeDef* node = graph_view.GetNode(dataset_node_name);
    OP_REQUIRES(ctx, node != nullptr,
                errors::InvalidArgument("Node does not exist in graph"));
    int64_t batch_size = GetBatchSize(*node, graph_view);
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &result));
    result->scalar<int64_t>()() = batch_size;
  }

 private:
  int64_t GetBatchSizeFromBatchNode(const NodeDef& node,
                                    const grappler::GraphView& graph) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompute_batch_size_opDTcc mht_4(mht_4_v, 299, "", "./tensorflow/core/kernels/data/experimental/compute_batch_size_op.cc", "GetBatchSizeFromBatchNode");

    int64_t arg_index;
    if (node.op() == kMapAndBatchOp ||
        node.op() == kExperimentalMapAndBatchOp) {
      arg_index = node.input_size() - 3;
    } else {
      arg_index = 1;
    }

    auto batch_size_node = GetInputNode(node, graph, arg_index);
    int64_t batch_size;
    auto s = GetScalarConstNodeValue(*batch_size_node, &batch_size);
    if (!s.ok()) {
      VLOG(1) << "Could not compute static batch size. Found batching dataset ("
              << node.name() << "), but failed to get its input batch size: "
              << s.error_message();
      return -1;
    }
    return batch_size;
  }

  // Helper function that returns the static 0th dimension of a given dataset
  // node in the graph. It starts from a node in the graph and recursively
  // traverses its inputs until it finds a valid BatchDataset operation,
  // and returns its batch size. If the batch size cannot be determined,
  // returns -1.
  //
  // During recursion, it handles four kinds of cases:
  // 1. BatchDataset type ops: Returns the value from its batch_size input node.
  // 2. Zip / Concatenate dataset ops: Recurses into all inputs to these ops,
  //    which are themselves all datasets, and returns the batch sizes computed
  //    by the inputs if they are all the same.
  // 3. Core dataset ops which cannot change the size of the 0th dimension of
  //    dataset output elements: Recurses into the first input parameter.
  // 4. All other ops: Fail, returning -1 for unknown.
  // TODO(rachelim): For FlatMap type mapping dataset ops, recurse into the
  // function definition.
  int64_t GetBatchSize(const NodeDef& node, const grappler::GraphView& graph) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPScompute_batch_size_opDTcc mht_5(mht_5_v, 339, "", "./tensorflow/core/kernels/data/experimental/compute_batch_size_op.cc", "GetBatchSize");

    if (IsDatasetNodeOfType(node, kBatchDatasetOps)) {
      return GetBatchSizeFromBatchNode(node, graph);
    }
    if (IsDatasetNodeOfType(node, kMultipleInputDatasetOps)) {
      const NodeDef* input_0 = GetInputNode(node, graph, 0);
      int64_t batch_size_0 = GetBatchSize(*input_0, graph);
      for (int i = 1; i < node.input_size(); ++i) {
        const NodeDef* input = GetInputNode(node, graph, i);
        auto batch_size_i = GetBatchSize(*input, graph);
        if (batch_size_i != batch_size_0) {
          VLOG(1) << "Could not compute batch size: inputs to " << node.name()
                  << " (" << node.op() << ") had different batch sizes."
                  << " Namely, input 0 had batch size " << batch_size_0
                  << " while input " << i << " had batch size " << batch_size_i
                  << ".";
          return -1;
        }
      }
      return batch_size_0;
    }
    if (IsDatasetNodeOfType(node, kPassThroughOps)) {
      const NodeDef* input = GetInputNode(node, graph, 0);
      return GetBatchSize(*input, graph);
    }
    VLOG(1) << "Encountered dataset node " << node.name() << " (" << node.op()
            << ") that prevented further static batch size analysis.";

    return -1;
  }
};

REGISTER_KERNEL_BUILDER(Name("ComputeBatchSize").Device(DEVICE_CPU),
                        ComputeBatchSizeOp);

}  // anonymous namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
