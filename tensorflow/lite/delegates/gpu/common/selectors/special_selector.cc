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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSspecial_selectorDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSspecial_selectorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSspecial_selectorDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/selectors/special_selector.h"

#include <string>
#include <utility>

#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/flops_util.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/depthwise_conv_plus_1x1_conv.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/fc_fc_add.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {
absl::Status TryDepthwiseConvPlus1x1Conv(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSspecial_selectorDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/delegates/gpu/common/selectors/special_selector.cc", "TryDepthwiseConvPlus1x1Conv");

  auto* dw_node = graph.GetNode(first_node_id);
  if (dw_node == nullptr) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  if (OperationTypeFromString(dw_node->operation.type) !=
      OperationType::DEPTHWISE_CONVOLUTION) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  auto dw_inputs = graph.FindInputs(dw_node->id);
  if (dw_inputs.size() != 1) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  auto dw_outputs = graph.FindOutputs(dw_node->id);
  auto consumers = graph.FindConsumers(dw_outputs[0]->id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }

  Node* next_node;
  next_node = consumers[0];
  if (next_node == nullptr) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  if (consumed_nodes->find(next_node->id) != consumed_nodes->end()) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  Node* relu_node = nullptr;
  ReLUAttributes relu_attributes;
  if (OperationTypeFromString(next_node->operation.type) ==
      OperationType::RELU) {
    relu_node = next_node;
    auto relu_outputs = graph.FindOutputs(relu_node->id);
    consumers = graph.FindConsumers(relu_outputs[0]->id);
    if (consumers.size() != 1) {
      return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
    }
    relu_attributes =
        absl::any_cast<ReLUAttributes>(relu_node->operation.attributes);
    next_node = consumers[0];
  }

  auto* conv_node = next_node;
  if (OperationTypeFromString(conv_node->operation.type) !=
      OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  if (graph.FindInputs(conv_node->id).size() != 1) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  auto dw_attr = absl::any_cast<DepthwiseConvolution2DAttributes>(
      dw_node->operation.attributes);
  auto conv_attr =
      absl::any_cast<Convolution2DAttributes>(conv_node->operation.attributes);
  auto conv_outputs = graph.FindOutputs(conv_node->id);
  OperationDef op_def;
  op_def.precision = precision;
  auto it = tensor_descriptors.find(dw_inputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(conv_outputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.dst_tensors.push_back(it->second);
  }
  if (!IsDepthwiseConvPlus1x1ConvSupported(op_def, gpu_info, dw_attr,
                                           conv_attr)) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph(dw_inputs, conv_outputs, gpu_subgraph);
  ReLUAttributes* relu_attr_ptr = relu_node ? &relu_attributes : nullptr;
  auto operation = CreateDepthwiseConvPlus1x1Conv(op_def, gpu_info, dw_attr,
                                                  conv_attr, relu_attr_ptr);
  *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
  (*gpu_op)->flops_ = GetDepthwiseConvolutionFlops(dw_outputs[0]->tensor.shape,
                                                   dw_attr.weights.shape) +
                      GetConvolutionFlops(conv_outputs[0]->tensor.shape,
                                          conv_attr.weights.shape);
  std::string fused_nodes = std::to_string(dw_node->id);
  if (relu_node) {
    fused_nodes += " " + std::to_string(relu_node->id);
  }
  fused_nodes += " " + std::to_string(conv_node->id);
  gpu_subgraph->operations[0].name =
      "depthwise_conv_plus_1x1_conv " + fused_nodes;
  consumed_nodes->insert(dw_node->id);
  if (relu_node) {
    consumed_nodes->insert(relu_node->id);
  }
  consumed_nodes->insert(conv_node->id);
  return absl::OkStatus();
}

// fully connected + fully connected + add
absl::Status TryFCFCAdd(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSspecial_selectorDTcc mht_1(mht_1_v, 311, "", "./tensorflow/lite/delegates/gpu/common/selectors/special_selector.cc", "TryFCFCAdd");

  auto* fc0_node = graph.GetNode(first_node_id);
  if (fc0_node == nullptr) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto first_op_type = OperationTypeFromString(fc0_node->operation.type);
  if (first_op_type != OperationType::FULLY_CONNECTED &&
      first_op_type != OperationType::FULLY_CONNECTED_INT8) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  const bool first_quantized =
      first_op_type == OperationType::FULLY_CONNECTED_INT8;
  auto fc0_inputs = graph.FindInputs(fc0_node->id);
  if (fc0_inputs.size() != 1) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto fc0_output_id = graph.FindOutputs(fc0_node->id)[0]->id;
  auto consumers = graph.FindConsumers(fc0_output_id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto* add_node = consumers[0];
  if (add_node == nullptr) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  if (consumed_nodes->find(add_node->id) != consumed_nodes->end()) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  if (OperationTypeFromString(add_node->operation.type) != OperationType::ADD) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto add_inputs = graph.FindInputs(add_node->id);
  if (add_inputs.size() != 2) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto fc1_output_id = add_inputs[0]->id + add_inputs[1]->id - fc0_output_id;
  auto* fc1_node = graph.FindProducer(fc1_output_id);
  if (fc1_node == nullptr) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto second_op_type = OperationTypeFromString(fc1_node->operation.type);
  if (second_op_type != OperationType::FULLY_CONNECTED &&
      second_op_type != OperationType::FULLY_CONNECTED_INT8) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  const bool second_quantized =
      second_op_type == OperationType::FULLY_CONNECTED_INT8;
  const bool both_quantized = first_quantized && second_quantized;
  const bool both_not_quantized = !first_quantized && !second_quantized;
  if (!(both_quantized || both_not_quantized)) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  if (consumed_nodes->find(fc1_node->id) != consumed_nodes->end()) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto fc1_inputs = graph.FindInputs(fc1_node->id);
  if (fc1_inputs.size() != 1) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto add_outputs = graph.FindOutputs(add_node->id);

  OperationDef op_def;
  op_def.precision = precision;
  auto it = tensor_descriptors.find(fc0_inputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(fc1_inputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(add_outputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.dst_tensors.push_back(it->second);
  }

  for (int i = 0; i < fc1_inputs.size(); ++i) {
    fc0_inputs.push_back(fc1_inputs[i]);
  }
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph(fc0_inputs, add_outputs, gpu_subgraph);
  FCFCAdd fc;
  if (both_not_quantized) {
    auto fc0_attr = absl::any_cast<FullyConnectedAttributes>(
        fc0_node->operation.attributes);
    auto fc1_attr = absl::any_cast<FullyConnectedAttributes>(
        fc1_node->operation.attributes);
    if (fc0_attr.weights.shape.o != fc1_attr.weights.shape.o) {
      return absl::NotFoundError("FCFCAdd not suitable.");
    }
    fc = CreateFCFCAdd(gpu_info, op_def, fc0_attr, fc1_attr);
  } else {
    // both_quantized
    auto fc0_attr = absl::any_cast<FullyConnectedInt8Attributes>(
        fc0_node->operation.attributes);
    auto fc1_attr = absl::any_cast<FullyConnectedInt8Attributes>(
        fc1_node->operation.attributes);
    if (fc0_attr.weights.shape.o != fc1_attr.weights.shape.o) {
      return absl::NotFoundError("FCFCAdd not suitable.");
    }
    fc = CreateFCFCAdd(gpu_info, op_def, fc0_attr, fc1_attr);
  }
  *gpu_op = absl::make_unique<FCFCAdd>(std::move(fc));
  const std::string fused_nodes = std::to_string(fc0_node->id) + " " +
                                  std::to_string(fc1_node->id) + " " +
                                  std::to_string(add_node->id);
  gpu_subgraph->operations[0].name =
      "fully_connected_x2_and_add " + fused_nodes;
  consumed_nodes->insert(fc0_node->id);
  consumed_nodes->insert(fc1_node->id);
  consumed_nodes->insert(add_node->id);
  return absl::OkStatus();
}
}  // namespace

absl::Status GPUSubgraphFromGraph(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSselectorsPSspecial_selectorDTcc mht_2(mht_2_v, 433, "", "./tensorflow/lite/delegates/gpu/common/selectors/special_selector.cc", "GPUSubgraphFromGraph");

  if ((gpu_info.IsAdreno() || gpu_info.IsNvidia() || gpu_info.IsMali() ||
       gpu_info.IsApple() || gpu_info.IsAMD()) &&
      TryDepthwiseConvPlus1x1Conv(gpu_info, precision, graph, first_node_id,
                                  tensor_descriptors, consumed_nodes,
                                  gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if ((gpu_info.IsIntel() || gpu_info.IsNvidia() || gpu_info.IsAMD()) &&
      TryFCFCAdd(gpu_info, precision, graph, first_node_id, tensor_descriptors,
                 consumed_nodes, gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if (TryFusedPointwiseConv(graph, first_node_id, precision, tensor_descriptors,
                            consumed_nodes, gpu_subgraph)
          .ok()) {
    gpu_subgraph->operations[0].name = "slice_mul_mean_concat";
    return absl::OkStatus();
  }
  return absl::NotFoundError("No special combination.");
}

}  // namespace gpu
}  // namespace tflite
