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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {

namespace {
std::string GenerateCode() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.cc", "GenerateCode");

  std::string c = R"(
MAIN_FUNCTION($0) {
  int X = GLOBAL_ID_0;
  int Y = GLOBAL_ID_1;
  int S = GLOBAL_ID_2;
  if (X >= args.dst_tensor.Width() ||
      Y >= args.dst_tensor.Height() ||
      S >= args.dst_tensor.Slices()) return;
  int4 offset0 = args.offsets.Read(S * 2 + 0, 0);
  int4 offset1 = args.offsets.Read(S * 2 + 1, 0);
  ACCUM_FLT4 res = INIT_ACCUM_FLT4(0.0f);
  FLT4 last_mask;
  int last_src_ch = (args.src_tensor.Slices() - 1) * 4;
  last_mask.x = INIT_FLT(1.0f);
  last_mask.y = last_src_ch + 1 < args.src_tensor.Channels() ? INIT_FLT(1.0f) : INIT_FLT(0.0f);
  last_mask.z = last_src_ch + 2 < args.src_tensor.Channels() ? INIT_FLT(1.0f) : INIT_FLT(0.0f);
  last_mask.w = last_src_ch + 3 < args.src_tensor.Channels() ? INIT_FLT(1.0f) : INIT_FLT(0.0f);
  for (int s = 0; s < args.src_tensor.Slices(); ++s) {
    FLT4 src = args.src_tensor.Read(X, Y, s);
    FLT4 w0 = args.weights_tensor.Read(X + offset0.x, Y + offset0.y, s);
    FLT4 w1 = args.weights_tensor.Read(X + offset0.z, Y + offset0.w, s);
    FLT4 w2 = args.weights_tensor.Read(X + offset1.x, Y + offset1.y, s);
    FLT4 w3 = args.weights_tensor.Read(X + offset1.z, Y + offset1.w, s);
    FLT4 mask = INIT_FLT4(1.0f);
    if (s == (args.src_tensor.Slices() - 1)) {
      mask = last_mask;
    }
    src *= mask;
    res.x += dot(src, w0);
    res.y += dot(src, w1);
    res.z += dot(src, w2);
    res.w += dot(src, w3);
  }
  FLT4 result = TO_FLT4(res) / INIT_FLT(args.src_tensor.Channels());
  args.dst_tensor.Write(result, X, Y, S);
})";
  return c;
}

struct NodeContext {
  Node* node;
  std::vector<Value*> inputs;
  std::vector<Value*> outputs;
};

absl::Status IsNode(const GraphFloat32& graph, OperationType op_type,
                    int inputs_count, int outputs_count, Node* node,
                    NodeContext* node_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc mht_1(mht_1_v, 247, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.cc", "IsNode");

  const std::string op_desc = ToString(op_type);
  node_context->node = node;
  if (node_context->node == nullptr) {
    return absl::NotFoundError(absl::StrCat("Invalid ", op_desc, " node."));
  }
  if (OperationTypeFromString(node_context->node->operation.type) != op_type) {
    return absl::InternalError(
        absl::StrCat("Not correct node type. Expected ", op_desc, ", received ",
                     node_context->node->operation.type));
  }
  node_context->inputs = graph.FindInputs(node_context->node->id);
  node_context->outputs = graph.FindOutputs(node_context->node->id);
  if (inputs_count != -1) {
    if (node_context->inputs.size() != inputs_count) {
      return absl::InternalError(
          absl::StrCat("Expected ", inputs_count, " input in a ", op_desc,
                       " node. Node has ", node_context->inputs.size()));
    }
  }
  if (node_context->outputs.size() != outputs_count) {
    return absl::InternalError(
        absl::StrCat("Expected ", outputs_count, " output in a ", op_desc,
                     " node. Node has ", node_context->outputs.size()));
  }
  return absl::OkStatus();
}

absl::Status IsMeanNode(const GraphFloat32& graph, Node* node,
                        NodeContext* node_context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc mht_2(mht_2_v, 279, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.cc", "IsMeanNode");

  RETURN_IF_ERROR(IsNode(graph, OperationType::MEAN, 1, 1, node, node_context));
  auto mean_attr =
      absl::any_cast<MeanAttributes>(node_context->node->operation.attributes);
  if (mean_attr.dims != std::set<Axis>{Axis::CHANNELS}) {
    return absl::InternalError("Expected mean node with channels reduction.");
  }
  return absl::OkStatus();
}

absl::Status IsMulNode(const GraphFloat32& graph, Node* node,
                       NodeContext* node_context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc mht_3(mht_3_v, 293, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.cc", "IsMulNode");

  RETURN_IF_ERROR(IsNode(graph, OperationType::MUL, 2, 1, node, node_context));
  if (node_context->inputs[0]->tensor.shape !=
      node_context->inputs[1]->tensor.shape) {
    return absl::InternalError("Expected mul node with 2 equal tensors.");
  }
  return absl::OkStatus();
}

absl::Status IsSliceNode(const GraphFloat32& graph, Node* node,
                         NodeContext* node_context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc mht_4(mht_4_v, 306, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.cc", "IsSliceNode");

  RETURN_IF_ERROR(
      IsNode(graph, OperationType::SLICE, 1, 1, node, node_context));
  auto slice_attr =
      absl::any_cast<SliceAttributes>(node_context->node->operation.attributes);
  if (slice_attr.strides != BHWC(1, 1, 1, 1)) {
    return absl::InternalError("Not valid attributes in slice node.");
  }
  return absl::OkStatus();
}

absl::Status IsConcatNode(const GraphFloat32& graph, Node* node,
                          NodeContext* node_context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc mht_5(mht_5_v, 321, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.cc", "IsConcatNode");

  RETURN_IF_ERROR(
      IsNode(graph, OperationType::CONCAT, -1, 1, node, node_context));
  auto concat_attr = absl::any_cast<ConcatAttributes>(
      node_context->node->operation.attributes);
  if (concat_attr.axis != Axis::CHANNELS) {
    return absl::InternalError("Not valid attributes in concat node.");
  }
  return absl::OkStatus();
}

absl::Status GetOffset(const GraphFloat32& graph, NodeId concat_input_node,
                       NodeId second_commom_input_id, int* offset_x,
                       int* offset_y, std::set<NodeId>* consumed_nodes) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc mht_6(mht_6_v, 337, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.cc", "GetOffset");

  NodeContext mean_node, mul_node, slice_node;
  RETURN_IF_ERROR(
      IsMeanNode(graph, graph.FindProducer(concat_input_node), &mean_node));
  RETURN_IF_ERROR(
      IsMulNode(graph, graph.FindProducer(mean_node.inputs[0]->id), &mul_node));
  const ValueId slice_output_id =
      mul_node.inputs[0]->id == second_commom_input_id ? mul_node.inputs[1]->id
                                                       : mul_node.inputs[0]->id;
  RETURN_IF_ERROR(
      IsSliceNode(graph, graph.FindProducer(slice_output_id), &slice_node));
  auto slice_attr =
      absl::any_cast<SliceAttributes>(slice_node.node->operation.attributes);
  *offset_x = slice_attr.starts.w;
  *offset_y = slice_attr.starts.h;
  consumed_nodes->insert(mean_node.node->id);
  consumed_nodes->insert(mul_node.node->id);
  consumed_nodes->insert(slice_node.node->id);
  return absl::OkStatus();
}
}  // namespace

GPUOperation CreateConvPointwise(const OperationDef& definition,
                                 const ConvPointwiseAttributes& attr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc mht_7(mht_7_v, 363, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.cc", "CreateConvPointwise");

  const int dst_channels = attr.offsets.size();
  const int dst_depth = DivideRoundUp(dst_channels, 4);
  std::vector<int32_t> offsets_data(dst_depth * 2 * 4, 0);
  for (int i = 0; i < attr.offsets.size(); ++i) {
    offsets_data[i * 2 + 0] = attr.offsets[i].x;
    offsets_data[i * 2 + 1] = attr.offsets[i].y;
  }
  for (int i = attr.offsets.size(); i < offsets_data.size() / 2; ++i) {
    offsets_data[i * 2 + 0] = attr.offsets.back().x;
    offsets_data[i * 2 + 1] = attr.offsets.back().y;
  }

  Texture2DDescriptor desc;
  desc.element_type = DataType::INT32;
  desc.size = int2(dst_depth * 2, 1);
  desc.data.resize(offsets_data.size() * 4);
  memcpy(desc.data.data(), offsets_data.data(), offsets_data.size() * 4);

  GPUOperation op(definition);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddSrcTensor("weights_tensor", definition.src_tensors[1]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GenerateCode();
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.args_.AddObject("offsets",
                     absl::make_unique<Texture2DDescriptor>(std::move(desc)));
  return op;
}

absl::Status TryFusedPointwiseConv(
    const GraphFloat32& graph, NodeId first_node_id,
    CalculationsPrecision precision,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStasksPSspecialPSconv_pointwiseDTcc mht_8(mht_8_v, 400, "", "./tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.cc", "TryFusedPointwiseConv");

  NodeContext slice_node;
  RETURN_IF_ERROR(
      IsSliceNode(graph, graph.GetNode(first_node_id), &slice_node));
  const auto& first_commom_input = slice_node.inputs[0];
  auto slice_consumers = graph.FindConsumers(slice_node.outputs[0]->id);
  if (slice_consumers.size() != 1) {
    return absl::NotFoundError("FusedPointwiseConv not suitable.");
  }
  NodeContext mul_node;
  RETURN_IF_ERROR(IsMulNode(graph, slice_consumers[0], &mul_node));
  const auto& second_commom_input =
      mul_node.inputs[0]->id == slice_node.outputs[0]->id ? mul_node.inputs[1]
                                                          : mul_node.inputs[0];
  auto mul_consumers = graph.FindConsumers(mul_node.outputs[0]->id);
  if (mul_consumers.size() != 1) {
    return absl::NotFoundError("FusedPointwiseConv not suitable.");
  }
  NodeContext mean_node;
  RETURN_IF_ERROR(IsMeanNode(graph, mul_consumers[0], &mean_node));
  auto mean_consumers = graph.FindConsumers(mean_node.outputs[0]->id);
  if (mean_consumers.size() != 1) {
    return absl::NotFoundError("FusedPointwiseConv not suitable.");
  }
  NodeContext concat_node;
  RETURN_IF_ERROR(IsConcatNode(graph, mean_consumers[0], &concat_node));
  ConvPointwiseAttributes op_attr;
  std::set<NodeId> temp_consumed_nodes;
  for (const auto& concat_input : concat_node.inputs) {
    int offset_x, offset_y;
    RETURN_IF_ERROR(GetOffset(graph, concat_input->id, second_commom_input->id,
                              &offset_x, &offset_y, &temp_consumed_nodes));
    op_attr.offsets.push_back(int2(offset_x, offset_y));
  }
  consumed_nodes->insert(temp_consumed_nodes.begin(),
                         temp_consumed_nodes.end());
  consumed_nodes->insert(concat_node.node->id);
  OperationDef op_def;
  op_def.precision = precision;
  auto it = tensor_descriptors.find(second_commom_input->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(first_commom_input->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(concat_node.outputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.dst_tensors.push_back(it->second);
  }
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph({second_commom_input, first_commom_input},
                           {concat_node.outputs[0]}, gpu_subgraph);
  auto operation = CreateConvPointwise(op_def, op_attr);
  *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
