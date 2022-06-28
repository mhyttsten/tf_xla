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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/gpu_model.h"

#include <algorithm>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/operation_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/special_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/subgraph.h"
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/add_bias.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/global_pooling_to_reduce_op.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"

namespace tflite {
namespace gpu {

namespace {
bool IsReady(const absl::flat_hash_set<ValueId>& ready_tensors,
             const GpuNode& node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "IsReady");

  for (const ValueId in_id : node.inputs) {
    if (ready_tensors.find(in_id) == ready_tensors.end()) {
      return false;
    }
  }
  return true;
}

absl::Status MergeGpuNodes(GpuNode* src, GpuNode* dst) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "MergeGpuNodes");

  for (int j = 1; j < src->inputs.size(); ++j) {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " linked : " + src->name;
  return dst->gpu_operation->AddOperation(src->gpu_operation.get());
}

flatbuffers::Offset<data::TensorDescWithId> Encode(
    const TensorDescriptor& desc, const ValueId& id,
    flatbuffers::FlatBufferBuilder* builder) {
  auto desc_fb = Encode(desc, builder);
  data::TensorDescWithIdBuilder desc_builder(*builder);
  desc_builder.add_desc(desc_fb);
  desc_builder.add_id(id);
  return desc_builder.Finish();
}

flatbuffers::Offset<data::GpuNode> Encode(
    const GpuNode& node, flatbuffers::FlatBufferBuilder* builder) {
  auto op_fb = Encode(*node.gpu_operation, builder);
  std::vector<int32_t> in_ids(node.inputs.size());
  for (int i = 0; i < in_ids.size(); ++i) {
    in_ids[i] = node.inputs[i];
  }
  std::vector<int32_t> out_ids(node.outputs.size());
  for (int i = 0; i < out_ids.size(); ++i) {
    out_ids[i] = node.outputs[i];
  }
  auto in_ids_fb = builder->CreateVector(in_ids);
  auto out_ids_fb = builder->CreateVector(out_ids);
  auto name_fb = builder->CreateString(node.name);
  data::GpuNodeBuilder node_builder(*builder);
  node_builder.add_gpu_op(op_fb);
  node_builder.add_input_ids(in_ids_fb);
  node_builder.add_output_ids(out_ids_fb);
  node_builder.add_name(name_fb);
  return node_builder.Finish();
}

absl::Status Decode(const data::GpuNode* fb_node, GpuNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_2(mht_2_v, 263, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Decode");

  GPUOperation op;
  RETURN_IF_ERROR(Decode(fb_node->gpu_op(), &op));
  node->gpu_operation = absl::make_unique<GPUOperation>(std::move(op));
  for (auto in_fb : *fb_node->input_ids()) {
    node->inputs.push_back(in_fb);
  }
  for (auto out_fb : *fb_node->output_ids()) {
    node->outputs.push_back(out_fb);
  }
  node->name = std::string(fb_node->name()->c_str(), fb_node->name()->size());

  return absl::OkStatus();
}

bool IsAssociativeLinkableOp(const Node& node,
                             const std::vector<Value*>& inputs,
                             const std::vector<Value*>& outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_3(mht_3_v, 283, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "IsAssociativeLinkableOp");

  if (inputs.size() == 1) {
    return false;
  }
  const OperationType op_type = OperationTypeFromString(node.operation.type);
  if (op_type != OperationType::ADD && op_type != OperationType::MUL) {
    return false;
  }

  const auto dst_shape = outputs[0]->tensor.shape;
  for (int i = 0; i < inputs.size(); ++i) {
    const auto src_shape = inputs[i]->tensor.shape;
    if (dst_shape.b != src_shape.b && src_shape.b == 1) {
      return false;
    }
    if (dst_shape.h != src_shape.h && src_shape.h == 1) {
      return false;
    }
    if (dst_shape.w != src_shape.w && src_shape.w == 1) {
      return false;
    }
    if (dst_shape.c != src_shape.c && src_shape.c == 1) {
      return false;
    }
  }
  return true;
}

absl::Status CheckExternalTensorDescription(const GpuInfo& gpu_info,
                                            const TensorDescriptor& tensor_desc,
                                            const BHWC& shape,
                                            DataType data_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_4(mht_4_v, 317, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "CheckExternalTensorDescription");

  if (tensor_desc.data_type != data_type) {
    return absl::InvalidArgumentError(
        "Global precision and precision of predefined/external tensors must be "
        "synchronized.");
  }
  const bool tensor_supported_layout = tensor_desc.layout == Layout::HWDC ||
                                       tensor_desc.layout == Layout::BHWDC ||
                                       tensor_desc.layout == Layout::HWC ||
                                       tensor_desc.layout == Layout::BHWC;
  if (!tensor_supported_layout) {
    return absl::InvalidArgumentError(
        "Currently no support of this layouts for spatial tensors.");
  }
  const bool has_depth =
      tensor_desc.layout == Layout::HWDC || tensor_desc.layout == Layout::BHWDC;
  if (has_depth) {
    return absl::InvalidArgumentError(
        "Currently no support of Depth dimension in predefined/external "
        "tensors.");
  }
  const bool has_batch =
      tensor_desc.layout == Layout::BHWC || tensor_desc.layout == Layout::BHWDC;
  if (has_batch && shape.b == 1) {
    return absl::InvalidArgumentError("Wrong layout, batch mismatch.");
  }
  if (!has_batch && shape.b != 1) {
    return absl::InvalidArgumentError("Wrong layout, batch mismatch.");
  }
  if (!tensor_desc.CanCreateTensorWithShape(gpu_info, shape).ok()) {
    return absl::UnavailableError(
        "Current device can not allocate tensor with this shape for "
        "predefined/external descriptor.");
  }
  return absl::OkStatus();
}

// Helper class for creating descriptors for appropriate tensors from
// GraphFloat32
// Also allows to create descriptors for new tensors(not present in
// GraphFloat32)
class TensorReserver {
 public:
  TensorReserver() : next_(0) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_5(mht_5_v, 363, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "TensorReserver");
}
  ValueId Add(const TensorDescriptor& dummy) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_6(mht_6_v, 367, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Add");

    reservations_[next_] = dummy;
    return next_++;
  }
  void Add(ValueId id, const TensorDescriptor& dummy) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_7(mht_7_v, 374, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Add");

    reservations_[id] = dummy;
  }
  ValueId GetNewId() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_8(mht_8_v, 380, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "GetNewId");
 return next_++; }
  void SetNext(ValueId id) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_9(mht_9_v, 384, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "SetNext");
 next_ = id; }
  TensorDescriptor Get(ValueId id) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_10(mht_10_v, 388, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Get");
 return reservations_[id]; }

 public:
  absl::flat_hash_map<ValueId, TensorDescriptor> reservations_;
  ValueId next_;
};

absl::Status ReserveGraphTensors(const CreateGpuModelInfo& create_info,
                                 const GpuInfo& gpu_info,
                                 const GraphFloat32& graph,
                                 TensorReserver* tensor_reserver) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_11(mht_11_v, 401, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "ReserveGraphTensors");

  ValueId max_id = 0;
  auto tensors = graph.values();
  auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
  for (auto& t : tensors) {
    const auto shape = graph.GetValue(t->id)->tensor.shape;
    auto it_predefined = create_info.predefined.find(t->id);
    auto it_immutable_external =
        create_info.external_immutable_tensors.find(t->id);
    auto it_mutable_external = create_info.external_mutable_tensors.find(t->id);
    int external_categories_count = 0;
    TensorDescriptor tensor_desc;
    if (it_predefined != create_info.predefined.end()) {
      external_categories_count++;
      tensor_desc = it_predefined->second;
    }
    if (it_immutable_external != create_info.external_immutable_tensors.end()) {
      external_categories_count++;
      tensor_desc = it_immutable_external->second->GetDescriptor();
    }
    if (it_mutable_external != create_info.external_mutable_tensors.end()) {
      external_categories_count++;
      tensor_desc = it_mutable_external->second;
    }
    if (external_categories_count > 1) {
      return absl::InvalidArgumentError(
          "Tensors ids from predefined / external_immutable_tensors / "
          "external_mutable_tensors should not intersect.");
    }
    if (external_categories_count == 1) {
      if (!(graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id))) {
        return absl::InvalidArgumentError(
            "Currently external can be used only for graph inputs/outputs");
      }
      RETURN_IF_ERROR(CheckExternalTensorDescription(gpu_info, tensor_desc,
                                                     shape, data_type));
    } else {
      TensorStorageType storage_type = create_info.storage_type;
      Layout layout = shape.b == 1 ? Layout::HWC : Layout::BHWC;
      const bool can_use_single_texture =
          storage_type == TensorStorageType::TEXTURE_2D ||
          storage_type == TensorStorageType::TEXTURE_3D ||
          storage_type == TensorStorageType::TEXTURE_ARRAY;
      if (graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id)) {
        if (shape.c < 4 && can_use_single_texture &&
            TensorDescriptor{data_type, TensorStorageType::SINGLE_TEXTURE_2D,
                             layout}
                .CanCreateTensorWithShape(gpu_info, shape)
                .ok()) {
          storage_type = TensorStorageType::SINGLE_TEXTURE_2D;
        }
      }
      RETURN_IF_ERROR(SelectBestStorageType(gpu_info, shape, storage_type,
                                            data_type, layout, &storage_type));
      tensor_desc = TensorDescriptor{data_type, storage_type, layout};
      if (gpu_info.IsApiMetal() &&
          storage_type == TensorStorageType::TEXTURE_2D) {
        tensor_desc.use_buffer_for_write_only_2d_texture = true;
      }
    }
    tensor_desc.SetBHWCShape(shape);
    tensor_reserver->Add(t->id, tensor_desc);
    max_id = std::max(max_id, t->id);
  }
  tensor_reserver->SetNext(max_id + 1);
  return absl::OkStatus();
}

absl::Status ConvertOperations(const GpuInfo& gpu_info,
                               const GraphFloat32& graph,
                               const CreateGpuModelInfo& create_info,
                               TensorReserver* tensor_reserver,
                               GpuModel* gpu_model) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_12(mht_12_v, 476, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "ConvertOperations");

  std::map<ValueId, TensorDescriptor> tensor_descriptors;
  const auto values = graph.values();
  for (auto value : values) {
    tensor_descriptors[value->id] = tensor_reserver->Get(value->id);
  }
  std::set<NodeId> consumed_nodes;
  std::vector<Node*> graph_nodes = graph.nodes();
  std::map<ValueId, int>
      tensor_usages;  // keeps latest index of operation that updated tensor
  for (const auto& input : gpu_model->input_ids_and_refs) {
    tensor_usages[input.first] = -1;  // so as inputs "updated" before operation
                                      // 0, we will mark them with -1
  }
  std::vector<SharedWeightsConvDesc> shared_conv_weights;
  std::vector<SharedWeightsConvDesc>* shared_conv_weights_ptr =
      create_info.hints.Check(ModelHints::kReuseConvWeights)
          ? &shared_conv_weights
          : nullptr;
  for (int i = 0; i < graph_nodes.size(); ++i) {
    const Node& node = *graph_nodes[i];
    if (consumed_nodes.find(node.id) != consumed_nodes.end()) {
      continue;
    }
    auto op_type = OperationTypeFromString(node.operation.type);
    if (op_type == OperationType::CONSTANT) {
      auto attr =
          absl::any_cast<ConstTensorAttributes>(node.operation.attributes);
      auto outputs = graph.FindOutputs(node.id);
      gpu_model->const_tensors[outputs[0]->id] =
          tensor_reserver->Get(outputs[0]->id);
      gpu_model->const_tensors[outputs[0]->id].UploadData(attr.tensor);
      continue;
    }
    GPUOperationsSubgraph gpu_subgraph;
    if (create_info.hints.Check(ModelHints::kAllowSpecialKernels) &&
        GPUSubgraphFromGraph(gpu_info, create_info.precision, graph, node.id,
                             tensor_descriptors, &consumed_nodes, &gpu_subgraph)
            .ok()) {
      // Mapping of subgraph (set of nodes) to GPU operations. Should happen
      // before straigtforward mapping.
    } else {
      // Straigtforward mapping of one graph node to GPU operations.
      auto inputs = graph.FindInputs(node.id);
      auto outputs = graph.FindOutputs(node.id);
      // Reordering of input ids and updating of temporary tensors_usage struct.
      // To have better linking we need linking tensor(latest written during
      // linear execution) on first position.
      if (IsAssociativeLinkableOp(node, inputs, outputs)) {
        int latest_written_tensor_index = 0;
        int last_usage = tensor_usages[inputs[0]->id];
        for (int j = 1; j < inputs.size(); ++j) {
          if (tensor_usages[inputs[j]->id] > last_usage) {
            last_usage = tensor_usages[inputs[j]->id];
            latest_written_tensor_index = j;
          }
        }
        std::swap(inputs[0], inputs[latest_written_tensor_index]);
      }
      consumed_nodes.insert(node.id);
      OperationDef op_def;
      op_def.precision = create_info.precision;
      for (int j = 0; j < inputs.size(); ++j) {
        op_def.src_tensors.push_back(tensor_reserver->Get(inputs[j]->id));
      }
      for (int j = 0; j < outputs.size(); ++j) {
        op_def.dst_tensors.push_back(tensor_reserver->Get(outputs[j]->id));
      }
      RETURN_IF_ERROR(GPUOperationFromNode(
          gpu_info, op_def, create_info.hints, inputs, outputs, node,
          shared_conv_weights_ptr, &gpu_subgraph));
    }
    absl::flat_hash_map<int, ValueId> mapping_to_global_ids;
    for (int j = 0; j < gpu_subgraph.new_tensors.size(); ++j) {
      const auto& t = gpu_subgraph.new_tensors[j];
      if (!t.second.GetData().empty()) {  // constant tensor
        auto global_id = tensor_reserver->GetNewId();
        gpu_model->const_tensors[global_id] =
            std::move(gpu_subgraph.new_tensors[j].second);
        const auto& shape = gpu_subgraph.new_tensors[j].first;
        gpu_model->const_tensors[global_id].SetBHWCShape(shape);
        mapping_to_global_ids[j] = global_id;
      } else {
        TensorDescriptor td = t.second;
        td.SetBHWCShape(t.first);
        auto global_id = tensor_reserver->Add(td);
        mapping_to_global_ids[j] = global_id;
      }
    }
    if (!shared_conv_weights.empty() && !mapping_to_global_ids.empty()) {
      shared_conv_weights.back().RemapIds(mapping_to_global_ids);
    }
    for (auto& gpu_op : gpu_subgraph.operations) {
      GpuNode gpu_node;
      gpu_node.gpu_operation = std::move(gpu_op.operation);
      gpu_node.inputs.resize(gpu_op.input_ids.size());
      for (int j = 0; j < gpu_op.input_ids.size(); ++j) {
        int id = gpu_op.input_ids[j];
        if (id >= 0) {
          gpu_node.inputs[j] = id;
        } else {
          gpu_node.inputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      gpu_node.outputs.resize(gpu_op.output_ids.size());
      for (int j = 0; j < gpu_op.output_ids.size(); ++j) {
        int id = gpu_op.output_ids[j];
        if (id >= 0) {
          gpu_node.outputs[j] = id;
          tensor_usages[id] = i;
        } else {
          gpu_node.outputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      gpu_node.name = gpu_op.name;
      gpu_model->nodes.push_back(std::move(gpu_node));
    }
  }

  return absl::OkStatus();
}

absl::Status MergeNodes(GpuModel* gpu_model) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_13(mht_13_v, 601, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "MergeNodes");

  absl::flat_hash_set<ValueId> ready_tensors;
  for (const auto& input : gpu_model->input_ids_and_refs) {
    ready_tensors.insert(input.first);
  }
  auto& nodes = gpu_model->nodes;
  for (int i = 0; i < nodes.size(); ++i) {
    auto& node = nodes[i];
    for (const auto& out_id : node.outputs) {
      ready_tensors.insert(out_id);
    }
    if (node.outputs.size() != 1) {
      continue;
    }
    std::vector<int> next_nodes;
    int link_index = 0;
    for (int j = i + 1; j < nodes.size(); ++j) {
      for (int k = 0; k < nodes[j].inputs.size(); ++k) {
        if (nodes[j].inputs[k] == node.outputs[0]) {
          next_nodes.push_back(j);
          link_index = k;
        }
      }
    }
    if (next_nodes.size() != 1 || link_index != 0) {
      continue;
    }
    auto& linkable_node = nodes[next_nodes[0]];
    if (!linkable_node.gpu_operation->IsLinkable() ||
        linkable_node.outputs.size() != 1 ||
        !IsReady(ready_tensors, linkable_node)) {
      continue;
    }
    const auto& original_dst_def =
        node.gpu_operation->GetDefinition().dst_tensors[0];
    const auto& link_dst_def =
        linkable_node.gpu_operation->GetDefinition().dst_tensors[0];
    if (original_dst_def != link_dst_def) {
      continue;
    }
    RETURN_IF_ERROR(MergeGpuNodes(&linkable_node, &node));
    nodes.erase(nodes.begin() + next_nodes[0]);
    i -= 1;
  }
  return absl::OkStatus();
}

void CopyExternals(const GraphFloat32& graph, GpuModel* gpu_model) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_14(mht_14_v, 651, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "CopyExternals");

  const auto inputs = graph.inputs();
  for (const auto& value : inputs) {
    gpu_model->input_ids_and_refs.push_back({value->id, value->tensor.ref});
  }

  const auto variable_inputs = graph.variable_inputs();
  for (const auto& value : variable_inputs) {
    gpu_model->variable_ids_and_refs.push_back({value->id, value->tensor.ref});
  }

  const auto outputs = graph.outputs();
  for (const auto& value : outputs) {
    gpu_model->output_ids_and_refs.push_back({value->id, value->tensor.ref});
  }
}

// Removing tensors that was fused in complex operations
void RemoveUnusedTensors(GpuModel* gpu_model) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_15(mht_15_v, 672, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "RemoveUnusedTensors");

  absl::flat_hash_set<ValueId> used_tensors;
  for (const auto& node : gpu_model->nodes) {
    for (const auto& id : node.inputs) {
      used_tensors.insert(id);
    }
    for (const auto& id : node.outputs) {
      used_tensors.insert(id);
    }
  }
  for (auto it = gpu_model->tensors.begin(); it != gpu_model->tensors.end();) {
    if (used_tensors.find(it->first) == used_tensors.end()) {
      gpu_model->tensors.erase(it++);
    } else {
      ++it;
    }
  }
}

// Serialized model will lose polymorphic properties for GpuOperations.
// Here we will retrieve some information needed for generic execution of
// GpuOperations. Specifically, BindArguments and RecalculateGridSize must be
// executed.
absl::Status ResolvePolymorphicArgs(GpuModel* gpu_model) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_16(mht_16_v, 698, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "ResolvePolymorphicArgs");

  class DummySpatialTensor : public GpuSpatialTensor {
   public:
    DummySpatialTensor() = default;
    explicit DummySpatialTensor(const BHWDC& shape,
                                const TensorDescriptor& tensor_desc)
        : shape_(shape), tensor_desc_(tensor_desc) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_17(mht_17_v, 707, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "DummySpatialTensor");
}
    ~DummySpatialTensor() override = default;

    int Width() const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_18(mht_18_v, 713, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Width");
 return shape_.w; }
    int Height() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_19(mht_19_v, 717, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Height");
 return shape_.h; }
    int Depth() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_20(mht_20_v, 721, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Depth");
 return shape_.d; }
    int Channels() const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_21(mht_21_v, 725, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Channels");
 return shape_.c; }
    int Slices() const override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_22(mht_22_v, 729, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Slices");
 return DivideRoundUp(shape_.c, 4); }
    int Batch() const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_23(mht_23_v, 733, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Batch");
 return shape_.b; }

    TensorDescriptor GetDescriptor() const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_24(mht_24_v, 738, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "GetDescriptor");
 return tensor_desc_; }

   private:
    BHWDC shape_;
    TensorDescriptor tensor_desc_;
  };

  for (auto& node : gpu_model->nodes) {
    std::vector<DummySpatialTensor> src_tensors(node.inputs.size());
    for (int i = 0; i < node.inputs.size(); ++i) {
      const auto& tensor_desc = gpu_model->tensors[node.inputs[i]];
      src_tensors[i] =
          DummySpatialTensor(tensor_desc.GetBHWDCShape(), tensor_desc);
      node.gpu_operation->SetSrc(&src_tensors[i], i);
    }
    std::vector<DummySpatialTensor> dst_tensors(node.outputs.size());
    for (int i = 0; i < node.outputs.size(); ++i) {
      const auto& tensor_desc = gpu_model->tensors[node.outputs[i]];
      dst_tensors[i] =
          DummySpatialTensor(tensor_desc.GetBHWDCShape(), tensor_desc);
      node.gpu_operation->SetDst(&dst_tensors[i], i);
    }
    RETURN_IF_ERROR(
        node.gpu_operation->BindArguments(&node.gpu_operation->args_));
    node.gpu_operation->RecalculateGridSize();
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status GraphToGpuModel(const GraphFloat32& graph,
                             const CreateGpuModelInfo& create_info,
                             const GpuInfo& gpu_info, GpuModel* gpu_model) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_25(mht_25_v, 774, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "GraphToGpuModel");

  TensorReserver tensor_reserver;
  RETURN_IF_ERROR(
      ReserveGraphTensors(create_info, gpu_info, graph, &tensor_reserver));
  CopyExternals(graph, gpu_model);
  RETURN_IF_ERROR(ConvertOperations(gpu_info, graph, create_info,
                                    &tensor_reserver, gpu_model));
  RETURN_IF_ERROR(MergeNodes(gpu_model));
  gpu_model->tensors = std::move(tensor_reserver.reservations_);
  RemoveUnusedTensors(gpu_model);

  for (auto& node : gpu_model->nodes) {
    RETURN_IF_ERROR(node.gpu_operation->AssembleCode(gpu_info));
  }

  return ResolvePolymorphicArgs(gpu_model);
}

flatbuffers::Offset<data::GpuModel> Encode(
    const GpuModel& gpu_model, flatbuffers::FlatBufferBuilder* builder) {
  std::vector<int32_t> in_ids(gpu_model.input_ids_and_refs.size());
  std::vector<int64_t> in_refs(gpu_model.input_ids_and_refs.size());
  for (int i = 0; i < in_ids.size(); ++i) {
    in_ids[i] = gpu_model.input_ids_and_refs[i].first;
    in_refs[i] = gpu_model.input_ids_and_refs[i].second;
  }
  auto in_ids_fb = builder->CreateVector(in_ids);
  auto in_refs_fb = builder->CreateVector(in_refs);

  std::vector<int32_t> out_ids(gpu_model.output_ids_and_refs.size());
  std::vector<int64_t> out_refs(gpu_model.output_ids_and_refs.size());
  for (int i = 0; i < out_ids.size(); ++i) {
    out_ids[i] = gpu_model.output_ids_and_refs[i].first;
    out_refs[i] = gpu_model.output_ids_and_refs[i].second;
  }
  auto out_ids_fb = builder->CreateVector(out_ids);
  auto out_refs_fb = builder->CreateVector(out_refs);

  std::vector<flatbuffers::Offset<data::GpuNode>> nodes_fb;
  for (int i = 0; i < gpu_model.nodes.size(); ++i) {
    auto node_fb = Encode(gpu_model.nodes[i], builder);
    nodes_fb.push_back(node_fb);
  }
  auto nodes_fb_vec = builder->CreateVector(nodes_fb);

  std::vector<flatbuffers::Offset<data::TensorDescWithId>> tensors_fb;
  for (const auto& tensor : gpu_model.tensors) {
    auto tensor_fb = Encode(tensor.second, tensor.first, builder);
    tensors_fb.push_back(tensor_fb);
  }
  auto tensors_fb_vec = builder->CreateVector(tensors_fb);

  std::vector<flatbuffers::Offset<data::TensorDescWithId>> const_tensors_fb;
  for (const auto& tensor : gpu_model.const_tensors) {
    auto tensor_fb = Encode(tensor.second, tensor.first, builder);
    const_tensors_fb.push_back(tensor_fb);
  }
  auto const_tensors_fb_vec = builder->CreateVector(const_tensors_fb);

  std::vector<flatbuffers::Offset<data::PairOfValueIds>>
      variable_ids_and_refs_fb;
  for (auto& pair : gpu_model.variable_ids_and_refs) {
    data::PairOfValueIdsBuilder pair_builder(*builder);
    pair_builder.add_first(pair.first);
    pair_builder.add_second(pair.second);
    variable_ids_and_refs_fb.push_back(pair_builder.Finish());
  }
  auto variable_ids_and_refs_fb_vec =
      builder->CreateVector(variable_ids_and_refs_fb);

  data::GpuModelBuilder gpu_model_builder(*builder);
  gpu_model_builder.add_nodes(nodes_fb_vec);
  gpu_model_builder.add_tensors(tensors_fb_vec);
  gpu_model_builder.add_const_tensors(const_tensors_fb_vec);
  gpu_model_builder.add_input_ids(in_ids_fb);
  gpu_model_builder.add_output_ids(out_ids_fb);
  gpu_model_builder.add_variable_ids_and_refs(variable_ids_and_refs_fb_vec);
  gpu_model_builder.add_input_refs(in_refs_fb);
  gpu_model_builder.add_output_refs(out_refs_fb);
  return gpu_model_builder.Finish();
}

absl::Status Decode(const data::GpuModel* fb_gpu_model, GpuModel* gpu_model) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_26(mht_26_v, 859, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "Decode");

  gpu_model->nodes.resize(fb_gpu_model->nodes()->size());
  int counter = 0;
  for (auto node_fb : *fb_gpu_model->nodes()) {
    RETURN_IF_ERROR(Decode(node_fb, &gpu_model->nodes[counter]));
    counter++;
  }

  for (const auto& tensor_fb : *fb_gpu_model->tensors()) {
    TensorDescriptor desc;
    Decode(tensor_fb->desc(), &desc);
    gpu_model->tensors[tensor_fb->id()] = std::move(desc);
  }
  for (const auto& tensor_fb : *fb_gpu_model->const_tensors()) {
    TensorDescriptor desc;
    Decode(tensor_fb->desc(), &desc);
    gpu_model->const_tensors[tensor_fb->id()] = std::move(desc);
  }
  for (int i = 0; i < fb_gpu_model->input_ids()->size(); ++i) {
    gpu_model->input_ids_and_refs.push_back(
        {(*fb_gpu_model->input_ids())[i], (*fb_gpu_model->input_refs())[i]});
  }
  for (int i = 0; i < fb_gpu_model->output_ids()->size(); ++i) {
    gpu_model->output_ids_and_refs.push_back(
        {(*fb_gpu_model->output_ids())[i], (*fb_gpu_model->output_refs())[i]});
  }

  for (auto variable_id : *fb_gpu_model->variable_ids_and_refs()) {
    gpu_model->variable_ids_and_refs.push_back(
        {variable_id->first(), variable_id->second()});
  }
  return absl::OkStatus();
}

absl::Status RunGraphTransformsForGpuModel(GraphFloat32* graph) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSgpu_modelDTcc mht_27(mht_27_v, 896, "", "./tensorflow/lite/delegates/gpu/common/gpu_model.cc", "RunGraphTransformsForGpuModel");

  auto merge_padding_transform = NewMergePaddingWithAdd();
  auto add_bias_transform = NewAddBias();
  auto pooling_to_reduce_op = NewGlobalPoolingToReduceOp();
  ModelTransformer transformer(graph);
  if (!transformer.Apply("add_bias", add_bias_transform.get())) {
    return absl::InternalError("Invalid add_bias transform");
  }
  if (!transformer.Apply("merge_padding", merge_padding_transform.get())) {
    return absl::InternalError("Invalid merge_padding transform");
  }
  if (!transformer.Apply("global pooling to mean",
                         pooling_to_reduce_op.get())) {
    return absl::InternalError("Invalid global pooling to mean transform");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
