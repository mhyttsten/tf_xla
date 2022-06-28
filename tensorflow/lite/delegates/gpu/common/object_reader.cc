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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/object_reader.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace gpu {

absl::Status ObjectReader::ReadNonConstantTensor(
    TfLiteContext* context, absl::flat_hash_map<int, Value*>* tensor_to_value,
    absl::flat_hash_map<int, int>* quant_conversion_map, GraphFloat32* graph,
    uint32_t tensor_idx, Value** value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::ReadNonConstantTensor");

  if (tensor_idx >= context->tensors_size) {
    return absl::OutOfRangeError(
        absl::StrCat("ReadNonConstTensor: input tensor index: ", tensor_idx));
  }

  if (tensor_to_value->find(tensor_idx) == tensor_to_value->end()) {
    TfLiteTensor* tflite_tensor = &context->tensors[tensor_idx];
    if (tflite::IsConstantTensor(tflite_tensor)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ReadNonConstantTensor: value is a constant tensor: ", tensor_idx));
    }

    if ((tflite_tensor->type == kTfLiteInt8 ||
         tflite_tensor->type == kTfLiteUInt8) &&
        quant_conversion_map) {
      // Quantized case
      if (quant_conversion_map->find(tensor_idx) ==
          quant_conversion_map->end()) {
        // Since the original tensor is fixed-point, add a new float tensor to
        // the TFLite graph to represent the dequantized data.
        int fp_tensor_index = 0;
        TfLiteTensor* fp_tflite_tensor;
        if (delegates::CreateNewTensorWithDifferentType(
                context, tensor_idx, kTfLiteFloat32, &fp_tflite_tensor,
                &fp_tensor_index) != kTfLiteOk) {
          return absl::InternalError("Could not add new tensor to graph");
        }
        // `tflite_tensor` value could be invalid when the `context->tensors`
        // is reallocated. Thus reassigning `tflite_tensor` with a fresh value.
        tflite_tensor = &context->tensors[tensor_idx];

        // Remember this tensor for later.
        (*quant_conversion_map)[fp_tensor_index] = tensor_idx;
        (*quant_conversion_map)[tensor_idx] = fp_tensor_index;
        // Add a new GPU Value for the new dequantized floating-point tensor.
        Value* value = graph->NewValue();
        RETURN_IF_ERROR(
            ConvertTfLiteTensorToTensorRef(*fp_tflite_tensor, &value->tensor));
        value->tensor.ref = fp_tensor_index;
        value->tensor.is_variable_input = tflite_tensor->is_variable;
        value->quant_params.emplace();
        RETURN_IF_ERROR(
            PopulateQuantParams(*tflite_tensor, &value->quant_params.value()));
        (*tensor_to_value)[fp_tensor_index] = value;
      }
      // We do not use the original tensor index as reference for the GPU
      // Value, instead pointing at the corresponding float version.
      tensor_idx = quant_conversion_map->at(tensor_idx);
    } else {
      // Floating-point case.
      Value* value = graph->NewValue();
      RETURN_IF_ERROR(
          ConvertTfLiteTensorToTensorRef(*tflite_tensor, &value->tensor));
      value->tensor.ref = tensor_idx;
      value->tensor.is_variable_input = tflite_tensor->is_variable;
      (*tensor_to_value)[tensor_idx] = value;
    }
  }

  if (value) {
    *value = (*tensor_to_value)[tensor_idx];
  }
  return absl::OkStatus();
}

absl::Status ObjectReader::ReadValue(uint32_t idx, Value** value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_1(mht_1_v, 276, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::ReadValue");

  if (idx >= node_->inputs->size) {
    return absl::OutOfRangeError(
        absl::StrCat("ReadValue: input tensor index: ", idx));
  }
  return ReadValueByTensorIdx(node_->inputs->data[idx], value);
}

absl::Status ObjectReader::ReadValueByTensorIdx(uint32_t tensor_idx,
                                                Value** value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_2(mht_2_v, 288, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::ReadValueByTensorIdx");

  // Constant tensors should be handled by ReadTensor.
  return ReadNonConstantTensor(context_, tensor_to_value_,
                               quant_conversion_map_, graph_, tensor_idx,
                               value);
}

int ObjectReader::GetNumberOfRuntimeInputs() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_3(mht_3_v, 298, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::GetNumberOfRuntimeInputs");

  return GetNumberOfRuntimeInputsForNode(context_, node_);
}

absl::Status ObjectReader::GetTensorId(uint32_t input_id,
                                       int* tensor_id) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_4(mht_4_v, 306, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::GetTensorId");

  if (input_id >= node_->inputs->size) {
    return absl::OutOfRangeError(
        absl::StrCat("Input tensor index: ", input_id));
  }
  *tensor_id = node_->inputs->data[input_id];
  if (*tensor_id < 0 || *tensor_id > context_->tensors_size) {
    return absl::OutOfRangeError(absl::StrCat("Tensor index: ", *tensor_id));
  }
  return absl::OkStatus();
}

absl::Status ObjectReader::GetTensorDims(uint32_t idx,
                                         TfLiteIntArray* dimensions) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_5(mht_5_v, 322, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::GetTensorDims");

  if (idx >= node_->inputs->size) {
    return absl::OutOfRangeError(absl::StrCat("Input tensor index: ", idx));
  }
  const int tensor_idx = node_->inputs->data[idx];
  if (tensor_idx < 0 || tensor_idx > context_->tensors_size) {
    return absl::OutOfRangeError(absl::StrCat("Tensor index: ", tensor_idx));
  }
  const TfLiteTensor& tflite_tensor = context_->tensors[tensor_idx];
  *dimensions = *tflite_tensor.dims;
  return absl::OkStatus();
}

absl::Status ObjectReader::AddOutput(const Node* node, int id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_6(mht_6_v, 338, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::AddOutput");

  if (node_->outputs->size <= id) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Data id ", id, " must be less than tflite node outputs size ",
        node_->outputs->size));
  }
  int output_tensor_idx = node_->outputs->data[id];
  Value* value;
  RETURN_IF_ERROR(ReadValueByTensorIdx(output_tensor_idx, &value));
  RETURN_IF_ERROR(graph_->SetProducer(node->id, value->id));
  return absl::OkStatus();
}

absl::Status ObjectReader::AddOutputs(const Node* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_7(mht_7_v, 354, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::AddOutputs");

  for (int i = 0; i < node_->outputs->size; ++i) {
    RETURN_IF_ERROR(AddOutput(node, i));
  }
  return absl::OkStatus();
}

absl::Status ObjectReader::AddInput(const Node* node, uint32_t idx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_8(mht_8_v, 364, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::AddInput");

  Value* input;
  RETURN_IF_ERROR(ReadValue(idx, &input));
  return graph_->AddConsumer(node->id, input->id);
}

absl::Status ObjectReader::AddUpdate(const Node* node, uint32_t idx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_9(mht_9_v, 373, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::AddUpdate");

  if (node_->inputs->size <= idx) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Data id ", idx, " must be less than tflite node inputs size ",
        node_->inputs->size));
  }

  int update_tensor_idx = node_->inputs->data[idx];
  TfLiteTensor* update_tensor = context_->tensors + update_tensor_idx;
  if (!update_tensor->is_variable) {
    return absl::InvalidArgumentError(
        "The tensor must be a variable tensor to update it in place");
  }

  Value* value;
  RETURN_IF_ERROR(ReadValueByTensorIdx(update_tensor_idx, &value));
  if (!value->tensor.is_variable_input) {
    return absl::InternalError(
        "Variable input tensor is not marked as variable");
  }

  // We cannot create a cycle in the graph. The way around this when a node
  // updates a tensor in place would be to add a new value to the graph that
  // points to the same tensor.
  Value* updated_value = graph_->NewValue();
  updated_value->tensor = value->tensor;
  updated_value->quant_params = value->quant_params;
  RETURN_IF_ERROR(graph_->SetProducer(node->id, updated_value->id));

  // We also need to update the tensor_to_value arrays so that the nodes added
  // after the current node will access the tensor with the updated value rather
  // than the initial value.
  if (quant_conversion_map_ != nullptr &&
      quant_conversion_map_->find(update_tensor_idx) !=
          quant_conversion_map_->end()) {
    // If quantization conversion map exists, then the index provided is not the
    // actual tensor idx. We need to find the float version of the tensor from
    // the map.
    tensor_to_value_->at(quant_conversion_map_->at(update_tensor_idx)) =
        updated_value;
  } else {
    tensor_to_value_->at(update_tensor_idx) = updated_value;
  }

  return absl::OkStatus();
}

TfLiteTensor* ObjectReader::GetInputTensor(int index) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_10(mht_10_v, 423, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::GetInputTensor");

  return index >= 0 && index < node_->inputs->size
             ? context_->tensors + node_->inputs->data[index]
             : nullptr;
}

TfLiteTensor* ObjectReader::GetOutputTensor(int index) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_11(mht_11_v, 432, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::GetOutputTensor");

  return index >= 0 && index < node_->outputs->size
             ? context_->tensors + node_->outputs->data[index]
             : nullptr;
}

absl::Status ObjectReader::VerifyInputsConstsOutputs(const TfLiteNode* node,
                                                     int runtime_inputs,
                                                     int const_inputs,
                                                     int outputs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSobject_readerDTcc mht_12(mht_12_v, 444, "", "./tensorflow/lite/delegates/gpu/common/object_reader.cc", "ObjectReader::VerifyInputsConstsOutputs");

  return CheckInputsConstsOutputs(context_, node, runtime_inputs, const_inputs,
                                  outputs);
}

}  // namespace gpu
}  // namespace tflite
