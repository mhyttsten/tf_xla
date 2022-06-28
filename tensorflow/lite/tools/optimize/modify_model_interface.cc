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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc() {
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
#include "tensorflow/lite/tools/optimize/modify_model_interface.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "flatbuffers/flexbuffers.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/tools/optimize/model_utils.h"

namespace tflite {
namespace optimize {

namespace {

// Structure to hold input tensor, op and output tensor.
// op must be either quantize or dequantize.
struct TensorOpTensor {
  size_t subgraph_index;  // index of the subgraph.
  int32_t input_index;    // index of the input tensor.
  int32_t op_index;       // index of the op.
  int32_t output_index;   // index of the output tensor.
  int32_t model_index;    // index of the added tensor in the model.
};

// Finds float tensors that are model inputs and is consumed by a quantize Op.
// The returned TensorOpTensor should have reverse order.
std::vector<TensorOpTensor> GetInputTensors(const TensorType& input_type,
                                            ModelT* model,
                                            ErrorReporter* error_reporter) {
  std::vector<TensorOpTensor> result;
  // Get all input tensors.
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    absl::flat_hash_map<TensorT*, int> input_tensors;
    for (size_t input_idx = 0; input_idx < subgraph->inputs.size();
         input_idx++) {
      TensorT* tensor = subgraph->tensors[subgraph->inputs[input_idx]].get();
      if (tensor->type == TensorType_FLOAT32) {
        input_tensors.insert({tensor, input_idx});
      }
    }

    for (int32_t op_idx = subgraph->operators.size() - 1; op_idx >= 0;
         op_idx--) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          GetBuiltinCode(model->operator_codes[op->opcode_index].get());
      TensorT* input_tensor = subgraph->tensors[op->inputs[0]].get();
      if (input_tensors.find(input_tensor) == input_tensors.end()) {
        continue;
      }
      if (op_code != BuiltinOperator_QUANTIZE) {
        // Currently only supports int8 and int16 quantized models.
        TF_LITE_REPORT_ERROR(
            error_reporter,
            "modify_model_interface called on a model without quant/dequant.");
        return {};
      }
      if (op->inputs.size() != 1) {
        continue;
      }
      if (op->outputs.size() != 1) {
        continue;
      }
      const int model_input_index = input_tensors[input_tensor];
      TensorT* quant_output = subgraph->tensors[op->outputs[0]].get();
      if (quant_output->type != TensorType_INT8 &&
          quant_output->type != TensorType_INT16) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "modify_model_interface currently only supports "
                             "int8 and int16 quantized models.");
      }

      // The input type must be the same as the model quantization type
      if (input_type != quant_output->type) {
        // An exception, allow for UINT8 input type for INT8 quantized model.
        if (!(input_type == TensorType_UINT8 &&
              quant_output->type == TensorType_INT8)) {
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "The %s input type is incompatible with %s quantized models. "
              "To resolve this error, change the input_type to a compatible "
              "one. "
              "See: modify_model_interface.cc",
              EnumNameTensorType(input_type),
              EnumNameTensorType(quant_output->type));
        }
      }
      if (quant_output->quantization == nullptr) {
        continue;
      }
      result.push_back({subgraph_idx, op->inputs[0], op_idx, op->outputs[0],
                        model_input_index});
    }
  }
  return result;
}

// Finds float tensors that are model output and is consumed by a dequantize Op.
// The returned TensorOpTensor should have reverse order.
std::vector<TensorOpTensor> GetOutputTensors(const TensorType& output_type,
                                             ModelT* model,
                                             ErrorReporter* error_reporter) {
  std::vector<TensorOpTensor> result;
  // Get all output tensors.
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    absl::flat_hash_map<TensorT*, int> output_tensors;
    for (size_t output_idx = 0; output_idx < subgraph->outputs.size();
         output_idx++) {
      TensorT* tensor = subgraph->tensors[subgraph->outputs[output_idx]].get();
      if (tensor->type == TensorType_FLOAT32) {
        output_tensors.insert({tensor, output_idx});
      }
    }

    for (int32_t op_idx = subgraph->operators.size() - 1; op_idx >= 0;
         op_idx--) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          GetBuiltinCode(model->operator_codes[op->opcode_index].get());
      TensorT* output_tensor = subgraph->tensors[op->outputs[0]].get();
      if (output_tensors.find(output_tensor) == output_tensors.end()) {
        continue;
      }
      if (op_code != BuiltinOperator_DEQUANTIZE) {
        // Currently only supports int8 and int16 quantized models.
        TF_LITE_REPORT_ERROR(
            error_reporter,
            "modify_model_interface called on a model without quant/dequant.");
        return {};
      }
      if (op->inputs.size() != 1) {
        continue;
      }
      if (op->outputs.size() != 1) {
        continue;
      }
      const int model_output_index = output_tensors[output_tensor];
      TensorT* dequant_input = subgraph->tensors[op->inputs[0]].get();
      if (dequant_input->type != TensorType_INT8 &&
          dequant_input->type != TensorType_INT16) {
        // Currently only supports int8 and int16 quantized models.
        TF_LITE_REPORT_ERROR(error_reporter,
                             "modify_model_interface currently only supports "
                             "int8 and int16 quantized models.");
        return {};
      }
      if (output_type != dequant_input->type) {
        // An exception, allow for UINT8 input type for INT8 quantized model.
        if (!(output_type == TensorType_UINT8 &&
              dequant_input->type == TensorType_INT8)) {
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "The %s output type is incompatible with %s quantized models. "
              "To resolve this error, change the output_type to a compatible "
              "one. "
              "See: modify_model_interface.cc",
              EnumNameTensorType(output_type),
              EnumNameTensorType(dequant_input->type));
        }
      }
      if (dequant_input->quantization == nullptr) {
        continue;
      }
      result.push_back({subgraph_idx, op->inputs[0], op_idx, op->outputs[0],
                        model_output_index});
    }
  }
  return result;
}

TfLiteStatus SetInputTypeToUINT8(ModelT* model,
                                 const std::vector<TensorOpTensor>& inputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_0(mht_0_v, 369, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "SetInputTypeToUINT8");

  // If the input type is uint8, change float to uint8.
  for (auto tot : inputs) {
    SubGraphT* subgraph = model->subgraphs.at(tot.subgraph_index).get();
    TensorT* quant_tensor = subgraph->tensors[tot.output_index].get();
    const float quant_tensor_scale = quant_tensor->quantization->scale[0];
    const int quant_tensor_zp = quant_tensor->quantization->zero_point[0];
    TensorT* float_tensor = subgraph->tensors[tot.input_index].get();
    float_tensor->type = TensorType_UINT8;
    if (float_tensor->quantization == nullptr) {
      float_tensor->quantization = absl::make_unique<QuantizationParametersT>();
    }
    float_tensor->quantization->scale.push_back(quant_tensor_scale);
    float_tensor->quantization->zero_point.push_back(quant_tensor_zp + 128);
  }
  return kTfLiteOk;
}

TfLiteStatus SetOutputTypeToUINT8(ModelT* model,
                                  const std::vector<TensorOpTensor>& outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_1(mht_1_v, 391, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "SetOutputTypeToUINT8");

  // Find Quant op code index.
  size_t quant_op_index = 0;
  for (size_t i = 0; i < model->operator_codes.size(); ++i) {
    if (GetBuiltinCode(model->operator_codes[i].get()) ==
        BuiltinOperator_QUANTIZE) {
      quant_op_index = i;
    }
  }
  // If the output type is uint8, change float to uint8.
  for (auto tot : outputs) {
    SubGraphT* subgraph = model->subgraphs.at(tot.subgraph_index).get();
    TensorT* quant_tensor = subgraph->tensors[tot.input_index].get();
    const float quant_tensor_scale = quant_tensor->quantization->scale[0];
    const int quant_tensor_zp = quant_tensor->quantization->zero_point[0];
    TensorT* float_tensor = subgraph->tensors[tot.output_index].get();
    float_tensor->type = TensorType_UINT8;
    if (float_tensor->quantization == nullptr) {
      float_tensor->quantization = absl::make_unique<QuantizationParametersT>();
    }
    float_tensor->quantization->scale.push_back(quant_tensor_scale);
    float_tensor->quantization->zero_point.push_back(quant_tensor_zp + 128);

    // Change op from dequant (int8 to float) to quant (int8 to uint8)
    OperatorT* op = subgraph->operators[tot.op_index].get();
    op->opcode_index = quant_op_index;
  }
  return kTfLiteOk;
}

TfLiteStatus RemoveInputTensor(ModelT* model,
                               const std::vector<TensorOpTensor>& inputs,
                               int32 original_number_tensors) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_2(mht_2_v, 426, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "RemoveInputTensor");

  // Consistency check to make sure that erase start from the end.
  int last_op_index = std::numeric_limits<int32_t>::max();
  int last_tensor_index = std::numeric_limits<int32_t>::max();
  for (auto tot : inputs) {
    TFLITE_DCHECK(tot.input_index < last_tensor_index);
    TFLITE_DCHECK(tot.op_index < last_op_index);
    last_tensor_index = tot.input_index;
    last_op_index = tot.op_index;
  }
  // Removes the input tensor and the related operator.
  for (auto tot : inputs) {
    SubGraphT* subgraph = model->subgraphs.at(tot.subgraph_index).get();
    TFLITE_DCHECK(tot.input_index < subgraph->tensors.size());
    TFLITE_DCHECK(tot.op_index < subgraph->operators.size());
    if (tot.input_index >= original_number_tensors) {
      subgraph->tensors.erase(subgraph->tensors.begin() + tot.input_index);
    }
    subgraph->operators.erase(subgraph->operators.begin() + tot.op_index);
    subgraph->inputs[tot.model_index] = tot.output_index;
  }
  return kTfLiteOk;
}

TfLiteStatus RemoveOutputTensor(ModelT* model,
                                const std::vector<TensorOpTensor>& outputs,
                                int32 original_number_tensors) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_3(mht_3_v, 455, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "RemoveOutputTensor");

  // Consistency check to make sure that erase start from the end.
  int last_op_index = std::numeric_limits<int32_t>::max();
  int last_tensor_index = std::numeric_limits<int32_t>::max();
  for (auto tot : outputs) {
    TFLITE_DCHECK(tot.output_index < last_tensor_index);
    TFLITE_DCHECK(tot.op_index < last_op_index);
    last_tensor_index = tot.output_index;
    last_op_index = tot.op_index;
  }
  // Removes the output tensor and the related operator.
  for (auto tot : outputs) {
    SubGraphT* subgraph = model->subgraphs.at(tot.subgraph_index).get();
    TFLITE_DCHECK(tot.output_index < subgraph->tensors.size());
    TFLITE_DCHECK(tot.op_index < subgraph->operators.size());
    if (tot.output_index >= original_number_tensors) {
      subgraph->tensors.erase(subgraph->tensors.begin() + tot.output_index);
    }
    subgraph->operators.erase(subgraph->operators.begin() + tot.op_index);
    subgraph->outputs[tot.model_index] = tot.input_index;
  }
  return kTfLiteOk;
}


int GetOriginalNumberOfTensors(const TensorType& input_type,
                               const TensorType& output_type, ModelT* model,
                               ErrorReporter* error_reporter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_4(mht_4_v, 485, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "GetOriginalNumberOfTensors");

  std::vector<TensorOpTensor> outputs =
      GetOutputTensors(output_type, model, error_reporter);
  std::vector<TensorOpTensor> inputs =
      GetInputTensors(input_type, model, error_reporter);
  return model->subgraphs[0]->tensors.size() - outputs.size() - inputs.size();
}

}  // namespace

TfLiteStatus ModifyModelInterface(flatbuffers::FlatBufferBuilder* builder,
                                  ModelT* model, const TensorType& input_type,
                                  const TensorType& output_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_5(mht_5_v, 500, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "ModifyModelInterface");

  tflite::StderrReporter error_reporter;
  const int original_number_tensors = GetOriginalNumberOfTensors(
      input_type, output_type, model, &error_reporter);
  // Finds float tensors that are model output and are consumed by a float to
  // int8/int16 quantize Op. Do output first since the tensors are added into
  // input first.,
  std::vector<TensorOpTensor> outputs =
      GetOutputTensors(output_type, model, &error_reporter);
  switch (output_type) {
    case TensorType_UINT8:
      SetOutputTypeToUINT8(model, outputs);
      break;
    case TensorType_INT8:
    case TensorType_INT16:
      RemoveOutputTensor(model, outputs, original_number_tensors);
      break;
    default:
      return kTfLiteError;
  }

  // Find float tensors that are model input and is consumed by a float to
  // int8/int16 quantize Op.
  std::vector<TensorOpTensor> inputs =
      GetInputTensors(input_type, model, &error_reporter);
  switch (input_type) {
    case TensorType_UINT8:
      SetInputTypeToUINT8(model, inputs);
      break;
    case TensorType_INT8:
    case TensorType_INT16:
      RemoveInputTensor(model, inputs, original_number_tensors);
      break;
    default:
      return kTfLiteError;
  }

  // Write to builder.
  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model);
  FinishModelBuffer(*builder, output_model_location);

  return kTfLiteOk;
}

TfLiteStatus ModifyModelInterface(const string& input_file,
                                  const string& output_file,
                                  const TensorType& input_type,
                                  const TensorType& output_type) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("input_file: \"" + input_file + "\"");
   mht_6_v.push_back("output_file: \"" + output_file + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_6(mht_6_v, 553, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "ModifyModelInterface");

  // Consistency Check
  if (input_type != tflite::TensorType_INT8 &&
      input_type != tflite::TensorType_UINT8 &&
      input_type != tflite::TensorType_INT16) {
    return kTfLiteError;
  }
  if (output_type != tflite::TensorType_INT8 &&
      output_type != tflite::TensorType_UINT8 &&
      output_type != tflite::TensorType_INT16) {
    return kTfLiteError;
  }

  // Create model.
  auto tflite_model = utils::CreateMutableModelFromFile(input_file);

  auto model_builder = utils::FinishModel(tflite_model.get());

  auto fixed_point_model_builder =
      absl::make_unique<flatbuffers::FlatBufferBuilder>();
  flatbuffers::FlatBufferBuilder builder;

  auto status = ModifyModelInterface(&builder, tflite_model.get(), input_type,
                                     output_type);
  TFLITE_DCHECK_EQ(status, kTfLiteOk);

  utils::WriteFile(output_file, builder.GetBufferPointer(), builder.GetSize());

  return kTfLiteOk;
}

namespace {
void AddUint8Dequant(
    const std::unordered_map<string, std::pair<float, int32_t>>& quant_params,
    ModelT* model) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_7(mht_7_v, 590, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "AddUint8Dequant");

  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    // Add dequant to input tensors.
    for (size_t input_idx = 0; input_idx < subgraph->inputs.size();
         input_idx++) {
      const int32_t tensor_idx = subgraph->inputs[input_idx];
      TensorT* tensor = subgraph->tensors[tensor_idx].get();
      if (tensor->type != TensorType_FLOAT32) {
        continue;
      }
      if (quant_params.find(tensor->name) != quant_params.end()) {
        // Add uint8 tensor
        const string added_tensor_name = tensor->name + "_uint8";
        std::unique_ptr<TensorT> leading_op_input;
        const std::pair<float, int32_t>& provided_quant_params =
            quant_params.at(string(tensor->name));
        utils::MakeTensorWithQuantParam(
            added_tensor_name, tensor->shape, tensor->shape_signature,
            TensorType_UINT8, provided_quant_params.first,
            provided_quant_params.second, &leading_op_input);
        const int32_t leading_op_input_idx = subgraph->tensors.size();
        subgraph->tensors.push_back(std::move(leading_op_input));

        // Create the leading op, which is deqantize Op.
        std::unique_ptr<OperatorT> leading_op;
        utils::MakeDequantizeOperator(model, &leading_op, leading_op_input_idx,
                                      tensor_idx);

        // Insert the new op at the start of the model.
        subgraph->operators.insert(subgraph->operators.begin(),
                                   std::move(leading_op));
      }
    }
  }
}

void AddUint8Quant(
    const std::unordered_map<string, std::pair<float, int32_t>>& quant_params,
    ModelT* model) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_8(mht_8_v, 633, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "AddUint8Quant");

  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    // Add quant to output tensors.
    for (size_t output_idx = 0; output_idx < subgraph->outputs.size();
         output_idx++) {
      const int32_t tensor_idx = subgraph->outputs[output_idx];
      TensorT* tensor = subgraph->tensors[tensor_idx].get();
      if (tensor->type != TensorType_FLOAT32) {
        continue;
      }
      if (quant_params.find(tensor->name) != quant_params.end()) {
        // Add uint8 tensor
        const string added_tensor_name = tensor->name + "_uint8";
        std::unique_ptr<TensorT> tailing_op_output;
        const std::pair<float, int32_t>& provided_quant_params =
            quant_params.at(string(tensor->name));
        utils::MakeTensorWithQuantParam(
            added_tensor_name, tensor->shape, tensor->shape_signature,
            TensorType_UINT8, provided_quant_params.first,
            provided_quant_params.second, &tailing_op_output);
        const int32_t tailing_op_output_idx = subgraph->tensors.size();
        subgraph->tensors.push_back(std::move(tailing_op_output));

        // Create the tailing op, which is Qantize Op.
        std::unique_ptr<OperatorT> tailing_op;
        utils::MakeQuantizeOperator(model, &tailing_op, tensor_idx,
                                    tailing_op_output_idx);

        // Insert the new op at the end of the model.
        subgraph->operators.push_back(std::move(tailing_op));
      }
    }
  }
}
}  // namespace

TfLiteStatus Uint8QuantizeModelInputsOutputs(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    const std::unordered_map<string, std::pair<float, int32_t>>&
        input_quant_params,
    const std::unordered_map<string, std::pair<float, int32_t>>&
        output_quant_params) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodify_model_interfaceDTcc mht_9(mht_9_v, 679, "", "./tensorflow/lite/tools/optimize/modify_model_interface.cc", "Uint8QuantizeModelInputsOutputs");

  std::unique_ptr<ModelT> model;
  model.reset(input_model->UnPack());
  // Add Dequant for inputs.
  AddUint8Dequant(input_quant_params, model.get());

  // Add Quant for outputs.
  AddUint8Quant(output_quant_params, model.get());

  // Output model.
  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model.get());
  FinishModelBuffer(*builder, output_model_location);

  return kTfLiteOk;
}

}  // namespace optimize
}  // namespace tflite
