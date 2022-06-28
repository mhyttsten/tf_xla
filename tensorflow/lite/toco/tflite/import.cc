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
class MHTracer_DTPStensorflowPSlitePStocoPStflitePSimportDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStflitePSimportDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/tflite/import.h"

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/toco/tflite/operator.h"
#include "tensorflow/lite/toco/tflite/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/lite/tools/verifier.h"

namespace toco {

namespace tflite {

namespace details {
void LoadTensorsTable(const ::tflite::Model& input_model,
                      TensorsTable* tensors_table) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimportDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/toco/tflite/import.cc", "LoadTensorsTable");

  // TODO(aselle): add support to toco for multiple subgraphs.
  auto tensors = (*input_model.subgraphs())[0]->tensors();
  if (!tensors) return;
  for (const auto* tensor : *tensors) {
    tensors_table->push_back(tensor->name()->c_str());
  }
}

void LoadOperatorsTable(const ::tflite::Model& input_model,
                        OperatorsTable* operators_table) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimportDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/toco/tflite/import.cc", "LoadOperatorsTable");

  auto opcodes = input_model.operator_codes();
  if (!opcodes) return;
  for (const auto* opcode : *opcodes) {
    auto builtin_code = GetBuiltinCode(opcode);
    if (builtin_code != ::tflite::BuiltinOperator_CUSTOM) {
      operators_table->push_back(EnumNameBuiltinOperator(builtin_code));
    } else {
      operators_table->push_back(opcode->custom_code()->c_str());
    }
  }
}
}  // namespace details

void ImportTensors(const ::tflite::Model& input_model, Model* model) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimportDTcc mht_2(mht_2_v, 231, "", "./tensorflow/lite/toco/tflite/import.cc", "ImportTensors");

  auto tensors = (*input_model.subgraphs())[0]->tensors();
  auto* buffers = input_model.buffers();
  // auto tensors = input_model.tensors();
  if (!tensors) return;
  for (const auto* input_tensor : *tensors) {
    Array& array = model->GetOrCreateArray(input_tensor->name()->c_str());
    array.data_type = DataType::Deserialize(input_tensor->type());
    int buffer_index = input_tensor->buffer();
    auto* buffer = buffers->Get(buffer_index);
    DataBuffer::Deserialize(*input_tensor, *buffer, &array);

    auto shape = input_tensor->shape();
    if (shape) {
      // If the shape is 0-dimensional, make sure to record it as such,
      // as oppose to leaving the array without a shape.
      array.mutable_shape()->mutable_dims()->clear();
      for (uint32_t i = 0; i < shape->Length(); ++i) {
        auto d = shape->Get(i);
        array.mutable_shape()->mutable_dims()->push_back(d);
      }
    }

    auto quantization = input_tensor->quantization();
    if (quantization) {
      // Note that tf.mini only supports a single quantization parameters for
      // the whole array.
      if (quantization->min() && quantization->max()) {
        CHECK_EQ(1, quantization->min()->Length());
        CHECK_EQ(1, quantization->max()->Length());
        MinMax& minmax = array.GetOrCreateMinMax();
        minmax.min = quantization->min()->Get(0);
        minmax.max = quantization->max()->Get(0);
      }
      if (quantization->scale() && quantization->zero_point()) {
        CHECK_EQ(1, quantization->scale()->Length());
        CHECK_EQ(1, quantization->zero_point()->Length());
        QuantizationParams& q = array.GetOrCreateQuantizationParams();
        q.scale = quantization->scale()->Get(0);
        q.zero_point = quantization->zero_point()->Get(0);
      }
    }
  }
}

void ImportOperators(
    const ::tflite::Model& input_model,
    const std::map<std::string, std::unique_ptr<BaseOperator>>& ops_by_name,
    const details::TensorsTable& tensors_table,
    const details::OperatorsTable& operators_table, Model* model) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimportDTcc mht_3(mht_3_v, 283, "", "./tensorflow/lite/toco/tflite/import.cc", "ImportOperators");

  // TODO(aselle): add support for multiple subgraphs.
  auto ops = (*input_model.subgraphs())[0]->operators();

  if (!ops) return;
  for (const auto* input_op : *ops) {
    uint32_t index = input_op->opcode_index();
    if (index > operators_table.size()) {
      LOG(FATAL) << "Index " << index << " must be between zero and "
                 << operators_table.size();
    }
    std::string opname = operators_table.at(index);

    // Find and use the appropriate operator deserialization factory.
    std::unique_ptr<Operator> new_op = nullptr;
    if (ops_by_name.count(opname) == 0) {
      std::string effective_opname = "TENSORFLOW_UNSUPPORTED";
      if (ops_by_name.count(effective_opname) == 0) {
        LOG(FATAL) << "Internal logic error: TENSORFLOW_UNSUPPORTED not found.";
      }
      new_op = ops_by_name.at(effective_opname)
                   ->Deserialize(input_op->builtin_options(),
                                 input_op->custom_options());
      if (new_op->type == OperatorType::kUnsupported) {
        auto* unsupported_op =
            static_cast<TensorFlowUnsupportedOperator*>(new_op.get());
        unsupported_op->tensorflow_op = opname;
        // TODO(b/109932940): Remove this when quantized is removed.
        // For now, we assume all ops are quantized.
        unsupported_op->quantized = true;
      } else {
        LOG(FATAL) << "Expected a TensorFlowUnsupportedOperator";
      }
    } else {
      new_op = ops_by_name.at(opname)->Deserialize(input_op->builtin_options(),
                                                   input_op->custom_options());
    }
    model->operators.emplace_back(new_op.release());
    auto* op = model->operators.back().get();

    // Make sure all the inputs and outputs are hooked up.
    auto inputs = input_op->inputs();
    for (uint32_t i = 0; i < inputs->Length(); i++) {
      auto input_index = inputs->Get(i);
      // input_index == -1 indicates optional tensor.
      if (input_index != -1) {
        const std::string& input_name = tensors_table.at(input_index);
        op->inputs.push_back(input_name);
      } else {
        const std::string& tensor_name =
            toco::AvailableArrayName(*model, "OptionalTensor");
        model->CreateOptionalArray(tensor_name);
        op->inputs.push_back(tensor_name);
      }
    }
    auto outputs = input_op->outputs();
    for (int i = 0, end = outputs->Length(); i < end; i++) {
      auto output_index = outputs->Get(i);
      const std::string& output_name = tensors_table.at(output_index);
      op->outputs.push_back(output_name);
    }
  }
}

void ImportIOTensors(const ModelFlags& model_flags,
                     const ::tflite::Model& input_model,
                     const details::TensorsTable& tensors_table, Model* model) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimportDTcc mht_4(mht_4_v, 352, "", "./tensorflow/lite/toco/tflite/import.cc", "ImportIOTensors");

  // Import from the first subgraph if input arrays have not been specified.
  if (model_flags.input_arrays().empty()) {
    auto inputs = (*input_model.subgraphs())[0]->inputs();
    if (inputs) {
      for (int input : *inputs) {
        const std::string& input_name = tensors_table.at(input);
        model->flags.add_input_arrays()->set_name(input_name);
      }
    }
  }

  // Import from the first subgraph if output arrays have not been specified.
  if (model_flags.output_arrays().empty()) {
    auto outputs = (*input_model.subgraphs())[0]->outputs();
    if (outputs) {
      for (int output : *outputs) {
        const std::string& output_name = tensors_table.at(output);
        model->flags.add_output_arrays(output_name);
      }
    }
  }
}

namespace {
bool Verify(const void* buf, size_t len) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimportDTcc mht_5(mht_5_v, 380, "", "./tensorflow/lite/toco/tflite/import.cc", "Verify");

  ::flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  return ::tflite::VerifyModelBuffer(verifier);
}
}  // namespace

std::unique_ptr<Model> Import(const ModelFlags& model_flags,
                              const std::string& input_file_contents) {
  ::tflite::AlwaysTrueResolver r;
  if (!::tflite::Verify(input_file_contents.data(), input_file_contents.size(),
                        r, ::tflite::DefaultErrorReporter())) {
    LOG(FATAL) << "Invalid flatbuffer.";
  }
  const ::tflite::Model* input_model =
      ::tflite::GetModel(input_file_contents.data());

  // Full list of all known operators.
  const auto ops_by_name = BuildOperatorByNameMap();

  if (!input_model->subgraphs() || input_model->subgraphs()->size() != 1) {
    LOG(FATAL) << "Number of subgraphs in tflite should be exactly 1.";
  }
  std::unique_ptr<Model> model;
  model.reset(new Model);

  details::TensorsTable tensors_table;
  details::LoadTensorsTable(*input_model, &tensors_table);

  details::OperatorsTable operators_table;
  details::LoadOperatorsTable(*input_model, &operators_table);

  ImportTensors(*input_model, model.get());
  ImportOperators(*input_model, ops_by_name, tensors_table, operators_table,
                  model.get());

  ImportIOTensors(model_flags, *input_model, tensors_table, model.get());

  UndoWeightsShuffling(model.get());

  return model;
}

}  // namespace tflite

}  // namespace toco
