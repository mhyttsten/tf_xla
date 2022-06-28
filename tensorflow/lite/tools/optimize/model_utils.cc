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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc() {
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
#include "tensorflow/lite/tools/optimize/model_utils.h"

#include <fstream>
#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/tools/optimize/operator_property.h"

namespace tflite {
namespace optimize {
namespace utils {

namespace {

// Returns the index of the OpCode.
// If a OpCode doesn't exist, adds it and returns its index.
int32_t GetOrInsertOpCodeIndex(ModelT* model, const BuiltinOperator& op_code,
                               int32_t version) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "GetOrInsertOpCodeIndex");

  for (size_t i = 0; i < model->operator_codes.size(); ++i) {
    if (GetBuiltinCode(model->operator_codes[i].get()) == op_code) {
      return i;
    }
  }
  model->operator_codes.push_back(absl::make_unique<OperatorCodeT>());
  int op_code_idx = model->operator_codes.size() - 1;
  model->operator_codes[op_code_idx]->builtin_code = op_code;
  model->operator_codes[op_code_idx]->deprecated_builtin_code =
      ConvertBuiltinCodeToDeprecatedBuiltinCode(op_code);
  // Version 2 and onwards supports INT8 inputs.
  model->operator_codes[op_code_idx]->version = version;

  // Return the index of the newly placed OperatorCodeT.
  return op_code_idx;
}

}  // namespace

// Creates a Dequantize OperatorT object.
void MakeDequantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                            int32_t input, int32_t output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_1(mht_1_v, 232, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "MakeDequantizeOperator");

  OperatorT* op_raw = new OperatorT;
  // Version 2 and onwards supports INT8 inputs.
  op_raw->opcode_index =
      GetOrInsertOpCodeIndex(model, BuiltinOperator_DEQUANTIZE, 2);
  op_raw->inputs = {input};
  op_raw->outputs = {output};

  op->reset(op_raw);
}

// Creates a Quantize OperatorT object.
void MakeQuantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                          int32_t input, int32_t output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_2(mht_2_v, 248, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "MakeQuantizeOperator");

  OperatorT* op_raw = new OperatorT;
  op_raw->opcode_index =
      GetOrInsertOpCodeIndex(model, BuiltinOperator_QUANTIZE, 1);
  op_raw->inputs = {input};
  op_raw->outputs = {output};

  op->reset(op_raw);
}

// Create a new TensorT object without quantization parameters.
void MakeTensor(const string& name, const std::vector<int32_t>& shape,
                const std::vector<int32_t>& shape_signature,
                const TensorType& type, std::unique_ptr<TensorT>* tensor) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_3(mht_3_v, 265, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "MakeTensor");

  TensorT* tensor_raw = new TensorT;
  tensor_raw->name = name;
  tensor_raw->shape = shape;
  if (!shape_signature.empty()) {
    tensor_raw->shape_signature = shape_signature;
  }
  tensor_raw->type = type;

  tensor->reset(tensor_raw);
}

// Create a new TensorT object with quantization parameters.
void MakeTensorWithQuantParam(const string& name,
                              const std::vector<int32_t>& shape,
                              const std::vector<int32_t>& shape_signature,
                              const TensorType& type, float scale,
                              int64_t zero_point,
                              std::unique_ptr<TensorT>* tensor) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_4(mht_4_v, 287, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "MakeTensorWithQuantParam");

  MakeTensor(name, shape, shape_signature, type, tensor);
  (*tensor)->quantization = absl::make_unique<QuantizationParametersT>();
  (*tensor)->quantization->scale.push_back(scale);
  (*tensor)->quantization->zero_point.push_back(zero_point);
}

bool QuantizationParametersExist(const TensorT* tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_5(mht_5_v, 297, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "QuantizationParametersExist");

  return tensor->quantization != nullptr &&
         !tensor->quantization->scale.empty() &&
         !tensor->quantization->zero_point.empty();
}

bool HasBuffer(const ModelT* model, const SubGraphT* subgraph,
               int tensor_index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_6(mht_6_v, 307, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "HasBuffer");

  const int buffer_index = subgraph->tensors[tensor_index]->buffer;
  BufferT* buffer = model->buffers[buffer_index].get();
  if (buffer == nullptr || buffer->data.empty()) {
    return false;
  }
  return true;
}

bool HasMinMax(const TensorT* tensor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_7(mht_7_v, 319, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "HasMinMax");

  return tensor->quantization && !tensor->quantization->min.empty() &&
         !tensor->quantization->max.empty();
}

void SetOperatorCodeVersion(ModelT* model) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_8(mht_8_v, 327, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "SetOperatorCodeVersion");

  for (int subgraph_idx = 0, end = model->subgraphs.size(); subgraph_idx < end;
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    // Iterate backward to avoid messing with index.
    for (int op_idx = subgraph->operators.size() - 1; op_idx >= 0; op_idx--) {
      OperatorT* op = subgraph->operators[op_idx].get();
      OperatorCodeT* op_code = model->operator_codes[op->opcode_index].get();
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(model, subgraph_idx, op_idx);
      if (property.quantizable && op_code->version < property.version) {
        // Only update the versions of quantizable operations if the original
        // version is lesser than minimum quantized one mentioned by
        // OperatorProperty.
        op_code->version = property.version;
      }
    }
  }
}

void WriteFile(const std::string& out_file, const uint8_t* bytes,
               size_t num_bytes) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("out_file: \"" + out_file + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSmodel_utilsDTcc mht_9(mht_9_v, 352, "", "./tensorflow/lite/tools/optimize/model_utils.cc", "WriteFile");

  std::fstream stream(out_file, std::ios::binary | std::ios::out);
  for (size_t i = 0; i < num_bytes; i++) {
    stream << bytes[i];
  }
  TFLITE_DCHECK(!stream.bad() && !stream.fail());
}

std::unique_ptr<flatbuffers::FlatBufferBuilder> FinishModel(
    const tflite::ModelT* model) {
  std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(
      new flatbuffers::FlatBufferBuilder());
  auto packed_model = tflite::Model::Pack(*builder, model);
  tflite::FinishModelBuffer(*builder, packed_model);
  return builder;
}

std::unique_ptr<tflite::ModelT> CreateMutableModelFromFile(
    const string& model_filepath) {
  auto fb_model =
      tflite::FlatBufferModel::BuildFromFile(model_filepath.c_str());
  auto tflite_model = fb_model->GetModel();
  auto copied_model = absl::make_unique<tflite::ModelT>();
  tflite_model->UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}

}  // namespace utils
}  // namespace optimize
}  // namespace tflite
