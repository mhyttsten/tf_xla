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
class MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc() {
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
#include "tensorflow/lite/tools/serialization/writer_lib.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/tools/serialization/enum_mapping.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace {

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<OperatorCode>>>
CreateOpCodeTableImpl(flatbuffers::FlatBufferBuilder* fbb,
                      std::vector<OpCode>* opcodes) {
  std::vector<flatbuffers::Offset<OperatorCode>> codes;
  for (const auto& it : *opcodes) {
    const char* custom_name = it.custom.empty() ? nullptr : it.custom.c_str();
    codes.push_back(CreateOperatorCodeDirect(
        *fbb, static_cast<BuiltinOperator>(it.builtin), custom_name));
  }
  return fbb->template CreateVector<flatbuffers::Offset<OperatorCode>>(codes);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
ExportBuffersImpl(flatbuffers::FlatBufferBuilder* fbb,
                  std::vector<std::pair<const uint8_t*, size_t>>* buffers) {
  std::vector<flatbuffers::Offset<Buffer>> buffer_vector;
  for (auto buffer : *buffers) {
    auto data_offset = fbb->CreateVector(buffer.first, buffer.second);
    buffer_vector.push_back(CreateBuffer(*fbb, data_offset));
  }
  return fbb->template CreateVector<flatbuffers::Offset<Buffer>>(buffer_vector);
}

TfLiteStatus WriteImpl(const std::string& filename, void* data, size_t size) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_0(mht_0_v, 228, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "WriteImpl");

  FILE* fp = fopen(filename.c_str(), "wb");
  if (!fp) return kTfLiteError;

  const int result_size = fwrite(data, 1, size, fp);
  fclose(fp);
  if (result_size != size) return kTfLiteError;

  return kTfLiteOk;
}

std::pair<BuiltinOptions, flatbuffers::Offset<void>> CreateBuiltinUnion(
    flatbuffers::FlatBufferBuilder* fbb, enum BuiltinOperator op,
    void* builtin_op_data, const TfLiteNode& node) {
  switch (op) {
#include "tensorflow/lite/tools/serialization/option_writer_generated.h"
  }
  return std::make_pair(BuiltinOptions_NONE, flatbuffers::Offset<void>());
}

}  // namespace

template <class T_OUTPUT, class T_INPUT>
flatbuffers::Offset<flatbuffers::Vector<T_OUTPUT>> SubgraphWriter::ExportVector(
    flatbuffers::FlatBufferBuilder* fbb, const T_INPUT& v) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_1(mht_1_v, 255, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::ExportVector");

  std::vector<T_OUTPUT> inputs(v.begin(), v.end());
  return fbb->template CreateVector<T_OUTPUT>(inputs);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Operator>>>
SubgraphWriter::ExportOperators(flatbuffers::FlatBufferBuilder* fbb) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_2(mht_2_v, 264, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::ExportOperators");

  std::vector<flatbuffers::Offset<Operator>> operators;

  std::vector<int> operator_to_opcode;
  // TODO(aselle): Augment this once we put execution plan in schema.
  operator_to_opcode.resize(subgraph_->nodes_size(), -1);
  for (int op_index : execution_plan_) {
    const auto* node_and_registration =
        subgraph_->node_and_registration(op_index);
    const TfLiteRegistration* registration = &node_and_registration->second;
    if (!registration->custom_name) {
      operator_to_opcode[op_index] =
          GetOpCodeForBuiltin(registration->builtin_code);
    } else {
      operator_to_opcode[op_index] =
          GetOpCodeForCustom(registration->custom_name);
    }
  }
  // second pass serialize operators
  for (int op_index : execution_plan_) {
    const auto* node_and_registration =
        subgraph_->node_and_registration(op_index);
    const TfLiteNode& node = node_and_registration->first;
    const TfLiteRegistration& registration = node_and_registration->second;
    flatbuffers::Offset<void> builtin_options;
    BuiltinOptions builtin_options_type = BuiltinOptions_NONE;
    // Custom data
    // TODO(aselle): Custom options format is not known by default. Just assume
    // for now.
    auto custom_options_format = CustomOptionsFormat_FLEXBUFFERS;
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> custom_options = 0;

    if (!registration.custom_name) {
      // builtin
      auto builtin_options_and_type = CreateBuiltinUnion(
          fbb, static_cast<enum BuiltinOperator>(registration.builtin_code),
          node.builtin_data, node);
      builtin_options = builtin_options_and_type.second;
      builtin_options_type = builtin_options_and_type.first;
    } else {
      auto custom_writer = custom_op_to_writer_.find(registration.custom_name);
      if (custom_writer != custom_op_to_writer_.end() &&
          custom_writer->second) {
        // delegate to custom writer if it exists
        custom_writer->second(fbb, subgraph_, op_index, &custom_options,
                              &custom_options_format);
      } else {
        // use the custom data as fact
        custom_options = fbb->CreateVector(
            reinterpret_cast<const uint8_t*>(node.custom_initial_data),
            node.custom_initial_data_size);
      }
    }

    int opcode_index = operator_to_opcode[op_index];
    std::vector<int> written_inputs =
        RemapTensorIndicesToWritten(TfLiteIntArrayView(node.inputs));
    std::vector<int> written_outputs =
        RemapTensorIndicesToWritten(TfLiteIntArrayView(node.outputs));
    auto inputs = ExportVector<int32_t>(fbb, written_inputs);
    auto outputs = ExportVector<int32_t>(fbb, written_outputs);
    operators.push_back(CreateOperator(*fbb, opcode_index, inputs, outputs,
                                       builtin_options_type, builtin_options,
                                       custom_options, custom_options_format));
  }

  return fbb->template CreateVector<flatbuffers::Offset<Operator>>(operators);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Tensor>>>
SubgraphWriter::ExportTensors(flatbuffers::FlatBufferBuilder* fbb) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_3(mht_3_v, 337, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::ExportTensors");

  // Initialized to -1.
  // A value of -1 means this tensor will not be exported.
  tensor_to_written_tensor_.resize(subgraph_->tensors_size(), -1);

  std::vector<flatbuffers::Offset<Tensor>> tensors;

  // Make a map from tensor index to whether the tensor is a temporary.
  std::vector<bool> tensor_is_temporary(subgraph_->tensors_size(), false);
  for (int op_index = 0; op_index < subgraph_->nodes_size(); ++op_index) {
    const auto* node_and_registration =
        subgraph_->node_and_registration(op_index);
    for (auto tensor_index :
         TfLiteIntArrayView(node_and_registration->first.temporaries))
      tensor_is_temporary[tensor_index] = true;
  }

  // Now we need to remap all used tensor indices
  int curr_output_index = 0;
  for (int tensor_index = 0; tensor_index < subgraph_->tensors_size();
       tensor_index++) {
    // Temporary tensors and unused tensors will not be written.
    if (!tensor_is_temporary[tensor_index] &&
        unused_tensors_.find(tensor_index) == unused_tensors_.end()) {
      tensor_to_written_tensor_[tensor_index] = curr_output_index++;
    }
  }

  for (int tensor_index = 0; tensor_index < subgraph_->tensors_size();
       ++tensor_index) {
    // Tensor not exported.
    if (tensor_to_written_tensor_[tensor_index] == -1) continue;

    if (TfLiteTensor* tensor = subgraph_->tensor(tensor_index)) {
      // Allocate a buffer index
      int buffer_index = 0;  // This is null
      if (tensor->allocation_type == kTfLiteMmapRo) {
        buffer_index = buffers_->size();
        buffers_->push_back(std::make_pair(
            reinterpret_cast<const uint8_t*>(tensor->data.raw), tensor->bytes));
      }
      // Primitive type.
      TensorType type = TfLiteTypeToSchemaType(tensor->type);
      // Handle quantization
      flatbuffers::Offset<QuantizationParameters> quantization_params;

      const flatbuffers::Offset<flatbuffers::Vector<float>> null_array;
      flatbuffers::Offset<flatbuffers::Vector<float>> scale_array;
      flatbuffers::Offset<flatbuffers::Vector<int64_t>> zero_point_array;

      if (tensor->quantization.type == kTfLiteAffineQuantization) {
        if (tensor->params.scale != 0.f) {
          // Quantization with a single argument array.
          scale_array = fbb->CreateVector<float>({tensor->params.scale});
          zero_point_array =
              fbb->CreateVector<int64_t>({tensor->params.zero_point});
          quantization_params = CreateQuantizationParameters(
              *fbb, null_array, null_array, scale_array, zero_point_array);
        } else {  // Multi channel quantization.
          const TfLiteAffineQuantization* params =
              reinterpret_cast<TfLiteAffineQuantization*>(
                  tensor->quantization.params);
          const size_t num_scales = params->scale->size;

          std::vector<float> scale_vector(params->scale->data,
                                          params->scale->data + num_scales);
          std::vector<int64_t> zero_point_vector(
              params->zero_point->data, params->zero_point->data + num_scales);
          scale_array = fbb->CreateVector<float>(scale_vector);
          zero_point_array = fbb->CreateVector<int64_t>(zero_point_vector);
          quantization_params = CreateQuantizationParameters(
              *fbb, null_array, null_array, scale_array, zero_point_array,
              QuantizationDetails_NONE, 0, params->quantized_dimension);
        }
      }

      // Shape
      // Some tensors added during op init are not registered formally as
      // node temporaries. Some didn't get memory allocated for them, and we
      // should avoid serializing those tensors.
      if (tensor->dims) {
        TfLiteIntArrayView shape_view(tensor->dims);
        std::vector<int> shape =
            std::vector<int>(shape_view.begin(), shape_view.end());

        Offset<flatbuffers::String> tensor_name_offset = 0;
        if (tensor->name != nullptr) {
          tensor_name_offset = fbb->CreateString(tensor->name);
        }

        tensors.push_back(CreateTensor(
            *fbb, ExportVector<int32_t>(fbb, shape), type, buffer_index,
            tensor_name_offset, quantization_params, tensor->is_variable));
      }
    }
  }
  return fbb->template CreateVector<flatbuffers::Offset<Tensor>>(tensors);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
SubgraphWriter::ExportBuffers(flatbuffers::FlatBufferBuilder* fbb) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_4(mht_4_v, 440, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::ExportBuffers");

  return ExportBuffersImpl(fbb, buffers_);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<OperatorCode>>>
SubgraphWriter::CreateOpCodeTable(flatbuffers::FlatBufferBuilder* fbb) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_5(mht_5_v, 448, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::CreateOpCodeTable");

  return CreateOpCodeTableImpl(fbb, opcodes_);
}

template <class T>
std::vector<int> SubgraphWriter::RemapTensorIndicesToWritten(const T& input) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_6(mht_6_v, 456, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::RemapTensorIndicesToWritten");

  std::vector<int> output;
  output.reserve(input.size());
  for (int x : input) {
    // Special value representing an optional tensor which is not present.
    if (x == -1) {
      output.push_back(x);
      continue;
    }
    if (tensor_to_written_tensor_[x] != -1) {
      output.push_back(tensor_to_written_tensor_[x]);
    }
  }
  return output;
}

TfLiteStatus SubgraphWriter::GetBuffer(std::unique_ptr<uint8_t[]>* out,
                                       size_t* size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_7(mht_7_v, 476, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::GetBuffer");

  if (!out || !size) return kTfLiteError;
  flatbuffers::FlatBufferBuilder builder(/*initial_size=*/10240);
  std::vector<flatbuffers::Offset<SubGraph>> subgraphs_as_vector;
  subgraphs_as_vector.push_back(
      PopulateAndGetOffset(&builder, subgraph_->GetName()));

  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
      buffers = ExportBuffers(&builder);

  auto description = builder.CreateString("Exported from Subgraph.");

  auto op_codes = CreateOpCodeTable(&builder);
  auto model = CreateModel(builder, TFLITE_SCHEMA_VERSION, op_codes,
                           builder.CreateVector(subgraphs_as_vector),
                           description, buffers);
  ::tflite::FinishModelBuffer(builder, model);
  const uint8_t* buffer = builder.GetBufferPointer();
  *size = builder.GetSize();
  (*out).reset(new uint8_t[*size]);
  memcpy(out->get(), buffer, *size);
  return kTfLiteOk;
}

flatbuffers::Offset<SubGraph> SubgraphWriter::PopulateAndGetOffset(
    flatbuffers::FlatBufferBuilder* builder, const std::string& subgraph_name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("subgraph_name: \"" + subgraph_name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_8(mht_8_v, 505, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::PopulateAndGetOffset");

  auto tensors = ExportTensors(builder);
  std::vector<int> written_inputs = RemapTensorIndicesToWritten(inputs_);
  std::vector<int> written_outputs = RemapTensorIndicesToWritten(outputs_);
  auto inputs = ExportVector<int32_t>(builder, written_inputs);
  auto outputs = ExportVector<int32_t>(builder, written_outputs);

  auto ops = ExportOperators(builder);
  auto name = builder->CreateString(subgraph_name);
  return CreateSubGraph(*builder, tensors, inputs, outputs, ops, name);
}

TfLiteStatus SubgraphWriter::Write(const std::string& filename) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_9(mht_9_v, 521, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::Write");

  std::unique_ptr<uint8_t[]> buffer;
  size_t size;
  TF_LITE_ENSURE_STATUS(GetBuffer(&buffer, &size));
  return WriteImpl(filename, buffer.get(), size);
}

TfLiteStatus SubgraphWriter::RegisterCustomWriter(
    const std::string& custom_name, CustomWriter custom_writer) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("custom_name: \"" + custom_name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_10(mht_10_v, 533, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::RegisterCustomWriter");

  if (custom_op_to_writer_.find(custom_name) != custom_op_to_writer_.end()) {
    return kTfLiteError;
  }
  custom_op_to_writer_.insert(std::make_pair(custom_name, custom_writer));
  return kTfLiteOk;
}

TfLiteStatus SubgraphWriter::CheckInputOutput(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const std::vector<int>& execution_plan) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_11(mht_11_v, 546, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::CheckInputOutput");

  absl::flat_hash_set<int> known_tensors(inputs.begin(), inputs.end());
  known_tensors.insert(subgraph_->variables().begin(),
                       subgraph_->variables().end());
  // Scan execution plan and confirm input tensors are known before each node
  // executes. Then append output tensors to known tensors.
  for (int op_index : execution_plan) {
    const auto* node_and_registration =
        subgraph_->node_and_registration(op_index);
    const TfLiteNode& node = node_and_registration->first;
    for (int tensor_index : TfLiteIntArrayView(node.inputs)) {
      if (tensor_index < 0) {
        // Skip if optional input not present.
        if (tensor_index == kTfLiteOptionalTensor) {
          continue;
        } else {
          return kTfLiteError;
        }
      }
      if (TfLiteTensor* tensor = subgraph_->tensor(tensor_index)) {
        // Skip constant tensors.
        if (tensor->allocation_type == kTfLiteMmapRo) {
          continue;
        }
      }

      if (known_tensors.find(tensor_index) == known_tensors.end()) {
        subgraph_->context()->ReportError(
            subgraph_->context(),
            "Node (%d) uses an input (%d) that is not provided.", op_index,
            tensor_index);
        return kTfLiteError;
      }
    }
    TfLiteIntArrayView outputs(node.outputs);
    known_tensors.insert(outputs.begin(), outputs.end());
  }

  // Check if outputs are known tensors or constants.
  for (int tensor_index : outputs) {
    if (TfLiteTensor* tensor = subgraph_->tensor(tensor_index)) {
      // Skip constant tensors.
      if (tensor->allocation_type == kTfLiteMmapRo) {
        continue;
      }
    }

    if (known_tensors.find(tensor_index) == known_tensors.end()) {
      subgraph_->context()->ReportError(
          subgraph_->context(),
          "Output (%d) is not produced by the execution plan.", tensor_index);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SubgraphWriter::SetCustomInputOutput(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const std::vector<int>& execution_plan) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_12(mht_12_v, 608, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "SubgraphWriter::SetCustomInputOutput");

  TF_LITE_ENSURE_STATUS(CheckInputOutput(inputs, outputs, execution_plan));
  inputs_ = inputs;
  outputs_ = outputs;
  execution_plan_ = execution_plan;
  return kTfLiteOk;
}

ModelWriter::ModelWriter(Interpreter* interpreter) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_13(mht_13_v, 619, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "ModelWriter::ModelWriter");

  std::vector<Subgraph*> subgraphs;

  // Retrieves the list of the subgraphs from the interpreter for constructing
  // a list of SubgraphWriters.
  subgraphs.reserve(interpreter->subgraphs_size());
  for (int i = 0; i < interpreter->subgraphs_size(); ++i) {
    subgraphs.push_back(interpreter->subgraph(i));
  }

  Init(subgraphs);
}

ModelWriter::ModelWriter(const std::vector<Subgraph*>& subgraphs) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_14(mht_14_v, 635, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "ModelWriter::ModelWriter");

  Init(subgraphs);
}

void ModelWriter::Init(const std::vector<Subgraph*>& subgraphs) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_15(mht_15_v, 642, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "ModelWriter::Init");

  buffers_.push_back(std::make_pair(nullptr, 0));
  subgraph_writers_.reserve(subgraphs.size());
  for (auto* subgraph : subgraphs) {
    SubgraphWriter writer(subgraph, &buffers_, &opcodes_,
                          &builtin_op_to_opcode_);
    subgraph_writers_.push_back(writer);
  }
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
ModelWriter::ExportBuffers(flatbuffers::FlatBufferBuilder* fbb) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_16(mht_16_v, 656, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "ModelWriter::ExportBuffers");

  return ExportBuffersImpl(fbb, &buffers_);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<OperatorCode>>>
ModelWriter::CreateOpCodeTable(flatbuffers::FlatBufferBuilder* fbb) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_17(mht_17_v, 664, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "ModelWriter::CreateOpCodeTable");

  return CreateOpCodeTableImpl(fbb, &opcodes_);
}

TfLiteStatus ModelWriter::GetBuffer(std::unique_ptr<uint8_t[]>* out,
                                    size_t* size) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_18(mht_18_v, 672, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "ModelWriter::GetBuffer");

  if (!out || !size) return kTfLiteError;
  flatbuffers::FlatBufferBuilder builder(/*initial_size=*/10240);

  std::vector<flatbuffers::Offset<SubGraph>> subgraphs_as_vector;
  for (auto& subgraph_writer : subgraph_writers_) {
    subgraphs_as_vector.push_back(subgraph_writer.PopulateAndGetOffset(
        &builder, subgraph_writer.subgraph_->GetName()));
  }

  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
      buffers = ExportBuffers(&builder);

  auto description = builder.CreateString("Exported from Subgraph.");

  auto op_codes = CreateOpCodeTable(&builder);
  auto model = CreateModel(builder, TFLITE_SCHEMA_VERSION, op_codes,
                           builder.CreateVector(subgraphs_as_vector),
                           description, buffers);
  ::tflite::FinishModelBuffer(builder, model);
  const uint8_t* buffer = builder.GetBufferPointer();
  *size = builder.GetSize();
  (*out).reset(new uint8_t[*size]);
  memcpy(out->get(), buffer, *size);
  return kTfLiteOk;
}

TfLiteStatus ModelWriter::Write(const std::string& filename) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_19(mht_19_v, 703, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "ModelWriter::Write");

  std::unique_ptr<uint8_t[]> buffer;
  size_t size;
  TF_LITE_ENSURE_STATUS(GetBuffer(&buffer, &size));
  return WriteImpl(filename, buffer.get(), size);
}

void ModelWriter::SetUnusedTensors(int subgraph_index,
                                   const std::set<int>& unused_tensors) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_20(mht_20_v, 714, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "ModelWriter::SetUnusedTensors");

  subgraph_writers_[subgraph_index].SetUnusedTensors(unused_tensors);
}

TfLiteStatus ModelWriter::SetCustomInputOutput(
    int subgraph_index, const std::vector<int>& inputs,
    const std::vector<int>& outputs, const std::vector<int>& execution_plan) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSwriter_libDTcc mht_21(mht_21_v, 723, "", "./tensorflow/lite/tools/serialization/writer_lib.cc", "ModelWriter::SetCustomInputOutput");

  return subgraph_writers_[subgraph_index].SetCustomInputOutput(inputs, outputs,
                                                                execution_plan);
}

}  // namespace tflite
