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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/grafter.h"

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "flatbuffers/reflection.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"

namespace fb = flatbuffers;

namespace tflite {
namespace acceleration {

namespace {

class Combiner : FlatbufferHelper {
 public:
  Combiner(flatbuffers::FlatBufferBuilder* fbb,
           std::vector<const Model*> models,
           std::vector<std::string> subgraph_names,
           const reflection::Schema* schema)
      : FlatbufferHelper(fbb, schema),
        fbb_(fbb),
        models_(models),
        subgraph_names_(subgraph_names),
        schema_(schema) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTcc mht_0(mht_0_v, 217, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/grafter.cc", "Combiner");
}
  absl::Status Combine() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/grafter.cc", "Combine");

    auto operator_codes = OperatorCodes();
    if (!operator_codes.ok()) {
      return operator_codes.status();
    }
    auto subgraphs = SubGraphs();
    if (!subgraphs.ok()) {
      return subgraphs.status();
    }
    auto buffers = Buffers();
    if (!buffers.ok()) {
      return buffers.status();
    }
    auto metadata = Metadatas();
    if (!metadata.ok()) {
      return metadata.status();
    }
    auto signature_defs = SignatureDefs();
    if (!signature_defs.ok()) {
      return signature_defs.status();
    }
    fb::Offset<Model> model = CreateModel(
        *fbb_, 3, *operator_codes, *subgraphs,
        fbb_->CreateString(models_[0]->description()->str()), *buffers,
        /* metadata_buffer */ 0, *metadata, *signature_defs);
    fbb_->Finish(model, "TFL3");
    return absl::OkStatus();
  }

 private:
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<OperatorCode>>>>
  OperatorCodes() {
    std::vector<fb::Offset<OperatorCode>> codes;
    for (const Model* model : models_) {
      for (int i = 0; i < model->operator_codes()->size(); i++) {
        auto status = CopyTableToVector(
            "tflite.OperatorCode", model->operator_codes()->Get(i), &codes);
        if (!status.ok()) {
          return status;
        }
      }
    }
    return fbb_->CreateVector(codes);
  }
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<SubGraph>>>> SubGraphs() {
    std::vector<fb::Offset<SubGraph>> graphs;
    int buffer_offset = 0;
    int operator_code_offset = 0;
    int subgraph_index = 0;
    for (const Model* model : models_) {
      if (model->subgraphs()->size() != 1) {
        return absl::InvalidArgumentError(
            "Every model to be combined must have a single subgraph.");
      }
      auto graph =
          AdjustSubGraph(model->subgraphs()->Get(0), buffer_offset,
                         operator_code_offset, subgraph_names_[subgraph_index]);
      if (!graph.ok()) {
        return graph.status();
      }
      graphs.push_back(*graph);
      buffer_offset += model->buffers()->size();
      operator_code_offset += model->operator_codes()->size();
      ++subgraph_index;
    }
    return fbb_->CreateVector(graphs);
  }
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<Buffer>>>> Buffers() {
    std::vector<fb::Offset<Buffer>> buffers;
    for (const Model* model : models_) {
      for (int i = 0; i < model->buffers()->size(); i++) {
        auto status = CopyTableToVector("tflite.Buffer",
                                        model->buffers()->Get(i), &buffers);
        if (!status.ok()) {
          return status;
        }
      }
    }
    return fbb_->CreateVector(buffers);
  }
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<Metadata>>>> Metadatas() {
    std::vector<fb::Offset<Metadata>> metadatas;
    int buffer_offset = 0;
    for (const Model* model : models_) {
      for (int i = 0; model->metadata() && i < model->metadata()->size(); i++) {
        auto metadata =
            AdjustMetadata(model->metadata()->Get(i), buffer_offset);
        if (!metadata.ok()) {
          return metadata.status();
        }
        metadatas.push_back(*metadata);
        buffer_offset += model->buffers()->size();
      }
    }
    return fbb_->CreateVector(metadatas);
  }
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<SignatureDef>>>>
  SignatureDefs() {
    std::vector<fb::Offset<SignatureDef>> signature_defs;
    const Model* model = models_[0];
    for (int i = 0;
         model->signature_defs() && i < model->signature_defs()->size(); i++) {
      auto status =
          CopyTableToVector("tflite.SignatureDef",
                            model->signature_defs()->Get(i), &signature_defs);
      if (!status.ok()) {
        return status;
      }
    }
    return fbb_->CreateVector(signature_defs);
  }

  absl::StatusOr<fb::Offset<SubGraph>> AdjustSubGraph(const SubGraph* graph,
                                                      int buffer_offset,
                                                      int operator_code_offset,
                                                      const std::string& name) {
    auto tensors = AdjustTensors(graph, buffer_offset);
    if (!tensors.ok()) {
      return tensors.status();
    }
    auto ops = AdjustOps(graph, operator_code_offset);
    if (!ops.ok()) {
      return ops.status();
    }
    return CreateSubGraph(*fbb_, fbb_->CreateVector(*tensors),
                          CopyIntVector(graph->inputs()),
                          CopyIntVector(graph->outputs()),
                          fbb_->CreateVector(*ops), fbb_->CreateString(name));
  }

  absl::StatusOr<std::vector<fb::Offset<Operator>>> AdjustOps(
      const SubGraph* graph, int operator_code_offset) {
    std::vector<fb::Offset<Operator>> ops;
    auto op_object = FindObject("tflite.Operator");
    const reflection::Field* builtin_options_field = nullptr;
    for (auto it = op_object->fields()->cbegin();
         it != op_object->fields()->cend(); it++) {
      auto candidate = *it;
      if (candidate->name()->str() == "builtin_options") {
        builtin_options_field = candidate;
        break;
      }
    }
    if (!builtin_options_field) {
      return absl::UnknownError(
          "Wasn't able to find the builtin_options field on tflite.Operator");
    }
    for (int i = 0; i < graph->operators()->size(); i++) {
      const Operator* op = graph->operators()->Get(i);
      fb::Offset<void> copied_builtin_options = 0;
      if (op->builtin_options() != nullptr) {
        const fb::Table* opt = (const fb::Table*)op;  // NOLINT
        auto& builtin_options_object = fb::GetUnionType(
            *schema_, *op_object, *builtin_options_field, *opt);
        copied_builtin_options =
            fb::CopyTable(*fbb_, *schema_, builtin_options_object,
                          *fb::GetFieldT(*opt, *builtin_options_field))
                .o;
      }
      ops.push_back(CreateOperator(
          *fbb_, op->opcode_index() + operator_code_offset,
          CopyIntVector(op->inputs()), CopyIntVector(op->outputs()),
          op->builtin_options_type(), copied_builtin_options,
          CopyIntVector(op->custom_options()), op->custom_options_format(),
          CopyIntVector(op->mutating_variable_inputs()),
          CopyIntVector(op->intermediates())));
    }
    return ops;
  }

  absl::StatusOr<std::vector<fb::Offset<Tensor>>> AdjustTensors(
      const SubGraph* graph, int buffer_offset) {
    std::vector<fb::Offset<Tensor>> tensors;
    auto orig_tensors = graph->tensors();
    for (auto iter = orig_tensors->cbegin(); iter != orig_tensors->cend();
         iter++) {
      auto i = *iter;
      std::vector<int32_t> shape{i->shape()->cbegin(), i->shape()->cend()};
      std::vector<int32_t> shape_signature;
      if (i->shape_signature()) {
        shape_signature.assign(i->shape_signature()->cbegin(),
                               i->shape_signature()->cend());
      }
      auto quantization =
          CopyTable("tflite.QuantizationParameters", i->quantization());
      if (!quantization.ok()) {
        return quantization.status();
      }
      auto sparsity = CopyTable("tflite.SparsityParameters", i->sparsity());
      if (!sparsity.ok()) {
        return sparsity.status();
      }
      tensors.push_back(CreateTensor(
          *fbb_, fbb_->CreateVector(shape), i->type(),
          i->buffer() + buffer_offset, fbb_->CreateString(i->name()->str()),
          *quantization, i->is_variable(), *sparsity,
          shape_signature.empty() ? 0 : fbb_->CreateVector(shape_signature)));
    }
    return tensors;
  }

  absl::StatusOr<fb::Offset<Metadata>> AdjustMetadata(const Metadata* metadata,
                                                      int buffer_offset) {
    return CreateMetadata(*fbb_,
                          metadata->name()
                              ? fbb_->CreateString(metadata->name()->str())
                              : 0,
                          metadata->buffer())
        .o;
  }

  flatbuffers::FlatBufferBuilder* fbb_;
  std::vector<const Model*> models_;
  std::vector<std::string> subgraph_names_;
  const reflection::Schema* schema_;
};

}  // namespace

absl::Status CombineModels(flatbuffers::FlatBufferBuilder* fbb,
                           std::vector<const Model*> models,
                           std::vector<std::string> subgraph_names,
                           const reflection::Schema* schema) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTcc mht_2(mht_2_v, 446, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/grafter.cc", "CombineModels");

  if (!fbb || !schema) {
    return absl::InvalidArgumentError(
        "Must provide FlatBufferBuilder and Schema");
  }
  if (models.size() < 2) {
    return absl::InvalidArgumentError("Must have 2+ models to combine");
  }
  Combiner combiner(fbb, models, subgraph_names, schema);
  return combiner.Combine();
}

FlatbufferHelper::FlatbufferHelper(flatbuffers::FlatBufferBuilder* fbb,
                                   const reflection::Schema* schema)
    : fbb_(fbb), schema_(schema) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTcc mht_3(mht_3_v, 463, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/grafter.cc", "FlatbufferHelper::FlatbufferHelper");
}

const reflection::Object* FlatbufferHelper::FindObject(
    const std::string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSgrafterDTcc mht_4(mht_4_v, 470, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/grafter.cc", "FlatbufferHelper::FindObject");

  for (auto candidate = schema_->objects()->cbegin();
       candidate != schema_->objects()->cend(); candidate++) {
    if (candidate->name()->str() == name) {
      return *candidate;
    }
  }
  return nullptr;
}

}  // namespace acceleration
}  // namespace tflite
