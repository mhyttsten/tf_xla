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
class MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc() {
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

#include "tensorflow/lite/tools/strip_buffers/stripping_lib.h"

#include <stdint.h>

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/model.h"

#define TFLITE_SCHEMA_VERSION 3

namespace tflite {

using ::flatbuffers::FlatBufferBuilder;
using ::flatbuffers::Offset;

// Parameters for a simple Gaussian distribution to generate values roughly in
// [0, 1).
constexpr float kGaussianFloatMean = 0.5;
constexpr float kGaussianStdDev = 1.0 / 3;

template <typename Type, typename TypeT>
void CopyToOffsetVector(FlatBufferBuilder* builder, const Type* data,
                        std::vector<Offset<Type>>& vec) {
  std::unique_ptr<TypeT> unpacked(data->UnPack());
  flatbuffers::Offset<Type> offset = Type::Pack(*builder, unpacked.get());
  vec.push_back(offset);
}

int GetNumElements(const std::vector<int>& dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc mht_0(mht_0_v, 221, "", "./tensorflow/lite/tools/strip_buffers/stripping_lib.cc", "GetNumElements");

  int num_elements = 1;
  for (int i = 0; i < dims.size(); i++) {
    num_elements *= dims[i];
  }
  return num_elements;
}

// TODO(b/141023954): Reconcile this with the function in
// inference_profiler_stage.
template <typename T>
void GenerateRandomGaussianData(int64_t num_elements, float min, float max,
                                std::vector<T>* data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc mht_1(mht_1_v, 236, "", "./tensorflow/lite/tools/strip_buffers/stripping_lib.cc", "GenerateRandomGaussianData");

  data->clear();
  data->reserve(num_elements);

  static std::normal_distribution<double> distribution(kGaussianFloatMean,
                                                       kGaussianStdDev);
  static std::default_random_engine generator;
  for (int i = 0; i < num_elements; ++i) {
    auto rand_n = distribution(generator);
    while (rand_n < 0 || rand_n >= 1) {
      rand_n = distribution(generator);
    }
    auto rand_float = min + (max - min) * static_cast<float>(rand_n);
    data->push_back(static_cast<T>(rand_float));
  }
}

TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc mht_2(mht_2_v, 256, "", "./tensorflow/lite/tools/strip_buffers/stripping_lib.cc", "ConvertTensorType");

  *type = kTfLiteNoType;
  switch (tensor_type) {
    case TensorType_FLOAT32:
      *type = kTfLiteFloat32;
      break;
    case TensorType_INT32:
      *type = kTfLiteInt32;
      break;
    case TensorType_UINT32:
      *type = kTfLiteUInt32;
      break;
    case TensorType_UINT8:
      *type = kTfLiteUInt8;
      break;
    case TensorType_INT8:
      *type = kTfLiteInt8;
      break;
    default:
      break;
  }
  if (*type == kTfLiteNoType) {
    VLOG(0) << "Unsupported data type %d in tensor: " << tensor_type;
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus StripWeightsFromFlatbuffer(
    const Model* input_model,
    flatbuffers::FlatBufferBuilder* new_model_builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc mht_3(mht_3_v, 289, "", "./tensorflow/lite/tools/strip_buffers/stripping_lib.cc", "StripWeightsFromFlatbuffer");

  // TODO(b/141023954): Generalize to N subgraphs.
  if (input_model->subgraphs()->size() != 1) {
    VLOG(0) << "Only 1 subgraph supported for now: "
            << input_model->subgraphs()->size();
    return kTfLiteError;
  }

  // Data structures for output flatbuffer.
  std::vector<Offset<SubGraph>> output_subgraphs;
  std::vector<Offset<OperatorCode>> output_opcodes;
  std::vector<Offset<Buffer>> output_buffers;

  const SubGraph* input_subgraph = (*input_model->subgraphs())[0];
  std::unique_ptr<SubGraphT> mutable_subgraph(input_subgraph->UnPack());

  // For constant tensors that meet requirements:
  // 1. Set the buffer-id to something larger than total number of buffers.
  // This indicates to reconstitute_weights_into_fb that random data should be
  // generated for them.
  // 2. Mark that buffer for not getting carried into the output flatbuffer.
  absl::flat_hash_set<uint32_t> erased_buffers;
  const int num_buffers = input_model->buffers()->size();
  int i = 0;
  for (auto& tensor : mutable_subgraph->tensors) {
    // We don't support Int32 tensors because they could contain
    // non-randomisable information like Reshape dims.
    if (tensor->type == TensorType_INT32 &&
        GetNumElements(tensor->shape) < 10) {
      // Int32 tensors of elements < 10 could be non-randomisable: for example,
      // 'shape' input to a Reshape op.
      continue;
    } else if (tensor->type != TensorType_INT32 &&
               tensor->type != TensorType_FLOAT32 &&
               tensor->type != TensorType_UINT8 &&
               tensor->type != TensorType_INT8) {
      continue;
    }

    if (auto* buffer = (*input_model->buffers())[tensor->buffer]) {
      if (auto* array = buffer->data()) {
        VLOG(1) << "Tensor " << i
                << " is constant, with buffer = " << tensor->buffer;
        // Set tensor buffer to a high value (num_buffers * 2) & put an empty
        // buffer in place of the original one.
        erased_buffers.insert(tensor->buffer);
        tensor->buffer = num_buffers * 2;
      }
    }
    ++i;
  }

  i = 0;
  for (const Buffer* buffer : *(input_model->buffers())) {
    if (erased_buffers.find(i) == erased_buffers.end()) {
      // If buffer is not to be erased, insert it into the output flatbuffer to
      // retain data.
      CopyToOffsetVector<Buffer, BufferT>(new_model_builder, buffer,
                                          output_buffers);
    } else {
      output_buffers.push_back(CreateBuffer(*new_model_builder));
    }
    ++i;
  }

  flatbuffers::Offset<SubGraph> output_subgraph =
      SubGraph::Pack(*new_model_builder, mutable_subgraph.get());
  output_subgraphs.push_back(output_subgraph);

  // Write all ops as they are.
  for (const OperatorCode* opcode : *(input_model->operator_codes())) {
    CopyToOffsetVector<OperatorCode, OperatorCodeT>(new_model_builder, opcode,
                                                    output_opcodes);
  }

  // Generate output model.
  auto description =
      new_model_builder->CreateString("Generated by strip_buffers_from_fb");
  auto new_model_offset =
      CreateModel(*new_model_builder, TFLITE_SCHEMA_VERSION,
                  new_model_builder->CreateVector(output_opcodes),
                  new_model_builder->CreateVector(output_subgraphs),
                  description, new_model_builder->CreateVector(output_buffers),
                  /* metadata_buffer */ 0, /* metadatas */ 0);
  FinishModelBuffer(*new_model_builder, new_model_offset);

  return kTfLiteOk;
}

string StripWeightsFromFlatbuffer(const absl::string_view input_flatbuffer) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("input_flatbuffer: \"" + std::string(input_flatbuffer.data(), input_flatbuffer.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc mht_4(mht_4_v, 382, "", "./tensorflow/lite/tools/strip_buffers/stripping_lib.cc", "StripWeightsFromFlatbuffer");

  auto input_model = FlatBufferModel::BuildFromBuffer(input_flatbuffer.data(),
                                                      input_flatbuffer.size());

  FlatBufferBuilder builder(/*initial_size=*/10240);
  if (StripWeightsFromFlatbuffer(input_model->GetModel(), &builder) !=
      kTfLiteOk) {
    return string();
  }

  return string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                builder.GetSize());
}

bool FlatbufferHasStrippedWeights(const Model* input_model) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc mht_5(mht_5_v, 399, "", "./tensorflow/lite/tools/strip_buffers/stripping_lib.cc", "FlatbufferHasStrippedWeights");

  if (input_model->subgraphs()->size() != 1) {
    VLOG(0) << "Only 1 subgraph supported for now";
    return false;
  }
  const SubGraph* input_subgraph = (*input_model->subgraphs())[0];
  std::unique_ptr<SubGraphT> mutable_subgraph(input_subgraph->UnPack());

  // For all tensors that have buffer > num_buffers + 1 (set to be so in
  // strip_buffers_from_fb), create a buffer with random data & assign to them.
  // For others, just copy over the original buffer from source model.
  const int num_buffers = input_model->buffers()->size();
  for (auto& tensor : mutable_subgraph->tensors) {
    if (tensor->buffer > num_buffers + 1) {
      return true;
    }
  }
  return false;
}

TfLiteStatus ReconstituteConstantTensorsIntoFlatbuffer(
    const Model* input_model,
    flatbuffers::FlatBufferBuilder* new_model_builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc mht_6(mht_6_v, 424, "", "./tensorflow/lite/tools/strip_buffers/stripping_lib.cc", "ReconstituteConstantTensorsIntoFlatbuffer");

  // TODO(b/141023954): Generalize to N subgraphs.
  if (input_model->subgraphs()->size() != 1) {
    VLOG(0) << "Only 1 subgraph supported for now";
    return kTfLiteError;
  }
  const SubGraph* input_subgraph = (*input_model->subgraphs())[0];
  std::unique_ptr<SubGraphT> mutable_subgraph(input_subgraph->UnPack());

  // Containers for output flatbuffer.
  std::vector<Offset<::tflite::SubGraph>> output_subgraphs;
  std::vector<Offset<::tflite::OperatorCode>> output_opcodes;
  std::vector<Offset<::tflite::Buffer>> output_buffers;

  // An empty buffer, needed as a TFLite convention.
  output_buffers.push_back(CreateBuffer(*new_model_builder));

  // For all tensors that have buffer > num_buffers + 1 (set to be so in
  // strip_buffers_from_fb), create a buffer with random data & assign to them.
  // For others, just copy over the original buffer from source model.
  const int num_buffers = input_model->buffers()->size();
  for (auto& tensor : mutable_subgraph->tensors) {
    int buffer_id = output_buffers.size();
    if (tensor->buffer > num_buffers + 1) {
      // Num tensor elements.
      int num_elements = GetNumElements(tensor->shape);
      // Tensor type.
      TfLiteType type;
      if (ConvertTensorType(tensor->type, &type) != kTfLiteOk) {
        return kTfLiteError;
      }
      // Generate buffer random data.
      // Use different min/max bounds per tensor-type to ensure that random data
      // 'appears' similar to typical values.
      if (type == kTfLiteUInt8) {
        std::vector<uint8_t> data;
        GenerateRandomGaussianData(num_elements,
                                   std::numeric_limits<uint8_t>::min(),
                                   std::numeric_limits<uint8_t>::max(), &data);
        auto data_buffer = new_model_builder->CreateVector(
            reinterpret_cast<const uint8_t*>(data.data()),
            sizeof(uint8_t) * data.size());
        output_buffers.push_back(CreateBuffer(*new_model_builder, data_buffer));
      } else if (type == kTfLiteInt8) {
        std::vector<int8_t> data;
        GenerateRandomGaussianData(num_elements,
                                   std::numeric_limits<int8_t>::min(),
                                   std::numeric_limits<int8_t>::max(), &data);
        auto data_buffer = new_model_builder->CreateVector(
            reinterpret_cast<const uint8_t*>(data.data()),
            sizeof(int8_t) * data.size());
        output_buffers.push_back(CreateBuffer(*new_model_builder, data_buffer));
      } else if (type == kTfLiteFloat32) {
        std::vector<float_t> data;
        GenerateRandomGaussianData(num_elements, -1, 1, &data);
        auto data_buffer = new_model_builder->CreateVector(
            reinterpret_cast<const uint8_t*>(data.data()),
            sizeof(float_t) * data.size());
        output_buffers.push_back(CreateBuffer(*new_model_builder, data_buffer));
      } else if (type == kTfLiteInt32) {
        std::vector<int32_t> data;
        GenerateRandomGaussianData(num_elements, 10, 10, &data);
        auto data_buffer = new_model_builder->CreateVector(
            reinterpret_cast<const uint8_t*>(data.data()),
            sizeof(int32_t) * data.size());
        output_buffers.push_back(CreateBuffer(*new_model_builder, data_buffer));
      }
    } else {
      // For intermediate tensors, create a new buffer & assign the id to them.
      // output_buffers.push_back(CreateBuffer(*new_model_builder));
      CopyToOffsetVector<Buffer, BufferT>(
          new_model_builder, input_model->buffers()->Get(tensor->buffer),
          output_buffers);
    }
    tensor->buffer = buffer_id;
  }

  for (const ::tflite::OperatorCode* opcode :
       *(input_model->operator_codes())) {
    CopyToOffsetVector<::tflite::OperatorCode, ::tflite::OperatorCodeT>(
        new_model_builder, opcode, output_opcodes);
  }

  flatbuffers::Offset<::tflite::SubGraph> output_subgraph =
      ::tflite::SubGraph::Pack(*new_model_builder, mutable_subgraph.get());
  output_subgraphs.push_back(output_subgraph);

  auto description = new_model_builder->CreateString(
      "Generated by TFLite strip_buffers/reconstitution");
  auto new_model_offset =
      CreateModel(*new_model_builder, TFLITE_SCHEMA_VERSION,
                  new_model_builder->CreateVector(output_opcodes),
                  new_model_builder->CreateVector(output_subgraphs),
                  description, new_model_builder->CreateVector(output_buffers),
                  /* metadata_buffer */ 0, /* metadatas */ 0);
  FinishModelBuffer(*new_model_builder, new_model_offset);

  return kTfLiteOk;
}

string ReconstituteConstantTensorsIntoFlatbuffer(
    const absl::string_view input_flatbuffer) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("input_flatbuffer: \"" + std::string(input_flatbuffer.data(), input_flatbuffer.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSstrip_buffersPSstripping_libDTcc mht_7(mht_7_v, 529, "", "./tensorflow/lite/tools/strip_buffers/stripping_lib.cc", "ReconstituteConstantTensorsIntoFlatbuffer");

  auto input_model = FlatBufferModel::BuildFromBuffer(input_flatbuffer.data(),
                                                      input_flatbuffer.size());

  FlatBufferBuilder builder(/*initial_size=*/10240);
  if (ReconstituteConstantTensorsIntoFlatbuffer(input_model->GetModel(),
                                                &builder) != kTfLiteOk) {
    return string();
  }

  return string(reinterpret_cast<const char*>(builder.GetBufferPointer()),
                builder.GetSize());
}

}  // namespace tflite
