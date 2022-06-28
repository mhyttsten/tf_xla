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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc() {
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
#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

// Note: branched from tensorflow/lite/tools/optimize/quantize_model_test.cc

namespace {
tensorflow::string* g_test_model_dir = nullptr;
}  // namespace

namespace tflite {
namespace optimize {
namespace {

TfLiteStatus QuantizeModel(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, const std::unordered_set<string>& operator_names,
    const TensorType& activations_type, ErrorReporter* error_reporter,
    bool disable_per_channel = false,
    const absl::flat_hash_set<std::string>& blocked_ops = {},
    const absl::flat_hash_set<std::string>& blocked_nodes = {}) {
  TensorType inference_tensor_type = activations_type;
  bool fully_quantize = !allow_float;
  auto status = mlir::lite::QuantizeModel(
      *model, input_type, output_type, inference_tensor_type,
      /*operator_names=*/{}, disable_per_channel, fully_quantize, builder,
      error_reporter, /*verify_numeric=*/false, /*whole_model_verify=*/false,
      /*legacy_float_scale=*/true, blocked_ops, blocked_nodes);
  if (status != kTfLiteOk) {
    return status;
  }
  std::string buffer(
      reinterpret_cast<const char*>(builder->GetCurrentBufferPointer()),
      builder->GetSize());

  auto flatbuffer_model =
      FlatBufferModel::BuildFromBuffer(buffer.c_str(), buffer.size());
  flatbuffer_model->GetModel()->UnPackTo(model);
  return kTfLiteOk;
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           ErrorReporter* error_reporter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_0(mht_0_v, 246, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeModel");

  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       /*operator_names=*/{}, TensorType_INT8, error_reporter);
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type,
                           ErrorReporter* error_reporter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_1(mht_1_v, 257, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeModel");

  return QuantizeModel(builder, model, input_type, output_type,
                       /*allow_float=*/false, error_reporter);
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_2(mht_2_v, 266, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeModel");

  return QuantizeModel(builder, model, TensorType_FLOAT32, TensorType_FLOAT32,
                       /*allow_float=*/true, error_reporter);
}

TfLiteStatus QuantizeModelAllOperators(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, const TensorType& activations_type,
    bool disable_per_channel, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_3(mht_3_v, 278, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeModelAllOperators");

  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       /*operator_names=*/{}, activations_type, error_reporter,
                       disable_per_channel);
}

TfLiteStatus QuantizeModelAllOperators(flatbuffers::FlatBufferBuilder* builder,
                                       ModelT* model,
                                       const TensorType& input_type,
                                       const TensorType& output_type,
                                       bool allow_float,
                                       const TensorType& activations_type,
                                       ErrorReporter* error_reporter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_4(mht_4_v, 293, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeModelAllOperators");

  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       /*operator_names=*/{}, activations_type, error_reporter);
}

std::unique_ptr<FlatBufferModel> ReadModel(const string& model_name) {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir, model_name);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

template <typename T>
std::vector<T> GetAsVector(const flatbuffers::Vector<T>* vec) {
  return std::vector<T>(vec->begin(), vec->end());
}

void VerifyAsymmetricQuantizationScale(
    const QuantizationParameters& float_quant_params,
    const QuantizationParametersT& quantized_quant_params) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_5(mht_5_v, 313, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "VerifyAsymmetricQuantizationScale");

  const float eps = 1e-7;
  ASSERT_EQ(float_quant_params.min()->size(), 1);
  ASSERT_EQ(float_quant_params.max()->size(), 1);
  float float_min = std::min(0.f, float_quant_params.min()->Get(0));
  float float_max = std::max(0.f, float_quant_params.max()->Get(0));

  ASSERT_EQ(quantized_quant_params.scale.size(), 1);
  ASSERT_EQ(quantized_quant_params.zero_point.size(), 1);
  float scale = (float_max - float_min) / 255;
  EXPECT_NEAR(scale, quantized_quant_params.scale[0], eps);
}

class QuantizeModelTest : public testing::Test {
 protected:
  QuantizeModelTest() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_6(mht_6_v, 331, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeModelTest");

    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  std::unique_ptr<FlatBufferModel> input_model_;
  const Model* readonly_model_;
  tflite::ModelT model_;
  flatbuffers::FlatBufferBuilder builder_;
  internal::FailOnErrorReporter error_reporter_;
};

void ExpectEqualTensor(TensorT* tensor, TensorT* expected_tensor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_7(mht_7_v, 347, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "ExpectEqualTensor");

  const float eps = 1e-7;
  EXPECT_NE(expected_tensor, nullptr);
  EXPECT_EQ(tensor->is_variable, expected_tensor->is_variable);
  EXPECT_EQ(tensor->shape, expected_tensor->shape);
  EXPECT_EQ(tensor->type, expected_tensor->type);
  const auto quantization_params = tensor->quantization.get();
  const auto expected_quantization_params = expected_tensor->quantization.get();
  if (quantization_params != nullptr &&
      expected_quantization_params != nullptr) {
    for (int i = 0; i < quantization_params->scale.size(); ++i) {
      if (quantization_params->scale[i] > 3e-5) {
        EXPECT_NEAR(quantization_params->scale[i],
                    expected_quantization_params->scale[i], eps);
      }
    }
    EXPECT_EQ(quantization_params->zero_point,
              expected_quantization_params->zero_point);
  }
}

TensorT* FindMatchingExpectedTensor(const SubGraphT& expected_graph,
                                    const ModelT& expected_model,
                                    const ModelT& quant_model,
                                    const OperatorT& quant_op, int idx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_8(mht_8_v, 374, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "FindMatchingExpectedTensor");

  const auto& builtin_code =
      GetBuiltinCode(quant_model.operator_codes[quant_op.opcode_index].get());
  for (const auto& expected_op : expected_graph.operators) {
    const auto& op_code =
        expected_model.operator_codes[expected_op->opcode_index].get();
    const auto& expected_code = GetBuiltinCode(op_code);
    if (expected_code == builtin_code) {
      return expected_graph.tensors[expected_op->inputs[idx]].get();
    }
  }
  return nullptr;
}

void ExpectSameModels(const ModelT& model, const ModelT& expected_model) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_9(mht_9_v, 391, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "ExpectSameModels");

  ASSERT_EQ(model.subgraphs.size(), expected_model.subgraphs.size());
  for (size_t subgraph_idx = 0; subgraph_idx < model.subgraphs.size();
       subgraph_idx++) {
    const auto graph = model.subgraphs[subgraph_idx].get();
    const auto expected_graph = expected_model.subgraphs[subgraph_idx].get();
    for (auto& op : graph->operators) {
      for (int idx = 0; idx < op->inputs.size(); idx++) {
        if (op->inputs[idx] < 0) {
          continue;
        }
        const auto& tensor = graph->tensors[op->inputs[idx]];
        auto* expected_tensor = FindMatchingExpectedTensor(
            *expected_graph, expected_model, model, *op, idx);
        if (!expected_tensor) {
          continue;
        }
        ExpectEqualTensor(tensor.get(), expected_tensor);
        if (tensor->buffer >= 0) {
          const int buffer_idx = tensor->buffer;
          const int expected_buffer_idx = expected_tensor->buffer;
          const auto buffer = model.buffers[buffer_idx].get()->data;
          const auto expected_buffer =
              expected_model.buffers[expected_buffer_idx].get()->data;
          EXPECT_EQ(buffer, expected_buffer);
        }
      }
    }
  }
}

class QuantizeConvModelTest : public QuantizeModelTest,
                              public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConvModelTest() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_10(mht_10_v, 428, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeConvModelTest");

    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
    // Flatbuffer is missing calibration data -- add dummy params.
    auto& subgraph = model_.subgraphs[0];
    auto* input = subgraph->tensors[subgraph->inputs[0]].get();
    auto* output = subgraph->tensors[subgraph->outputs[0]].get();
    input->quantization.reset(new QuantizationParametersT());
    output->quantization.reset(new QuantizationParametersT());
    input->quantization->min.push_back(0.0);
    output->quantization->min.push_back(0.0);
    input->quantization->max.push_back(6.0);
    output->quantization->max.push_back(6.0);
  }
  TensorType tensor_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeConvModelTestInst, QuantizeConvModelTest,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeConvModelTest, QuantizationSucceeds) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  const uint8_t* buffer = builder_.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);
}

TEST_P(QuantizeConvModelTest, SkipUnspecifiedLayer) {
  auto status = QuantizeModel(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
      /*allow_float=*/true, /*operator_names=*/{}, TensorType_FLOAT32,
      &error_reporter_, /*disable_per_channel=*/false, {"CONV_2D"});
  EXPECT_EQ(status, kTfLiteOk);

  ModelT expected_model;
  readonly_model_->UnPackTo(&expected_model);
  // The resulting model should be the same.
  ExpectSameModels(model_, expected_model);
}

TEST_P(QuantizeConvModelTest, SkipUnspecifiedLayerByName) {
  auto status = QuantizeModel(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
      /*allow_float=*/true, /*operator_names=*/{}, TensorType_FLOAT32,
      &error_reporter_, /*disable_per_channel=*/false, /*blocked_ops=*/{},
      {"output"});
  EXPECT_EQ(status, kTfLiteOk);

  ModelT expected_model;
  readonly_model_->UnPackTo(&expected_model);
  // The resulting model should be the same.
  ExpectSameModels(model_, expected_model);
}

TEST_P(QuantizeConvModelTest, GraphIsFullyQuantized) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_,
      /*allow_float=*/false, tensor_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  for (const auto& subgraph : model_.subgraphs) {
    for (const auto& tensor : subgraph->tensors) {
      EXPECT_TRUE(tensor->type == TensorType_INT32 ||
                  tensor->type == TensorType_INT8);
    }
  }
}

class QuantizeConvNoBiasModelTest : public QuantizeModelTest {
 protected:
  QuantizeConvNoBiasModelTest() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_11(mht_11_v, 506, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeConvNoBiasModelTest");

    input_model_ = ReadModel(internal::kConvModelWithNoBias);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeConvNoBiasModelTest, QuantizationSucceeds) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);
  const uint8_t* buffer = builder_.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);
}

class QuantizeSplitModelTest : public QuantizeModelTest {
 protected:
  QuantizeSplitModelTest() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_12(mht_12_v, 528, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeSplitModelTest");

    input_model_ = ReadModel(internal::kModelSplit);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

// There are two outputs for split with different scales, the resulting model
// should have the scales be hardcodes to the input scale value.
TEST_F(QuantizeSplitModelTest, QuantizeSplit) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be two ops: the split and add in the original model.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 2);
  EXPECT_EQ(subgraph->operators.size(), 2);
  const auto& split = subgraph->operators[0];
  const auto& add = subgraph->operators[1];
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[split->opcode_index].get()),
            BuiltinOperator_SPLIT);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[add->opcode_index].get()),
            BuiltinOperator_ADD);

  // There should be 5 tensors: input, output, split, split/split_dim, split:1.
  // Tensor indices could be different between original and quantized.
  EXPECT_EQ(subgraph->tensors.size(), 5);
  const int input_idx = 0;
  EXPECT_EQ(subgraph->tensors[input_idx]->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[input_idx]->name, "input");
  EXPECT_EQ(subgraph->tensors[input_idx]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[input_idx]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[input_idx]->quantization->scale[0], 1.0);
  EXPECT_FLOAT_EQ(subgraph->tensors[input_idx]->quantization->zero_point[0],
                  -128);
  const int output_idx = 4;
  EXPECT_EQ(subgraph->tensors[output_idx]->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[output_idx]->name, "output");
  EXPECT_EQ(subgraph->tensors[output_idx]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[output_idx]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[output_idx]->quantization->scale[0], 1.0);
  EXPECT_FLOAT_EQ(subgraph->tensors[output_idx]->quantization->zero_point[0],
                  -128);
  const int split0_idx = 2;
  EXPECT_EQ(subgraph->tensors[split0_idx]->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[split0_idx]->name, "split;split:1");
  EXPECT_EQ(subgraph->tensors[split0_idx]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[split0_idx]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[split0_idx]->quantization->scale[0], 1.0);
  EXPECT_FLOAT_EQ(subgraph->tensors[split0_idx]->quantization->zero_point[0],
                  -128);
  const int split1_idx = 3;
  EXPECT_EQ(subgraph->tensors[split1_idx]->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[split1_idx]->name, "split;split:11");
  EXPECT_EQ(subgraph->tensors[split1_idx]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[split1_idx]->quantization->zero_point.size(), 1);
  EXPECT_FLOAT_EQ(subgraph->tensors[split1_idx]->quantization->scale[0], 1.0);
  EXPECT_FLOAT_EQ(subgraph->tensors[split1_idx]->quantization->zero_point[0],
                  -128);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_SPLIT);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

class QuantizeConvModel2Test : public QuantizeModelTest,
                               public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConvModel2Test() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_13(mht_13_v, 608, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeConvModel2Test");

    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kConvModelWith0Plus10Weights);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
    auto& subgraph = model_.subgraphs[0];
    auto* input = subgraph->tensors[subgraph->inputs[0]].get();
    auto* output = subgraph->tensors[subgraph->outputs[0]].get();
    input->quantization.reset(new QuantizationParametersT());
    output->quantization.reset(new QuantizationParametersT());
    input->quantization->min.push_back(0.0);
    output->quantization->min.push_back(0.0);
    input->quantization->max.push_back(6.0);
    output->quantization->max.push_back(6.0);
  }

  TensorType tensor_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeConvModel2TestInst, QuantizeConvModel2Test,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeConvModel2Test, VerifyConvQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);
  const auto& subgraph = model_.subgraphs[0];
  auto conv_op = subgraph->operators[0].get();
  const int input_tensor_idx = 0;
  const int weights_tensor_idx = 1;
  const int bias_tensor_index = 2;
  const int output_tensor_idx = 0;
  const auto bias_tensor =
      subgraph->tensors[conv_op->inputs[bias_tensor_index]].get();
  const auto input_tensor =
      subgraph->tensors[conv_op->inputs[input_tensor_idx]].get();
  const auto weights_tensor =
      subgraph->tensors[conv_op->inputs[weights_tensor_idx]].get();
  const auto output_tensor =
      subgraph->tensors[conv_op->outputs[output_tensor_idx]].get();

  EXPECT_EQ(bias_tensor->type, tensor_type_ == TensorType_INT8
                                   ? TensorType_INT32
                                   : TensorType_INT64);
  EXPECT_EQ(input_tensor->type, tensor_type_);
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);

  ASSERT_TRUE(weights_tensor->quantization);
  ASSERT_TRUE(bias_tensor->quantization);
  ASSERT_TRUE(weights_tensor->quantization);
  const std::vector<float>& bias_scales = bias_tensor->quantization->scale;
  const std::vector<float>& weights_scales =
      weights_tensor->quantization->scale;
  const std::vector<int64_t>& weights_zero_points =
      weights_tensor->quantization->zero_point;
  const int out_channel_size = weights_tensor->shape[0];
  ASSERT_EQ(bias_scales.size(), out_channel_size);
  ASSERT_EQ(weights_scales.size(), out_channel_size);
  ASSERT_EQ(weights_zero_points.size(), out_channel_size);
  ASSERT_EQ(input_tensor->quantization->scale.size(), 1);
  ASSERT_EQ(output_tensor->quantization->scale.size(), 1);

  const float eps = 1e-7;

  // Bias scale should be input * per_channel_weight_scale.
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_NEAR(bias_scales[i],
                input_tensor->quantization->scale[0] * weights_scales[i], eps);
  }

  const auto bias_buffer = model_.buffers[bias_tensor->buffer].get();
  auto control_size = tensor_type_ == TensorType_INT8
                          ? sizeof(int32_t) * bias_tensor->shape[0]
                          : sizeof(int64_t) * bias_tensor->shape[0];

  const auto float_op =
      readonly_model_->subgraphs()->Get(0)->operators()->Get(0);
  const auto original_bias_tensor =
      readonly_model_->subgraphs()->Get(0)->tensors()->Get(
          float_op->inputs()->Get(2));
  ASSERT_EQ(bias_buffer->data.size(), control_size);
  const auto original_bias_buffer =
      readonly_model_->buffers()->Get(original_bias_tensor->buffer());
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  if (tensor_type_ == TensorType_INT8) {
    int32_t* bias_values = reinterpret_cast<int32_t*>(bias_buffer->data.data());
    for (size_t i = 0; i < out_channel_size; i++) {
      auto dequantized_value = bias_values[i] * bias_scales[i];
      EXPECT_NEAR(dequantized_value, bias_float_buffer[i], bias_scales[i] / 2);
    }
  }

  const auto weights_buffer = model_.buffers[weights_tensor->buffer].get();
  const auto original_weights_tensor =
      readonly_model_->subgraphs()->Get(0)->tensors()->Get(
          float_op->inputs()->Get(1));
  const auto original_weights_buffer =
      readonly_model_->buffers()->Get(original_weights_tensor->buffer());
  const int8_t* weight_values =
      reinterpret_cast<int8_t*>(weights_buffer->data.data());
  const float* weights_float_buffer =
      reinterpret_cast<const float*>(original_weights_buffer->data()->data());
  ASSERT_EQ(sizeof(float) * weights_buffer->data.size(),
            original_weights_buffer->data()->size());
  int num_values_in_channel = weights_buffer->data.size() / out_channel_size;
  for (size_t channel_idx = 0; channel_idx < out_channel_size; channel_idx++) {
    for (size_t j = 0; j < num_values_in_channel; j++) {
      size_t element_idx = channel_idx * out_channel_size + j;
      auto scale = weights_scales[channel_idx];
      auto zero_point = weights_zero_points[channel_idx];
      auto dequantized_value = weight_values[element_idx] * scale;
      EXPECT_NEAR(dequantized_value, weights_float_buffer[element_idx],
                  scale / 2);
      EXPECT_EQ(zero_point, 0);
    }
  }

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_CONV_2D);
  EXPECT_EQ(model_.operator_codes[0]->version, 3);
}

TEST_P(QuantizeConvModel2Test, VerifyConvDisablePerChannelQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, /*disable_per_channel=*/true, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);
  const auto& subgraph = model_.subgraphs[0];
  auto conv_op = subgraph->operators[0].get();
  const int input_tensor_idx = 0;
  const int weights_tensor_idx = 1;
  const int bias_tensor_index = 2;
  const int output_tensor_idx = 0;
  const auto bias_tensor =
      subgraph->tensors[conv_op->inputs[bias_tensor_index]].get();
  const auto input_tensor =
      subgraph->tensors[conv_op->inputs[input_tensor_idx]].get();
  const auto weights_tensor =
      subgraph->tensors[conv_op->inputs[weights_tensor_idx]].get();
  const auto output_tensor =
      subgraph->tensors[conv_op->outputs[output_tensor_idx]].get();

  EXPECT_EQ(bias_tensor->type, tensor_type_ == TensorType_INT8
                                   ? TensorType_INT32
                                   : TensorType_INT64);
  EXPECT_EQ(input_tensor->type, tensor_type_);
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);

  ASSERT_TRUE(weights_tensor->quantization);
  ASSERT_TRUE(bias_tensor->quantization);
  ASSERT_TRUE(weights_tensor->quantization);
  const std::vector<float>& bias_scales = bias_tensor->quantization->scale;
  const std::vector<float>& weights_scales =
      weights_tensor->quantization->scale;
  const std::vector<int64_t>& weights_zero_points =
      weights_tensor->quantization->zero_point;

  const int out_channel_size = 1;
  ASSERT_EQ(bias_scales.size(), out_channel_size);
  ASSERT_EQ(weights_scales.size(), out_channel_size);
  ASSERT_EQ(weights_zero_points.size(), out_channel_size);
  ASSERT_EQ(input_tensor->quantization->scale.size(), 1);
  ASSERT_EQ(output_tensor->quantization->scale.size(), 1);

  const float eps = 1e-7;

  // Bias scale should be input * per_channel_weight_scale.
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_NEAR(bias_scales[i],
                input_tensor->quantization->scale[0] * weights_scales[i], eps);
  }

  const auto bias_buffer = model_.buffers[bias_tensor->buffer].get();
  auto control_size = tensor_type_ == TensorType_INT8
                          ? sizeof(int32_t) * bias_tensor->shape[0]
                          : sizeof(int64_t) * bias_tensor->shape[0];

  ASSERT_EQ(bias_buffer->data.size(), control_size);
  const auto float_op =
      readonly_model_->subgraphs()->Get(0)->operators()->Get(0);
  const auto original_bias_tensor =
      readonly_model_->subgraphs()->Get(0)->tensors()->Get(
          float_op->inputs()->Get(2));
  ASSERT_EQ(bias_buffer->data.size(), control_size);
  const auto original_bias_buffer =
      readonly_model_->buffers()->Get(original_bias_tensor->buffer());
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  if (tensor_type_ == TensorType_INT8) {
    int32_t* bias_values = reinterpret_cast<int32_t*>(bias_buffer->data.data());
    for (size_t i = 0; i < out_channel_size; i++) {
      auto dequantized_value = bias_values[i] * bias_scales[i];
      EXPECT_NEAR(dequantized_value, bias_float_buffer[i], bias_scales[i] / 2);
    }
  }

  const auto weights_buffer = model_.buffers[weights_tensor->buffer].get();
  const auto original_weights_tensor =
      readonly_model_->subgraphs()->Get(0)->tensors()->Get(
          float_op->inputs()->Get(1));
  const auto original_weights_buffer =
      readonly_model_->buffers()->Get(original_weights_tensor->buffer());
  const int8_t* weight_values =
      reinterpret_cast<int8_t*>(weights_buffer->data.data());
  const float* weights_float_buffer =
      reinterpret_cast<const float*>(original_weights_buffer->data()->data());
  ASSERT_EQ(sizeof(float) * weights_buffer->data.size(),
            original_weights_buffer->data()->size());
  int num_values_in_channel = weights_buffer->data.size() / out_channel_size;
  for (size_t channel_idx = 0; channel_idx < out_channel_size; channel_idx++) {
    for (size_t j = 0; j < num_values_in_channel; j++) {
      size_t element_idx = channel_idx * out_channel_size + j;
      auto scale = weights_scales[channel_idx];
      auto zero_point = weights_zero_points[channel_idx];
      auto dequantized_value = weight_values[element_idx] * scale;
      EXPECT_NEAR(dequantized_value, weights_float_buffer[element_idx],
                  scale / 2);
      EXPECT_EQ(zero_point, 0);
    }
  }

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_CONV_2D);
  EXPECT_EQ(model_.operator_codes[0]->version, 3);
}

class QuantizeSoftmaxTest : public QuantizeModelTest {
 protected:
  QuantizeSoftmaxTest() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_14(mht_14_v, 847, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeSoftmaxTest");

    input_model_ = ReadModel(internal::kSingleSoftmaxModelMinMinus5MaxPlus5);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeSoftmaxTest, VerifySoftmaxQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  // Model has a single softmax op.
  ASSERT_EQ(op->opcode_index, 0);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_SOFTMAX);

  ASSERT_EQ(op->inputs.size(), 1);
  ASSERT_EQ(op->outputs.size(), 1);
  auto float_graph = readonly_model_->subgraphs()->Get(0);

  // Verify input.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);

  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                    *input_quant_params);

  // Verify output.
  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);

  ASSERT_EQ(output_quant_params->scale.size(), 1);
  ASSERT_EQ(output_quant_params->zero_point.size(), 1);
  ASSERT_EQ(1.0f / 256.0f, output_quant_params->scale[0]);
  ASSERT_EQ(-128, output_quant_params->zero_point[0]);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_SOFTMAX);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

class QuantizeAvgPoolTest : public QuantizeModelTest {
 protected:
  QuantizeAvgPoolTest() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_15(mht_15_v, 912, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeAvgPoolTest");

    input_model_ = ReadModel(internal::kSingleAvgPoolModelMinMinus5MaxPlus5);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeAvgPoolTest, VerifyAvgPoolQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  // Model has a single AveragePool op.
  ASSERT_EQ(op->opcode_index, 0);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_AVERAGE_POOL_2D);

  ASSERT_EQ(op->inputs.size(), 1);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);

  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                    *input_quant_params);

  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);
  ASSERT_EQ(output_quant_params->scale.size(), 1);

  // Make sure the input min/maxes are propagated to outputs.
  EXPECT_EQ(input_quant_params->scale[0], output_quant_params->scale[0]);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_AVERAGE_POOL_2D);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

class QuantizeMultiInputAddWithReshapeTest : public QuantizeModelTest {
 protected:
  QuantizeMultiInputAddWithReshapeTest() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_16(mht_16_v, 974, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeMultiInputAddWithReshapeTest");

    input_model_ = ReadModel(internal::kMultiInputAddWithReshape);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeMultiInputAddWithReshapeTest, VerifyReshapeQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);

  ASSERT_EQ(kTfLiteOk, status);

  // Verify Reshape is quantized.
  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[1].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_RESHAPE);

  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  auto float_op = float_graph->operators()->Get(1);
  ASSERT_EQ(float_graph->tensors()->Get(float_op->inputs()->Get(0))->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(float_op->outputs()->Get(0))->type(),
            TensorType_FLOAT32);

  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);
  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                    *input_quant_params);

  auto float_output_quant_params =
      float_graph->tensors()->Get(float_op->outputs()->Get(0))->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);
  ASSERT_EQ(output_quant_params->scale.size(), 1);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_ADD);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[1].get()),
            BuiltinOperator_RESHAPE);
  EXPECT_EQ(model_.operator_codes[1]->version, 1);
}

TEST_F(QuantizeMultiInputAddWithReshapeTest, VerifyAddQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Verify ADD is quantized.
  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_ADD);

  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  auto float_op = float_graph->operators()->Get(0);
  const int float_input0_idx = float_op->inputs()->Get(0);
  const int float_input1_idx = float_op->inputs()->Get(1);
  const int float_output_idx = float_op->outputs()->Get(0);
  ASSERT_EQ(float_graph->tensors()->Get(float_input0_idx)->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(float_input1_idx)->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(float_output_idx)->type(),
            TensorType_FLOAT32);

  for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
    EXPECT_EQ(subgraph->tensors[op->inputs[input_idx]].get()->type,
              TensorType_INT8);
    auto float_input_quant_params =
        float_graph->tensors()
            ->Get(float_op->inputs()->Get(input_idx))
            ->quantization();
    auto input_quant_params =
        subgraph->tensors[op->inputs[input_idx]]->quantization.get();
    VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                      *input_quant_params);
  }

  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);
  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);
  ASSERT_EQ(output_quant_params->scale.size(), 1);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_ADD);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[1].get()),
            BuiltinOperator_RESHAPE);
  EXPECT_EQ(model_.operator_codes[1]->version, 1);
}

class QuantizeConstInputTest : public QuantizeModelTest,
                               public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeConstInputTest() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_17(mht_17_v, 1096, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeConstInputTest");

    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kConstInputAddModel);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  TensorType tensor_type_;
};
INSTANTIATE_TEST_SUITE_P(QuantizeConstInputTestInst, QuantizeConstInputTest,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeConstInputTest, VerifyConstOpInput) {
  auto status =
      QuantizeModelAllOperators(
          &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
          tensor_type_, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Verify ConstOp is quantized.
  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_ADD);

  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
    EXPECT_EQ(subgraph->tensors[op->inputs[input_idx]].get()->type,
              tensor_type_);
  }

  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, tensor_type_);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_ADD);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

class QuantizeArgMaxTest : public QuantizeModelTest {
 protected:
  QuantizeArgMaxTest() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_18(mht_18_v, 1149, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeArgMaxTest");

    input_model_ = ReadModel(internal::kModelWithArgMaxOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeArgMaxTest, VerifyArgMax) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_ARG_MAX);

  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  auto float_op = float_graph->operators()->Get(0);
  // Verify ArgMax input is quantized.
  ASSERT_EQ(float_graph->tensors()->Get(float_op->inputs()->Get(0))->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);

  // Verify ArgMax input axis should still be the same type.
  ASSERT_EQ(float_graph->tensors()->Get(float_op->inputs()->Get(1))->type(),
            subgraph->tensors[op->inputs[1]].get()->type);

  // The output of ArgMax should still be the same type.
  ASSERT_EQ(float_graph->tensors()->Get(float_op->outputs()->Get(0))->type(),
            subgraph->tensors[op->outputs[0]].get()->type);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_ARG_MAX);
  EXPECT_EQ(model_.operator_codes[0]->version, 2);
}

class QuantizeLSTMTest : public QuantizeModelTest {
 protected:
  QuantizeLSTMTest() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_19(mht_19_v, 1197, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeLSTMTest");

    input_model_ = ReadModel(internal::kLstmCalibrated);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeLSTMTest, VerifyLSTM) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32, true,
      TensorType_INT8, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Read expected model.
  auto expected_fb_model = ReadModel(internal::kLstmQuantized);
  auto expected_read_only_model = expected_fb_model->GetModel();
  ModelT expected_model;
  expected_read_only_model->UnPackTo(&expected_model);

  ExpectSameModels(model_, expected_model);
}

class QuantizeLSTM2Test : public QuantizeModelTest {
 protected:
  QuantizeLSTM2Test() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_20(mht_20_v, 1225, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeLSTM2Test");

    input_model_ = ReadModel(internal::kLstmCalibrated2);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeLSTM2Test, VerifyLSTM) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Read expected model.
  auto expected_fb_model = ReadModel(internal::kLstmQuantized2);
  auto expected_read_only_model = expected_fb_model->GetModel();
  ModelT expected_model;
  expected_read_only_model->UnPackTo(&expected_model);

  ExpectSameModels(model_, expected_model);
}

class QuantizeUnidirectionalSequenceLSTMTest : public QuantizeModelTest {
 protected:
  QuantizeUnidirectionalSequenceLSTMTest() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_21(mht_21_v, 1253, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeUnidirectionalSequenceLSTMTest");

    input_model_ = ReadModel(internal::kUnidirectionalSequenceLstmCalibrated);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeUnidirectionalSequenceLSTMTest,
       VerifyUnidirectionalSequenceLSTM) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_FLOAT32, TensorType_FLOAT32,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Read expected model.
  auto expected_fb_model =
      ReadModel(internal::kUnidirectionalSequenceLstmQuantized);
  auto expected_read_only_model = expected_fb_model->GetModel();
  ModelT expected_model;
  expected_read_only_model->UnPackTo(&expected_model);

  ExpectSameModels(model_, expected_model);
}

class QuantizeSVDFTest : public QuantizeModelTest {
 protected:
  QuantizeSVDFTest() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_22(mht_22_v, 1283, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeSVDFTest");

    input_model_ = ReadModel(internal::kSvdfCalibrated);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeSVDFTest, VerifySVDF) {
  // Quantize model.
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  // Read expected model.
  auto expected_fb_model = ReadModel(internal::kSvdfQuantized);
  auto expected_read_only_model = expected_fb_model->GetModel();
  ModelT expected_model;
  expected_read_only_model->UnPackTo(&expected_model);

  ExpectSameModels(model_, expected_model);
}

class QuantizeFCTest : public QuantizeModelTest {
 protected:
  QuantizeFCTest() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_23(mht_23_v, 1311, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeFCTest");

    input_model_ = ReadModel(internal::kModelWithFCOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeFCTest, VerifyFC) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, TensorType_INT8, TensorType_INT8,
      /*allow_float=*/false, TensorType_INT8, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);

  const auto& subgraph = model_.subgraphs[0];
  auto op = subgraph->operators[0].get();
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_FULLY_CONNECTED);

  ASSERT_EQ(op->inputs.size(), 3);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model_->subgraphs()->Get(0);
  // Verify FC input and weight is quantized.
  auto float_op = float_graph->operators()->Get(0);
  ASSERT_EQ(float_graph->tensors()->Get(float_op->inputs()->Get(0))->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  ASSERT_EQ(float_graph->tensors()->Get(float_op->inputs()->Get(1))->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->inputs[1]].get()->type, TensorType_INT8);

  // Verify FC bias should be int32 quantized.
  ASSERT_EQ(float_graph->tensors()->Get(float_op->inputs()->Get(2))->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->inputs[2]].get()->type, TensorType_INT32);

  // The output of FC should be quantized.
  ASSERT_EQ(float_graph->tensors()->Get(float_op->outputs()->Get(0))->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 2);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[0].get()),
            BuiltinOperator_FULLY_CONNECTED);
  EXPECT_EQ(model_.operator_codes[0]->version, 4);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[1].get()),
            BuiltinOperator_RESHAPE);
  EXPECT_EQ(model_.operator_codes[1]->version, 1);
}

class QuantizeCustomOpTest
    : public QuantizeModelTest,
      public ::testing::WithParamInterface<tflite::TensorType> {
 protected:
  QuantizeCustomOpTest() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_24(mht_24_v, 1369, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeCustomOpTest");

    input_model_ = ReadModel(internal::kModelMixed);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_P(QuantizeCustomOpTest, VerifyMixedQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, GetParam(), GetParam(),
      /*allow_float=*/true, GetParam(), &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);
  const auto& subgraph = model_.subgraphs[0];
  auto float_graph = readonly_model_->subgraphs()->Get(0);
  // The original model reshape->custom->custom->squeeze.
  ASSERT_EQ(float_graph->operators()->size(), 4);
  // The resulting model should be:
  // reshape->dequantize->custom->custom->quantize->squeeze.
  ASSERT_EQ(subgraph->operators.size(), 6);
  const std::vector<BuiltinOperator> op_codes = {
      BuiltinOperator_RESHAPE,  BuiltinOperator_DEQUANTIZE,
      BuiltinOperator_CUSTOM,   BuiltinOperator_CUSTOM,
      BuiltinOperator_QUANTIZE, BuiltinOperator_SQUEEZE};
  const std::vector<TensorType> op_input_types = {
      GetParam(),         GetParam(),         TensorType_FLOAT32,
      TensorType_FLOAT32, TensorType_FLOAT32, GetParam()};
  for (int i = 0; i < subgraph->operators.size(); ++i) {
    OperatorT* op = subgraph->operators[i].get();
    ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
              op_codes[i]);
    ASSERT_EQ(subgraph->tensors[op->inputs[0]]->type, op_input_types[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(QuantizeCustomOpTest, QuantizeCustomOpTest,
                         ::testing::Values(TensorType_INT8));

class QuantizePackTest : public QuantizeModelTest {
 protected:
  QuantizePackTest() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_25(mht_25_v, 1411, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizePackTest");

    input_model_ = ReadModel(internal::kModelPack);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizePackTest, VerifyPack) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);

  ASSERT_EQ(kTfLiteOk, status);

  const auto subgraph = model_.subgraphs[0].get();

  // The model should only have 3 inputs and 1 output.
  EXPECT_EQ(subgraph->inputs.size(), 3);
  EXPECT_EQ(subgraph->outputs.size(), 1);

  const auto& op1 = subgraph->operators[1].get();
  const auto& op2 = subgraph->operators[2].get();
  const auto& op3 = subgraph->operators[3].get();
  const auto& op4 = subgraph->operators[4].get();

  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op1->opcode_index].get()),
            BuiltinOperator_QUANTIZE);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op2->opcode_index].get()),
            BuiltinOperator_QUANTIZE);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op3->opcode_index].get()),
            BuiltinOperator_PACK);
  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op4->opcode_index].get()),
            BuiltinOperator_DEQUANTIZE);

  const auto& pack_input0 = subgraph->tensors[op3->inputs[0]].get();
  const auto& pack_input1 = subgraph->tensors[op3->inputs[1]].get();
  const auto& pack_input2 = subgraph->tensors[op3->inputs[2]].get();

  const auto& pack_output = subgraph->tensors[op3->outputs[0]].get();

  // Check quantization parameters for input and output.
  EXPECT_FLOAT_EQ(pack_input0->quantization->scale[0],
                  pack_input1->quantization->scale[0]);
  EXPECT_FLOAT_EQ(pack_input1->quantization->scale[0],
                  pack_input2->quantization->scale[0]);
  EXPECT_FLOAT_EQ(pack_input0->quantization->zero_point[0],
                  pack_input1->quantization->zero_point[0]);
  EXPECT_FLOAT_EQ(pack_input1->quantization->zero_point[0],
                  pack_input2->quantization->zero_point[0]);

  EXPECT_FLOAT_EQ(pack_input1->quantization->scale[0],
                  pack_output->quantization->scale[0]);
  EXPECT_FLOAT_EQ(pack_input1->quantization->zero_point[0],
                  pack_output->quantization->zero_point[0]);

  // Check type of input and output.
  EXPECT_EQ(pack_output->type, TensorType_INT8);
  EXPECT_EQ(pack_input0->type, TensorType_INT8);
  EXPECT_EQ(pack_input1->type, TensorType_INT8);
  EXPECT_EQ(pack_input2->type, TensorType_INT8);
}

class QuantizeMinimumMaximumTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<const char*> {
 protected:
  QuantizeMinimumMaximumTest() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_26(mht_26_v, 1478, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeMinimumMaximumTest");

    input_model_ = ReadModel(GetParam());
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_P(QuantizeMinimumMaximumTest, VerifyMinimumMaximum) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);
  ASSERT_EQ(kTfLiteOk, status);
  const auto& subgraph = model_.subgraphs[0];
  // Check that the first op is Quantize and the last is Dequant.
  const auto& quant_op = subgraph->operators[0];
  const auto& dequant_op = subgraph->operators[subgraph->operators.size() - 1];
  const int32_t quant_idx = quant_op->opcode_index;
  const int32_t dequant_idx = dequant_op->opcode_index;
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[quant_idx].get()),
            BuiltinOperator_QUANTIZE);
  EXPECT_EQ(GetBuiltinCode(model_.operator_codes[dequant_idx].get()),
            BuiltinOperator_DEQUANTIZE);

  const auto& op = subgraph->operators[1].get();

  // Check that we have MINIMUM or MAXIMUM operator.
  auto op_builtin_code =
      GetBuiltinCode(model_.operator_codes[op->opcode_index].get());
  ASSERT_TRUE(op_builtin_code == tflite::BuiltinOperator_MINIMUM ||
              op_builtin_code == tflite::BuiltinOperator_MAXIMUM);

  // Check that we have two inputs and one output.
  ASSERT_EQ(op->inputs.size(), 2);
  ASSERT_EQ(op->outputs.size(), 1);

  // Check that all is quantized.
  auto output = subgraph->tensors[op->outputs[0]].get();
  auto input1 = subgraph->tensors[op->inputs[0]].get();
  auto input2 = subgraph->tensors[op->inputs[1]].get();

  EXPECT_EQ(output->type, TensorType_INT8);
  EXPECT_EQ(input1->type, TensorType_INT8);
  EXPECT_EQ(input2->type, TensorType_INT8);

  // Check if the quantization params of the minimum/maximum inputs match
  // after requantization
  EXPECT_EQ(input1->quantization->scale, input2->quantization->scale);
  EXPECT_EQ(input1->quantization->zero_point, input2->quantization->zero_point);

  // Check the input quantization params match the output ones.
  EXPECT_EQ(output->quantization->scale, input1->quantization->scale);
  EXPECT_EQ(output->quantization->zero_point, input1->quantization->zero_point);
  EXPECT_EQ(output->quantization->scale, input2->quantization->scale);
  EXPECT_EQ(output->quantization->zero_point, input2->quantization->zero_point);
}

INSTANTIATE_TEST_SUITE_P(MinimumMaximumTestInst, QuantizeMinimumMaximumTest,
                         testing::ValuesIn({internal::kModelWithMinimumOp,
                                            internal::kModelWithMaximumOp}));

class QuantizeUnpackTest : public QuantizeModelTest {
 protected:
  QuantizeUnpackTest() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_27(mht_27_v, 1541, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeUnpackTest");

    input_model_ = ReadModel(internal::kModelWithUnpack);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeUnpackTest, VerifyUnpack) {
  auto status = QuantizeModel(&builder_, &model_, &error_reporter_);

  ASSERT_EQ(kTfLiteOk, status);

  const auto subgraph = model_.subgraphs[0].get();
  auto op = subgraph->operators[1].get();

  auto float_graph = readonly_model_->subgraphs()->Get(0);

  ASSERT_EQ(GetBuiltinCode(model_.operator_codes[op->opcode_index].get()),
            BuiltinOperator_UNPACK);

  // Get unpack input and output tensors
  auto unpack_input = subgraph->tensors[op->inputs[0]].get();
  auto unpack_output_0 = subgraph->tensors[op->outputs[0]].get();
  auto unpack_output_1 = subgraph->tensors[op->outputs[1]].get();

  // Verify Unpack input is quantized.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  EXPECT_EQ(unpack_input->type, TensorType_INT8);

  // The model should only have one input and 2 outputs.
  EXPECT_EQ(subgraph->inputs.size(), 1);
  EXPECT_EQ(subgraph->outputs.size(), 2);

  // Ensure quantization parameters before and after unpack
  // are preserved after quantization for all outputs of
  // unpack.
  EXPECT_FLOAT_EQ(unpack_input->quantization->scale[0],
                  unpack_output_0->quantization->scale[0]);
  EXPECT_FLOAT_EQ(unpack_input->quantization->scale[0],
                  unpack_output_1->quantization->scale[0]);
  EXPECT_FLOAT_EQ(unpack_input->quantization->zero_point[0],
                  unpack_output_0->quantization->zero_point[0]);
  EXPECT_FLOAT_EQ(unpack_input->quantization->zero_point[0],
                  unpack_output_1->quantization->zero_point[0]);
}

class QuantizeBroadcastToModelTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeBroadcastToModelTest() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_28(mht_28_v, 1595, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeBroadcastToModelTest");

    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kModelWithBroadcastToOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
  TensorType tensor_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeBroadcastToModelTestInst,
                         QuantizeBroadcastToModelTest,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeBroadcastToModelTest, VerifyBroadcastToQuantization) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be a single broadcast_to op.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 1);
  EXPECT_EQ(subgraph->operators.size(), 1);
  const auto& broadcast_to = subgraph->operators[0];
  EXPECT_EQ(model_.operator_codes[broadcast_to->opcode_index]->builtin_code,
            BuiltinOperator_BROADCAST_TO);

  // There should be 3 tensors: input, output, and BroadcastTo/shape.
  EXPECT_EQ(subgraph->tensors.size(), 3);

  // Input Tensor
  EXPECT_EQ(subgraph->tensors[0]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[0]->name, "input_1");
  EXPECT_EQ(subgraph->tensors[0]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[0]->quantization->zero_point.size(), 1);

  // Output Tensor. The name given in the generated
  // .bin test file is 'Identity' and should be preserved
  EXPECT_EQ(subgraph->tensors[2]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[2]->name, "Identity");
  EXPECT_EQ(subgraph->tensors[2]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[2]->quantization->zero_point.size(), 1);

  // The BroadCastTo shape is of type INT32 and should not be quantized
  EXPECT_EQ(subgraph->tensors[1]->type, TensorType_INT32);
  EXPECT_EQ(subgraph->tensors[1]->name,
            "model/tf.broadcast_to/BroadcastTo/shape");
  EXPECT_EQ(subgraph->tensors[1]->quantization->scale.size(), 0);
  EXPECT_EQ(subgraph->tensors[1]->quantization->zero_point.size(), 0);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(model_.operator_codes[0]->builtin_code,
            BuiltinOperator_BROADCAST_TO);
  EXPECT_EQ(model_.operator_codes[0]->version, 3);
}

class QuantizeGatherNDModelTest
    : public QuantizeModelTest,
      public testing::WithParamInterface<TensorType> {
 protected:
  QuantizeGatherNDModelTest() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_29(mht_29_v, 1664, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeGatherNDModelTest");

    tensor_type_ = GetParam();
    input_model_ = ReadModel(internal::kModelWithGatherNDOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }

  TensorType tensor_type_;
};

INSTANTIATE_TEST_SUITE_P(QuantizeGatherNDModelTestInst,
                         QuantizeGatherNDModelTest,
                         testing::ValuesIn({TensorType_INT8}));

TEST_P(QuantizeGatherNDModelTest, QuantizeGatherND) {
  auto status = QuantizeModelAllOperators(
      &builder_, &model_, tensor_type_, tensor_type_, /*allow_float=*/false,
      tensor_type_, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be a single gather_nd op.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 1);
  EXPECT_EQ(subgraph->operators.size(), 1);
  const auto& gather_nd = subgraph->operators[0];
  EXPECT_EQ(model_.operator_codes[gather_nd->opcode_index]->builtin_code,
            BuiltinOperator_GATHER_ND);

  // There should be 3 tensors: input, output, and indices.
  EXPECT_EQ(subgraph->tensors.size(), 3);

  // Input Tensor
  EXPECT_EQ(subgraph->tensors[0]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[0]->name, "input");
  EXPECT_EQ(subgraph->tensors[0]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[0]->quantization->zero_point.size(), 1);

  // Output Tensor
  EXPECT_EQ(subgraph->tensors[2]->type, tensor_type_);
  EXPECT_EQ(subgraph->tensors[2]->name, "output");
  EXPECT_EQ(subgraph->tensors[2]->quantization->scale.size(), 1);
  EXPECT_EQ(subgraph->tensors[2]->quantization->zero_point.size(), 1);

  // The gather indices are of type INT32 and should not be quantized
  EXPECT_EQ(subgraph->tensors[1]->type, TensorType_INT32);
  EXPECT_EQ(subgraph->tensors[1]->name, "indices");
  EXPECT_EQ(subgraph->tensors[1]->quantization->scale.size(), 0);
  EXPECT_EQ(subgraph->tensors[1]->quantization->zero_point.size(), 0);

  // Check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(model_.operator_codes[0]->builtin_code, BuiltinOperator_GATHER_ND);
  EXPECT_EQ(model_.operator_codes[0]->version, 1);
}

class QuantizeWhereModelTest : public QuantizeModelTest {
 protected:
  QuantizeWhereModelTest() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_30(mht_30_v, 1729, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "QuantizeWhereModelTest");

    input_model_ = ReadModel(internal::kModelWithWhereOp);
    readonly_model_ = input_model_->GetModel();
    readonly_model_->UnPackTo(&model_);
  }
};

TEST_F(QuantizeWhereModelTest, QuantizeWhere) {
  // Where operator takes a BOOL tensor as input
  // and outputs INT64 indices, both of which
  // should not be quantized
  auto status = QuantizeModel(&builder_, &model_, TensorType_BOOL,
                              TensorType_INT64, &error_reporter_);
  EXPECT_EQ(status, kTfLiteOk);

  // There is only one subgraph.
  const int32_t subgraph_idx = 0;
  const auto& subgraph = model_.subgraphs[subgraph_idx];
  const auto& readonly_subgraph =
      readonly_model_->subgraphs()->Get(subgraph_idx);

  // There should be a single where op.
  EXPECT_EQ(readonly_subgraph->operators()->size(), 1);
  EXPECT_EQ(subgraph->operators.size(), 1);
  const auto& where = subgraph->operators[0];
  EXPECT_EQ(model_.operator_codes[where->opcode_index]->builtin_code,
            BuiltinOperator_WHERE);

  // There should be 2 tensors: input and output.
  EXPECT_EQ(subgraph->tensors.size(), 2);

  // Testing input tensor type and ensuring it
  // was not quantized
  EXPECT_EQ(subgraph->tensors[0]->type, TensorType_BOOL);
  EXPECT_EQ(subgraph->tensors[0]->name, "input");
  EXPECT_EQ(subgraph->tensors[0]->quantization->scale.size(), 0);
  EXPECT_EQ(subgraph->tensors[0]->quantization->zero_point.size(), 0);

  // Testing output (indices) tensor type and ensuring it
  // was not quantized
  EXPECT_EQ(subgraph->tensors[1]->type, TensorType_INT64);
  EXPECT_EQ(subgraph->tensors[1]->name, "indices");
  EXPECT_EQ(subgraph->tensors[1]->quantization->scale.size(), 0);
  EXPECT_EQ(subgraph->tensors[1]->quantization->zero_point.size(), 0);

  // check op and versioning.
  EXPECT_EQ(model_.operator_codes.size(), 1);
  EXPECT_EQ(model_.operator_codes[0]->builtin_code, BuiltinOperator_WHERE);
  EXPECT_EQ(model_.operator_codes[0]->version, 1);
}

}  // namespace
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSlitePSquantize_model_testDTcc mht_31(mht_31_v, 1787, "", "./tensorflow/compiler/mlir/lite/quantization/lite/quantize_model_test.cc", "main");

  tensorflow::string model_file;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("test_model_file", &model_file,
                       "Path to test tflite model file."),
  };

  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cerr << "Required test_model_file\n";
    std::abort();
  }
  g_test_model_dir =
      new tensorflow::string(tensorflow::io::Dirname(model_file));
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
