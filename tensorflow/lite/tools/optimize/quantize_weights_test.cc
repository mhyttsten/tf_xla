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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc() {
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
#include "tensorflow/lite/tools/optimize/quantize_weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>

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

namespace {
tensorflow::string* g_test_model_dir = nullptr;
}  // namespace

namespace tflite {
namespace optimize {
namespace {

std::unique_ptr<FlatBufferModel> ReadTestModel() {
  auto model_path = tensorflow::io::JoinPath(
      *g_test_model_dir, internal::kConvModelWith0Plus10Weights);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModel> ReadSharedWeightsTestModel() {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir,
                                             internal::kModelWithSharedWeights);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModel> ReadGatherTestModel() {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir,
                                             internal::kQuantizedWithGather);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModel> ReadCustomOpTestModel() {
  auto model_path =
      tensorflow::io::JoinPath(*g_test_model_dir, internal::kModelWithCustomOp);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

template <typename T>
std::vector<T> GetAsVector(const flatbuffers::Vector<T>* vec) {
  return std::vector<T>(vec->begin(), vec->end());
}

class QuantizeWeightsTest : public testing::Test {
 protected:
  QuantizeWeightsTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc mht_0(mht_0_v, 241, "", "./tensorflow/lite/tools/optimize/quantize_weights_test.cc", "QuantizeWeightsTest");
}

  void LoadBasicModel() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc mht_1(mht_1_v, 246, "", "./tensorflow/lite/tools/optimize/quantize_weights_test.cc", "LoadBasicModel");

    input_model_ = ReadTestModel();
    model_ = input_model_->GetModel();
  }

  void LoadSharedWeightsModel() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc mht_2(mht_2_v, 254, "", "./tensorflow/lite/tools/optimize/quantize_weights_test.cc", "LoadSharedWeightsModel");

    input_model_ = ReadSharedWeightsTestModel();
    model_ = input_model_->GetModel();
  }

  void LoadGatherTestModel() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc mht_3(mht_3_v, 262, "", "./tensorflow/lite/tools/optimize/quantize_weights_test.cc", "LoadGatherTestModel");

    input_model_ = ReadGatherTestModel();
    model_ = input_model_->GetModel();
  }

  void LoadCustomOpTestModel() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc mht_4(mht_4_v, 270, "", "./tensorflow/lite/tools/optimize/quantize_weights_test.cc", "LoadCustomOpTestModel");

    input_model_ = ReadCustomOpTestModel();
    model_ = input_model_->GetModel();
  }

  std::unique_ptr<FlatBufferModel> input_model_;
  const Model* model_;

  bool IsModelInputOrOutput(const Model* model, uint32_t tensor_idx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc mht_5(mht_5_v, 281, "", "./tensorflow/lite/tools/optimize/quantize_weights_test.cc", "IsModelInputOrOutput");

    for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
         ++subgraph_idx) {
      const auto subgraph = model->subgraphs()->Get(subgraph_idx);
      for (size_t i = 0; i < subgraph->inputs()->size(); ++i) {
        if (subgraph->inputs()->Get(i) == tensor_idx) {
          return true;
        }
      }
      for (size_t i = 0; i < subgraph->outputs()->size(); ++i) {
        if (subgraph->outputs()->Get(i) == tensor_idx) {
          return true;
        }
      }
    }
    return false;
  }

  // Returns the producer op code of the specified tensor_idx.
  bool GetProducerOpCode(const Model* model, uint32_t subgraph_idx,
                         uint32_t tensor_idx,
                         tflite::BuiltinOperator* op_code) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc mht_6(mht_6_v, 305, "", "./tensorflow/lite/tools/optimize/quantize_weights_test.cc", "GetProducerOpCode");

    const auto subgraph = model->subgraphs()->Get(subgraph_idx);
    for (size_t op_idx = 0; op_idx < subgraph->operators()->size(); ++op_idx) {
      const auto op = subgraph->operators()->Get(op_idx);
      for (size_t i = 0; i < op->outputs()->size(); ++i) {
        if (op->outputs()->Get(i) == tensor_idx) {
          const uint32_t op_code_idx = op->opcode_index();
          *op_code = GetBuiltinCode(model->operator_codes()->Get(op_code_idx));
          return true;
        }
      }
    }
    return false;
  }
};

TEST_F(QuantizeWeightsTest, QuantizationSucceeds) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status =
      QuantizeWeights(&builder, model_, 0, QuantizerType::OLD_QUANTIZER);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);
}

TEST_F(QuantizeWeightsTest, WeightsMinNumElements) {
  LoadBasicModel();
  // Make weights_min_size sufficiently large such that no quantization should
  // happen, i.e. the original model is the same size as the old one.
  flatbuffers::FlatBufferBuilder builder;
  const uint64_t kWeightsMinNumElements = 1000000;
  EXPECT_EQ(QuantizeWeights(&builder, model_, kWeightsMinNumElements,
                            QuantizerType::OLD_QUANTIZER),
            kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       subgraph_idx++) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size());
    for (size_t i = 0; i < quantized_graph->tensors()->size(); i++) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      const auto float_tensor = float_graph->tensors()->Get(i);
      // Everything should remain equal between the two graphs.
      EXPECT_EQ(quant_tensor->buffer(), float_tensor->buffer());
      EXPECT_EQ(quant_tensor->is_variable(), float_tensor->is_variable());
      EXPECT_EQ(GetAsVector(quant_tensor->shape()),
                GetAsVector(float_tensor->shape()));
      EXPECT_EQ(quant_tensor->name()->str(), float_tensor->name()->str());
      EXPECT_EQ(quant_tensor->type(), float_tensor->type());
    }
  }
}

TEST_F(QuantizeWeightsTest, HybridConv) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status =
      QuantizeWeights(&builder, model_, 0, QuantizerType::OLD_QUANTIZER);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  // Nothing should change.
  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       subgraph_idx++) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size());
    // Make sure the graph only has one Conv operation.
    ASSERT_EQ(quantized_graph->operators()->size(), 1);
    const auto op = quantized_graph->operators()->Get(0);
    const uint32_t op_code_idx = op->opcode_index();
    ASSERT_EQ(GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)),
              BuiltinOperator_CONV_2D);
    for (size_t i = 0; i < quantized_graph->tensors()->size(); i++) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      const auto float_tensor = float_graph->tensors()->Get(i);
      EXPECT_EQ(quant_tensor->buffer(), float_tensor->buffer());
      EXPECT_EQ(quant_tensor->is_variable(), float_tensor->is_variable());
      EXPECT_EQ(GetAsVector(quant_tensor->shape()),
                GetAsVector(float_tensor->shape()));
      EXPECT_EQ(quant_tensor->name()->str(), float_tensor->name()->str());
      // If the tensor is a weight, it should have type INT8, otherwise it
      // should stay with type FLOAT32.
      // If the tensor is a bias, it should have type FLOAT32.
      if (quant_tensor->name()->str() == "conv_bias") {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (IsModelInputOrOutput(output_model, i)) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->buffer() != 0) {
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8)
            << quant_tensor->name()->str();
        auto shape = GetAsVector(quant_tensor->shape());
        if (kUseUpdatedHybridSchemeDefault) {
          EXPECT_EQ(quant_tensor->quantization()->scale()->size(), shape[0]);
        } else {
          EXPECT_EQ(quant_tensor->quantization()->scale()->size(), 1);
        }
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, DequantizeConv) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status = internal::QuantizeWeights(&builder, model_, 0,
                                          /*use_hybrid_evaluation=*/false,
                                          QuantizerType::OLD_QUANTIZER);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    // The output graph should have an extra tensor from the added dequantize
    // op.
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size() + 1);
    // Check that a dequantize op exists.
    int32_t dequant_input_idx = -1;
    int32_t dequant_output_idx = -1;
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      if (GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)) ==
          BuiltinOperator_DEQUANTIZE) {
        dequant_input_idx = op->inputs()->Get(0);
        dequant_output_idx = op->outputs()->Get(0);
      }
    }
    ASSERT_GT(dequant_input_idx, -1);
    ASSERT_GT(dequant_output_idx, -1);
    for (size_t i = 0; i < quantized_graph->tensors()->size(); ++i) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      // If the tensor is a weight, it should have type INT8.
      // If the tensor is a bias, it should have type FLOAT32.
      // If the tensor is an input or output it should have type FLOAT32.
      // The input to dequantize should be INT8, and all other tensors should be
      // FLOAT32.
      if (i == dequant_input_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8);
      } else if (i == dequant_output_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (IsModelInputOrOutput(output_model, i)) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->name()->str() == "conv_bias") {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->buffer() != 0) {
        // If it's a non-bias constant tensor, it must be the weight.
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8);
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, DequantizeConvFloat16) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status = tflite::optimize::QuantizeWeights(
      &builder, model_, BufferType::QUANTIZED_FLOAT16,
      kUseUpdatedHybridSchemeDefault, QuantizerType::OLD_QUANTIZER);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    // The output graph should have two extra tensors from the added dequantize
    // op.
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size() + 2);
    // Check that a dequantize op exists.
    int32_t dequant_input_idx = -1;
    int32_t dequant_output_idx = -1;
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      if (GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)) ==
          BuiltinOperator_DEQUANTIZE) {
        dequant_input_idx = op->inputs()->Get(0);
        dequant_output_idx = op->outputs()->Get(0);
      }
    }
    ASSERT_GT(dequant_input_idx, -1);
    ASSERT_GT(dequant_output_idx, -1);
    for (size_t i = 0; i < quantized_graph->tensors()->size(); ++i) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      // If the tensor is a weight, it should have type FLOAT16.
      // If the tensor is a bias, it should have type FLOAT16.
      // If the tensor is an input or output it should have type FLOAT32.
      // The input to dequantize should be FLOAT16, and all other tensors should
      // be FLOAT32.
      if (i == dequant_input_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT16);
      } else if (i == dequant_output_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (IsModelInputOrOutput(output_model, i)) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->name()->str() == "conv_bias") {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT16);
      } else if (quant_tensor->buffer() != 0) {
        // If it's a non-bias constant tensor, it must be the weight.
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT16);
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, SharedWeights_Hybrid) {
  LoadSharedWeightsModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status =
      QuantizeWeights(&builder, model_, 0, QuantizerType::OLD_QUANTIZER);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  uint32_t num_conv_ops = 0;
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      const auto op_code =
          GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
      if (op_code == BuiltinOperator_CONV_2D) {
        num_conv_ops++;
        // Ensure that each convolution's weights tensor is now INT8.
        const auto weights_tensor =
            quantized_graph->tensors()->Get(op->inputs()->Get(1));
        EXPECT_EQ(weights_tensor->type(), TensorType_INT8);
      }
    }
  }
  // Ensure that there were exactly two convolutions in the model.
  EXPECT_EQ(num_conv_ops, 2);
}

TEST_F(QuantizeWeightsTest, SharedWeights_Dequantize) {
  LoadSharedWeightsModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status = internal::QuantizeWeights(&builder, model_, 0,
                                          /*use_hybrid_evaluation*/ false,
                                          QuantizerType::OLD_QUANTIZER);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  uint32_t num_conv_ops = 0;
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      const auto op_code =
          GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
      if (op_code == BuiltinOperator_CONV_2D) {
        num_conv_ops++;
        // Ensure that each convolution's weights tensor is still FLOAT
        // (the output of the dequantize).
        uint32_t weights_tensor_index = op->inputs()->Get(1);
        const auto weights_tensor =
            quantized_graph->tensors()->Get(weights_tensor_index);
        EXPECT_EQ(weights_tensor->type(), TensorType_FLOAT32);

        // Check that it comes from a dequantize operation.
        BuiltinOperator producer_op_code;
        ASSERT_TRUE(GetProducerOpCode(output_model, subgraph_idx,
                                      weights_tensor_index, &producer_op_code));
        EXPECT_EQ(producer_op_code, BuiltinOperator_DEQUANTIZE);
      }
    }
  }
  // Ensure that there were exactly two convolutions in the model.
  EXPECT_EQ(num_conv_ops, 2);
}

TEST_F(QuantizeWeightsTest, VerifyGatherQuantization) {
  LoadGatherTestModel();
  flatbuffers::FlatBufferBuilder builder;
  auto status =
      QuantizeWeights(&builder, model_, 0, QuantizerType::OLD_QUANTIZER);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      const auto op_code =
          GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
      if (op_code == BuiltinOperator_GATHER) {
        uint32_t input_tensor_index = op->inputs()->Get(0);
        const auto weights_tensor =
            quantized_graph->tensors()->Get(input_tensor_index);
        EXPECT_EQ(weights_tensor->type(), TensorType_INT8);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, VerifyCustomOpQuantizationDequantize) {
  LoadCustomOpTestModel();

  // The custom op is not hybrid, and the second input is a constant that can
  // be quantized.
  CustomOpMap custom_op_map;
  custom_op_map["CustomTestOp"] = {
      .quantizable_input_indices = {1},
      .is_hybrid = false,
  };

  flatbuffers::FlatBufferBuilder builder;
  auto status = QuantizeWeights(&builder, model_, 0, custom_op_map,
                                QuantizerType::OLD_QUANTIZER);
  ASSERT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  const auto quantized_graph = output_model->subgraphs()->Get(0);
  // A dequantize op should be added.
  ASSERT_EQ(quantized_graph->operators()->size(),
            model_->subgraphs()->Get(0)->operators()->size() + 1);
  int num_custom_ops_found = 0;
  for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
    const auto op = quantized_graph->operators()->Get(i);
    const uint32_t op_code_idx = op->opcode_index();
    const auto op_code =
        GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
    if (op_code == BuiltinOperator_CUSTOM) {
      uint32_t weights_tensor_index = op->inputs()->Get(1);
      const auto weights_tensor =
          quantized_graph->tensors()->Get(weights_tensor_index);
      EXPECT_EQ(weights_tensor->type(), TensorType_FLOAT32);

      // Check that it comes from a dequantize operation.
      BuiltinOperator producer_op_code;
      ASSERT_TRUE(GetProducerOpCode(output_model, 0, weights_tensor_index,
                                    &producer_op_code));
      EXPECT_EQ(producer_op_code, BuiltinOperator_DEQUANTIZE);
      num_custom_ops_found++;
    }
  }
  EXPECT_EQ(num_custom_ops_found, 1);
}

TEST_F(QuantizeWeightsTest, VerifyCustomOpQuantizationHybrid) {
  LoadCustomOpTestModel();

  // The custom op is hybrid, and the second input is a constant that can
  // be quantized.
  CustomOpMap custom_op_map;
  custom_op_map["CustomTestOp"] = {
      .quantizable_input_indices = {1},
      .is_hybrid = true,
  };

  flatbuffers::FlatBufferBuilder builder;
  auto status = QuantizeWeights(&builder, model_, 0, custom_op_map,
                                QuantizerType::OLD_QUANTIZER);
  ASSERT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  const auto quantized_graph = output_model->subgraphs()->Get(0);
  ASSERT_EQ(quantized_graph->operators()->size(),
            model_->subgraphs()->Get(0)->operators()->size());
  int num_custom_ops_found = 0;
  for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
    const auto op = quantized_graph->operators()->Get(i);
    const uint32_t op_code_idx = op->opcode_index();
    const auto op_code =
        GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx));
    if (op_code == BuiltinOperator_CUSTOM) {
      uint32_t weights_tensor_index = op->inputs()->Get(1);
      const auto weights_tensor =
          quantized_graph->tensors()->Get(weights_tensor_index);
      EXPECT_EQ(weights_tensor->type(), TensorType_INT8);
      num_custom_ops_found++;
    }
  }
  EXPECT_EQ(num_custom_ops_found, 1);
}

TEST_F(QuantizeWeightsTest, VerifyUpdatedHybridSchemeFalseQuantizationHybrid) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  const CustomOpMap custom_op_map;
  auto status = QuantizeWeights(
      &builder, model_, 0, custom_op_map, /*use_updated_hybrid_scheme=*/false,
      /*op_denylist=*/{}, QuantizerType::OLD_QUANTIZER);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  // Nothing should change.
  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       subgraph_idx++) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size());
    // Make sure the graph only has one Conv operation.
    ASSERT_EQ(quantized_graph->operators()->size(), 1);
    const auto op = quantized_graph->operators()->Get(0);
    const uint32_t op_code_idx = op->opcode_index();
    ASSERT_EQ(GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)),
              BuiltinOperator_CONV_2D);
    for (size_t i = 0; i < quantized_graph->tensors()->size(); i++) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      const auto float_tensor = float_graph->tensors()->Get(i);
      EXPECT_EQ(quant_tensor->buffer(), float_tensor->buffer());
      EXPECT_EQ(quant_tensor->is_variable(), float_tensor->is_variable());
      EXPECT_EQ(GetAsVector(quant_tensor->shape()),
                GetAsVector(float_tensor->shape()));
      EXPECT_EQ(quant_tensor->name()->str(), float_tensor->name()->str());
      // If the tensor is a weight, it should have type INT8, otherwise it
      // should stay with type FLOAT32.
      // If the tensor is a bias, it should have type FLOAT32.
      if (quant_tensor->name()->str() == "conv_bias") {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (IsModelInputOrOutput(output_model, i)) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->buffer() != 0) {
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8)
            << quant_tensor->name()->str();
        auto shape = GetAsVector(quant_tensor->shape());
        EXPECT_EQ(quant_tensor->quantization()->scale()->size(), 1);
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

TEST_F(QuantizeWeightsTest, DequantizeConvBlocklisted) {
  LoadBasicModel();
  flatbuffers::FlatBufferBuilder builder;
  const CustomOpMap custom_op_map;
  auto status = QuantizeWeights(&builder, model_, 0, custom_op_map,
                                /*use_updated_hybrid_scheme=*/true,
                                /*op_denylist*/ {BuiltinOperator_CONV_2D},
                                QuantizerType::OLD_QUANTIZER);
  EXPECT_EQ(status, kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  ASSERT_TRUE(output_model);

  ASSERT_EQ(output_model->subgraphs()->size(), model_->subgraphs()->size());
  for (size_t subgraph_idx = 0; subgraph_idx < model_->subgraphs()->size();
       ++subgraph_idx) {
    const auto quantized_graph = output_model->subgraphs()->Get(subgraph_idx);
    const auto float_graph = model_->subgraphs()->Get(subgraph_idx);
    // The output graph should have an extra tensor from the added dequantize
    // op.
    ASSERT_EQ(quantized_graph->tensors()->size(),
              float_graph->tensors()->size() + 1);
    // Check that a dequantize op exists.
    int32_t dequant_input_idx = -1;
    int32_t dequant_output_idx = -1;
    for (size_t i = 0; i < quantized_graph->operators()->size(); ++i) {
      const auto op = quantized_graph->operators()->Get(i);
      const uint32_t op_code_idx = op->opcode_index();
      if (GetBuiltinCode(output_model->operator_codes()->Get(op_code_idx)) ==
          BuiltinOperator_DEQUANTIZE) {
        dequant_input_idx = op->inputs()->Get(0);
        dequant_output_idx = op->outputs()->Get(0);
      }
    }
    ASSERT_GT(dequant_input_idx, -1);
    ASSERT_GT(dequant_output_idx, -1);
    for (size_t i = 0; i < quantized_graph->tensors()->size(); ++i) {
      const auto quant_tensor = quantized_graph->tensors()->Get(i);
      // If the tensor is a weight, it should have type INT8.
      // If the tensor is a bias, it should have type FLOAT32.
      // If the tensor is an input or output it should have type FLOAT32.
      // The input to dequantize should be INT8, and all other tensors should be
      // FLOAT32.
      if (i == dequant_input_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8);
        // The dequantize should still be quantized per-channel
        EXPECT_EQ(quant_tensor->quantization()->scale()->size(), 5);
        EXPECT_EQ(quant_tensor->quantization()->quantized_dimension(), 0);
      } else if (i == dequant_output_idx) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (IsModelInputOrOutput(output_model, i)) {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->name()->str() == "conv_bias") {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      } else if (quant_tensor->buffer() != 0) {
        // If it's a non-bias constant tensor, it must be the weight.
        EXPECT_EQ(quant_tensor->type(), TensorType_INT8);
      } else {
        EXPECT_EQ(quant_tensor->type(), TensorType_FLOAT32);
      }
    }
  }
}

}  // namespace
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weights_testDTcc mht_7(mht_7_v, 865, "", "./tensorflow/lite/tools/optimize/quantize_weights_test.cc", "main");

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
