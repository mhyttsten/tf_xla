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
class MHTracer_DTPStensorflowPSlitePStocoPStflitePSimport_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimport_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStflitePSimport_testDTcc() {
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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace toco {

namespace tflite {
namespace {

using ::testing::ElementsAre;

using flatbuffers::Offset;
using flatbuffers::Vector;
class ImportTest : public ::testing::Test {
 protected:
  template <typename T>
  Offset<Vector<unsigned char>> CreateDataVector(const std::vector<T>& data) {
    return builder_.CreateVector(reinterpret_cast<const uint8_t*>(data.data()),
                                 sizeof(T) * data.size());
  }

  Offset<Vector<Offset<::tflite::Buffer>>> BuildBuffers() {
    auto buf0 = ::tflite::CreateBuffer(builder_, CreateDataVector<float>({}));
    auto buf1 = ::tflite::CreateBuffer(
        builder_, CreateDataVector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
    auto buf2 =
        ::tflite::CreateBuffer(builder_, CreateDataVector<float>({3.0f, 4.0f}));
    return builder_.CreateVector(
        std::vector<Offset<::tflite::Buffer>>({buf0, buf1, buf2}));
  }

  Offset<Vector<Offset<::tflite::Tensor>>> BuildTensors() {
    auto q = ::tflite::CreateQuantizationParameters(
        builder_,
        /*min=*/builder_.CreateVector<float>({0.1f}),
        /*max=*/builder_.CreateVector<float>({0.2f}),
        /*scale=*/builder_.CreateVector<float>({0.3f}),
        /*zero_point=*/builder_.CreateVector<int64_t>({100LL}));
    auto t1 =
        ::tflite::CreateTensor(builder_, builder_.CreateVector<int>({1, 2, 2}),
                               ::tflite::TensorType_FLOAT32, 1,
                               builder_.CreateString("tensor_one"), q);
    auto t2 =
        ::tflite::CreateTensor(builder_, builder_.CreateVector<int>({2, 1}),
                               ::tflite::TensorType_FLOAT32, 0,
                               builder_.CreateString("tensor_two"), q);
    return builder_.CreateVector(
        std::vector<Offset<::tflite::Tensor>>({t1, t2}));
  }

  Offset<Vector<Offset<::tflite::OperatorCode>>> BuildOpCodes(
      std::initializer_list<::tflite::BuiltinOperator> op_codes) {
    std::vector<Offset<::tflite::OperatorCode>> op_codes_vector;
    for (auto op : op_codes) {
      op_codes_vector.push_back(::tflite::CreateOperatorCode(builder_, op, 0));
    }
    return builder_.CreateVector(op_codes_vector);
  }

  Offset<Vector<Offset<::tflite::OperatorCode>>> BuildOpCodes() {
    return BuildOpCodes({::tflite::BuiltinOperator_MAX_POOL_2D,
                         ::tflite::BuiltinOperator_CONV_2D});
  }

  Offset<Vector<Offset<::tflite::Operator>>> BuildOperators(
      std::initializer_list<int> inputs, std::initializer_list<int> outputs) {
    auto is = builder_.CreateVector<int>(inputs);
    if (inputs.size() == 0) is = 0;
    auto os = builder_.CreateVector<int>(outputs);
    if (outputs.size() == 0) os = 0;
    auto op = ::tflite::CreateOperator(
        builder_, 0, is, os, ::tflite::BuiltinOptions_Conv2DOptions,
        ::tflite::CreateConv2DOptions(builder_, ::tflite::Padding_VALID, 1, 1,
                                      ::tflite::ActivationFunctionType_NONE)
            .Union(),
        /*custom_options=*/0, ::tflite::CustomOptionsFormat_FLEXBUFFERS);

    return builder_.CreateVector(std::vector<Offset<::tflite::Operator>>({op}));
  }

  Offset<Vector<Offset<::tflite::Operator>>> BuildOperators() {
    return BuildOperators({0}, {1});
  }

  Offset<Vector<Offset<::tflite::SubGraph>>> BuildSubGraphs(
      Offset<Vector<Offset<::tflite::Tensor>>> tensors,
      Offset<Vector<Offset<::tflite::Operator>>> operators,
      int num_sub_graphs = 1) {
    std::vector<int32_t> inputs = {0};
    std::vector<int32_t> outputs = {1};
    std::vector<Offset<::tflite::SubGraph>> v;
    for (int i = 0; i < num_sub_graphs; ++i) {
      v.push_back(::tflite::CreateSubGraph(
          builder_, tensors, builder_.CreateVector(inputs),
          builder_.CreateVector(outputs), operators,
          builder_.CreateString("subgraph")));
    }
    return builder_.CreateVector(v);
  }

  // This is a very simplistic model. We are not interested in testing all the
  // details here, since tf.mini's testing framework will be exercising all the
  // conversions multiple times, and the conversion of operators is tested by
  // separate unittests.
  void BuildTestModel() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimport_testDTcc mht_0(mht_0_v, 293, "", "./tensorflow/lite/toco/tflite/import_test.cc", "BuildTestModel");

    auto buffers = BuildBuffers();
    auto tensors = BuildTensors();
    auto opcodes = BuildOpCodes();
    auto operators = BuildOperators();
    auto subgraphs = BuildSubGraphs(tensors, operators);
    auto s = builder_.CreateString("");

    ::tflite::FinishModelBuffer(
        builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION,
                                        opcodes, subgraphs, s, buffers));

    input_model_ = ::tflite::GetModel(builder_.GetBufferPointer());
  }
  std::string InputModelAsString() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSimport_testDTcc mht_1(mht_1_v, 310, "", "./tensorflow/lite/toco/tflite/import_test.cc", "InputModelAsString");

    return std::string(reinterpret_cast<char*>(builder_.GetBufferPointer()),
                       builder_.GetSize());
  }
  flatbuffers::FlatBufferBuilder builder_;
  const ::tflite::Model* input_model_ = nullptr;
};

TEST_F(ImportTest, LoadTensorsTable) {
  BuildTestModel();

  details::TensorsTable tensors;
  details::LoadTensorsTable(*input_model_, &tensors);
  EXPECT_THAT(tensors, ElementsAre("tensor_one", "tensor_two"));
}

TEST_F(ImportTest, LoadOperatorsTable) {
  BuildTestModel();

  details::OperatorsTable operators;
  details::LoadOperatorsTable(*input_model_, &operators);
  EXPECT_THAT(operators, ElementsAre("MAX_POOL_2D", "CONV_2D"));
}

TEST_F(ImportTest, Tensors) {
  BuildTestModel();

  auto model = Import(ModelFlags(), InputModelAsString());

  ASSERT_GT(model->HasArray("tensor_one"), 0);
  Array& a1 = model->GetArray("tensor_one");
  EXPECT_EQ(ArrayDataType::kFloat, a1.data_type);
  EXPECT_THAT(a1.GetBuffer<ArrayDataType::kFloat>().data,
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  ASSERT_TRUE(a1.has_shape());
  EXPECT_THAT(a1.shape().dims(), ElementsAre(1, 2, 2));

  const auto& mm = a1.minmax;
  ASSERT_TRUE(mm.get());
  EXPECT_FLOAT_EQ(0.1, mm->min);
  EXPECT_FLOAT_EQ(0.2, mm->max);

  const auto& q = a1.quantization_params;
  ASSERT_TRUE(q.get());
  EXPECT_FLOAT_EQ(0.3, q->scale);
  EXPECT_EQ(100, q->zero_point);
}

TEST_F(ImportTest, NoBuffers) {
  auto buffers = 0;
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes();
  auto operators = BuildOperators();
  auto subgraphs = BuildSubGraphs(tensors, operators);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));
  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Missing 'buffers' section.");
}

TEST_F(ImportTest, NoInputs) {
  auto buffers = BuildBuffers();
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes();
  auto operators = BuildOperators({}, {1});
  auto subgraphs = BuildSubGraphs(tensors, operators);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));
  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Missing 'inputs' for operator.");
}

TEST_F(ImportTest, NoOutputs) {
  auto buffers = BuildBuffers();
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes();
  auto operators = BuildOperators({0}, {});
  auto subgraphs = BuildSubGraphs(tensors, operators);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));
  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Missing 'outputs' for operator.");
}

TEST_F(ImportTest, InvalidOpCode) {
  auto buffers = BuildBuffers();
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes({static_cast<::tflite::BuiltinOperator>(-1),
                               ::tflite::BuiltinOperator_CONV_2D});
  auto operators = BuildOperators();
  auto subgraphs = BuildSubGraphs(tensors, operators);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));
  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Operator id '-1' is out of range.");
}

TEST_F(ImportTest, MultipleSubGraphs) {
  auto buffers = BuildBuffers();
  auto tensors = BuildTensors();
  auto opcodes = BuildOpCodes();
  auto operators = BuildOperators();
  auto subgraphs = BuildSubGraphs(tensors, operators, 2);
  auto comment = builder_.CreateString("");
  ::tflite::FinishModelBuffer(
      builder_, ::tflite::CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                                      subgraphs, comment, buffers));

  input_model_ = ::tflite::GetModel(builder_.GetBufferPointer());

  EXPECT_DEATH(Import(ModelFlags(), InputModelAsString()),
               "Number of subgraphs in tflite should be exactly 1.");
}

// TODO(ahentz): still need tests for Operators and IOTensors.

}  // namespace
}  // namespace tflite

}  // namespace toco
