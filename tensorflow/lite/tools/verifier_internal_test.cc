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
class MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc() {
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
#include "tensorflow/lite/tools/verifier_internal.h"

#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/util.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/version.h"

namespace tflite {

// Build single subgraph model.
class TfLiteFlatbufferModelBuilder {
 public:
  TfLiteFlatbufferModelBuilder() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/lite/tools/verifier_internal_test.cc", "TfLiteFlatbufferModelBuilder");

    buffers_.push_back(
        CreateBuffer(builder_, builder_.CreateVector(std::vector<uint8_t>{})));
  }

  TfLiteFlatbufferModelBuilder(const std::vector<BuiltinOperator>& builtin_ops,
                               const std::vector<std::string>& custom_ops) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/tools/verifier_internal_test.cc", "TfLiteFlatbufferModelBuilder");

    buffers_.push_back(
        CreateBuffer(builder_, builder_.CreateVector(std::vector<uint8_t>{})));

    for (const auto& iter : builtin_ops) {
      resolver_.AddBuiltin(iter, &fake_op_);
    }
    for (const auto& iter : custom_ops) {
      resolver_.AddCustom(iter.data(), &fake_op_);
    }
  }

  void AddTensor(const std::vector<int>& shape, tflite::TensorType type,
                 const std::vector<uint8_t>& buffer, const char* name,
                 const bool is_variable = false) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/lite/tools/verifier_internal_test.cc", "AddTensor");

    int buffer_index = 0;
    if (!buffer.empty()) {
      buffer_index = buffers_.size();
      buffers_.push_back(CreateBuffer(builder_, builder_.CreateVector(buffer)));
    }
    if (shape.empty()) {
      tensors_.push_back(CreateTensorDirect(builder_, /*shape=*/nullptr, type,
                                            buffer_index, name,
                                            /*quantization=*/0, is_variable));
      return;
    }
    tensors_.push_back(CreateTensorDirect(builder_, &shape, type, buffer_index,
                                          name, /*quantization=*/0,
                                          is_variable));
  }

  void AddOperator(const std::vector<int32_t>& inputs,
                   const std::vector<int32_t>& outputs,
                   tflite::BuiltinOperator builtin_op, const char* custom_op) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("custom_op: \"" + (custom_op == nullptr ? std::string("nullptr") : std::string((char*)custom_op)) + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc mht_3(mht_3_v, 262, "", "./tensorflow/lite/tools/verifier_internal_test.cc", "AddOperator");

    operator_codes_.push_back(
        CreateOperatorCodeDirect(builder_, builtin_op, custom_op));
    operators_.push_back(CreateOperator(
        builder_, operator_codes_.size() - 1, builder_.CreateVector(inputs),
        builder_.CreateVector(outputs), BuiltinOptions_NONE,
        /*builtin_options=*/0,
        /*custom_options=*/0, tflite::CustomOptionsFormat_FLEXBUFFERS));
  }

  enum BuilderMode {
    kBuilderModeEmptyVectorIsEmpty,
    kBuilderModeEmptyVectorIsNull,
    kBuilderModeDefault = kBuilderModeEmptyVectorIsEmpty,
  };
  void FinishModel(const std::vector<int32_t>& inputs,
                   const std::vector<int32_t>& outputs,
                   BuilderMode mode = kBuilderModeDefault) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc mht_4(mht_4_v, 282, "", "./tensorflow/lite/tools/verifier_internal_test.cc", "FinishModel");

    auto subgraph = std::vector<flatbuffers::Offset<SubGraph>>({CreateSubGraph(
        builder_, CreateVector(tensors_, mode), CreateVector(inputs, mode),
        CreateVector(outputs, mode), CreateVector(operators_, mode),
        builder_.CreateString("test_subgraph"))});
    auto result = CreateModel(
        builder_, TFLITE_SCHEMA_VERSION, CreateVector(operator_codes_, mode),
        CreateVector(subgraph, mode), builder_.CreateString("test_model"),
        CreateVector(buffers_, mode));
    tflite::FinishModelBuffer(builder_, result);
  }

  bool Verify(const void* buf, size_t length) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc mht_5(mht_5_v, 297, "", "./tensorflow/lite/tools/verifier_internal_test.cc", "Verify");

    return tflite::internal::VerifyFlatBufferAndGetModel(buf, length);
  }

  bool Verify() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSverifier_internal_testDTcc mht_6(mht_6_v, 304, "", "./tensorflow/lite/tools/verifier_internal_test.cc", "Verify");

    return Verify(builder_.GetBufferPointer(), builder_.GetSize());
  }

 private:
  template <typename T>
  flatbuffers::Offset<flatbuffers::Vector<T>> CreateVector(
      const std::vector<T>& v, BuilderMode mode) {
    if (mode == kBuilderModeEmptyVectorIsNull && v.empty()) {
      return 0;
    }
    return builder_.CreateVector(v);
  }

  flatbuffers::FlatBufferBuilder builder_;
  MutableOpResolver resolver_;
  TfLiteRegistration fake_op_;
  std::vector<flatbuffers::Offset<Operator>> operators_;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes_;
  std::vector<flatbuffers::Offset<Tensor>> tensors_;
  std::vector<flatbuffers::Offset<Buffer>> buffers_;
};

TEST(VerifyModel, TestEmptyModel) {
  flatbuffers::FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/TFLITE_SCHEMA_VERSION,
                           /*operator_codes=*/0, /*subgraphs=*/0,
                           /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);

  ASSERT_TRUE(::tflite::internal::VerifyFlatBufferAndGetModel(
      builder.GetBufferPointer(), builder.GetSize()));
}

TEST(VerifyModel, TestSimpleModel) {
  TfLiteFlatbufferModelBuilder builder({}, {"test"});
  builder.AddOperator({0, 1}, {2}, BuiltinOperator_CUSTOM, "test");
  builder.AddTensor({2, 3}, TensorType_UINT8, {1, 2, 3, 4, 5, 6}, "input");
  builder.AddTensor(
      {2}, TensorType_STRING,
      {2, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 19, 0, 0, 0, 'A', 'B', 'C'},
      "data");
  builder.AddTensor({2, 3}, TensorType_INT32, {}, "output");
  builder.FinishModel({0, 1}, {2});
  ASSERT_TRUE(builder.Verify());
}

TEST(VerifyModel, TestCorruptedData) {
  std::string model = "123";
  ASSERT_FALSE(::tflite::internal::VerifyFlatBufferAndGetModel(model.data(),
                                                               model.size()));
}

TEST(VerifyModel, TestRandomModificationIsNotAllowed) {
  flatbuffers::FlatBufferBuilder builder;
  auto model = CreateModel(builder, /*version=*/TFLITE_SCHEMA_VERSION,
                           /*operator_codes=*/0,
                           /*subgraphs=*/0, /*description=*/0, /*buffers=*/0);
  ::tflite::FinishModelBuffer(builder, model);

  std::string model_content(reinterpret_cast<char*>(builder.GetBufferPointer()),
                            builder.GetSize());
  for (size_t i = 0; i < model_content.size(); i++) {
    model_content[i] = (model_content[i] + 137) % 255;
    EXPECT_FALSE(tflite::internal::VerifyFlatBufferAndGetModel(
        model_content.data(), model_content.size()))
        << "Fail at position: " << i;
  }
}

}  // namespace tflite
