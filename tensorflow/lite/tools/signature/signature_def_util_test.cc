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
class MHTracer_DTPStensorflowPSlitePStoolsPSsignaturePSsignature_def_util_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSsignaturePSsignature_def_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSsignaturePSsignature_def_util_testDTcc() {
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
#include "tensorflow/lite/tools/signature/signature_def_util.h"

#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

using tensorflow::kClassifyMethodName;
using tensorflow::kDefaultServingSignatureDefKey;
using tensorflow::kPredictMethodName;
using tensorflow::SignatureDef;
using tensorflow::Status;

constexpr char kSignatureInput[] = "input";
constexpr char kSignatureOutput[] = "output";
constexpr char kTestFilePath[] = "tensorflow/lite/testdata/add.bin";

class SimpleSignatureDefUtilTest : public testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSsignaturePSsignature_def_util_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/lite/tools/signature/signature_def_util_test.cc", "SetUp");

    flatbuffer_model_ = FlatBufferModel::BuildFromFile(kTestFilePath);
    ASSERT_NE(flatbuffer_model_, nullptr);
    model_ = flatbuffer_model_->GetModel();
    ASSERT_NE(model_, nullptr);
  }

  SignatureDef GetTestSignatureDef() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSsignaturePSsignature_def_util_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/tools/signature/signature_def_util_test.cc", "GetTestSignatureDef");

    auto signature_def = SignatureDef();
    tensorflow::TensorInfo input_tensor;
    tensorflow::TensorInfo output_tensor;
    *input_tensor.mutable_name() = kSignatureInput;
    *output_tensor.mutable_name() = kSignatureOutput;
    *signature_def.mutable_method_name() = kClassifyMethodName;
    (*signature_def.mutable_inputs())[kSignatureInput] = input_tensor;
    (*signature_def.mutable_outputs())[kSignatureOutput] = output_tensor;
    return signature_def;
  }
  std::unique_ptr<FlatBufferModel> flatbuffer_model_;
  const Model* model_;
};

TEST_F(SimpleSignatureDefUtilTest, SetSignatureDefTest) {
  SignatureDef expected_signature_def = GetTestSignatureDef();
  std::string model_output;
  const std::map<string, SignatureDef> expected_signature_def_map = {
      {kDefaultServingSignatureDefKey, expected_signature_def}};
  EXPECT_EQ(Status::OK(), SetSignatureDefMap(model_, expected_signature_def_map,
                                             &model_output));
  const Model* add_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_TRUE(HasSignatureDef(add_model, kDefaultServingSignatureDefKey));
  std::map<string, SignatureDef> test_signature_def_map;
  EXPECT_EQ(Status::OK(),
            GetSignatureDefMap(add_model, &test_signature_def_map));
  SignatureDef test_signature_def =
      test_signature_def_map[kDefaultServingSignatureDefKey];
  EXPECT_EQ(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
}

TEST_F(SimpleSignatureDefUtilTest, OverwriteSignatureDefTest) {
  auto expected_signature_def = GetTestSignatureDef();
  std::string model_output;
  std::map<string, SignatureDef> expected_signature_def_map = {
      {kDefaultServingSignatureDefKey, expected_signature_def}};
  EXPECT_EQ(Status::OK(), SetSignatureDefMap(model_, expected_signature_def_map,
                                             &model_output));
  const Model* add_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_TRUE(HasSignatureDef(add_model, kDefaultServingSignatureDefKey));
  std::map<string, SignatureDef> test_signature_def_map;
  EXPECT_EQ(Status::OK(),
            GetSignatureDefMap(add_model, &test_signature_def_map));
  SignatureDef test_signature_def =
      test_signature_def_map[kDefaultServingSignatureDefKey];
  EXPECT_EQ(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
  *expected_signature_def.mutable_method_name() = kPredictMethodName;
  expected_signature_def_map.erase(
      expected_signature_def_map.find(kDefaultServingSignatureDefKey));
  constexpr char kTestSignatureDefKey[] = "ServingTest";
  expected_signature_def_map[kTestSignatureDefKey] = expected_signature_def;
  EXPECT_EQ(
      Status::OK(),
      SetSignatureDefMap(add_model, expected_signature_def_map, &model_output));
  const Model* final_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_FALSE(HasSignatureDef(final_model, kDefaultServingSignatureDefKey));
  EXPECT_EQ(Status::OK(),
            GetSignatureDefMap(final_model, &test_signature_def_map));
  EXPECT_NE(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
  EXPECT_TRUE(HasSignatureDef(final_model, kTestSignatureDefKey));
  EXPECT_EQ(Status::OK(),
            GetSignatureDefMap(final_model, &test_signature_def_map));
  test_signature_def = test_signature_def_map[kTestSignatureDefKey];
  EXPECT_EQ(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
}

TEST_F(SimpleSignatureDefUtilTest, GetSignatureDefTest) {
  std::map<string, SignatureDef> test_signature_def_map;
  EXPECT_EQ(Status::OK(), GetSignatureDefMap(model_, &test_signature_def_map));
  EXPECT_FALSE(HasSignatureDef(model_, kDefaultServingSignatureDefKey));
}

TEST_F(SimpleSignatureDefUtilTest, ClearSignatureDefTest) {
  const int expected_num_buffers = model_->buffers()->size();
  auto expected_signature_def = GetTestSignatureDef();
  std::string model_output;
  std::map<string, SignatureDef> expected_signature_def_map = {
      {kDefaultServingSignatureDefKey, expected_signature_def}};
  EXPECT_EQ(Status::OK(), SetSignatureDefMap(model_, expected_signature_def_map,
                                             &model_output));
  const Model* add_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_TRUE(HasSignatureDef(add_model, kDefaultServingSignatureDefKey));
  SignatureDef test_signature_def;
  std::map<string, SignatureDef> test_signature_def_map;
  EXPECT_EQ(Status::OK(),
            GetSignatureDefMap(add_model, &test_signature_def_map));
  test_signature_def = test_signature_def_map[kDefaultServingSignatureDefKey];
  EXPECT_EQ(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
  EXPECT_EQ(Status::OK(), ClearSignatureDefMap(add_model, &model_output));
  const Model* clear_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_FALSE(HasSignatureDef(clear_model, kDefaultServingSignatureDefKey));
  EXPECT_EQ(expected_num_buffers, clear_model->buffers()->size());
}

TEST_F(SimpleSignatureDefUtilTest, SetSignatureDefErrorsTest) {
  std::map<string, SignatureDef> test_signature_def_map;
  std::string model_output;
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(
      SetSignatureDefMap(model_, test_signature_def_map, &model_output)));
  SignatureDef test_signature_def;
  test_signature_def_map[kDefaultServingSignatureDefKey] = test_signature_def;
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(
      SetSignatureDefMap(model_, test_signature_def_map, nullptr)));
}

}  // namespace
}  // namespace tflite
