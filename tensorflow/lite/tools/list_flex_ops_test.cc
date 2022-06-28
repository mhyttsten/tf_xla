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
class MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_ops_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_ops_testDTcc() {
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

#include "tensorflow/lite/tools/list_flex_ops.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace flex {

class FlexOpsListTest : public ::testing::Test {
 protected:
  FlexOpsListTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_ops_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/tools/list_flex_ops_test.cc", "FlexOpsListTest");
}

  void ReadOps(const string& path) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_ops_testDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/tools/list_flex_ops_test.cc", "ReadOps");

    std::string full_path = tensorflow::GetDataDependencyFilepath(path);
    auto model = FlatBufferModel::BuildFromFile(full_path.data());
    AddFlexOpsFromModel(model->GetModel(), &flex_ops_);
    output_text_ = OpListToJSONString(flex_ops_);
  }

  void ReadOps(const tflite::Model* model) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_ops_testDTcc mht_2(mht_2_v, 218, "", "./tensorflow/lite/tools/list_flex_ops_test.cc", "ReadOps");

    AddFlexOpsFromModel(model, &flex_ops_);
    output_text_ = OpListToJSONString(flex_ops_);
  }

  std::string output_text_;
  OpKernelSet flex_ops_;
};

TfLiteRegistration* Register_TEST() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_ops_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/lite/tools/list_flex_ops_test.cc", "Register_TEST");

  static TfLiteRegistration r = {nullptr, nullptr, nullptr, nullptr};
  return &r;
}

std::vector<uint8_t> CreateFlexCustomOptions(std::string nodedef_raw_string) {
  tensorflow::NodeDef node_def;
  tensorflow::protobuf::TextFormat::ParseFromString(nodedef_raw_string,
                                                    &node_def);
  std::string node_def_str = node_def.SerializeAsString();
  auto flex_builder = std::make_unique<flexbuffers::Builder>();
  flex_builder->Vector([&]() {
    flex_builder->String(node_def.op());
    flex_builder->String(node_def_str);
  });
  flex_builder->Finish();
  return flex_builder->GetBuffer();
}

class FlexOpModel : public SingleOpModel {
 public:
  FlexOpModel(const std::string& op_name, const TensorData& input1,
              const TensorData& input2, const TensorType& output,
              const std::vector<uint8_t>& custom_options) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSlist_flex_ops_testDTcc mht_4(mht_4_v, 257, "", "./tensorflow/lite/tools/list_flex_ops_test.cc", "FlexOpModel");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetCustomOp(op_name, custom_options, Register_TEST);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST_F(FlexOpsListTest, TestModelsNoFlex) {
  ReadOps("tensorflow/lite/testdata/test_model.bin");
  EXPECT_EQ(output_text_, "[]\n");
}

TEST_F(FlexOpsListTest, TestBrokenModel) {
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps("tensorflow/lite/testdata/test_model_broken.bin"), "");
}

TEST_F(FlexOpsListTest, TestZeroSubgraphs) {
  ReadOps("tensorflow/lite/testdata/0_subgraphs.bin");
  EXPECT_EQ(output_text_, "[]\n");
}

TEST_F(FlexOpsListTest, TestFlexAdd) {
  ReadOps("tensorflow/lite/testdata/multi_add_flex.bin");
  EXPECT_EQ(output_text_,
            "[[\"AddV2\",\"BinaryOp<CPUDevice, functor::add<float>>\"]]\n");
}

TEST_F(FlexOpsListTest, TestTwoModel) {
  ReadOps("tensorflow/lite/testdata/multi_add_flex.bin");
  ReadOps("tensorflow/lite/testdata/softplus_flex.bin");
  EXPECT_EQ(output_text_,
            "[[\"AddV2\",\"BinaryOp<CPUDevice, "
            "functor::add<float>>\"],[\"Softplus\",\"SoftplusOp<CPUDevice, "
            "float>\"]]\n");
}

TEST_F(FlexOpsListTest, TestDuplicatedOp) {
  ReadOps("tensorflow/lite/testdata/multi_add_flex.bin");
  ReadOps("tensorflow/lite/testdata/multi_add_flex.bin");
  EXPECT_EQ(output_text_,
            "[[\"AddV2\",\"BinaryOp<CPUDevice, functor::add<float>>\"]]\n");
}

TEST_F(FlexOpsListTest, TestInvalidCustomOptions) {
  // Using a invalid custom options, expected to fail.
  std::vector<uint8_t> random_custom_options(20);
  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        random_custom_options);
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps(tflite::GetModel(max_model.GetModelBuffer())),
      "Failed to parse data into a valid NodeDef");
}

TEST_F(FlexOpsListTest, TestOpNameEmpty) {
  // NodeDef with empty opname.
  std::string nodedef_raw_str =
      "name: \"node_1\""
      "op: \"\""
      "input: [ \"b\", \"c\" ]"
      "attr: { key: \"T\" value: { type: DT_FLOAT } }";
  std::string random_fieldname = "random string";
  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        CreateFlexCustomOptions(nodedef_raw_str));
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps(tflite::GetModel(max_model.GetModelBuffer())), "Invalid NodeDef");
}

TEST_F(FlexOpsListTest, TestOpNotFound) {
  // NodeDef with invalid opname.
  std::string nodedef_raw_str =
      "name: \"node_1\""
      "op: \"FlexInvalidOp\""
      "input: [ \"b\", \"c\" ]"
      "attr: { key: \"T\" value: { type: DT_FLOAT } }";

  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        CreateFlexCustomOptions(nodedef_raw_str));
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps(tflite::GetModel(max_model.GetModelBuffer())),
      "Op FlexInvalidOp not found");
}

TEST_F(FlexOpsListTest, TestKernelNotFound) {
  // NodeDef with non-supported type.
  std::string nodedef_raw_str =
      "name: \"node_1\""
      "op: \"Add\""
      "input: [ \"b\", \"c\" ]"
      "attr: { key: \"T\" value: { type: DT_BOOL } }";

  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        CreateFlexCustomOptions(nodedef_raw_str));
  EXPECT_DEATH_IF_SUPPORTED(
      ReadOps(tflite::GetModel(max_model.GetModelBuffer())),
      "Failed to find kernel class for op: Add");
}

TEST_F(FlexOpsListTest, TestFlexAddWithSingleOpModel) {
  std::string nodedef_raw_str =
      "name: \"node_1\""
      "op: \"Add\""
      "input: [ \"b\", \"c\" ]"
      "attr: { key: \"T\" value: { type: DT_FLOAT } }";

  FlexOpModel max_model("FlexAdd", {TensorType_FLOAT32, {3, 1, 2, 2}},
                        {TensorType_FLOAT32, {3, 1, 2, 1}}, TensorType_FLOAT32,
                        CreateFlexCustomOptions(nodedef_raw_str));
  ReadOps(tflite::GetModel(max_model.GetModelBuffer()));
  EXPECT_EQ(output_text_,
            "[[\"Add\",\"BinaryOp<CPUDevice, functor::add<float>>\"]]\n");
}
}  // namespace flex
}  // namespace tflite
