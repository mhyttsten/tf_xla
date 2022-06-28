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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSremove_successive_transpose_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSremove_successive_transpose_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSremove_successive_transpose_testDTcc() {
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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace {

using ::testing::Test;

class RemoveSuccessiveTransposeTest : public Test {
 protected:
  RemoveSuccessiveTransposeTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSremove_successive_transpose_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/tests/remove_successive_transpose_test.cc", "RemoveSuccessiveTransposeTest");
}

  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSremove_successive_transpose_testDTcc mht_1(mht_1_v, 204, "", "./tensorflow/lite/toco/graph_transformations/tests/remove_successive_transpose_test.cc", "SetUp");
 model_.reset(new toco::Model); }

  void CreateArray(const std::string& name, const std::vector<int>& shape) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSremove_successive_transpose_testDTcc mht_2(mht_2_v, 210, "", "./tensorflow/lite/toco/graph_transformations/tests/remove_successive_transpose_test.cc", "CreateArray");

    toco::Array& array = model_->GetOrCreateArray(name);
    array.data_type = toco::ArrayDataType::kFloat;
    toco::Shape* array_shape = array.mutable_shape();
    *(array_shape->mutable_dims()) = shape;
  }

  void CreateConstantArray(const std::string& name,
                           const std::vector<int>& shape,
                           const std::vector<float>& data) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSremove_successive_transpose_testDTcc mht_3(mht_3_v, 223, "", "./tensorflow/lite/toco/graph_transformations/tests/remove_successive_transpose_test.cc", "CreateConstantArray");

    CreateArray(name, shape);
    toco::Array& array = model_->GetOrCreateArray(name);
    auto& array_buffer = array.GetMutableBuffer<toco::ArrayDataType::kFloat>();
    int bufsize = 1;
    for (int dim : shape) {
      bufsize *= dim;
    }
    array_buffer.data.resize(bufsize);
    float* buf_ptr = array_buffer.data.data();
    for (int i = 0; i < bufsize; ++i) {
      buf_ptr[i] = data[i];
    }
  }

  void CreateGraph(const std::vector<int>& perm1,
                   const std::vector<int>& perm2) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPStestsPSremove_successive_transpose_testDTcc mht_4(mht_4_v, 242, "", "./tensorflow/lite/toco/graph_transformations/tests/remove_successive_transpose_test.cc", "CreateGraph");

    CreateArray("InputA", {2, 2});
    CreateArray("InputB", {2, 2});
    CreateArray("Input", {2, 2});
    CreateArray("InputTranspose", {2, 2});
    CreateArray("InputTransposeTranspose", {2, 2});
    CreateArray("InputTransposeTransposePlusB", {2, 2});

    auto* add_op = new toco::AddOperator;
    add_op->inputs = {"InputA", "InputB"};
    add_op->outputs = {"Input"};
    model_->operators.push_back(std::unique_ptr<toco::Operator>(add_op));

    auto* transpose_op = new toco::TransposeOperator;
    transpose_op->inputs = {"Input"};
    transpose_op->perm = perm1;
    transpose_op->outputs = {"InputTranspose"};
    model_->operators.push_back(std::unique_ptr<toco::Operator>(transpose_op));

    auto* transpose2_op = new toco::TransposeOperator;
    transpose2_op->inputs = {"InputTranspose"};
    transpose2_op->perm = perm2;
    transpose2_op->outputs = {"InputTransposeTranspose"};
    model_->operators.push_back(std::unique_ptr<toco::Operator>(transpose2_op));

    auto* add2_op = new toco::AddOperator;
    add2_op->inputs = {"InputTransposeTranspose", "InputB"};
    add2_op->outputs = {"InputTransposeTransposePlusB"};
    model_->operators.push_back(std::unique_ptr<toco::Operator>(add2_op));
  }

  std::unique_ptr<toco::Model> model_;
};

TEST_F(RemoveSuccessiveTransposeTest, RemoveTranspose) {
  // Creating a model.
  CreateGraph({1, 0}, {1, 0});

  toco::RemoveSuccessiveTranspose transformation;
  bool modified;
  ASSERT_TRUE(transformation.Run(model_.get(), /*op_index=*/1, &modified).ok());
  EXPECT_TRUE(modified);

  ASSERT_EQ(model_->operators.size(), 2);
  ASSERT_EQ(model_->operators[0]->type, toco::OperatorType::kAdd);
  ASSERT_EQ(model_->operators[1]->type, toco::OperatorType::kAdd);
  ASSERT_EQ(model_->operators[1]->inputs[0], model_->operators[0]->outputs[0]);
}

TEST_F(RemoveSuccessiveTransposeTest, DontRemoveNotIdentityTranspose) {
  // Creating a model.
  CreateGraph({0, 2, 1}, {1, 0, 2});

  toco::RemoveSuccessiveTranspose transformation;
  bool modified;
  ASSERT_TRUE(transformation.Run(model_.get(), /*op_index=*/1, &modified).ok());
  EXPECT_FALSE(modified);
}

TEST_F(RemoveSuccessiveTransposeTest, DontRemoveTransposeOutputUnused) {
  CreateArray("InputA", {2, 2});
  CreateArray("InputB", {2, 2});
  CreateArray("Input", {2, 2});
  CreateArray("InputTranspose", {2, 2});
  CreateArray("InputTransposeTranspose", {2, 2});

  auto* add_op = new toco::AddOperator;
  add_op->inputs = {"InputA", "InputB"};
  add_op->outputs = {"Input"};
  model_->operators.push_back(std::unique_ptr<toco::Operator>(add_op));

  auto* transpose_op = new toco::TransposeOperator;
  transpose_op->inputs = {"Input"};
  transpose_op->perm = {0, 2, 1};
  transpose_op->outputs = {"InputTranspose"};
  model_->operators.push_back(std::unique_ptr<toco::Operator>(transpose_op));

  auto* transpose2_op = new toco::TransposeOperator;
  transpose2_op->inputs = {"InputTranspose"};
  transpose2_op->perm = {0, 2, 1};
  transpose2_op->outputs = {"InputTransposeTranspose"};
  model_->operators.push_back(std::unique_ptr<toco::Operator>(transpose2_op));

  toco::RemoveSuccessiveTranspose transformation;
  bool modified;
  ASSERT_TRUE(transformation.Run(model_.get(), /*op_index=*/1, &modified).ok());
  EXPECT_FALSE(modified);
}
}  // namespace
