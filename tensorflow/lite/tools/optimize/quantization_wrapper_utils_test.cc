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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_wrapper_utils_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_wrapper_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_wrapper_utils_testDTcc() {
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
#include "tensorflow/lite/tools/optimize/quantization_wrapper_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {
namespace optimize {
namespace {

using ::testing::ElementsAreArray;

TEST(LstmPreprocess, Add2Tensors) {
  // Create a model with 1 lstm layer.
  auto model = absl::make_unique<ModelT>();
  auto subgraph = absl::make_unique<tflite::SubGraphT>();
  auto buffer = absl::make_unique<tflite::BufferT>();
  auto lstm_op_code = absl::make_unique<OperatorCodeT>();
  auto lstm_op = absl::make_unique<OperatorT>();

  lstm_op_code->builtin_code = BuiltinOperator_LSTM;
  lstm_op_code->deprecated_builtin_code =
      static_cast<int8_t>(BuiltinOperator_LSTM);
  lstm_op_code->version = 2;
  lstm_op->opcode_index = 0;
  lstm_op->inputs = {0, 1,  2,  3,  4,  5,  6,  7,  8,  -1, -1, -1,
                     9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  lstm_op->outputs = {24};

  model->subgraphs.push_back(std::move(subgraph));
  for (int i = 0; i < lstm_op->inputs.size(); ++i) {
    const int index = lstm_op->inputs[i];
    if (index == -1) {
      continue;
    }
    auto tensor = absl::make_unique<TensorT>();
    tensor->name = "lstm_tensor" + std::to_string(index);
    tensor->shape = {2, 3, 4};
    tensor->type = TensorType_FLOAT32;
    model->subgraphs[0]->tensors.push_back(std::move(tensor));
  }
  model->subgraphs[0]->operators.push_back(std::move(lstm_op));
  model->operator_codes.push_back(std::move(lstm_op_code));
  model->buffers.push_back(std::move(buffer));

  // Add 2 tensors.
  flatbuffers::FlatBufferBuilder builder;
  tflite::optimize::AddIntermediateTensorsToFusedOp(&builder, model.get());

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 1);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 26);
  EXPECT_EQ(model->buffers.size(), 1);

  EXPECT_EQ(GetBuiltinCode(model->operator_codes[0].get()),
            BuiltinOperator_LSTM);
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->name, "lstm_tensor0");
  EXPECT_EQ(model->subgraphs[0]->tensors[21]->name, "intermediate_0_0");
  EXPECT_EQ(model->subgraphs[0]->tensors[22]->name, "intermediate_0_1");
  EXPECT_EQ(model->subgraphs[0]->tensors[23]->name, "intermediate_0_2");
  EXPECT_EQ(model->subgraphs[0]->tensors[24]->name, "intermediate_0_3");
  EXPECT_EQ(model->subgraphs[0]->tensors[25]->name, "intermediate_0_4");
  EXPECT_THAT(
      model->subgraphs[0]->operators[0]->inputs,
      ElementsAreArray({0, 1,  2,  3,  4,  5,  6,  7,  8,  -1, -1, -1,
                        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->outputs,
              ElementsAreArray({24}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->intermediates,
              ElementsAreArray({21, 22, 23, 24, 25}));

  // Call AddIntermediateTensorsToFusedOp again and expect no change in model.
  tflite::optimize::AddIntermediateTensorsToFusedOp(&builder, model.get());

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 1);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 26);
  EXPECT_EQ(model->buffers.size(), 1);

  EXPECT_EQ(GetBuiltinCode(model->operator_codes[0].get()),
            BuiltinOperator_LSTM);
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->name, "lstm_tensor0");
  EXPECT_EQ(model->subgraphs[0]->tensors[21]->name, "intermediate_0_0");
  EXPECT_EQ(model->subgraphs[0]->tensors[22]->name, "intermediate_0_1");
  EXPECT_EQ(model->subgraphs[0]->tensors[23]->name, "intermediate_0_2");
  EXPECT_EQ(model->subgraphs[0]->tensors[24]->name, "intermediate_0_3");
  EXPECT_EQ(model->subgraphs[0]->tensors[25]->name, "intermediate_0_4");
  EXPECT_THAT(
      model->subgraphs[0]->operators[0]->inputs,
      ElementsAreArray({0, 1,  2,  3,  4,  5,  6,  7,  8,  -1, -1, -1,
                        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->outputs,
              ElementsAreArray({24}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->intermediates,
              ElementsAreArray({21, 22, 23, 24, 25}));
}

}  // namespace
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantization_wrapper_utils_testDTcc mht_0(mht_0_v, 292, "", "./tensorflow/lite/tools/optimize/quantization_wrapper_utils_test.cc", "main");
 return RUN_ALL_TESTS(); }
