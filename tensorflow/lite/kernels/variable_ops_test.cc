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
class MHTracer_DTPStensorflowPSlitePSkernelsPSvariable_ops_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSvariable_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSvariable_ops_testDTcc() {
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
#include <stdint.h>

#include <memory>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace {
const char kContainer[] = "c";
const char kSharedName[] = "a";

class VariableOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSvariable_ops_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/kernels/variable_ops_test.cc", "SetUp");

    assign_registration_ = ::tflite::ops::builtin::Register_ASSIGN_VARIABLE();
    ASSERT_NE(assign_registration_, nullptr);
    read_registration_ = ::tflite::ops::builtin::Register_READ_VARIABLE();
    ASSERT_NE(read_registration_, nullptr);
    var_handle_registration_ = ::tflite::ops::builtin::Register_VAR_HANDLE();
    ASSERT_NE(var_handle_registration_, nullptr);

    ConstructGraph();
  }

  void ConstructInvalidGraph() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSvariable_ops_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/kernels/variable_ops_test.cc", "ConstructInvalidGraph");

    interpreter_.reset(new Interpreter);
    // Invalid graph, variable is read before it is assigned a value.

    // Construct a graph like this:
    //   Input: %0
    //   Output: %2
    //   %1 = var_handle()
    //   %2 = read(%1)

    TfLiteVarHandleParams* var_handle_params = GetVarHandleParams();

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs({2}), kTfLiteOk);
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteResource, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    int node_index;
    interpreter_->AddNodeWithParameters({}, {1}, nullptr, 0, var_handle_params,
                                        var_handle_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                        read_registration_, &node_index);
  }

  TfLiteVarHandleParams* GetVarHandleParams() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSvariable_ops_testDTcc mht_2(mht_2_v, 248, "", "./tensorflow/lite/kernels/variable_ops_test.cc", "GetVarHandleParams");

    TfLiteVarHandleParams* var_handle_params =
        reinterpret_cast<TfLiteVarHandleParams*>(
            malloc(sizeof(TfLiteVarHandleParams)));
    var_handle_params->container = kContainer;
    var_handle_params->shared_name = kSharedName;
    return var_handle_params;
  }

  void ConstructGraph() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSvariable_ops_testDTcc mht_3(mht_3_v, 260, "", "./tensorflow/lite/kernels/variable_ops_test.cc", "ConstructGraph");

    interpreter_.reset(new Interpreter);
    // Construct a graph like this:
    //   Input: %0
    //   Output: %2
    //   %1 = var_handle()
    //   variable_assign(%1, %0)
    //   %2 = read(%1)

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs({2}), kTfLiteOk);
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteResource, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    int node_index;

    TfLiteVarHandleParams* var_handle_params = GetVarHandleParams();
    interpreter_->AddNodeWithParameters({}, {1}, nullptr, 0, var_handle_params,
                                        var_handle_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1, 0}, {}, nullptr, 0, nullptr,
                                        assign_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                        read_registration_, &node_index);
  }

  // Similar with `ConstructGraph`, but with static tensor shapes.
  void ConstructGraphWithKnownShape() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSvariable_ops_testDTcc mht_4(mht_4_v, 294, "", "./tensorflow/lite/kernels/variable_ops_test.cc", "ConstructGraphWithKnownShape");

    interpreter_.reset(new Interpreter);
    // Construct a graph like this:
    //   Input: %0
    //   Output: %2
    //   %1 = var_handle()
    //   variable_assign(%1, %0)
    //   %2 = read(%1)

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs({2}), kTfLiteOk);
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {2, 2},
                                               TfLiteQuantization());
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteResource, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {2, 2},
                                               TfLiteQuantization());
    int node_index;

    TfLiteVarHandleParams* var_handle_params = GetVarHandleParams();
    interpreter_->AddNodeWithParameters({}, {1}, nullptr, 0, var_handle_params,
                                        var_handle_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1, 0}, {}, nullptr, 0, nullptr,
                                        assign_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                        read_registration_, &node_index);
  }

  TfLiteRegistration* assign_registration_;
  TfLiteRegistration* read_registration_;
  TfLiteRegistration* var_handle_registration_;
  std::unique_ptr<Interpreter> interpreter_;
};

TEST_F(VariableOpsTest, TestAssignThenReadVariable) {
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1717;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  // Verify output.
  TfLiteTensor* output = interpreter_->tensor(2);
  ASSERT_EQ(output->dims->size, 0);
  EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
}

TEST_F(VariableOpsTest, TestAssignThenReadVariableWithKnownShape) {
  ConstructGraphWithKnownShape();
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1.0;
  GetTensorData<float>(input_data_index)[1] = 2.0;
  GetTensorData<float>(input_data_index)[2] = 3.0;
  GetTensorData<float>(input_data_index)[3] = 4.0;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  // Verify output.
  TfLiteTensor* output = interpreter_->tensor(2);
  ASSERT_EQ(output->dims->size, 2);
  EXPECT_EQ(GetTensorData<float>(output)[0], 1.0);
  EXPECT_EQ(GetTensorData<float>(output)[1], 2.0);
  EXPECT_EQ(GetTensorData<float>(output)[2], 3.0);
  EXPECT_EQ(GetTensorData<float>(output)[3], 4.0);
}

TEST_F(VariableOpsTest, TestReadVariableBeforeAssign) {
  ConstructInvalidGraph();
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1717;

  // Error because variable 2 is never initialized.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteError);
}

TEST_F(VariableOpsTest, TestReassignToDifferentSize) {
  // 1st invocation. The variable is assigned as a scalar.
  {
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    TfLiteTensor* input_data_index = interpreter_->tensor(0);
    GetTensorData<float>(input_data_index)[0] = 1717;
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    // Verify output.
    TfLiteTensor* output = interpreter_->tensor(2);
    ASSERT_EQ(output->dims->size, 0);
    EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
  }

  // 2nd invocation. The variable is assigned as a 1D vector with 2 elements.
  {
    interpreter_->ResizeInputTensor(0, {2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    TfLiteTensor* input_data_index = interpreter_->tensor(0);
    GetTensorData<float>(input_data_index)[0] = 1717;
    GetTensorData<float>(input_data_index)[1] = 2121;
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    // Verify output.
    TfLiteTensor* output = interpreter_->tensor(2);
    ASSERT_EQ(output->dims->size, 1);
    ASSERT_EQ(output->dims->data[0], 2);
    EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
    EXPECT_EQ(GetTensorData<float>(output)[1], 2121);
  }
}

}  // namespace
}  // namespace tflite
