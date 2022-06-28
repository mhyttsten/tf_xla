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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_nnapi_failure_handling_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_nnapi_failure_handling_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_nnapi_failure_handling_testDTcc() {
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
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace {

struct NnApiFailureHandlingTest
    : ::tflite::delegate::nnapi::NnApiDelegateMockTest {};

// This is a model with two ops:
//
//  input1 ---->
//                ADD --
//  input2   -->        |
//                       -->
//                          SUB --> output
//  input3 ---------------->
//
class AddSubOpsAcceleratedModel : public MultiOpModel {
 public:
  AddSubOpsAcceleratedModel(const TensorData& input1, const TensorData& input2,
                            const TensorData& input3, const TensorData& output,
                            ActivationFunctionType activation_type,
                            const NnApi* nnapi,
                            const std::string& accelerator_name,
                            bool allow_fp32_relax_to_fp16 = false)
      : MultiOpModel() {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("accelerator_name: \"" + accelerator_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_nnapi_failure_handling_testDTcc mht_0(mht_0_v, 231, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_nnapi_failure_handling_test.cc", "AddSubOpsAcceleratedModel");

    StatefulNnApiDelegate::Options options;
    options.accelerator_name = accelerator_name.c_str();
    stateful_delegate_.reset(new StatefulNnApiDelegate(nnapi, options));
    SetDelegate(stateful_delegate_.get());
    Init(input1, input2, input3, output, activation_type,
         allow_fp32_relax_to_fp16);
  }

  int input1() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_nnapi_failure_handling_testDTcc mht_1(mht_1_v, 243, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_nnapi_failure_handling_test.cc", "input1");
 return input1_; }
  int input2() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_nnapi_failure_handling_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_nnapi_failure_handling_test.cc", "input2");
 return input2_; }
  int input3() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_nnapi_failure_handling_testDTcc mht_3(mht_3_v, 251, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_nnapi_failure_handling_test.cc", "input3");
 return input3_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int input3_;
  int output_;

 private:
  std::unique_ptr<StatefulNnApiDelegate> stateful_delegate_;

  // Performs initialization logic shared across all constructors.
  void Init(const TensorData& input1, const TensorData& input2,
            const TensorData& input3, const TensorData& output,
            ActivationFunctionType activation_type,
            bool allow_fp32_relax_to_fp16 = false) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_nnapi_failure_handling_testDTcc mht_4(mht_4_v, 271, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_nnapi_failure_handling_test.cc", "Init");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    input3_ = AddInput(input3);
    const int add_output = AddInnerTensor<float>(output);
    output_ = AddOutput(output);
    AddBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union(),
                 {input1_, input2_}, {add_output});
    AddBuiltinOp(BuiltinOperator_SUB, BuiltinOptions_SubOptions,
                 CreateSubOptions(builder_, activation_type).Union(),
                 {add_output, input3_}, {output_});
    BuildInterpreter({GetShape(input1_), GetShape(input2_), GetShape(input3_)},
                     /*num_threads=*/-1, allow_fp32_relax_to_fp16,
                     /*apply_delegate=*/false);
    ApplyDelegate();
  }
};

TEST_F(NnApiFailureHandlingTest, DelegateShouldFailImmediatelyIfUnableToAddOp) {
  static int add_op_invocation_count = 0;
  nnapi_mock_->SetNnapiSupportedDevice("test-device");

  nnapi_mock_->StubAddOperationWith(
      [](ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
         uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
         const uint32_t* outputs) -> int {
        ++add_op_invocation_count;
        return ANEURALNETWORKS_BAD_DATA;
      });

  AddSubOpsAcceleratedModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
      ActivationFunctionType_NONE, nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device");
  std::vector<float> input1{-2.0, 0.2, 0.7, 0.9};
  std::vector<float> input2{0.1, 0.2, 0.3, 0.5};
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  m.PopulateTensor<float>(m.input3(), input2);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_EQ(add_op_invocation_count, 1);
}

}  // namespace
}  // namespace tflite
