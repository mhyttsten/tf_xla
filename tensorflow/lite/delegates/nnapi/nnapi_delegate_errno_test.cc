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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc() {
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

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace {

class SingleOpModelWithNNAPI : public SingleOpModel {
 public:
  explicit SingleOpModelWithNNAPI(const NnApi* nnapi) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_errno_test.cc", "SingleOpModelWithNNAPI");

    options_.disallow_nnapi_cpu = false;
    stateful_delegate_.reset(new StatefulNnApiDelegate(nnapi, options_));
    this->SetDelegate(stateful_delegate_.get());
  }

  StatefulNnApiDelegate* GetDelegate() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_errno_test.cc", "GetDelegate");
 return stateful_delegate_.get(); }

  void SetBufferHandle(int index, TfLiteBufferHandle handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc mht_2(mht_2_v, 216, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_errno_test.cc", "SetBufferHandle");

    interpreter_->SetBufferHandle(index, handle, stateful_delegate_.get());
  }

 private:
  std::unique_ptr<StatefulNnApiDelegate> stateful_delegate_;
  StatefulNnApiDelegate::Options options_;
};

class FloatAddOpModel : public SingleOpModelWithNNAPI {
 public:
  FloatAddOpModel(const NnApi* nnapi, const TensorData& input1,
                  const TensorData& input2, const TensorData& output,
                  ActivationFunctionType activation_type,
                  bool allow_fp32_relax_to_fp16 = false)
      : SingleOpModelWithNNAPI(nnapi) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc mht_3(mht_3_v, 234, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_errno_test.cc", "FloatAddOpModel");

    Init(input1, input2, output, activation_type, allow_fp32_relax_to_fp16);
  }

  int input1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc mht_4(mht_4_v, 241, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_errno_test.cc", "input1");
 return input1_; }
  int input2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc mht_5(mht_5_v, 245, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_errno_test.cc", "input2");
 return input2_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;

 private:
  // Performs initialization logic shared across all constructors.
  void Init(const TensorData& input1, const TensorData& input2,
            const TensorData& output, ActivationFunctionType activation_type,
            bool allow_fp32_relax_to_fp16 = false) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_errno_testDTcc mht_6(mht_6_v, 261, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_errno_test.cc", "Init");

    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)}, /*num_threads=*/-1,
                     allow_fp32_relax_to_fp16, /*apply_delegate=*/true);
  }
};

struct NnApiErrnoTest : ::tflite::delegate::nnapi::NnApiDelegateMockTest {};

TEST_F(NnApiErrnoTest, IsZeroWhenNoErrorOccurs) {
  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), 0);
}

TEST_F(NnApiErrnoTest, HasTheStatusOfTheNnApiCallFailedCallingInit) {
  nnapi_mock_->ExecutionCreateReturns<8>();

  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);

  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});

  EXPECT_EQ(m.InvokeUnchecked(), kTfLiteError);
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), 8);
}

TEST_F(NnApiErrnoTest, HasTheStatusOfTheNnApiCallFailedCallingInvoke) {
  nnapi_mock_->ModelFinishReturns<-4>();

  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);

  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});

  // Failure is detected and the delegate is disabled.
  // Execution runs without it and succeeds
  EXPECT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // The delegate should store the value of the failure
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), -4);
}

TEST_F(NnApiErrnoTest, ErrnoIsResetWhenRestoringDelegateForModel) {
  nnapi_mock_->ModelFinishReturns<-4>();

  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);

  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});

  // Failure is detected and the delegate is disabled.
  // Execution runs without it and succeeds
  EXPECT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // The delegate should store the value of the failure
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), -4);

  nnapi_mock_->ModelFinishReturns<0>();

  // Need to restore the delegate since it was disabled because of the
  // previous failure.
  m.ApplyDelegate();
  EXPECT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  // The error is still the last one recorded
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), 0);
}

TEST_F(NnApiErrnoTest, ErrnoIsUpdatedInCaseOfAnotherFailure) {
  nnapi_mock_->ModelFinishReturns<-4>();

  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);

  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});

  // Failure is detected and the delegate is disabled.
  // Execution runs without it and succeeds
  EXPECT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // The delegate should store the value of the failure
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), -4);

  nnapi_mock_->ModelFinishReturns<-5>();

  // Need to restore the delegate since it was disabled because of the
  // previous failure.
  m.ApplyDelegate();
  EXPECT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  // The error is still the last one recorded
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), -5);
}

}  // namespace
}  // namespace tflite
