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
class MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc() {
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

#include "tensorflow/lite/profiling/profile_summarizer.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/profiling/buffered_profiler.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace profiling {

namespace {

const char* kOpName = "SimpleOpEval";

TfLiteStatus SimpleOpEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/profiling/profile_summarizer_test.cc", "SimpleOpEval");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, /*index=*/0, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, /*index=*/1, &input2));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, /*index=*/0, &output));

  int32_t* output_data = output->data.i32;
  *output_data = *(input1->data.i32) + *(input2->data.i32);
  return kTfLiteOk;
}

const char* SimpleOpProfilingString(const TfLiteContext* context,
                                    const TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/profiling/profile_summarizer_test.cc", "SimpleOpProfilingString");

  return "Profile";
}

TfLiteRegistration* RegisterSimpleOp() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc mht_2(mht_2_v, 233, "", "./tensorflow/lite/profiling/profile_summarizer_test.cc", "RegisterSimpleOp");

  static TfLiteRegistration registration = {
      nullptr,        nullptr, nullptr,
      SimpleOpEval,   nullptr, tflite::BuiltinOperator_CUSTOM,
      "SimpleOpEval", 1};
  return &registration;
}

TfLiteRegistration* RegisterSimpleOpWithProfilingDetails() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc mht_3(mht_3_v, 244, "", "./tensorflow/lite/profiling/profile_summarizer_test.cc", "RegisterSimpleOpWithProfilingDetails");

  static TfLiteRegistration registration = {nullptr,
                                            nullptr,
                                            nullptr,
                                            SimpleOpEval,
                                            SimpleOpProfilingString,
                                            tflite::BuiltinOperator_CUSTOM,
                                            kOpName,
                                            1};
  return &registration;
}

class SimpleOpModel : public SingleOpModel {
 public:
  void Init(const std::function<TfLiteRegistration*()>& registration);
  tflite::Interpreter* GetInterpreter() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc mht_4(mht_4_v, 262, "", "./tensorflow/lite/profiling/profile_summarizer_test.cc", "GetInterpreter");
 return interpreter_.get(); }
  void SetInputs(int32_t x, int32_t y) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc mht_5(mht_5_v, 266, "", "./tensorflow/lite/profiling/profile_summarizer_test.cc", "SetInputs");

    PopulateTensor(inputs_[0], {x});
    PopulateTensor(inputs_[1], {y});
  }
  int32_t GetOutput() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc mht_6(mht_6_v, 273, "", "./tensorflow/lite/profiling/profile_summarizer_test.cc", "GetOutput");
 return ExtractVector<int32_t>(output_)[0]; }

 private:
  int inputs_[2];
  int output_;
};

void SimpleOpModel::Init(
    const std::function<TfLiteRegistration*()>& registration) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc mht_7(mht_7_v, 284, "", "./tensorflow/lite/profiling/profile_summarizer_test.cc", "SimpleOpModel::Init");

  inputs_[0] = AddInput({TensorType_INT32, {1}});
  inputs_[1] = AddInput({TensorType_INT32, {1}});
  output_ = AddOutput({TensorType_INT32, {}});
  SetCustomOp(kOpName, {}, registration);
  BuildInterpreter({GetShape(inputs_[0]), GetShape(inputs_[1])});
}

TEST(ProfileSummarizerTest, Empty) {
  ProfileSummarizer summarizer;
  std::string output = summarizer.GetOutputString();
  EXPECT_GT(output.size(), 0);
}

TEST(ProfileSummarizerTest, Interpreter) {
  BufferedProfiler profiler(1024);
  SimpleOpModel m;
  m.Init(RegisterSimpleOp);
  auto interpreter = m.GetInterpreter();
  interpreter->SetProfiler(&profiler);
  profiler.StartProfiling();
  m.SetInputs(1, 2);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // 3 = 1 + 2
  EXPECT_EQ(m.GetOutput(), 3);
  profiler.StopProfiling();
  ProfileSummarizer summarizer;
  auto events = profiler.GetProfileEvents();
  EXPECT_EQ(2, events.size());
  summarizer.ProcessProfiles(profiler.GetProfileEvents(), *interpreter);
  auto output = summarizer.GetOutputString();
  // TODO(shashishekhar): Add a better test here.
  ASSERT_TRUE(output.find("SimpleOpEval") != std::string::npos) << output;
  ASSERT_TRUE(output.find("Invoke") == std::string::npos) << output;  // NOLINT
}

TEST(ProfileSummarizerTest, InterpreterPlusProfilingDetails) {
  BufferedProfiler profiler(1024);
  SimpleOpModel m;
  m.Init(RegisterSimpleOpWithProfilingDetails);
  auto interpreter = m.GetInterpreter();
  interpreter->SetProfiler(&profiler);
  profiler.StartProfiling();
  m.SetInputs(1, 2);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // 3 = 1 + 2
  EXPECT_EQ(m.GetOutput(), 3);
  profiler.StopProfiling();
  ProfileSummarizer summarizer;
  auto events = profiler.GetProfileEvents();
  EXPECT_EQ(2, events.size());
  summarizer.ProcessProfiles(profiler.GetProfileEvents(), *interpreter);
  auto output = summarizer.GetOutputString();
  // TODO(shashishekhar): Add a better test here.
  ASSERT_TRUE(output.find("SimpleOpEval/Profile") != std::string::npos)
      << output;
}

// A simple test that performs `ADD` if condition is true, and `MUL` otherwise.
// The computation is: `cond ? a + b : a * b`.
class ProfileSummarizerIfOpTest : public subgraph_test_util::ControlFlowOpTest {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSprofilingPSprofile_summarizer_testDTcc mht_8(mht_8_v, 349, "", "./tensorflow/lite/profiling/profile_summarizer_test.cc", "SetUp");

    AddSubgraphs(2);
    builder_->BuildAddSubgraph(interpreter_->subgraph(1));
    builder_->BuildMulSubgraph(interpreter_->subgraph(2));
    builder_->BuildIfSubgraph(&interpreter_->primary_subgraph());

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1, 2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});
    subgraph_test_util::FillIntTensor(
        interpreter_->tensor(interpreter_->inputs()[2]), {1, 2});
  }
};

TEST_F(ProfileSummarizerIfOpTest, TestIfTrue) {
  BufferedProfiler profiler(1024);
  interpreter_->SetProfiler(&profiler);

  interpreter_->typed_input_tensor<bool>(0)[0] = true;
  profiler.StartProfiling();
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  profiler.StopProfiling();
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  subgraph_test_util::CheckIntTensor(output, {1, 2}, {6, 9});

  auto events = profiler.GetProfileEvents();
  EXPECT_EQ(4, events.size());
  int event_count_of_subgraph_zero = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 0; });
  int event_count_of_subgraph_one = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 1; });
  int event_count_of_subgraph_two = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 2; });
  EXPECT_EQ(2, event_count_of_subgraph_zero);
  EXPECT_EQ(2, event_count_of_subgraph_one);
  EXPECT_EQ(0, event_count_of_subgraph_two);
}

TEST_F(ProfileSummarizerIfOpTest, TestIfFalse) {
  BufferedProfiler profiler(1024);
  interpreter_->SetProfiler(&profiler);

  interpreter_->typed_input_tensor<bool>(0)[0] = false;
  profiler.StartProfiling();
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  profiler.StopProfiling();
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  subgraph_test_util::CheckIntTensor(output, {1, 2}, {5, 14});

  auto events = profiler.GetProfileEvents();
  EXPECT_EQ(4, events.size());
  int event_count_of_subgraph_zero = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 0; });
  int event_count_of_subgraph_one = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 1; });
  int event_count_of_subgraph_two = std::count_if(
      events.begin(), events.end(),
      [](auto event) { return event->extra_event_metadata == 2; });
  EXPECT_EQ(2, event_count_of_subgraph_zero);
  EXPECT_EQ(0, event_count_of_subgraph_one);
  EXPECT_EQ(2, event_count_of_subgraph_two);
}

}  // namespace
}  // namespace profiling
}  // namespace tflite
