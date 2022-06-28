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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_testDTcc() {
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
#include "tensorflow/lite/delegates/flex/delegate.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/flex/test_util.h"
#include "tensorflow/lite/shared_library.h"

namespace tflite {
namespace flex {
namespace {

using ::testing::ElementsAre;

class DelegateTest : public testing::FlexModelTest {
 public:
  DelegateTest() : delegate_(FlexDelegate::Create()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/delegates/flex/delegate_test.cc", "DelegateTest");

    flex_delegate_ = static_cast<FlexDelegate*>(delegate_->data_);
    interpreter_.reset(new Interpreter(&error_reporter_));
  }

  ~DelegateTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/lite/delegates/flex/delegate_test.cc", "~DelegateTest");

    // The delegate needs to be destructed after the interpreter because the
    // interpreter references data contained in the delegate.
    interpreter_.reset();
    delegate_.reset();
  }

  void ConfigureDelegate() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_testDTcc mht_2(mht_2_v, 220, "", "./tensorflow/lite/delegates/flex/delegate_test.cc", "ConfigureDelegate");

    interpreter_->SetCancellationFunction(flex_delegate_,
                                          FlexDelegate::HasCancelled);
    ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
              kTfLiteOk);
  }

  void Cancel() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/lite/delegates/flex/delegate_test.cc", "Cancel");
 flex_delegate_->Cancel(); }

 private:
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_;
  FlexDelegate* flex_delegate_;
};

TEST_F(DelegateTest, FullGraph) {
  // Define the graph.
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfOp(testing::kMul, {6, 7}, {8});

  // Apply the delegate.
  ConfigureDelegate();

  // Define inputs.
  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(2, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));
  ASSERT_EQ(GetType(8), kTfLiteFloat32);
}

TEST_F(DelegateTest, NonFloatTypeInference) {
  AddTensors(3, {0, 1}, {2}, kTfLiteInt32, {2});

  AddTfOp(testing::kAdd, {0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2});
  SetTypedValues<int>(0, {1, 2, 3, 4});
  SetShape(1, {2, 2});
  SetTypedValues<int>(1, {4, 3, 2, 1});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(2), ElementsAre(2, 2));
  ASSERT_THAT(GetTypedValues<int>(2), ElementsAre(5, 5, 5, 5));
  ASSERT_EQ(GetType(2), kTfLiteInt32);
}

TEST_F(DelegateTest, StringInference) {
  AddTensors(3, {0, 1}, {2}, kTfLiteString, {2});

  AddTfOp(testing::kAdd, {0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2});
  SetStringValues(0, {"1", "2", "3", "4"});
  SetShape(1, {2, 2});
  SetStringValues(1, {"4", "3", "2", "1"});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(2), ElementsAre(2, 2));
  ASSERT_THAT(GetStringValues(2), ElementsAre("14", "23", "32", "41"));
  ASSERT_EQ(GetType(2), kTfLiteString);
}

TEST_F(DelegateTest, MixedGraph) {
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfLiteMulOp({6, 7}, {8});

  ConfigureDelegate();

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(2, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));
}

TEST_F(DelegateTest, SplitGraph) {
  AddTensors(10, {0}, {9}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kAdd, {1, 2}, {3});
  AddTfOp(testing::kUnpack, {3}, {4, 5});

  AddTfLiteMulOp({4, 5}, {6});

  AddTfOp(testing::kUnpack, {6}, {7, 8});
  AddTfOp(testing::kAdd, {7, 8}, {9});

  ConfigureDelegate();

  SetShape(0, {2, 2, 2, 1});
  SetValues(0, {3.0f, 1.0f, 0.5f, -1.0f, 0.0f, 1.0f, 1.5f, 3.0f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(9), ElementsAre(1));
  ASSERT_THAT(GetValues(9), ElementsAre(10.0f));
}

TEST_F(DelegateTest, OnlyTFLite) {
  // Only TFLite single op model.
  AddTensors(10, {0, 1}, {2}, kTfLiteFloat32, {3});
  AddTfLiteMulOp({0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(1, {2, 2, 1});
  SetValues(1, {1.0f, 2.0f, 3.0f, 4.0f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(2), ElementsAre(2, 2, 1));
  ASSERT_THAT(GetValues(2), ElementsAre(1.1f, 4.4f, 9.9f, 17.6f));
}

TEST_F(DelegateTest, MultipleInvokeCalls) {
  // Call Invoke() multiple times on the same model.
  AddTensors(10, {0, 1}, {2}, kTfLiteFloat32, {3});
  AddTfLiteMulOp({0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(1, {2, 2, 1});
  SetValues(1, {1.0f, 2.0f, 3.0f, 4.0f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(2), ElementsAre(2, 2, 1));
  ASSERT_THAT(GetValues(2), ElementsAre(1.1f, 4.4f, 9.9f, 17.6f));

  SetShape(0, {2, 2, 1});
  SetValues(1, {4.0f, 3.0f, 2.0f, 1.0f});
  SetShape(1, {2, 2, 1});
  SetValues(0, {4.4f, 3.3f, 2.2f, 1.1f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(2), ElementsAre(2, 2, 1));
  ASSERT_THAT(GetValues(2), ElementsAre(17.6f, 9.9f, 4.4f, 1.1f));
}

TEST_F(DelegateTest, MultipleInterpretersSameDelegate) {
  // Build a graph, configure the delegate and set inputs.
  {
    AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});
    AddTfOp(testing::kUnpack, {0}, {1, 2});
    AddTfOp(testing::kUnpack, {3}, {4, 5});
    AddTfOp(testing::kAdd, {1, 4}, {6});
    AddTfOp(testing::kAdd, {2, 5}, {7});
    AddTfOp(testing::kMul, {6, 7}, {8});
    ConfigureDelegate();
    SetShape(0, {2, 2, 1});
    SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
    SetShape(3, {2, 2, 1});
    SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});
  }

  // Create a new interpreter, inject into the test framework and build
  // a different graph using the *same* delegate.
  std::unique_ptr<Interpreter> interpreter(new Interpreter(&error_reporter_));
  interpreter_.swap(interpreter);
  {
    AddTensors(10, {0}, {9}, kTfLiteFloat32, {3});
    AddTfOp(testing::kUnpack, {0}, {1, 2});
    AddTfOp(testing::kAdd, {1, 2}, {3});
    AddTfOp(testing::kUnpack, {3}, {4, 5});
    AddTfLiteMulOp({4, 5}, {6});
    AddTfOp(testing::kUnpack, {6}, {7, 8});
    AddTfOp(testing::kAdd, {7, 8}, {9});
    ConfigureDelegate();
    SetShape(0, {2, 2, 2, 1});
    SetValues(0, {3.0f, 1.0f, 0.5f, -1.0f, 0.0f, 1.0f, 1.5f, 3.0f});
  }

  // Swap back in the first interpreter and validate inference.
  interpreter_.swap(interpreter);
  {
    ASSERT_TRUE(Invoke());
    EXPECT_THAT(GetShape(8), ElementsAre(2, 1));
    EXPECT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));
  }

  // Swap in the second interpreter and validate inference.
  interpreter_.swap(interpreter);
  {
    ASSERT_TRUE(Invoke());
    EXPECT_THAT(GetShape(9), ElementsAre(1));
    EXPECT_THAT(GetValues(9), ElementsAre(10.0f));
  }
}

TEST_F(DelegateTest, SingleThreaded) {
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});
  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfOp(testing::kMul, {6, 7}, {8});

  // Explicitly disable multi-threading before installing the delegate.
  interpreter_->SetNumThreads(1);
  ConfigureDelegate();

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  // Invocation should behave as expected.
  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(2, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));
  ASSERT_EQ(GetType(8), kTfLiteFloat32);
}

TEST_F(DelegateTest, MultiThreaded) {
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});
  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfOp(testing::kMul, {6, 7}, {8});

  // Explicitly enable multi-threading before installing the delegate.
  interpreter_->SetNumThreads(4);
  ConfigureDelegate();

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  // Invocation should behave as expected.
  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(2, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));
  ASSERT_EQ(GetType(8), kTfLiteFloat32);
}

#if !defined(__ANDROID__)
TEST_F(DelegateTest, TF_AcquireFlexDelegate) {
  auto TF_AcquireFlexDelegate =
      reinterpret_cast<Interpreter::TfLiteDelegatePtr (*)()>(
          SharedLibrary::GetSymbol("TF_AcquireFlexDelegate"));
  ASSERT_TRUE(TF_AcquireFlexDelegate);
  auto delegate_ptr = TF_AcquireFlexDelegate();
  ASSERT_TRUE(delegate_ptr != nullptr);
}
#endif  // !defined(__ANDROID__)

TEST_F(DelegateTest, StaticOutput) {
  // Define the graph with input, output shapes of [2].
  AddTensors(7, {0, 1, 2, 3}, {6}, kTfLiteFloat32, {2});

  AddTfOp(testing::kAdd, {0, 2}, {4});
  AddTfOp(testing::kAdd, {1, 3}, {5});
  AddTfOp(testing::kMul, {4, 5}, {6});

  // Apply the delegate.
  ConfigureDelegate();

  // Define inputs which matech with the original shapes.
  SetShape(0, {2});
  SetShape(1, {2});
  SetShape(2, {2});
  SetShape(3, {2});
  SetValues(0, {1.1f, 2.2f});
  SetValues(1, {3.3f, 4.4f});
  SetValues(2, {1.1f, 2.2f});
  SetValues(3, {3.3f, 4.4f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(6), ElementsAre(2));
  ASSERT_THAT(GetValues(6), ElementsAre(14.52f, 38.72f));
  ASSERT_EQ(GetType(6), kTfLiteFloat32);
  // Since shapes are consistent, static output tensor is used.
  ASSERT_FALSE(IsDynamicTensor(6));
}

TEST_F(DelegateTest, StaticOutputRFFT) {
  // Define the graph with input, output shapes of [3, 257].
  AddTensors(4, {0, 1}, {3}, kTfLiteFloat32, {3, 257});
  int32_t rfft_length[] = {512};
  SetConstTensor(1, {1}, kTfLiteInt32,
                 reinterpret_cast<const char*>(&rfft_length),
                 sizeof(rfft_length));

  AddTfOp(testing::kRfft, {0, 1}, {2});
  AddTfOp(testing::kImag, {2}, {3});

  // Apply the delegate.
  ConfigureDelegate();

  // Define inputs.
  SetShape(0, {3, 512});
  SetValues(0, std::vector<float>(3 * 512, 1.0f));

  ASSERT_TRUE(Invoke());

  ASSERT_EQ(GetType(3), kTfLiteFloat32);
  // Since shapes are consistent, static output tensor is used.
  ASSERT_FALSE(IsDynamicTensor(3));
}

TEST_F(DelegateTest, DynamicOutputAfterReshape) {
  // Define the graph.
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfOp(testing::kMul, {6, 7}, {8});

  // Apply the delegate.
  ConfigureDelegate();

  // Define inputs with reshape.
  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(2, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));
  ASSERT_EQ(GetType(8), kTfLiteFloat32);
  // Since shapes are inconsistent, dynamic output tensor is used.
  ASSERT_TRUE(IsDynamicTensor(8));
}

TEST_F(DelegateTest, TestCancellation1) {
  AddTensors(3, {0, 1}, {2}, kTfLiteInt32, {2});

  AddTfOp(testing::kAdd, {0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2});
  SetTypedValues<int>(0, {1, 2, 3, 4});
  SetShape(1, {2, 2});
  SetTypedValues<int>(1, {4, 3, 2, 1});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(2), ElementsAre(2, 2));
  ASSERT_THAT(GetTypedValues<int>(2), ElementsAre(5, 5, 5, 5));
  ASSERT_EQ(GetType(2), kTfLiteInt32);

  Cancel();
  // Op should be cancelled.
  ASSERT_FALSE(Invoke());
  // TODO(b/205345340): We shouldn't do raw string matching here. Instead we
  // need to introduce fine-grained error codes to represent cancellation
  // status.
  EXPECT_EQ(error_reporter_.error_messages(),
            "Client requested cancel during Invoke()");
}

TEST_F(DelegateTest, TestCancellation2) {
  // Define the graph.
  AddTensors(2, {0}, {1}, kTfLiteBool, {1});

  // We need an op that checks the CancellationManager status.
  AddTfOp(testing::kLoopCond, {0}, {1});

  // Apply the delegate.
  ConfigureDelegate();

  // Define inputs.
  SetShape(0, {1});

  ASSERT_TRUE(Invoke());

  Cancel();
  // Op should be cancelled.
  ASSERT_FALSE(Invoke());
  // TODO(b/205345340): We shouldn't do raw string matching here. Instead we
  // need to introduce fine-grained error codes to represent cancellation
  // status.
  EXPECT_EQ(error_reporter_.error_messages(),
            "Client requested cancel during Invoke()");
}

TEST_F(DelegateTest, TestCancellationTwoThreads) {
  AddTensors(3, {0, 1}, {2}, kTfLiteInt32, {2});

  AddTfOp(testing::kAdd, {0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2});
  SetTypedValues<int>(0, {1, 2, 3, 4});
  SetShape(1, {2, 2});
  SetTypedValues<int>(1, {4, 3, 2, 1});

  std::thread invoke_thread([this]() {
    bool result = true;
    result = this->Invoke();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    result = this->Invoke();
    ASSERT_FALSE(result);
    // TODO(b/205345340): Check returned error code.
  });

  std::thread cancel_thread([this]() { this->Cancel(); });

  invoke_thread.join();
  cancel_thread.join();
}

}  // namespace
}  // namespace flex
}  // namespace tflite
