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
class MHTracer_DTPStensorflowPSlitePSkernelsPScumsum_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScumsum_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScumsum_testDTcc() {
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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace {

template <typename T>
class CumsumOpModel : public SingleOpModel {
 public:
  CumsumOpModel(const TensorData& input, const TensorData& output,
                bool exclusive, bool reverse) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScumsum_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/cumsum_test.cc", "CumsumOpModel");

    input_ = AddInput(input);
    axis_ = AddInput({TensorType_INT32, {1}});

    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_CUMSUM, BuiltinOptions_CumsumOptions,
                 CreateCumsumOptions(builder_, exclusive, reverse).Union());

    BuildInterpreter({GetShape(input_), GetShape(axis_)});
  }

  int input() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScumsum_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/kernels/cumsum_test.cc", "input");
 return input_; }
  int axis() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScumsum_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/kernels/cumsum_test.cc", "axis");
 return axis_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }

 private:
  int input_;
  int axis_;
  int output_;
};

TEST(CumsumOpTest, SimpleIntTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 5, 11, 18, 26}));
}

TEST(CumsumOpTest, SimpleInt64Test) {
  CumsumOpModel<int64_t> m({TensorType_INT64, {2, 4}}, {TensorType_INT64, {}},
                           false, false);

  m.PopulateTensor<int64_t>(
      m.input(), {100000000001l, 100000000002l, 100000000003l, 100000000004l,
                  100000000005l, 100000000006l, 100000000007l, 100000000008l});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(
                                 {100000000001l, 200000000003l, 300000000006l,
                                  400000000010l, 100000000005l, 200000000011l,
                                  300000000018l, 400000000026l}));
}

TEST(CumsumOpTest, SimpleIntAxis0Test) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {0});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 2, 3, 4, 6, 8, 10, 12}));
}

TEST(CumsumOpTest, Simple1DIntTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {8}}, {TensorType_INT32, {}},
                           false, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {0});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({1, 3, 6, 10, 15, 21, 28, 36}));
}

TEST(CumsumOpTest, SimpleIntReverseTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           false, true);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({10, 9, 7, 4, 26, 21, 15, 8}));
}

TEST(CumsumOpTest, SimpleIntExclusiveTest) {
  CumsumOpModel<int32_t> m({TensorType_INT32, {2, 4}}, {TensorType_INT32, {}},
                           true, false);

  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              testing::ElementsAreArray({0, 1, 3, 6, 0, 5, 11, 18}));
}

TEST(CumsumOpTest, SimpleFloatTest) {
  CumsumOpModel<float> m({TensorType_FLOAT32, {2, 4}}, {TensorType_FLOAT32, {}},
                         false, false);

  m.PopulateTensor<float>(m.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.axis(), {1});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), testing::ElementsAreArray(
                                 ArrayFloatNear({1, 3, 6, 10, 5, 11, 18, 26})));
}

}  // namespace
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
