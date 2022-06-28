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
class MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_args_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_args_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_args_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {
using ::testing::ElementsAreArray;

template <class ShapeType = int32_t>
class BroadcastArgsOpModel : public SingleOpModel {
 public:
  BroadcastArgsOpModel(std::initializer_list<ShapeType> input1,
                       std::initializer_list<ShapeType> input2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_args_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/kernels/broadcast_args_test.cc", "BroadcastArgsOpModel");

    int input1_length = input1.size();
    int input2_length = input2.size();
    shape1_ = AddInput({GetTensorType<ShapeType>(), {input1_length}});
    shape2_ = AddInput({GetTensorType<ShapeType>(), {input2_length}});
    output_ = AddOutput(GetTensorType<ShapeType>());
    SetBuiltinOp(BuiltinOperator_BROADCAST_ARGS, BuiltinOptions_NONE, 0);
    BuildInterpreter({{input1_length}, {input2_length}});
    if (input1.size() > 0) SetInput1(input1);
    if (input2.size() > 0) SetInput2(input2);
  }

  void SetInput1(std::initializer_list<ShapeType> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_args_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/kernels/broadcast_args_test.cc", "SetInput1");

    PopulateTensor(shape1_, data);
  }

  void SetInput2(std::initializer_list<ShapeType> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_args_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/kernels/broadcast_args_test.cc", "SetInput2");

    PopulateTensor(shape2_, data);
  }

  std::vector<ShapeType> GetOutput() {
    return ExtractVector<ShapeType>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int shape1_;
  int shape2_;
  int output_;
};

template <typename T>
class BroadcastArgsOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<int64_t, int32_t>;
TYPED_TEST_SUITE(BroadcastArgsOpTest, DataTypes);

#ifdef GTEST_HAS_DEATH_TEST
TYPED_TEST(BroadcastArgsOpTest, ShapeNotBroadcastable) {
  BroadcastArgsOpModel<TypeParam> m({2, 3, 4, 4}, {2, 2});
  EXPECT_DEATH(ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk), "");
}
#endif

TYPED_TEST(BroadcastArgsOpTest, BroadcastArgsWithScalar) {
  BroadcastArgsOpModel<TypeParam> m({}, {2, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 4}));
}

TYPED_TEST(BroadcastArgsOpTest, BroadcastArgsDifferentDims) {
  BroadcastArgsOpModel<TypeParam> m({1}, {2, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 4}));
}

TYPED_TEST(BroadcastArgsOpTest, BroadcastArgsSameDims) {
  BroadcastArgsOpModel<TypeParam> m({1, 4, 6, 3, 1, 5}, {4, 4, 1, 3, 4, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 4, 6, 3, 4, 5}));
}

TYPED_TEST(BroadcastArgsOpTest, BroadcastArgsComplex) {
  BroadcastArgsOpModel<TypeParam> m({6, 3, 1, 5}, {4, 4, 1, 3, 4, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 4, 6, 3, 4, 5}));
}

}  // namespace
}  // namespace tflite
