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
class MHTracer_DTPStensorflowPSlitePSkernelsPSslice_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSslice_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSslice_testDTcc() {
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
#include <stdint.h>

#include <initializer_list>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename input_type, typename index_type>
class SliceOpModel : public SingleOpModel {
 public:
  SliceOpModel(std::initializer_list<int> input_shape,
               std::initializer_list<int> begin_shape,
               std::initializer_list<index_type> begin_data,
               std::initializer_list<int> size_shape,
               std::initializer_list<index_type> size_data,
               TensorType tensor_index_type, TensorType tensor_input_type,
               TestType input_tensor_types) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSslice_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/kernels/slice_test.cc", "SliceOpModel");

    input_ = AddInput(tensor_input_type);
    if (input_tensor_types == TestType::kDynamic) {
      begin_ = AddInput(tensor_index_type);
      size_ = AddInput(tensor_index_type);
    } else {
      begin_ =
          AddConstInput(GetTensorType<index_type>(), begin_data, begin_shape);
      size_ = AddConstInput(GetTensorType<index_type>(), size_data, size_shape);
    }
    output_ = AddOutput(tensor_input_type);
    SetBuiltinOp(BuiltinOperator_SLICE, BuiltinOptions_SliceOptions,
                 CreateSliceOptions(builder_).Union());
    BuildInterpreter({input_shape, begin_shape, size_shape});

    if (input_tensor_types == TestType::kDynamic) {
      PopulateTensor<index_type>(begin_, begin_data);
      PopulateTensor<index_type>(size_, size_data);
    }
  }

  void SetInput(std::initializer_list<input_type> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSslice_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/lite/kernels/slice_test.cc", "SetInput");

    PopulateTensor<input_type>(input_, data);
  }
  void SetStringInput(std::vector<string> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSslice_testDTcc mht_2(mht_2_v, 244, "", "./tensorflow/lite/kernels/slice_test.cc", "SetStringInput");

    PopulateStringTensor(input_, data);
  }

  std::vector<input_type> GetOutput() {
    return ExtractVector<input_type>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int begin_;
  int size_;
  int output_;
};

class SliceOpTest : public ::testing::TestWithParam<TestType> {};

TEST_P(SliceOpTest, In1D) {
  SliceOpModel<float, int32_t> m({4}, {1}, {1}, {1}, {2}, TensorType_INT32,
                                 TensorType_FLOAT32, GetParam());
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TEST_P(SliceOpTest, In2D) {
  SliceOpModel<float, int32_t> m({2, 3}, {2}, {1, 0}, {2}, {1, 2},
                                 TensorType_INT32, TensorType_FLOAT32,
                                 GetParam());
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5}));
}

TEST_P(SliceOpTest, In3D) {
  SliceOpModel<float, int32_t> m({2, 3, 2}, {3}, {0, 0, 0}, {3}, {2, 3, 2},
                                 TensorType_INT32, TensorType_FLOAT32,
                                 GetParam());
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST_P(SliceOpTest, In5D) {
  SliceOpModel<float, int32_t> m({5, 1, 1, 1, 1}, {5}, {1, 0, 0, 0, 0}, {5},
                                 {3, 1, 1, 1, 1}, TensorType_INT32,
                                 TensorType_FLOAT32, GetParam());
  m.SetInput({1, 2, 3, 4, 5});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

TEST_P(SliceOpTest, InputFloat) {
  SliceOpModel<float, int32_t> m({4, 1, 1, 1}, {4}, {1, 0, 0, 0}, {4},
                                 {3, 1, 1, 1}, TensorType_INT32,
                                 TensorType_FLOAT32, GetParam());
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

TEST_P(SliceOpTest, IndexInt64) {
  SliceOpModel<float, int64_t> m({4, 1, 1, 1}, {4}, {1, 0, 0, 0}, {4},
                                 {3, 1, 1, 1}, TensorType_INT64,
                                 TensorType_FLOAT32, GetParam());
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

// See these test cases under:
// https://www.tensorflow.org/versions/master/api_docs/python/tf/slice
TEST_P(SliceOpTest, InputInteger1) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {1, 1, 3, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3}));
}

TEST_P(SliceOpTest, InputInteger2) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {1, 2, 3, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 4, 4, 4}));
}

TEST_P(SliceOpTest, InputInteger3) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, 3, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SizeMinus1) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, BeginNonZeroSizeMinus1Axis1) {
  SliceOpModel<int32_t, int32_t> m({3, 3, 2, 1}, {4}, {1, 1, 0, 0}, {4},
                                   {2, -1, 1, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6, 8, 9}));
}

TEST_P(SliceOpTest, BeginNonZeroSizeMinus1Axis2) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 1, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 5, 5}));
}

TEST_P(SliceOpTest, BeginNonZeroSizeMinus1Axis3) {
  SliceOpModel<int32_t, int32_t> m({3, 1, 2, 3}, {4}, {1, 0, 0, 1}, {4},
                                   {2, 1, 1, -1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 5, 5}));
}

TEST_P(SliceOpTest, SliceUint8) {
  SliceOpModel<uint8_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_UINT8, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceInt8) {
  SliceOpModel<int8_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                  {2, 1, -1, 1}, TensorType_INT32,
                                  TensorType_INT8, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceInt16) {
  SliceOpModel<int16_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_INT16, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceString) {
  SliceOpModel<string, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                  {2, 1, -1, 1}, TensorType_INT32,
                                  TensorType_STRING, GetParam());
  m.SetStringInput({"0,0,0,0", "0,0,1,0", "0,0,2,0",  //
                    "0,1,0,0", "0,1,1,0", "0,1,2,0",  //
                    "1,0,0,0", "1,0,1,0", "1,0,2,0",  //
                    "1,1,0,0", "1,1,1,0", "1,1,2,0",  //
                    "2,0,0,0", "2,0,1,0", "2,0,2,0",  //
                    "2,1,0,0", "2,1,1,0", "2,1,2,0"});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({"1,0,0,0", "1,0,1,0", "1,0,2,0",  //
                                "2,0,0,0", "2,0,1,0", "2,0,2,0"}));
}

TEST_P(SliceOpTest, SliceInt64) {
  SliceOpModel<int64_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_INT64, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceBool) {
  SliceOpModel<bool, int32_t> m({2, 3}, {2}, {1, 0}, {2}, {-1, 2},
                                TensorType_INT32, TensorType_BOOL, GetParam());
  m.SetInput({true, false, true, false, true, true});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({false, true}));
}

INSTANTIATE_TEST_SUITE_P(SliceOpTest, SliceOpTest,
                         ::testing::Values(TestType::kConst,
                                           TestType::kDynamic));

}  // namespace
}  // namespace tflite
