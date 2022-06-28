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
class MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_nd_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_nd_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_nd_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BatchToSpaceNDOpModel : public SingleOpModel {
 public:
  template <typename T>
  void SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_nd_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/batch_to_space_nd_test.cc", "SetInput");

    PopulateTensor<T>(input_, data);
  }

  void SetBlockShape(std::initializer_list<int> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_nd_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/kernels/batch_to_space_nd_test.cc", "SetBlockShape");

    PopulateTensor<int>(block_shape_, data);
  }

  void SetCrops(std::initializer_list<int> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_nd_testDTcc mht_2(mht_2_v, 218, "", "./tensorflow/lite/kernels/batch_to_space_nd_test.cc", "SetCrops");

    PopulateTensor<int>(crops_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  int32_t GetOutputSize() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_nd_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/lite/kernels/batch_to_space_nd_test.cc", "GetOutputSize");
 return GetTensorSize(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int block_shape_;
  int crops_;
  int output_;
};

// Tests case where block_shape and crops are const tensors.
//
// Example usage is as follows:
//    BatchToSpaceNDOpConstModel m(input_shape, block_shape, crops);
//    m.SetInput(input_data);
//    m.Invoke();
class BatchToSpaceNDOpConstModel : public BatchToSpaceNDOpModel {
 public:
  BatchToSpaceNDOpConstModel(std::initializer_list<int> input_shape,
                             std::initializer_list<int> block_shape,
                             std::initializer_list<int> crops,
                             const TensorType& type = TensorType_FLOAT32) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_nd_testDTcc mht_4(mht_4_v, 254, "", "./tensorflow/lite/kernels/batch_to_space_nd_test.cc", "BatchToSpaceNDOpConstModel");

    int spatial_dims = static_cast<int>(block_shape.size());
    input_ = AddInput({type, input_shape});
    block_shape_ = AddConstInput(TensorType_INT32, block_shape, {spatial_dims});
    crops_ = AddConstInput(TensorType_INT32, crops, {spatial_dims, 2});
    output_ = AddOutput(type);

    SetBuiltinOp(BuiltinOperator_BATCH_TO_SPACE_ND,
                 BuiltinOptions_BatchToSpaceNDOptions,
                 CreateBatchToSpaceNDOptions(builder_).Union());
    BuildInterpreter({input_shape});
  }
};

// Tests case where block_shape and crops are non-const tensors.
//
// Example usage is as follows:
//    BatchToSpaceNDOpDynamicModel m(input_shape);
//    m.SetInput(input_data);
//    m.SetBlockShape(block_shape);
//    m.SetPaddings(crops);
//    m.Invoke();
class BatchToSpaceNDOpDynamicModel : public BatchToSpaceNDOpModel {
 public:
  BatchToSpaceNDOpDynamicModel(std::initializer_list<int> input_shape,
                               const TensorType& type = TensorType_FLOAT32) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_to_space_nd_testDTcc mht_5(mht_5_v, 282, "", "./tensorflow/lite/kernels/batch_to_space_nd_test.cc", "BatchToSpaceNDOpDynamicModel");

    input_ = AddInput({type, input_shape});
    block_shape_ = AddInput(TensorType_INT32);
    crops_ = AddInput(TensorType_INT32);
    output_ = AddOutput(type);

    int spatial_dims = static_cast<int>(input_shape.size()) - 2;
    SetBuiltinOp(BuiltinOperator_BATCH_TO_SPACE_ND,
                 BuiltinOptions_BatchToSpaceNDOptions,
                 CreateBatchToSpaceNDOptions(builder_).Union());
    BuildInterpreter({input_shape, {spatial_dims}, {spatial_dims, 2}});
  }
};

TEST(BatchToSpaceNDOpTest, SimpleConstTest) {
  BatchToSpaceNDOpConstModel m({4, 2, 2, 1}, {2, 2}, {0, 0, 0, 0});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  {1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, SimpleConstTestInt8) {
  BatchToSpaceNDOpConstModel m({4, 2, 2, 1}, {2, 2}, {0, 0, 0, 0},
                               TensorType_INT8);
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(
                  {1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, BatchOneConstTest) {
  BatchToSpaceNDOpConstModel m({1, 2, 2, 1}, {1, 1}, {0, 0, 0, 0});
  m.SetInput<float>({1, 2, 3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1, 2, 3, 4}));
}

TEST(BatchToSpaceNDOpTest, SimpleConstTestInt8EmptyOutput) {
  if (SingleOpModel::GetForceUseNnapi()) {
    // NNAPI doesn't currently support non-zero crop values.
    return;
  }

  BatchToSpaceNDOpConstModel m({4, 2, 2, 1}, {2, 2}, {0, 0, 2, 2},
                               TensorType_INT8);
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 0, 1}));
  EXPECT_THAT(m.GetOutputSize(), 0);
}

TEST(BatchToSpaceNDOpTest, SimpleDynamicTest) {
  BatchToSpaceNDOpDynamicModel m({4, 2, 2, 1});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetCrops({0, 0, 0, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  {1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, SimpleDynamicTestInt8) {
  BatchToSpaceNDOpDynamicModel m({4, 2, 2, 1}, TensorType_INT8);
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetCrops({0, 0, 0, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(
                  {1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, InvalidCropsDynamicTest) {
  BatchToSpaceNDOpDynamicModel m({4, 2, 2, 1});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetCrops({0, 0, -1, 0});
  ASSERT_NE(m.InvokeUnchecked(), kTfLiteOk) << "crops.i. >= 0 was not true.";
}

TEST(BatchToSpaceNDOpTest, SimpleDynamicTestInt8EmptyOutput) {
  if (SingleOpModel::GetForceUseNnapi()) {
    // NNAPI doesn't currently support non-zero crop values.
    return;
  }

  BatchToSpaceNDOpDynamicModel m({4, 2, 2, 1}, TensorType_INT8);
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetCrops({2, 2, 0, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 4, 1}));
  EXPECT_THAT(m.GetOutput<int8_t>(), ::testing::IsEmpty());
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(BatchToSpaceNDOpTest, InvalidShapeTest) {
  EXPECT_DEATH(BatchToSpaceNDOpConstModel({3, 2, 2, 1}, {2, 2}, {0, 0, 0, 0}),
               "Cannot allocate tensors");
}

TEST(BatchToSpaceNDOpTest, InvalidCropsConstTest) {
  EXPECT_DEATH(BatchToSpaceNDOpConstModel({3, 2, 2, 1}, {2, 2}, {0, 0, 0, -1}),
               "crops.i. >= 0 was not true.");
}
#endif

TEST(BatchToSpaceNDOpTest, Simple3DConstTest) {
  BatchToSpaceNDOpConstModel m({4, 4, 1}, {2}, {0, 0});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 8, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  {1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16}));
}

TEST(BatchToSpaceNDOpTest, Simple3DConstTestWithCrops) {
  BatchToSpaceNDOpConstModel m({4, 4, 1}, {2}, {1, 1});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 6, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({9, 2, 10, 3, 11, 4, 13, 6, 14, 7, 15, 8}));
}

TEST(BatchToSpaceNDOpTest, Simple3DDynamicTest) {
  BatchToSpaceNDOpDynamicModel m({4, 4, 1});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2});
  m.SetCrops({0, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 8, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  {1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16}));
}

TEST(BatchToSpaceNDOpTest, Simple3DDynamicTestWithCrops) {
  BatchToSpaceNDOpDynamicModel m({4, 4, 1});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2});
  m.SetCrops({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 6, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({9, 2, 10, 3, 11, 4, 13, 6, 14, 7, 15, 8}));
}

}  // namespace
}  // namespace tflite
