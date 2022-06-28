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
class MHTracer_DTPStensorflowPSlitePSkernelsPSgather_nd_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_nd_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSgather_nd_testDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class GatherNdOpModel : public SingleOpModel {
 public:
  GatherNdOpModel(const TensorData& params, const TensorData& indices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_nd_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/gather_nd_test.cc", "GatherNdOpModel");

    params_ = AddInput(params);
    indices_ = AddInput(indices);
    output_ = AddOutput(params.type);
    SetBuiltinOp(BuiltinOperator_GATHER_ND, BuiltinOptions_GatherNdOptions,
                 CreateGatherNdOptions(builder_).Union());
    BuildInterpreter({GetShape(params_), GetShape(indices_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_nd_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/kernels/gather_nd_test.cc", "SetInput");

    PopulateTensor<T>(params_, data);
  }

  template <typename T>
  void SetPositions(std::initializer_list<T> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSgather_nd_testDTcc mht_2(mht_2_v, 225, "", "./tensorflow/lite/kernels/gather_nd_test.cc", "SetPositions");

    PopulateTensor<T>(indices_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int params_;
  int indices_;
  int output_;
};

TEST(GatherNdOpTest, ElementIndexingIntoMatrix) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT32, {2, 2}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({0, 0, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.1, 2.2}));
}

TEST(GatherNdOpTest, SliceIndexingIntoMatrix) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT32, {2, 1}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({2.1, 2.2, 1.1, 1.2}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoMatrix1) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}},
                    {TensorType_INT32, {2, 1, 1}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({2.1, 2.2, 1.1, 1.2}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoMatrix2) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}},
                    {TensorType_INT32, {2, 1, 2}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({0, 0, 1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.1, 2.2}));
}

TEST(GatherNdOpTest, DuplicateIndexingIntoMatrix) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT32, {2, 2}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({0, 0, 0, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.1, 1.1}));
}

TEST(GatherNdOpTest, ElementIndexingIntoRank3Tensor) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {1, 2, 3}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 0, 1, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({-1.2, -4.1}));
}

TEST(GatherNdOpTest, SliceIndexingIntoRank3Tensor) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 1}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({1.1, -1.2, 1.3, -2.1, 2.2, 2.3, 5.1, -5.2, 5.3,
                                6.1, -6.2, 6.3}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoRank3Tensor1) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 1, 3}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 0, 1, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({-1.2, -4.1}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoRank3Tensor2) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 1, 1}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({3.1, 3.2, -3.3, -4.1, -4.2, 4.3, 1.1, -1.2, 1.3,
                                -2.1, 2.2, 2.3}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoRank3Tensor3) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 2, 2}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 1, 1, 0, 0, 0, 2, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-2.1, 2.2, 2.3, 3.1, 3.2, -3.3, 1.1, -1.2, 1.3,
                                6.1, -6.2, 6.3}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoRank3Tensor4) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 2, 3}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({-1.2, 3.2, 4.3, 6.3}));
}

TEST(GatherNdOpTest, DuplicateIndexingIntoRank3Tensor) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 2}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 1, 0, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-2.1, 2.2, 2.3, -2.1, 2.2, 2.3}));
}

TEST(GatherNdOpTest, Float32Int32) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 2}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-2.1, 2.2, 2.3, 3.1, 3.2, -3.3}));
}

TEST(GatherNdOpTest, Float32Int64) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT64, {2, 2}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-2.1, 2.2, 2.3, 3.1, 3.2, -3.3}));
}

TEST(GatherNdOpTest, Int32Int32) {
  GatherNdOpModel m({TensorType_INT32, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<int32_t>({1, -1, 1, -2, 2, 2,   //
                       3, 3, -3, -4, -4, 4,  //
                       5, -5, 5, 6, -6, 6});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Int32Int64) {
  GatherNdOpModel m({TensorType_INT32, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<int32_t>({1, -1, 1, -2, 2, 2,   //
                       3, 3, -3, -4, -4, 4,  //
                       5, -5, 5, 6, -6, 6});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Uint8Int32) {
  GatherNdOpModel m({TensorType_UINT8, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<uint8_t>({1, 1, 1, 2, 2, 2,  //
                       3, 3, 3, 4, 4, 4,  //
                       5, 5, 5, 6, 6, 6});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({2, 2, 2, 3, 3, 3}));
}

TEST(GatherNdOpTest, Uint8Int64) {
  GatherNdOpModel m({TensorType_UINT8, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<uint8_t>({1, 1, 1, 2, 2, 2,  //
                       3, 3, 3, 4, 4, 4,  //
                       5, 5, 5, 6, 6, 6});
  m.SetPositions<int64_t>({0, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({2, 2, 2, 3, 3, 3}));
}

TEST(GatherNdOpTest, Int8Int32) {
  GatherNdOpModel m({TensorType_INT8, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<int8_t>({1, -1, 1, -2, 2, 2,   //
                      3, 3, -3, -4, -4, 4,  //
                      5, -5, 5, 6, -6, 6});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Int8Int64) {
  GatherNdOpModel m({TensorType_INT8, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<int8_t>({1, -1, 1, -2, 2, 2,   //
                      3, 3, -3, -4, -4, 4,  //
                      5, -5, 5, 6, -6, 6});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Int16Int32) {
  GatherNdOpModel m({TensorType_INT16, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<int16_t>({1, -1, 1, -2, 2, 2,   //
                       3, 3, -3, -4, -4, 4,  //
                       5, -5, 5, 6, -6, 6});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int16_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Int16Int64) {
  GatherNdOpModel m({TensorType_INT16, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<int16_t>({1, -1, 1, -2, 2, 2,   //
                       3, 3, -3, -4, -4, 4,  //
                       5, -5, 5, 6, -6, 6});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int16_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Int64Int32) {
  GatherNdOpModel m({TensorType_INT64, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<int64_t>({1LL, -1LL, 1LL, -2LL, 2LL, 2LL,   //
                       3LL, 3LL, -3LL, -4LL, -4LL, 4LL,  //
                       5LL, -5LL, 5LL, 6LL, -6LL, 6LL});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int64_t>(),
              ElementsAreArray({-2LL, 2LL, 2LL, 3LL, 3LL, -3LL}));
}

TEST(GatherNdOpTest, Int64Int64) {
  GatherNdOpModel m({TensorType_INT64, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<int64_t>({1LL, -1LL, 1LL, -2LL, 2LL, 2LL,   //
                       3LL, 3LL, -3LL, -4LL, -4LL, 4LL,  //
                       5LL, -5LL, 5LL, 6LL, -6LL, 6LL});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int64_t>(),
              ElementsAreArray({-2LL, 2LL, 2LL, 3LL, 3LL, -3LL}));
}

TEST(GatherNdOpTest, StringInt32) {
  GatherNdOpModel m({TensorType_STRING, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<std::string>({"A", "B", "C", "D", "E", "F",  //
                           "G", "H", "I", "J", "K", "L",  //
                           "M", "N", "O", "P", "Q", "R"});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<std::string>(),
              ElementsAreArray({"D", "E", "F", "G", "H", "I"}));
}

TEST(GatherNdOpTest, StringInt64) {
  GatherNdOpModel m({TensorType_STRING, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<std::string>({"A", "B", "C", "D", "E", "F",  //
                           "G", "H", "I", "J", "K", "L",  //
                           "M", "N", "O", "P", "Q", "R"});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<std::string>(),
              ElementsAreArray({"D", "E", "F", "G", "H", "I"}));
}

TEST(GatherNdOpTest, EmptyParamsAndIndex) {
  GatherNdOpModel m({TensorType_FLOAT32, {1, 0}}, {TensorType_INT32, {0, 2}});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0}));
}

}  // namespace
}  // namespace tflite
