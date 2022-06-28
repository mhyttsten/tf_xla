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
class MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc() {
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
#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_BATCH_MATMUL_REF();
TfLiteRegistration* Register_BATCH_MATMUL_GENERIC_OPTIMIZED();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
tflite::TensorType GetTFLiteType() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "GetTFLiteType");

  if (std::is_same<T, int8_t>::value) {
    return TensorType_INT8;
  }
  if (std::is_same<T, int16_t>::value) {
    return TensorType_INT16;
  }
  if (std::is_same<T, int32_t>::value) {
    return TensorType_INT32;
  }
  return TensorType_FLOAT32;
}

template <typename T>
class BatchMatMulOpModel : public SingleOpModel {
 public:
  BatchMatMulOpModel(const TensorData& lhs, const TensorData& rhs,
                     bool adj_x = false, bool adj_y = false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "BatchMatMulOpModel");

    lhs_id_ = AddInput(lhs);
    rhs_id_ = AddInput(rhs);
    output_id_ = AddOutput(GetTFLiteType<T>());
    SetBuiltinOp(BuiltinOperator_BATCH_MATMUL,
                 BuiltinOptions_BatchMatMulOptions,
                 CreateBatchMatMulOptions(builder_, adj_x, adj_y).Union());
    BuildInterpreter({GetShape(lhs_id_), GetShape(rhs_id_)});
  }

  int lhs() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_2(mht_2_v, 245, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "lhs");
 return lhs_id_; }
  int rhs() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_3(mht_3_v, 249, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "rhs");
 return rhs_id_; }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }
  std::vector<int32_t> GetOutputShape() { return GetTensorShape(output_id_); }

 protected:
  int lhs_id_;
  int rhs_id_;
  int output_id_;
};

const auto kKernelMap = new std::map<string, TfLiteRegistration*>({
    {"Reference", ops::builtin::Register_BATCH_MATMUL_REF()},
    {"GenericOptimized",
     ops::builtin::Register_BATCH_MATMUL_GENERIC_OPTIMIZED()},
});

class BatchMatMulOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(BatchMatMulOpTest, Float32Test_Simple) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {1, 2, 3}},
                                  {TensorType_FLOAT32, {1, 3, 4}});
  model.PopulateTensor<float>(model.lhs(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TEST_P(BatchMatMulOpTest, Int8Test_Simple) {
  BatchMatMulOpModel<int32_t> model({TensorType_INT8, {1, 2, 3}},
                                    {TensorType_INT8, {1, 3, 4}});
  model.PopulateTensor<int8_t>(model.lhs(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int8_t>(model.rhs(),
                               {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74, 80, 86, 92, 173, 188, 203, 218}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_SimpleRHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {1, 2, 3}},
                                  {TensorType_FLOAT32, {1, 4, 3}}, false, true);
  model.PopulateTensor<float>(model.lhs(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 11, 15, 8, 12, 16, 9, 13, 17, 10, 14, 18});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_SimpleLHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {1, 3, 2}},
                                  {TensorType_FLOAT32, {1, 3, 4}}, true, false);
  model.PopulateTensor<float>(model.lhs(), {1, 4, 2, 5, 3, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_BatchSizeTwo) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 2, 3}},
                                  {TensorType_FLOAT32, {2, 3, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218., 560., 584.,
                        608., 632., 767., 800., 833., 866.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 2, 3}},
                                  {TensorType_FLOAT32, {3, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218., 272., 296.,
                        320., 344., 371., 404., 437., 470.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_BroadcastLHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 3, 2}},
                                  {TensorType_FLOAT32, {3, 4}}, true, false);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218., 272., 296.,
                        320., 344., 371., 404., 437., 470.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast2) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 1, 3, 2}},
                                  {TensorType_FLOAT32, {3, 2, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast2LHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 1, 2, 3}},
                                  {TensorType_FLOAT32, {3, 2, 4}}, true, false);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast2RHSAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 1, 3, 2}},
                                  {TensorType_FLOAT32, {3, 4, 2}}, false, true);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  11, 8,  12, 9,  13, 10, 14, 15, 19, 16, 20,
                               17, 21, 18, 22, 23, 27, 24, 28, 25, 29, 26, 30});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_Broadcast2BothAdjoint) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 1, 2, 3}},
                                  {TensorType_FLOAT32, {3, 4, 2}}, true, true);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  11, 8,  12, 9,  13, 10, 14, 15, 19, 16, 20,
                               17, 21, 18, 22, 23, 27, 24, 28, 25, 29, 26, 30});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TEST_P(BatchMatMulOpTest, Float32Test_BroadcastFromRHS) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {4, 5}},
                                  {TensorType_FLOAT32, {3, 1, 5, 2}});
  model.PopulateTensor<float>(
      model.lhs(),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  model.PopulateTensor<float>(
      model.rhs(),
      {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
       22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36});

  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({185., 200., 460.,  500.,  735.,  800.,  1010., 1100.,
                        335., 350., 860.,  900.,  1385., 1450., 1910., 2000.,
                        485., 500., 1260., 1300., 2035., 2100., 2810., 2900.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 1, 4, 2}));
}

INSTANTIATE_TEST_SUITE_P(
    BatchMatMulOpTest, BatchMatMulOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

class ConstRHSBatchMatMulOpModel : public MultiOpModel {
 public:
  ConstRHSBatchMatMulOpModel(const TensorData& lhs,
                             std::initializer_list<int> rhs_shape,
                             std::initializer_list<float> rhs_data,
                             bool adj_x = false, bool adj_y = false) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_4(mht_4_v, 494, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "ConstRHSBatchMatMulOpModel");

    lhs_id_ = AddInput(lhs);
    rhs_id_ = AddConstInput<float>(TensorType_FLOAT32, rhs_data, rhs_shape);
    matmul_output_id_ = AddOutput(lhs.type);
    std::vector<int> matmul_inputs{lhs_id_, rhs_id_};
    std::vector<int> matmul_outputs{matmul_output_id_};
    AddBuiltinOp(BuiltinOperator_BATCH_MATMUL,
                 BuiltinOptions_BatchMatMulOptions,
                 CreateBatchMatMulOptions(builder_, adj_x, adj_y).Union(),
                 matmul_inputs, matmul_outputs);

    // Without following ops (not limited to neg), temporary allocation with
    // kTfLiteArenaRw tends to re-claim the same memory across each evaluation,
    // and no other ops will modify values at that memory address because no
    // other memory allocations take place. Therefore, it's likely that results
    // are correct even if constant transposed RHS is allocated with
    // kTfLiteArenaRw. We thus use a dummy op to make sure constant transposed
    // RHS behaves correctly.
    neg_output_id_ = AddOutput(lhs.type);
    std::vector<int> neg_inputs{matmul_output_id_};
    std::vector<int> neg_outputs{neg_output_id_};
    AddBuiltinOp(BuiltinOperator_NEG, BuiltinOptions_NegOptions,
                 CreateNegOptions(builder_).Union(), neg_inputs, neg_outputs);
    BuildInterpreter({GetShape(lhs_id_), GetShape(rhs_id_)});
  }

  int lhs() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_5(mht_5_v, 523, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "lhs");
 return lhs_id_; }
  std::vector<float> GetOutput() {
    return ExtractVector<float>(neg_output_id_);
  }
  std::vector<int32_t> GetOutputShape() {
    return GetTensorShape(neg_output_id_);
  }

 protected:
  int lhs_id_;
  int rhs_id_;
  int matmul_output_id_;
  int neg_output_id_;
};

TEST(ConstRHSBatchMatMulOpModel, RHSNotAdjoint) {
  ConstRHSBatchMatMulOpModel model({TensorType_FLOAT32, {1, 6, 2}}, {2, 3},
                                   {6, 3, 7, 4, 6, 9});
  model.PopulateTensor<float>(model.lhs(),
                              {6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({-48, -36, -69, -58, -45, -85, -72, -72, -123,
                                -36, -42, -68, -58, -45, -85, -46, -51, -84}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6, 3}));
  // Eval twice to make sure constant transposed RHS is persistent.
  model.PopulateTensor<float>(model.lhs(),
                              {6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({-48, -36, -69, -58, -45, -85, -72, -72, -123,
                                -36, -42, -68, -58, -45, -85, -46, -51, -84}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 6, 3}));
}

// In the hybrid model the weights are quantized int8. But the input
// and output are expected to be in float precision.
class HybridBatchMatMulOpModel : public SingleOpModel {
 public:
  HybridBatchMatMulOpModel(int units, int batches, const TensorData& lhs,
                           const TensorData& rhs,
                           const TensorData& output = {TensorType_FLOAT32},
                           bool asymmetric_quantize_inputs = true)
      : units_(units), batches_(batches) {
    int total_input_size = 1;
    for (size_t i = 0; i < lhs.shape.size(); ++i) {
      total_input_size *= lhs.shape[i];
    }
    input_size_ = total_input_size / batches_;

    lhs_id_ = AddInput(lhs);
    rhs_id_ = AddInput(rhs);

    output_id_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_BATCH_MATMUL, BuiltinOptions_BatchMatMulOptions,
        CreateBatchMatMulOptions(builder_, /*adj_x=*/false, /*adj_y=*/false,
                                 asymmetric_quantize_inputs)
            .Union());
    BuildInterpreter({GetShape(lhs_id_), GetShape(rhs_id_)});
  }
  void SetWeights(const std::vector<float>& data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_6(mht_6_v, 588, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "SetWeights");

    SymmetricQuantizeAndPopulate(rhs_id_, data);
  }

  void SetSignedWeights(std::initializer_list<float> f) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_7(mht_7_v, 595, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "SetSignedWeights");

    SignedSymmetricQuantizeAndPopulate(rhs_id_, f);
  }

  void SetInput(const std::vector<float>& f) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_8(mht_8_v, 602, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "SetInput");
 PopulateTensor(lhs_id_, f); }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_id_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_id_); }

  int input_size() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_9(mht_9_v, 609, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "input_size");
 return input_size_; }
  int num_units() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_10(mht_10_v, 613, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "num_units");
 return units_; }
  int num_batches() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_11(mht_11_v, 617, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "num_batches");
 return batches_; }

  int lhs() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_12(mht_12_v, 622, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "lhs");
 return lhs_id_; }
  int rhs() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_13(mht_13_v, 626, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "rhs");
 return rhs_id_; }

 protected:
  int lhs_id_;
  int rhs_id_;
  int output_id_;
  int units_;
  int batches_;
  int input_size_;
};

class HybridAsymmetricBatchMatMulOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(HybridAsymmetricBatchMatMulOpTest, SimpleTestQuantizedInt8) {
  HybridBatchMatMulOpModel m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 10}},
      /*rhs=*/{TensorType_INT8, {10, 3}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     196,
                                     196,
                                     196,
                                     246,
                                     246,
                                     246,
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
}

TEST_P(HybridAsymmetricBatchMatMulOpTest, MultipleNumBatchQuantizedInt8) {
  // need 4 scale factors
  HybridBatchMatMulOpModel m(
      /*units=*/10, /*batches=*/4,
      /*lhs=*/{TensorType_FLOAT32, {1, 2, 2, 3}},
      /*rhs=*/{TensorType_INT8, {3, 10}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
      1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
  });

  m.SetInput({
      11, 12, 13,  // batch 1, 0
      11, 12, 13,  // batch 1, 1
      11, 12, 13,  // batch 1, 2
      11, 12, 13,  // batch 1, 3
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
                      73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
                      73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
                  },
                  /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 10}));
}

TEST_P(HybridAsymmetricBatchMatMulOpTest, RegressionTestQuantizedInt8) {
  HybridBatchMatMulOpModel m(
      /*units=*/10, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 3}},
      /*rhs=*/{TensorType_INT8, {3, 10}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
      1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
  });

  m.SetInput({
      11, 12, 13,  // batch 1, 0
      11, 12, 13,  // batch 1, 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
                                     73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 10}));
}

TEST_P(HybridAsymmetricBatchMatMulOpTest, QuantizedInt8BroadcastWeights) {
  HybridBatchMatMulOpModel m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 2, 10}},
      /*rhs=*/{TensorType_INT8, {10, 3}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 24, 24,     //
                                     58, 58, 58,     //
                                     196, 196, 196,  //
                                     246, 246, 246,  //
                                 },
                                 /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

TEST_P(HybridAsymmetricBatchMatMulOpTest, QuantizedInt8BroadcastBigWeights) {
  HybridBatchMatMulOpModel m(
      /*units=*/9, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 2, 10}},
      /*rhs=*/{TensorType_INT8, {10, 9}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 17, 17, 17, 26, 26, 26, 2,  2,  2,  18, 18, 18, 27, 27, 27,
      3, 3, 3, 19, 19, 19, 28, 28, 28, 4,  4,  4,  20, 20, 20, 29, 29, 29,
      5, 5, 5, 21, 21, 21, 30, 30, 30, 6,  6,  6,  22, 22, 22, 31, 31, 31,
      7, 7, 7, 23, 23, 23, 32, 32, 32, 8,  8,  8,  24, 24, 24, 33, 33, 33,
      9, 9, 9, 25, 25, 25, 34, 34, 34, 10, 10, 10, 26, 26, 26, 35, 35, 35,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      23,  23,  23,  295,  295,  295,  449,  449,  449,   //
                      60,  60,  60,  364,  364,  364,  533,  533,  533,   //
                      195, 195, 195, 1429, 1429, 1429, 2124, 2124, 2124,  //
                      250, 250, 250, 1512, 1512, 1512, 2213, 2213, 2213   //
                  },
                  /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 9}));
}

TEST_P(HybridAsymmetricBatchMatMulOpTest, QuantizedInt8BroadcastInputs) {
  HybridBatchMatMulOpModel m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 10}},
      /*rhs=*/{TensorType_INT8, {2, 10, 3}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, -3, 1, 2, -2, 2, 3, -1, 3, 4,  0, 4, 5, 1, 5, 6, 2, 6,  7,  3,
      7, 8,  4, 8, 9,  5, 9, 10, 6, 10, 1, 1, 1, 2, 2, 2, 3, 3,  3,  4,
      4, 4,  5, 5, 5,  6, 6, 6,  7, 7,  7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // batch 0, 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // batch 0, 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, -45, 24,  //
                                     58, -18, 58,  //
                                     24, 24, 24,   //
                                     58, 58, 58,   //
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

INSTANTIATE_TEST_SUITE_P(
    HybridAsymmetricBatchMatMulOpTest, HybridAsymmetricBatchMatMulOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

class HybridSymmetricBatchMatMulOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(HybridSymmetricBatchMatMulOpTest, SimpleTestQuantizedInt8) {
  HybridBatchMatMulOpModel m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 10}},
      /*rhs=*/{TensorType_INT8, {10, 3}, 0, 0, 10.0 / 127.0, 0},
      /*output=*/{TensorType_FLOAT32}, /*asymmetric_quantize_inputs=*/false);

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     194,
                                     194,
                                     194,
                                     248,
                                     248,
                                     248,
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
}

TEST_P(HybridSymmetricBatchMatMulOpTest, QuantizedInt8BroadcastWeights) {
  HybridBatchMatMulOpModel m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 2, 10}},
      /*rhs=*/{TensorType_INT8, {10, 3}, 0, 0, 10.0 / 127.0, 0},
      /*output=*/{TensorType_FLOAT32}, /*asymmetric_quantize_inputs=*/false);

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 24, 24,     //
                                     56, 56, 56,     //
                                     194, 194, 194,  //
                                     248, 248, 248,  //
                                 },
                                 /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

TEST_P(HybridSymmetricBatchMatMulOpTest, QuantizedInt8BroadcastBigWeights) {
  HybridBatchMatMulOpModel m(
      /*units=*/9, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 2, 10}},
      /*rhs=*/{TensorType_INT8, {10, 9}, 0, 0, 10.0 / 127.0, 0},
      {TensorType_FLOAT32}, false);

  m.SetSignedWeights({
      1, 1, 1, 17, 17, 17, 26, 26, 26, 2,  2,  2,  18, 18, 18, 27, 27, 27,
      3, 3, 3, 19, 19, 19, 28, 28, 28, 4,  4,  4,  20, 20, 20, 29, 29, 29,
      5, 5, 5, 21, 21, 21, 30, 30, 30, 6,  6,  6,  22, 22, 22, 31, 31, 31,
      7, 7, 7, 23, 23, 23, 32, 32, 32, 8,  8,  8,  24, 24, 24, 33, 33, 33,
      9, 9, 9, 25, 25, 25, 34, 34, 34, 10, 10, 10, 26, 26, 26, 35, 35, 35,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      23,  23,  23,  296,  296,  296,  451,  451,  451,   //
                      58,  58,  58,  362,  362,  362,  529,  529,  529,   //
                      193, 193, 193, 1424, 1424, 1424, 2118, 2118, 2118,  //
                      253, 253, 253, 1519, 1519, 1519, 2223, 2223, 2223   //
                  },
                  /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 9}));
}

TEST_P(HybridSymmetricBatchMatMulOpTest, QuantizedInt8BroadcastInputs) {
  HybridBatchMatMulOpModel m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{TensorType_FLOAT32, {2, 10}},
      /*rhs=*/{TensorType_INT8, {2, 10, 3}, 0, 0, 10.0 / 127.0, 0},
      {TensorType_FLOAT32}, false);

  m.SetSignedWeights({
      1, -3, 1, 2, -2, 2, 3, -1, 3, 4,  0, 4, 5, 1, 5, 6, 2, 6,  7,  3,
      7, 8,  4, 8, 9,  5, 9, 10, 6, 10, 1, 1, 1, 2, 2, 2, 3, 3,  3,  4,
      4, 4,  5, 5, 5,  6, 6, 6,  7, 7,  7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // batch 0, 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // batch 0, 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, -45, 24,  //
                                     56, -19, 56,  //
                                     24, 24, 24,   //
                                     56, 56, 56,   //
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

INSTANTIATE_TEST_SUITE_P(
    HybridSymmetricBatchMatMulOpTest, HybridSymmetricBatchMatMulOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

class QuantizedBatchMatMulOpModel : public SingleOpModel {
 public:
  QuantizedBatchMatMulOpModel(int units, int batches, const TensorData& lhs,
                              const TensorData& output = {TensorType_INT8},
                              bool adj_x = false, bool adj_y = false)
      : units_(units), batches_(batches) {
    int total_input_size = 1;
    for (size_t i = 0; i < lhs.shape.size(); ++i) {
      total_input_size *= lhs.shape[i];
    }
    input_size_ = total_input_size / batches_;

    lhs_id_ = AddInput(lhs);
    rhs_id_ = AddInput({lhs.type,
                        {input_size_, units_},
                        0,
                        0,
                        GetScale(lhs_id_),
                        GetZeroPoint(lhs_id_)});

    output_id_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_BATCH_MATMUL,
                 BuiltinOptions_BatchMatMulOptions,
                 CreateBatchMatMulOptions(builder_, adj_x, adj_y).Union());
    BuildInterpreter({GetShape(lhs_id_), GetShape(rhs_id_)});
  }

  template <typename T>
  void SetWeights(const std::vector<float>& data) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_14(mht_14_v, 1008, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "SetWeights");

    QuantizeAndPopulate<T>(rhs_id_, data);
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbatch_matmul_testDTcc mht_15(mht_15_v, 1016, "", "./tensorflow/lite/kernels/batch_matmul_test.cc", "SetInput");

    QuantizeAndPopulate<T>(lhs_id_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_id_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_id_), GetScale(output_id_),
                         GetZeroPoint(output_id_));
  }

 protected:
  int lhs_id_;
  int rhs_id_;
  int output_id_;
  int units_;
  int batches_;
  int input_size_;
};

class QuantizedBatchMatMulOpTest : public SingleOpTest {
 protected:
  const std::map<string, TfLiteRegistration*>& GetKernelMap() override {
    return *kKernelMap;
  }
};

TEST_P(QuantizedBatchMatMulOpTest, SimpleTestQuantizedInt8) {
  QuantizedBatchMatMulOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*lhs=*/{TensorType_INT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_INT8, {}, -127, 128});

  m.SetWeights<int8_t>({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({23, 23, 23, 57, 57, 57})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(22, 22, 22, 56, 56, 56));
}

TEST_P(QuantizedBatchMatMulOpTest, SimpleTestQuantizedInt16) {
  const float inputs_scale = 10.0 / std::numeric_limits<int16_t>::max();
  const float output_scale = 1.0;
  const int32_t zero_point = 0;

  QuantizedBatchMatMulOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*lhs=*/
      {TensorType_INT16, {2, 10}, 0, 0, inputs_scale, zero_point},
      /*output=*/
      {TensorType_INT16, {}, 0, 0, output_scale, zero_point});

  m.SetWeights<int16_t>({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput<int16_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetDequantizedOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({23, 23, 23, 57, 57, 57})));
  EXPECT_THAT(m.GetOutput<int16_t>(), ElementsAre(23, 23, 23, 57, 57, 57));
}

INSTANTIATE_TEST_SUITE_P(
    QuantizedBatchMatMulOpTest, QuantizedBatchMatMulOpTest,
    ::testing::ValuesIn(SingleOpTest::GetKernelTags(*kKernelMap)));

}  // namespace
}  // namespace tflite
