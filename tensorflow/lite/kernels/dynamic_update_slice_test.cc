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
class MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_slice_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_slice_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_slice_testDTcc() {
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

#include <algorithm>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class DynamicUpdateSliceOpModel : public SingleOpModel {
 public:
  DynamicUpdateSliceOpModel(const TensorData& operand, const TensorData& update,
                            const TensorData& start_indices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_slice_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/dynamic_update_slice_test.cc", "DynamicUpdateSliceOpModel");

    input_ = AddInput(operand);
    update_ = AddInput(update);
    start_indices_ = AddInput(start_indices);
    output_ = AddOutput(operand.type);
    SetBuiltinOp(BuiltinOperator_DYNAMIC_UPDATE_SLICE,
                 BuiltinOptions_DynamicUpdateSliceOptions,
                 CreateDynamicUpdateSliceOptions(builder_).Union());
    BuildInterpreter(
        {GetShape(input_), GetShape(update_), GetShape(start_indices_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_slice_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/lite/kernels/dynamic_update_slice_test.cc", "SetInput");

    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  void SetUpdate(std::initializer_list<T> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_slice_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/kernels/dynamic_update_slice_test.cc", "SetUpdate");

    PopulateTensor<T>(update_, data);
  }

  void SetStringInput(std::initializer_list<string> data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_slice_testDTcc mht_3(mht_3_v, 237, "", "./tensorflow/lite/kernels/dynamic_update_slice_test.cc", "SetStringInput");

    PopulateStringTensor(input_, data);
  }

  template <typename T>
  void SetStartIndices(std::initializer_list<T> data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSdynamic_update_slice_testDTcc mht_4(mht_4_v, 245, "", "./tensorflow/lite/kernels/dynamic_update_slice_test.cc", "SetStartIndices");

    PopulateTensor<T>(start_indices_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<string> GetStringOutput() {
    return ExtractVector<string>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int update_;
  int start_indices_;
  int output_;
};

TEST(DynamicUpdateSliceOpTest, SimpleTestF32) {
  DynamicUpdateSliceOpModel m({TensorType_FLOAT32, {3, 3}},
                              {TensorType_FLOAT32, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<float>({1, 2, 3,  //
                     4, 5, 6,  //
                     7, 8, 9});
  m.SetUpdate<float>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 2, 3,   //
                                               4, -1, 6,  //
                                               7, -2, 9})));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI1) {
  DynamicUpdateSliceOpModel m({TensorType_BOOL, {3, 3}},
                              {TensorType_BOOL, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<bool>({true, true, true,  //
                    true, true, true,  //
                    true, true, true});
  m.SetUpdate<bool>({false, false});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, true, true,   //
                                                     true, false, true,  //
                                                     true, false, true}));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI8) {
  DynamicUpdateSliceOpModel m({TensorType_INT8, {3, 3}},
                              {TensorType_INT8, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<int8_t>({1, 2, 3,  //
                      4, 5, 6,  //
                      7, 8, 9});
  m.SetUpdate<int8_t>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({1, 2, 3,   //
                                                       4, -1, 6,  //
                                                       7, -2, 9}));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI32) {
  DynamicUpdateSliceOpModel m({TensorType_INT32, {3, 3}},
                              {TensorType_INT32, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<int32_t>({1, 2, 3,  //
                       4, 5, 6,  //
                       7, 8, 9});
  m.SetUpdate<int32_t>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({1, 2, 3,   //
                                                        4, -1, 6,  //
                                                        7, -2, 9}));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI64) {
  DynamicUpdateSliceOpModel m({TensorType_INT64, {3, 3}},
                              {TensorType_INT64, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<int64_t>({1, 2, 3,  //
                       4, 5, 6,  //
                       7, 8, 9});
  m.SetUpdate<int64_t>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({1, 2, 3,   //
                                                        4, -1, 6,  //
                                                        7, -2, 9}));
}

TEST(DynamicUpdateSliceOpTest, BoundaryTest) {
  DynamicUpdateSliceOpModel m({TensorType_FLOAT32, {3, 3}},
                              {TensorType_FLOAT32, {2, 2}},
                              {TensorType_INT32, {2}});
  m.SetInput<float>({1, 2, 3,  //
                     4, 5, 6,  //
                     7, 8, 9});
  m.SetUpdate<float>({-1, -2,  //
                      -3, -4});
  m.SetStartIndices<int32_t>({2, 2});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 2, 3,    //
                                               4, -1, -2,  //
                                               7, -3, -4})));
}

TEST(DynamicUpdateSliceOpTest, UpdateShapeTooLargeTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      DynamicUpdateSliceOpModel({TensorType_FLOAT32, {3, 3}},
                                {TensorType_FLOAT32, {4, 2}},
                                {TensorType_INT32, {2}}),
      "SizeOfDimension\\(update, i\\) <= SizeOfDimension\\(operand, "
      "i\\) was not true.");
}

}  // namespace
}  // namespace tflite
