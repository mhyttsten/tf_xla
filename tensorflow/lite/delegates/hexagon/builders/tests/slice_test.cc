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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSslice_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSslice_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSslice_testDTcc() {
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
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

template <typename index_type>
class SliceOpModel : public SingleOpModelWithHexagon {
 public:
  SliceOpModel(const TensorData& input, const TensorData& output,
               const TensorData& begin, const TensorData& size,
               std::initializer_list<index_type> begin_data,
               std::initializer_list<index_type> size_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSslice_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/delegates/hexagon/builders/tests/slice_test.cc", "SliceOpModel");

    input_ = AddInput(input);
    begin_ = AddConstInput(begin, begin_data);
    size_ = AddConstInput(size, size_data);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SLICE, BuiltinOptions_SliceOptions,
                 CreateSliceOptions(builder_).Union());
    BuildInterpreter({GetShape(input_), GetShape(begin_), GetShape(size_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSslice_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/lite/delegates/hexagon/builders/tests/slice_test.cc", "SetInput");

    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int begin_;
  int size_;
  int output_;
};

TEST(SliceOpTest, Input_1D_Uint8) {
  SliceOpModel<int> m(/*input=*/{TensorType_UINT8, {4}, -10, 10},
                      /*output=*/{TensorType_UINT8, {2}, -10, 10},
                      {TensorType_INT32, {1}}, {TensorType_INT32, {1}}, {1},
                      {2});
  m.SetInput<uint8_t>({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({2, 3}, 0.1)));
}

TEST(SliceOpTest, Input_2D_Uint8) {
  SliceOpModel<int> m(
      /*input=*/{TensorType_UINT8, {2, 3}, -10, 10},
      /*output=*/{TensorType_UINT8, {1, 2}, -10, 10}, {TensorType_INT32, {2}},
      {TensorType_INT32, {2}}, {1, 0}, {1, 2});
  m.SetInput<uint8_t>({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, SizeInt64_Uint8) {
  SliceOpModel<int64_t> m(/*input=*/{TensorType_UINT8, {4, 1, 1, 1}, -10, 10},
                          /*output=*/{TensorType_UINT8, {3, 1, 1, 1}, -10, 10},
                          {TensorType_INT64, {4}}, {TensorType_INT64, {4}},
                          {1, 0, 0, 0}, {3, 1, 1, 1});
  m.SetInput<uint8_t>({1, 2, 3, 4});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, SizeMinus1) {
  SliceOpModel<int64_t> m(
      /*input=*/{TensorType_UINT8, {3, 2, 3, 1}, -10, 10},
      /*output=*/{TensorType_UINT8, {2, 1, 3, 1}, -10, 10},
      {TensorType_INT64, {4}}, {TensorType_INT64, {4}}, {1, 0, 0, 0},
      {2, 1, -1, 1});
  m.SetInput<uint8_t>({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, BeginNonZeroSizeMinus1Axis1) {
  SliceOpModel<int64_t> m(
      /*input=*/{TensorType_UINT8, {3, 3, 2, 1}, -10, 10},
      /*output=*/{TensorType_UINT8, {2, 2, 1, 1}, -10, 10},
      {TensorType_INT64, {4}}, {TensorType_INT64, {4}}, {1, 1, 0, 0},
      {2, -1, 1, 1});
  m.SetInput<uint8_t>({1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, BeginNonZeroSizeMinus1Axis2) {
  SliceOpModel<int64_t> m(
      /*input=*/{TensorType_UINT8, {3, 2, 3, 1}, -10, 10},
      /*output=*/{TensorType_UINT8, {2, 1, 2, 1}, -10, 10},
      {TensorType_INT64, {4}}, {TensorType_INT64, {4}}, {1, 0, 1, 0},
      {2, 1, -1, 1});
  m.SetInput<uint8_t>({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<uint8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

TEST(SliceOpTest, BeginNonZeroSizeMinus1Axis2_Int8) {
  SliceOpModel<int64_t> m(
      /*input=*/{TensorType_INT8, {3, 2, 3, 1}, -10, 10},
      /*output=*/{TensorType_INT8, {2, 1, 2, 1}, -10, 10},
      {TensorType_INT64, {4}}, {TensorType_INT64, {4}}, {1, 0, 1, 0},
      {2, 1, -1, 1});
  m.SetInput<int8_t>({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  auto reference_output = m.GetDequantizedOutput<int8_t>();
  auto reference_output_shape = m.GetOutputShape();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray(reference_output_shape));
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output, 0.1)));
}

}  // namespace tflite
