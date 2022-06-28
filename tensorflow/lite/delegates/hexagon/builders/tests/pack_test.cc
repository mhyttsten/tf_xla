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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpack_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpack_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpack_testDTcc() {
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

class PackOpModel : public SingleOpModelWithHexagon {
 public:
  PackOpModel(const TensorData& input_template, int axis, int values_count) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpack_testDTcc mht_0(mht_0_v, 192, "", "./tensorflow/lite/delegates/hexagon/builders/tests/pack_test.cc", "PackOpModel");

    std::vector<std::vector<int>> all_input_shapes;
    for (int i = 0; i < values_count; ++i) {
      all_input_shapes.push_back(input_template.shape);
      AddInput(input_template);
    }
    output_ = AddOutput({input_template.type, /*shape=*/{}, input_template.min,
                         input_template.max});
    SetBuiltinOp(BuiltinOperator_PACK, BuiltinOptions_PackOptions,
                 CreatePackOptions(builder_, values_count, axis).Union());
    BuildInterpreter(all_input_shapes);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename integer_type>
  void SetInput(int index, std::initializer_list<float> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSpack_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/delegates/hexagon/builders/tests/pack_test.cc", "SetInput");

    QuantizeAndPopulate<integer_type>(index, data);
  }

  template <typename integer_type>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_type>(ExtractVector<integer_type>(output_),
                                    GetScale(output_), GetZeroPoint(output_));
  }

 private:
  int output_;
};

template <typename InputType>
struct PackOpTest : public ::testing::Test {
  using TypeToTest = InputType;
  TensorType TENSOR_TYPE =
      (std::is_same<InputType, int16_t>::value
           ? TensorType_INT16
           : (std::is_same<InputType, uint8_t>::value ? TensorType_UINT8
                                                      : TensorType_INT8));
};

using TestTypes = testing::Types<int8_t, uint8_t>;
TYPED_TEST_CASE(PackOpTest, TestTypes);

TYPED_TEST(PackOpTest, ThreeInputs) {
  PackOpModel model({TestFixture::TENSOR_TYPE, {2}, -10, 10}, 0, 3);
  model.SetInput<typename TestFixture::TypeToTest>(0, {1, 4});
  model.SetInput<typename TestFixture::TypeToTest>(1, {2, 5});
  model.SetInput<typename TestFixture::TypeToTest>(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  auto ref_output_shape = model.GetOutputShape();
  auto ref_output =
      model.GetDequantizedOutput<typename TestFixture::TypeToTest>();
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(ref_output_shape));
  EXPECT_THAT(model.GetDequantizedOutput<typename TestFixture::TypeToTest>(),
              ElementsAreArray(ArrayFloatNear(ref_output)));
}

TYPED_TEST(PackOpTest, ThreeInputsDifferentAxis) {
  PackOpModel model({TestFixture::TENSOR_TYPE, {2}, -10, 10}, 1, 3);
  model.SetInput<typename TestFixture::TypeToTest>(0, {1, 4});
  model.SetInput<typename TestFixture::TypeToTest>(1, {2, 5});
  model.SetInput<typename TestFixture::TypeToTest>(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  auto ref_output_shape = model.GetOutputShape();
  auto ref_output =
      model.GetDequantizedOutput<typename TestFixture::TypeToTest>();
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(ref_output_shape));
  EXPECT_THAT(model.GetDequantizedOutput<typename TestFixture::TypeToTest>(),
              ElementsAreArray(ArrayFloatNear(ref_output)));
}

TYPED_TEST(PackOpTest, ThreeInputsNegativeAxis) {
  PackOpModel model({TestFixture::TENSOR_TYPE, {2}, -10, 10}, -1, 3);
  model.SetInput<typename TestFixture::TypeToTest>(0, {1, 4});
  model.SetInput<typename TestFixture::TypeToTest>(1, {2, 5});
  model.SetInput<typename TestFixture::TypeToTest>(2, {3, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  auto ref_output_shape = model.GetOutputShape();
  auto ref_output =
      model.GetDequantizedOutput<typename TestFixture::TypeToTest>();
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(ref_output_shape));
  EXPECT_THAT(model.GetDequantizedOutput<typename TestFixture::TypeToTest>(),
              ElementsAreArray(ArrayFloatNear(ref_output)));
}

TYPED_TEST(PackOpTest, MultilDimensions) {
  PackOpModel model({TestFixture::TENSOR_TYPE, {2, 3}, -10, 20}, 1, 2);
  model.SetInput<typename TestFixture::TypeToTest>(0, {1, 2, 3, 4, 5, 6});
  model.SetInput<typename TestFixture::TypeToTest>(1, {7, 8, 9, 10, 11, 12});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  auto ref_output_shape = model.GetOutputShape();
  auto ref_output =
      model.GetDequantizedOutput<typename TestFixture::TypeToTest>();
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(ref_output_shape));
  EXPECT_THAT(model.GetDequantizedOutput<typename TestFixture::TypeToTest>(),
              ElementsAreArray(ArrayFloatNear(ref_output)));
}

}  // namespace tflite
