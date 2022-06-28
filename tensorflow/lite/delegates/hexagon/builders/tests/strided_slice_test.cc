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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSstrided_slice_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSstrided_slice_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSstrided_slice_testDTcc() {
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

template <typename input_type>
class StridedSliceOpModel : public SingleOpModelWithHexagon {
 public:
  StridedSliceOpModel(const TensorData& input, const TensorData& output,
                      const TensorData& begin,
                      std::initializer_list<int> begin_data,
                      const TensorData& end,
                      std::initializer_list<int> end_data,
                      const TensorData& strides,
                      std::initializer_list<int> strides_data, int begin_mask,
                      int end_mask, int shrink_axis_mask) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSstrided_slice_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/delegates/hexagon/builders/tests/strided_slice_test.cc", "StridedSliceOpModel");

    input_ = AddInput(input);
    begin_ = AddConstInput(begin, begin_data);
    end_ = AddConstInput(end, end_data);
    strides_ = AddConstInput(strides, strides_data);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_STRIDED_SLICE,
                 BuiltinOptions_StridedSliceOptions,
                 CreateStridedSliceOptions(
                     builder_, begin_mask, end_mask, /*ellipsis_mask*/ 0,
                     /*new_axis_mask*/ 0, shrink_axis_mask)
                     .Union());
    BuildInterpreter({GetShape(input_), GetShape(begin_), GetShape(end_),
                      GetShape(strides_)});
  }

  void SetInput(std::initializer_list<input_type> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSstrided_slice_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/delegates/hexagon/builders/tests/strided_slice_test.cc", "SetInput");

    PopulateTensor<input_type>(input_, data);
  }

  std::vector<input_type> GetOutput() {
    return ExtractVector<input_type>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int begin_;
  int end_;
  int strides_;
  int output_;
};

TEST(StridedSliceOpModel, In1D_UInt8) {
  StridedSliceOpModel<uint8_t> m(
      /*input=*/{TensorType_UINT8, {4}, -10, 10},
      /*output=*/{TensorType_UINT8, {2}, -10, 10},
      /*begin*/ {TensorType_INT32, {1}},
      /*begin_data*/ {1},
      /*end*/ {TensorType_INT32, {1}},
      /*end_data*/ {3},
      /*strides*/ {TensorType_INT32, {1}},
      /*strides_data*/ {1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TEST(StridedSliceOpModel, In1D_NegativeBegin_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {4}, -10, 10},
      /*output=*/{TensorType_INT8, {2}, -10, 10},
      /*begin*/ {TensorType_INT32, {1}},
      /*begin_data*/ {-3},
      /*end*/ {TensorType_INT32, {1}},
      /*end_data*/ {3},
      /*strides*/ {TensorType_INT32, {1}},
      /*strides_data*/ {1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TEST(StridedSliceOpModel, In1D_NegativeEnd_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {4}, -10, 10},
      /*output=*/{TensorType_INT8, {1}, -10, 10},
      /*begin*/ {TensorType_INT32, {1}},
      /*begin_data*/ {1},
      /*end*/ {TensorType_INT32, {1}},
      /*end_data*/ {-2},
      /*strides*/ {TensorType_INT32, {1}},
      /*strides_data*/ {1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2}));
}

TEST(StridedSliceOpModel, In2D_MultipleStrides_UInt8) {
  StridedSliceOpModel<uint8_t> m(
      /*input=*/{TensorType_UINT8, {2, 3}, -10, 10},
      /*output=*/{TensorType_UINT8, {1, 3}, -10, 10},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {1, -1},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {2, -4},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {2, -1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 5, 4}));
}

TEST(StridedSliceOpModel, In2D_EndMask_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {2, 3}, -127, 128},
      /*output=*/{TensorType_INT8, {1, 3}, -127, 128},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {1, 0},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {2, 2},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {1, 1},
      /*begin_mask*/ 0, /*end_mask*/ 2, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5, 6}));
}

TEST(StridedSliceOpModel, In2D_NegStrideBeginMask_UInt8) {
  StridedSliceOpModel<uint8_t> m(
      /*input=*/{TensorType_UINT8, {2, 3}, -10, 10},
      /*output=*/{TensorType_UINT8, {1, 3}, -10, 10},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {1, -2},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {2, -4},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {1, -1},
      /*begin_mask*/ 2, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 5, 4}));
}

TEST(StridedSliceOpModel, In2D_ShrinkAxis2_BeginEndAxis1_NegativeSlice_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {4, 1}, -10, 10},
      /*output=*/{TensorType_INT8, {4}, -10, 10},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {0, -1},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {0, 0},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {1, 1},
      /*begin_mask*/ 1, /*end_mask*/ 1, /*shrink_axis_mask*/ 2);
  m.SetInput({0, 1, 2, 3});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2, 3}));
}

TEST(StridedSliceOpModel, In2D_ShrinkAxisMask3_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {2, 3}, -10, 10},
      /*output=*/{TensorType_INT8, {}, -10, 10},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {0, 0},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {1, 1},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {1, 1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 3);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

TEST(StridedSliceOpModel, In3D_Identity_UInt8) {
  StridedSliceOpModel<uint8_t> m(
      /*input=*/{TensorType_UINT8, {2, 3, 2}, -15, 15},
      /*output=*/{TensorType_UINT8, {2, 3, 2}, -15, 15},
      /*begin*/ {TensorType_INT32, {3}},
      /*begin_data*/ {0, 0, 0},
      /*end*/ {TensorType_INT32, {3}},
      /*end_data*/ {2, 3, 2},
      /*strides*/ {TensorType_INT32, {3}},
      /*strides_data*/ {1, 1, 1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST(StridedSliceOpModel, In3D_IdentityShrinkAxis4_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {2, 3, 2}, -15, 15},
      /*output=*/{TensorType_INT8, {2, 3, 2}, -15, 15},
      /*begin*/ {TensorType_INT32, {3}},
      /*begin_data*/ {0, 0, 0},
      /*end*/ {TensorType_INT32, {3}},
      /*end_data*/ {2, 3, 1},
      /*strides*/ {TensorType_INT32, {3}},
      /*strides_data*/ {1, 1, 1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 4);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5, 7, 9, 11}));
}

}  // namespace tflite
