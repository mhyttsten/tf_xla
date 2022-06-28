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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSresize_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSresize_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSresize_testDTcc() {
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

class ResizeOpModel : public SingleOpModelWithHexagon {
 public:
  explicit ResizeOpModel(BuiltinOperator op_type, const TensorData& input,
                         std::initializer_list<int> size_data,
                         const TensorData& output, bool align_corners = false,
                         bool half_pixel_centers = false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSresize_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/hexagon/builders/tests/resize_test.cc", "ResizeOpModel");

    input_ = AddInput(input);
    size_ = AddConstInput(TensorType_INT32, size_data, {2});
    output_ = AddOutput(output);
    if (op_type == BuiltinOperator_RESIZE_NEAREST_NEIGHBOR) {
      SetBuiltinOp(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                   BuiltinOptions_ResizeNearestNeighborOptions,
                   CreateResizeNearestNeighborOptions(
                       builder_, /*align_corners*/ align_corners,
                       /*half_pixel_centers*/ half_pixel_centers)
                       .Union());
    } else {
      SetBuiltinOp(op_type, BuiltinOptions_ResizeBilinearOptions,
                   CreateResizeBilinearOptions(
                       builder_, /**align_corners**/ align_corners,
                       /**half_pixel_centers**/ half_pixel_centers)
                       .Union());
    }
    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSresize_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/lite/delegates/hexagon/builders/tests/resize_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  void SetQuantizedInput(std::initializer_list<float> data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSresize_testDTcc mht_2(mht_2_v, 233, "", "./tensorflow/lite/delegates/hexagon/builders/tests/resize_test.cc", "SetQuantizedInput");

    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  int input() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSresize_testDTcc mht_3(mht_3_v, 246, "", "./tensorflow/lite/delegates/hexagon/builders/tests/resize_test.cc", "input");
 return input_; }

 private:
  int input_;
  int size_;
  int output_;
};

TEST(ResizeOpModel, HorizontalResizeBiliear_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_UINT8, {1, 1, 2, 1}, -2.0, 10}, {1, 3},
                  {TensorType_UINT8, {}, -2.0, 10});
  m.SetQuantizedInput<uint8_t>({3, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6}, /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, HorizontalResizeNearestNeighbor_Int8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_INT8, {1, 1, 2, 1}, -2.0, 10}, {1, 3},
                  {TensorType_INT8, {}, -2.0, 10});
  m.SetQuantizedInput<int8_t>({3, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3.01176, 3.01176, 6.02353},
                                              /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, VerticalResizeBiliear_Int8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_INT8, {1, 2, 1, 1}, -2.0, 20}, {3, 1},
                  {TensorType_INT8, {}, -2.0, 20});
  m.SetQuantizedInput<int8_t>({3, 9});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 7, 9}, /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, VerticalResizeNearestNeighbor_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_UINT8, {1, 2, 1, 1}, -2.0, 20}, {3, 1},
                  {TensorType_UINT8, {}, -2.0, 20});
  m.SetQuantizedInput<uint8_t>({3, 9});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3.01961, 3.01961, 8.97255},
                                              /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, ThreeDimensionalResizeBiliear_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_UINT8, {1, 2, 2, 2}, -2, 30}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 30.0});
  m.SetQuantizedInput<uint8_t>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      3, 4, 5, 8, 6, 10,       //
                      7, 9, 10, 12, 11, 14,    //
                      10, 12, 12, 14, 14, 16,  //
                  },
                  /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, ThreeDimensionalResizeNearestNeighbor_Int8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_INT8, {1, 2, 2, 2}, -2, 30}, {3, 3},
                  {TensorType_INT8, {}, -2.0, 30.0});
  m.SetQuantizedInput<int8_t>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      3.01177, 4.01569, 3.01177, 4.01569, 6.02353, 10.0392,  //
                      3.01177, 4.01569, 3.01177, 4.01569, 6.02353, 10.0392,  //
                      10.0392, 12.0471, 10.0392, 12.0471, 14.0549, 16.0627,  //
                  },
                  /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, TwoDimensionalResizeBilinearWithTwoBatches_Int8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_INT8, {2, 2, 2, 1}, -2, 30}, {3, 3},
                  {TensorType_INT8, {}, -2.0, 30.0});
  m.SetQuantizedInput<int8_t>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(), ElementsAreArray(ArrayFloatNear(
                                                    {
                                                        3, 5, 6,     //
                                                        7, 9, 10,    //
                                                        9, 11, 12,   //
                                                        4, 8, 10,    //
                                                        9, 12, 14,   //
                                                        12, 14, 16,  //
                                                    },
                                                    /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, TwoDimensionalResizeNNWithTwoBatches_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_UINT8, {2, 2, 2, 1}, -2, 30}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 30.0});
  m.SetQuantizedInput<uint8_t>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      3.01177, 3.01177, 6.02353,  //
                      3.01177, 3.01177, 6.02353,  //
                      9.03529, 9.03529, 12.0471,  //
                      4.01569, 4.01569, 10.0392,  //
                      4.01569, 4.01569, 10.0392,  //
                      12.0471, 12.0471, 16.0627,  //
                  },
                  /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, TwoDimResizeBilinearWithTwoBatches_HalfPixelCenters_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_UINT8, {2, 2, 2, 1}, -2.0, 20}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 20}, /**align_corners**/ false,
                  /**half_pixel_centers**/ true);
  m.SetQuantizedInput<uint8_t>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({2, 4, 6,    //
                                               6, 7, 9,    //
                                               9, 10, 12,  //
                                               4, 7, 10,   //
                                               8, 10, 13,  //
                                               12, 14, 16},
                                              /*max_abs_error=*/2)));
}

TEST(ResizeOpModel, TwoDimResizeBilinearWithTwoBatches_AlignCorners_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_BILINEAR,
                  {TensorType_UINT8, {2, 2, 2, 1}, -2.0, 20}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 20}, /**align_corners**/ true,
                  /**half_pixel_centers**/ false);
  m.SetQuantizedInput<uint8_t>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6,    //
                                               7, 9, 10,   //
                                               9, 11, 12,  //
                                               4, 8, 10,   //
                                               9, 12, 13,  //
                                               12, 15, 16},
                                              /*max_abs_error=*/2)));
}

TEST(ResizeOpModel, ThreeDimensionalResizeNN_AlignCorners_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_UINT8, {1, 2, 2, 2}, -2.0, 20}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 20}, /**align_corners**/ true);
  m.SetQuantizedInput<uint8_t>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 4, 6, 10, 6, 10,      //
                                               10, 12, 14, 16, 14, 16,  //
                                               10, 12, 14, 16, 14, 16},
                                              /*max_abs_error=*/1)));
}

TEST(ResizeOpModel, ThreeDimensionalResizeNN_HalfPixelCenters_UInt8) {
  ResizeOpModel m(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                  {TensorType_UINT8, {1, 2, 2, 2}, -2.0, 20}, {3, 3},
                  {TensorType_UINT8, {}, -2.0, 20}, /**align_corners**/ false,
                  /**half_pixel_centers**/ true);
  m.SetQuantizedInput<uint8_t>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 4, 6, 10, 6, 10,      //
                                               10, 12, 14, 16, 14, 16,  //
                                               10, 12, 14, 16, 14, 16},
                                              /*max_abs_error=*/1)));
}

}  // namespace tflite
