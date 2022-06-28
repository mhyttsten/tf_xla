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
class MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighbor_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighbor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighbor_testDTcc() {
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
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using uint8 = std::uint8_t;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

class ResizeNearestNeighborOpModel : public SingleOpModel {
 public:
  explicit ResizeNearestNeighborOpModel(const TensorData& input,
                                        std::initializer_list<int> size_data,
                                        TestType test_type,
                                        bool align_corners = false,
                                        bool half_pixel_centers = false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighbor_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/lite/kernels/resize_nearest_neighbor_test.cc", "ResizeNearestNeighborOpModel");

    bool const_size = (test_type == TestType::kConst);

    input_ = AddInput(input);
    if (const_size) {
      size_ = AddConstInput(TensorType_INT32, size_data, {2});
    } else {
      size_ = AddInput({TensorType_INT32, {2}});
    }
    output_ = AddOutput(input.type);
    SetBuiltinOp(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                 BuiltinOptions_ResizeNearestNeighborOptions,
                 CreateResizeNearestNeighborOptions(
                     builder_, /*align_corners*/ align_corners,
                     /*half_pixel_centers*/ half_pixel_centers)
                     .Union());
    if (const_size) {
      BuildInterpreter({GetShape(input_)});
    } else {
      BuildInterpreter({GetShape(input_), GetShape(size_)});
      PopulateTensor(size_, size_data);
    }
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSresize_nearest_neighbor_testDTcc mht_1(mht_1_v, 239, "", "./tensorflow/lite/kernels/resize_nearest_neighbor_test.cc", "SetInput");

    PopulateTensor(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 private:
  int input_;
  int size_;
  int output_;
};

class ResizeNearestNeighborOpTest : public ::testing::TestWithParam<TestType> {
};

TEST_P(ResizeNearestNeighborOpTest, HorizontalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 1, 2, 1}}, {1, 3},
                                 GetParam());
  m.SetInput<float>({3, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 6})));
}
TEST_P(ResizeNearestNeighborOpTest, HorizontalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 1, 2, 1}}, {1, 3},
                                 GetParam());
  m.SetInput<uint8>({3, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 6})));
}
TEST_P(ResizeNearestNeighborOpTest, HorizontalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 1, 2, 1}}, {1, 3},
                                 GetParam());
  m.SetInput<int8_t>({-3, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({-3, -3, 6})));
}
TEST_P(ResizeNearestNeighborOpTest, HorizontalResizeInt16) {
  ResizeNearestNeighborOpModel m({TensorType_INT16, {1, 1, 2, 1}}, {1, 3},
                                 GetParam());
  m.SetInput<int16_t>({-3, 6});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({-3, -3, 6})));
}
TEST_P(ResizeNearestNeighborOpTest, VerticalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 2, 1, 1}}, {3, 1},
                                 GetParam());
  m.SetInput<float>({3, 9});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 9})));
}
TEST_P(ResizeNearestNeighborOpTest, VerticalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 1, 1}}, {3, 1},
                                 GetParam());
  m.SetInput<uint8>({3, 9});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8>(),
              ElementsAreArray(ArrayFloatNear({3, 3, 9})));
}
TEST_P(ResizeNearestNeighborOpTest, VerticalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 2, 1, 1}}, {3, 1},
                                 GetParam());
  m.SetInput<int8_t>({3, -9});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({3, 3, -9})));
}
TEST_P(ResizeNearestNeighborOpTest, VerticalResizeInt16) {
  ResizeNearestNeighborOpModel m({TensorType_INT16, {1, 2, 1, 1}}, {3, 1},
                                 GetParam());
  m.SetInput<int16_t>({3, -9});
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray(ArrayFloatNear({3, 3, -9})));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<float>({
      3, 6,  //
      9, 12  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,   //
                                        3, 3, 6,   //
                                        9, 9, 12,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<uint8>({
      3, 6,  //
      9, 12  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,   //
                                        3, 3, 6,   //
                                        9, 9, 12,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<int8_t>({
      3, -6,  //
      9, 12   //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 3, -6,  //
                                         3, 3, -6,  //
                                         9, 9, 12,  //
                                     })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeInt16) {
  ResizeNearestNeighborOpModel m({TensorType_INT16, {1, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<int16_t>({
      3, -6,  //
      9, 12   //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(), ElementsAreArray(ArrayFloatNear({
                                          3, 3, -6,  //
                                          3, 3, -6,  //
                                          9, 9, 12,  //
                                      })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatches) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {2, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<float>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,     //
                                        3, 3, 6,     //
                                        9, 9, 12,    //
                                        4, 4, 10,    //
                                        4, 4, 10,    //
                                        10, 10, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest,
       TwoDimensionalResizeWithTwoBatches_AlignCorners) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {2, 2, 2, 1}}, {3, 3},
                                 GetParam(), /**align_corners**/ true);
  m.SetInput<float>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 6, 6,     //
                                        9, 12, 12,   //
                                        9, 12, 12,   //
                                        4, 10, 10,   //
                                        10, 16, 16,  //
                                        10, 16, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest,
       TwoDimensionalResizeWithTwoBatches_HalfPixelCenters) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {2, 2, 2, 1}}, {3, 3},
                                 GetParam(), /**align_corners**/ false,
                                 /**half_pixel_centers**/ true);
  m.SetInput<float>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 6, 6,     //
                                        9, 12, 12,   //
                                        9, 12, 12,   //
                                        4, 10, 10,   //
                                        10, 16, 16,  //
                                        10, 16, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, ThreeDimensionalResize) {
  ResizeNearestNeighborOpModel m({TensorType_FLOAT32, {1, 2, 2, 2}}, {3, 3},
                                 GetParam());
  m.SetInput<float>({
      3, 4, 6, 10,    //
      9, 10, 12, 16,  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 4, 3, 4, 6, 10,     //
                                        3, 4, 3, 4, 6, 10,     //
                                        9, 10, 9, 10, 12, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatchesUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {2, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<uint8>({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      12, 16  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 3, 6,     //
                                        3, 3, 6,     //
                                        9, 9, 12,    //
                                        4, 4, 10,    //
                                        4, 4, 10,    //
                                        12, 12, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatchesInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {2, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<int8_t>({
      3, 6,    //
      9, -12,  //
      -4, 10,  //
      12, 16   //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 3, 6,     //
                                         3, 3, 6,     //
                                         9, 9, -12,   //
                                         -4, -4, 10,  //
                                         -4, -4, 10,  //
                                         12, 12, 16,  //
                                     })));
}
TEST_P(ResizeNearestNeighborOpTest, TwoDimensionalResizeWithTwoBatchesInt16) {
  ResizeNearestNeighborOpModel m({TensorType_INT16, {2, 2, 2, 1}}, {3, 3},
                                 GetParam());
  m.SetInput<int16_t>({
      3, 6,    //
      9, -12,  //
      -4, 10,  //
      12, 16   //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(), ElementsAreArray(ArrayFloatNear({
                                          3, 3, 6,     //
                                          3, 3, 6,     //
                                          9, 9, -12,   //
                                          -4, -4, 10,  //
                                          -4, -4, 10,  //
                                          12, 12, 16,  //
                                      })));
}
TEST_P(ResizeNearestNeighborOpTest, ThreeDimensionalResizeUInt8) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 2, 2}}, {3, 3},
                                 GetParam());
  m.SetInput<uint8>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 4, 3, 4, 6, 10,       //
                                        3, 4, 3, 4, 6, 10,       //
                                        10, 12, 10, 12, 14, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, ThreeDimensionalResizeUInt8_AlignCorners) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 2, 2}}, {3, 3},
                                 GetParam(), /**align_corners**/ true);
  m.SetInput<uint8>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 4, 6, 10, 6, 10,      //
                                        10, 12, 14, 16, 14, 16,  //
                                        10, 12, 14, 16, 14, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest,
       ThreeDimensionalResizeUInt8_HalfPixelCenters) {
  ResizeNearestNeighborOpModel m({TensorType_UINT8, {1, 2, 2, 2}}, {3, 3},
                                 GetParam(), /**align_corners**/ false,
                                 /**half_pixel_centers**/ true);
  m.SetInput<uint8>({
      3, 4, 6, 10,     //
      10, 12, 14, 16,  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<uint8>(), ElementsAreArray(ArrayFloatNear({
                                        3, 4, 6, 10, 6, 10,      //
                                        10, 12, 14, 16, 14, 16,  //
                                        10, 12, 14, 16, 14, 16,  //
                                    })));
}
TEST_P(ResizeNearestNeighborOpTest, ThreeDimensionalResizeInt8) {
  ResizeNearestNeighborOpModel m({TensorType_INT8, {1, 2, 2, 2}}, {3, 3},
                                 GetParam());
  m.SetInput<int8_t>({
      3, 4, -6, 10,     //
      10, 12, -14, 16,  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray(ArrayFloatNear({
                                         3, 4, 3, 4, -6, 10,       //
                                         3, 4, 3, 4, -6, 10,       //
                                         10, 12, 10, 12, -14, 16,  //
                                     })));
}
TEST_P(ResizeNearestNeighborOpTest, ThreeDimensionalResizeInt16) {
  ResizeNearestNeighborOpModel m({TensorType_INT16, {1, 2, 2, 2}}, {3, 3},
                                 GetParam());
  m.SetInput<int16_t>({
      3, 4, -6, 10,     //
      10, 12, -14, 16,  //
  });
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int16_t>(), ElementsAreArray(ArrayFloatNear({
                                          3, 4, 3, 4, -6, 10,       //
                                          3, 4, 3, 4, -6, 10,       //
                                          10, 12, 10, 12, -14, 16,  //
                                      })));
}
INSTANTIATE_TEST_SUITE_P(ResizeNearestNeighborOpTest,
                         ResizeNearestNeighborOpTest,
                         testing::Values(TestType::kConst, TestType::kDynamic));

}  // namespace
}  // namespace tflite
