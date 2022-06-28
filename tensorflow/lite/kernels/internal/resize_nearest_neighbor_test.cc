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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSresize_nearest_neighbor_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSresize_nearest_neighbor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSresize_nearest_neighbor_testDTcc() {
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
#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/test_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace {

template <typename T>
void TestReferenceResizeNearestNeighbor(
    const RuntimeShape& input_shape, const std::vector<T>& input_data,
    const std::vector<int32>& output_size_data,
    const RuntimeShape& output_shape,
    const std::vector<T>& expected_output_data, bool align_corners = false,
    bool half_pixel_centers = false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSresize_nearest_neighbor_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/internal/resize_nearest_neighbor_test.cc", "TestReferenceResizeNearestNeighbor");

  ResizeNearestNeighborParams op_params{align_corners, half_pixel_centers};
  RuntimeShape output_size_shape({1, 1, 1, 2});

  std::vector<T> output_data(expected_output_data.size());
  reference_ops::ResizeNearestNeighbor(
      op_params, input_shape, input_data.data(), output_size_shape,
      output_size_data.data(), output_shape, output_data.data());
  ASSERT_EQ(expected_output_data, output_data);
}

// Consistency test values are from
// third_party/tensorflow/core/kernels/resize_nearest_neighbor_op_test.cc.

TEST(ResizeNearestNeighborReference, Test2x2To1x1) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<float> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {1, 1};
  RuntimeShape output_shape = {1, 1, 1, 1};
  std::vector<float> output_data = {1};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data);
}

TEST(ResizeNearestNeighborReference, Test2x2To1x1_AlignCorners) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<float> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {1, 1};
  RuntimeShape output_shape = {1, 1, 1, 1};
  std::vector<float> output_data = {1};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data,
                                     /*align_corners=*/true);
}

TEST(ResizeNearestNeighborReference, Test2x2To1x1_HalfPixelCenters) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<float> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {1, 1};
  RuntimeShape output_shape = {1, 1, 1, 1};
  std::vector<float> output_data = {4};

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/false, /*half_pixel_centers=*/true);
}

TEST(ResizeNearestNeighborReference, Test2x2To3x3) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<uint8> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {1, 3, 3, 1};
  std::vector<uint8> output_data = {1, 1, 2, 1, 1, 2, 3, 3, 4};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data);
}

TEST(ResizeNearestNeighborReference, Test2x2To3x3Int16) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<int16_t> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {1, 3, 3, 1};
  std::vector<int16_t> output_data = {1, 1, 2, 1, 1, 2, 3, 3, 4};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data);
}

TEST(ResizeNearestNeighborReference, Test2x2To3x3_AlignCorners) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<uint8> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {1, 3, 3, 1};
  std::vector<uint8> output_data = {1, 2, 2, 3, 4, 4, 3, 4, 4};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data,
                                     /*align_corners=*/true);
}

TEST(ResizeNearestNeighborReference, Test2x2To3x3_HalfPixelCenters) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<uint8> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {1, 3, 3, 1};
  std::vector<uint8> output_data = {1, 2, 2, 3, 4, 4, 3, 4, 4};

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/false, /*half_pixel_centers=*/true);
}

TEST(ResizeNearestNeighborReference, Test3x3To2x2) {
  RuntimeShape input_shape = {1, 3, 3, 1};
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int32> output_size_data = {2, 2};
  RuntimeShape output_shape = {1, 2, 2, 1};
  std::vector<float> output_data = {1, 2, 4, 5};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data);
}

TEST(ResizeNearestNeighborReference, Test3x3To2x2_AlignCorners) {
  RuntimeShape input_shape = {1, 3, 3, 1};
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int32> output_size_data = {2, 2};
  RuntimeShape output_shape = {1, 2, 2, 1};
  std::vector<float> output_data = {1, 3, 7, 9};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data,
                                     /*align_corners=*/true);
}

TEST(ResizeNearestNeighborReference, Test3x3To2x2_HalfPixelCenters) {
  RuntimeShape input_shape = {1, 3, 3, 1};
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int32> output_size_data = {2, 2};
  RuntimeShape output_shape = {1, 2, 2, 1};
  std::vector<float> output_data = {1, 3, 7, 9};

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/false, /*half_pixel_centers=*/true);
}

TEST(ResizeNearestNeighborReference, Test2x2To2x5) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<uint8> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {2, 5};
  RuntimeShape output_shape = {1, 2, 5, 1};
  std::vector<uint8> output_data = {1, 1, 1, 2, 2, 3, 3, 3, 4, 4};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data);
}

TEST(ResizeNearestNeighborReference, Test2x2To2x5_HalfPixelCenters) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<uint8> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {2, 5};
  RuntimeShape output_shape = {1, 2, 5, 1};
  std::vector<uint8> output_data = {1, 1, 2, 2, 2, 3, 3, 4, 4, 4};

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/false, /*half_pixel_centers=*/true);
}

TEST(ResizeNearestNeighborReference, Test4x4To3x3) {
  RuntimeShape input_shape = {1, 4, 4, 1};
  std::vector<uint8> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                   9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {1, 3, 3, 1};
  std::vector<uint8> output_data = {1, 2, 3, 5, 6, 7, 9, 10, 11};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data);
}

TEST(ResizeNearestNeighborReference, Test4x4To3x3_AlignCorners) {
  RuntimeShape input_shape = {1, 4, 4, 1};
  std::vector<uint8> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                   9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {1, 3, 3, 1};
  std::vector<uint8> output_data = {1, 3, 4, 9, 11, 12, 13, 15, 16};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data,
                                     /*align_corners=*/true);
}

TEST(ResizeNearestNeighborReference, Test4x4To3x3_HalfPixelCenters) {
  RuntimeShape input_shape = {1, 4, 4, 1};
  std::vector<uint8> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                   9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {1, 3, 3, 1};
  std::vector<uint8> output_data = {1, 3, 4, 9, 11, 12, 13, 15, 16};

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/false, /*half_pixel_centers=*/true);
}

TEST(ResizeNearestNeighborReference, Test2x2To5x2) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<float> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {5, 2};
  RuntimeShape output_shape = {1, 5, 2, 1};
  std::vector<float> output_data = {1, 2, 1, 2, 1, 2, 3, 4, 3, 4};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data);
}

TEST(ResizeNearestNeighborReference, Test2x2To5x2_HalfPixelCenters) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<float> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {5, 2};
  RuntimeShape output_shape = {1, 5, 2, 1};
  std::vector<float> output_data = {1, 2, 1, 2, 3, 4, 3, 4, 3, 4};

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/false, /*half_pixel_centers=*/true);
}

TEST(ResizeNearestNeighborReference,
     Test2x2To5x2_HalfPixelCenters_AlignCorners) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<float> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {5, 2};
  RuntimeShape output_shape = {1, 5, 2, 1};
  std::vector<float> output_data = {2, 2, 2, 2, 4, 4, 4, 4, 4, 4};

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/true, /*half_pixel_centers=*/true);
}

TEST(ResizeNearestNeighborReference, Test2x2To4x4) {
  RuntimeShape input_shape = {1, 2, 2, 1};
  std::vector<uint8> input_data = {1, 2, 3, 4};
  std::vector<int32> output_size_data = {4, 4};
  RuntimeShape output_shape = {1, 4, 4, 1};
  std::vector<uint8> output_data = {1, 1, 2, 2, 1, 1, 2, 2,
                                    3, 3, 4, 4, 3, 3, 4, 4};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data);
}

TEST(ResizeNearestNeighborReference, Test2x2x2x2To2x3x3x2) {
  // Input:
  //  [ [ 1, 1 ], [ 2, 2 ],
  //    [ 3, 3 ], [ 4, 4 ] ],
  //  [ [ 5, 5 ], [ 6, 6 ],
  //    [ 7, 7 ], [ 8, 8 ] ]
  RuntimeShape input_shape = {2, 2, 2, 2};
  std::vector<float> input_data = {1, 1, 2, 2, 3, 3, 4, 4,
                                   5, 5, 6, 6, 7, 7, 8, 8};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {2, 3, 3, 2};
  // Output:
  //  [ [ 1, 1 ], [ 1, 1 ], [ 2, 2 ],
  //    [ 1, 1 ], [ 1, 1 ], [ 2, 2 ],
  //    [ 3, 3 ], [ 3, 3 ], [ 4, 4 ] ],
  //  [ [ 5, 5 ], [ 5, 5 ], [ 6, 6 ],
  //    [ 5, 5 ], [ 5, 5 ], [ 6, 6 ],
  //    [ 7, 7 ], [ 7, 7 ], [ 8, 8 ] ]
  std::vector<float> output_data = {1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2,
                                    3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6,
                                    5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8};

  TestReferenceResizeNearestNeighbor(input_shape, input_data, output_size_data,
                                     output_shape, output_data);
}

TEST(ResizeNearestNeighborReference, Test2x2x2x2To2x3x3x2_AlignCorners) {
  RuntimeShape input_shape = {2, 2, 2, 2};
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8,
                                   1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {2, 3, 3, 2};
  std::vector<float> output_data = {
      1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 7, 8, 5, 6, 7, 8, 7, 8,
      1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 7, 8, 5, 6, 7, 8, 7, 8,
  };

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/true, /*half_pixel_centers=*/false);
}

TEST(ResizeNearestNeighborReference, Test2x2x2x2To2x3x3x2_HalfPixelCenters) {
  RuntimeShape input_shape = {2, 2, 2, 2};
  std::vector<float> input_data = {1, 1, 2, 2, 3, 3, 4, 4,
                                   5, 5, 6, 6, 7, 7, 8, 8};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {2, 3, 3, 2};
  std::vector<float> output_data = {1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4,
                                    3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6,
                                    7, 7, 8, 8, 8, 8, 7, 7, 8, 8, 8, 8};

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/false, /*half_pixel_centers=*/true);
}

TEST(ResizeNearestNeighborReference,
     Test2x2x2x2To2x3x3x2_HalfPixelCenters_AlignCorners) {
  RuntimeShape input_shape = {2, 2, 2, 2};
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8,
                                   1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32> output_size_data = {3, 3};
  RuntimeShape output_shape = {2, 3, 3, 2};
  std::vector<float> output_data = {1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 7, 8,
                                    5, 6, 7, 8, 7, 8, 1, 2, 3, 4, 3, 4,
                                    5, 6, 7, 8, 7, 8, 5, 6, 7, 8, 7, 8};

  TestReferenceResizeNearestNeighbor(
      input_shape, input_data, output_size_data, output_shape, output_data,
      /*align_corners=*/true, /*half_pixel_centers=*/true);
}

void TestOptimizedResizeNearestNeighbor(int batch, int depth, int input_width,
                                        int input_height, int output_width,
                                        int output_height) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSresize_nearest_neighbor_testDTcc mht_1(mht_1_v, 520, "", "./tensorflow/lite/kernels/internal/resize_nearest_neighbor_test.cc", "TestOptimizedResizeNearestNeighbor");

  RuntimeShape output_size_shape({1, 1, 1, 2});

  RuntimeShape input_shape({batch, input_height, input_width, depth});
  RuntimeShape output_shape({batch, output_height, output_width, depth});

  std::vector<uint8> input_data(input_shape.FlatSize(), 0);
  FillRandom(&input_data, static_cast<uint8>(0), static_cast<uint8>(255));

  std::vector<uint8> reference_output_data(output_shape.FlatSize(), 0);
  // Initialize the output data with something other than zero, so we can catch
  // issue with kernels failing to initialize the output.
  std::vector<uint8> output_data(output_shape.FlatSize(), 3);
  std::vector<int32> output_size_data = {output_height, output_width};

  ResizeNearestNeighborParams op_params{/*align_corners=*/false,
                                        /*half_pixel_centers=*/false};

  // Test the optimized version against the reference version.
  reference_ops::ResizeNearestNeighbor(
      op_params, input_shape, input_data.data(), output_size_shape,
      output_size_data.data(), output_shape, reference_output_data.data());
  optimized_ops::ResizeNearestNeighbor(
      op_params, input_shape, input_data.data(), output_size_shape,
      output_size_data.data(), output_shape, output_data.data());
  ASSERT_EQ(reference_output_data, output_data);

  op_params.align_corners = true;
  reference_ops::ResizeNearestNeighbor(
      op_params, input_shape, input_data.data(), output_size_shape,
      output_size_data.data(), output_shape, reference_output_data.data());
  optimized_ops::ResizeNearestNeighbor(
      op_params, input_shape, input_data.data(), output_size_shape,
      output_size_data.data(), output_shape, output_data.data());
  ASSERT_EQ(reference_output_data, output_data);

  op_params.align_corners = false;
  op_params.half_pixel_centers = true;
  reference_ops::ResizeNearestNeighbor(
      op_params, input_shape, input_data.data(), output_size_shape,
      output_size_data.data(), output_shape, reference_output_data.data());
  optimized_ops::ResizeNearestNeighbor(
      op_params, input_shape, input_data.data(), output_size_shape,
      output_size_data.data(), output_shape, output_data.data());
  ASSERT_EQ(reference_output_data, output_data);

  op_params.align_corners = true;
  op_params.half_pixel_centers = true;
  reference_ops::ResizeNearestNeighbor(
      op_params, input_shape, input_data.data(), output_size_shape,
      output_size_data.data(), output_shape, reference_output_data.data());
  optimized_ops::ResizeNearestNeighbor(
      op_params, input_shape, input_data.data(), output_size_shape,
      output_size_data.data(), output_shape, output_data.data());
  ASSERT_EQ(reference_output_data, output_data);
}

// Since the optimized version uses fixed-point and the reference version uses
// float, offsets may differ. Test if the input/output image combination results
// in the same offsets before running parity tests.
bool is_valid_scale(int input_width, int input_height, int output_width,
                    int output_height) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSresize_nearest_neighbor_testDTcc mht_2(mht_2_v, 584, "", "./tensorflow/lite/kernels/internal/resize_nearest_neighbor_test.cc", "is_valid_scale");

  const float height_scale_float =
      static_cast<float>(input_height) / output_height;
  const float width_scale_float =
      static_cast<float>(input_width) / output_width;

  int32 height_scale_int = (input_height << 16) / output_height + 1;
  int32 width_scale_int = (input_width << 16) / output_width + 1;

  for (int y = 0; y < output_height; ++y) {
    int32 in_y_float =
        std::min(static_cast<int32>(std::floor(y * height_scale_float)),
                 input_height - 1);
    int32 in_y_int = std::min((y * height_scale_int) >> 16, input_height - 1);
    if (in_y_int != in_y_float) {
      return false;
    }
    for (int x = 0; x < output_width; ++x) {
      int32 in_x_float =
          std::min(static_cast<int32>(std::floor(x * width_scale_float)),
                   input_width - 1);
      int32 in_x_int = std::min((x * width_scale_int) >> 16, input_width - 1);
      if (in_x_int != in_x_float) {
        return false;
      }
    }
  }
  return true;
}

TEST(ResizeNearestNeighborOptimized, TestReferenceParity) {
  int invalid_count = 0;
  const int kTestsToRun = 10000;
  for (int i = 0; i < kTestsToRun; i++) {
    const int batch = ExponentialRandomPositiveInt(0.9f, 3, 20);
    const int depth = ExponentialRandomPositiveInt(0.9f, 6, 50);
    const int input_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int input_height = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_width = ExponentialRandomPositiveInt(0.9f, 20, 200);
    const int output_height = ExponentialRandomPositiveInt(0.9f, 20, 200);

    if (is_valid_scale(input_width, input_height, output_width,
                       output_height)) {
      TestOptimizedResizeNearestNeighbor(
          batch, depth, input_width, input_height, output_width, output_height);
    } else {
      invalid_count++;
    }
  }
  // Test that the total number of invalid tests are a small percentage.
  ASSERT_LT(static_cast<float>(invalid_count) / kTestsToRun, 0.001f);
}

}  // namespace
}  // namespace tflite
