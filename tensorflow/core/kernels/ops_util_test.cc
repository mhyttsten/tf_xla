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
class MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/ops_util.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class OpsUtilTest : public ::testing::Test {
 protected:
  OpsUtilTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/ops_util_test.cc", "OpsUtilTest");
}
  ~OpsUtilTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc mht_1(mht_1_v, 201, "", "./tensorflow/core/kernels/ops_util_test.cc", "~OpsUtilTest");
}

  // Padding structure.
  struct padding_struct {
    // Input parameters.
    struct {
      int in_height;
      int in_width;
      int filter_height;
      int filter_width;
      int row_stride;
      int col_stride;
      Padding padding;
    } input;
    // Output.
    struct {
      int new_height;
      int new_width;
      int pad_top;
      int pad_bottom;
      int pad_left;
      int pad_right;
    } output;
  };

  // Broadcast structure.
  struct bcast_struct {
    // Input parameters.
    struct {
      int index;     // Current index.
      int in_size;   // Size of the dimension.
      int ksize;     // Kernel size.
      int stride;    // Stride.
      int pad_size;  // Padding size.
    } input;
    // Output.
    struct {
      int new_index;  // New starting index.
      int new_size;   // New broadcast size.
    } output;
  };

  static void VerifyGet2dOutputSizeBoundaries(padding_struct pad_struct,
                                              error::Code code) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/kernels/ops_util_test.cc", "VerifyGet2dOutputSizeBoundaries");

    int64_t new_height, new_width, pad_rows, pad_cols;
    Status status = GetWindowedOutputSize(
        pad_struct.input.in_height, pad_struct.input.filter_height,
        pad_struct.input.row_stride, pad_struct.input.padding, &new_height,
        &pad_rows);
    EXPECT_EQ(status.code(), code) << status;
    status = GetWindowedOutputSize(
        pad_struct.input.in_width, pad_struct.input.filter_width,
        pad_struct.input.col_stride, pad_struct.input.padding, &new_width,
        &pad_cols);
    EXPECT_EQ(status.code(), code) << status;
  }

  static void VerifyGet2dOutputSizeValues(padding_struct pad_struct,
                                          error::Code code) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc mht_3(mht_3_v, 265, "", "./tensorflow/core/kernels/ops_util_test.cc", "VerifyGet2dOutputSizeValues");

    int64_t new_height, new_width, pad_rows, pad_cols;
    Status status = GetWindowedOutputSize(
        pad_struct.input.in_height, pad_struct.input.filter_height,
        pad_struct.input.row_stride, pad_struct.input.padding, &new_height,
        &pad_rows);
    EXPECT_EQ(status.code(), code) << status;
    status = GetWindowedOutputSize(
        pad_struct.input.in_width, pad_struct.input.filter_width,
        pad_struct.input.col_stride, pad_struct.input.padding, &new_width,
        &pad_cols);
    EXPECT_EQ(status.code(), code) << status;
    EXPECT_EQ(pad_struct.output.new_height, new_height);
    EXPECT_EQ(pad_struct.output.new_width, new_width);
    EXPECT_EQ(pad_struct.output.pad_top, pad_rows);
    EXPECT_EQ(pad_struct.output.pad_left, pad_cols);
  }

  static void VerifyGet2dOutputVerboseSizeValues(padding_struct pad_struct,
                                                 error::Code code) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc mht_4(mht_4_v, 287, "", "./tensorflow/core/kernels/ops_util_test.cc", "VerifyGet2dOutputVerboseSizeValues");

    int64_t new_height, new_width, pad_top, pad_bottom, pad_left, pad_right;
    Status status = GetWindowedOutputSizeVerbose(
        pad_struct.input.in_height, pad_struct.input.filter_height,
        pad_struct.input.row_stride, pad_struct.input.padding, &new_height,
        &pad_top, &pad_bottom);
    EXPECT_EQ(status.code(), code) << status;
    status = GetWindowedOutputSizeVerbose(
        pad_struct.input.in_width, pad_struct.input.filter_width,
        pad_struct.input.col_stride, pad_struct.input.padding, &new_width,
        &pad_left, &pad_right);
    EXPECT_EQ(status.code(), code) << status;
    EXPECT_EQ(pad_struct.output.new_height, new_height);
    EXPECT_EQ(pad_struct.output.new_width, new_width);
    EXPECT_EQ(pad_struct.output.pad_top, pad_top);
    EXPECT_EQ(pad_struct.output.pad_bottom, pad_bottom);
    EXPECT_EQ(pad_struct.output.pad_left, pad_left);
    EXPECT_EQ(pad_struct.output.pad_right, pad_right);
  }

  static void VerifyBoundaries(bcast_struct bcast, error::Code code) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc mht_5(mht_5_v, 310, "", "./tensorflow/core/kernels/ops_util_test.cc", "VerifyBoundaries");

    int new_index, new_size;
    Status status = GetBroadcastSize(
        bcast.input.index, bcast.input.in_size, bcast.input.ksize,
        bcast.input.stride, bcast.input.pad_size, &new_index, &new_size);
    EXPECT_EQ(status.code(), code) << status;
  }

  static void VerifyBcastValues(bcast_struct bcast) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSops_util_testDTcc mht_6(mht_6_v, 321, "", "./tensorflow/core/kernels/ops_util_test.cc", "VerifyBcastValues");

    int new_index, new_size;
    EXPECT_EQ(Status::OK(),
              GetBroadcastSize(bcast.input.index, bcast.input.in_size,
                               bcast.input.ksize, bcast.input.stride,
                               bcast.input.pad_size, &new_index, &new_size));
    EXPECT_EQ(bcast.output.new_index, new_index);
    EXPECT_EQ(bcast.output.new_size, new_size);
  }
};

TEST_F(OpsUtilTest, Get2dOutputSizeNegativeSizeTest) {
  padding_struct pad_struct = {{1, 1, 3, 3, 1, 1, VALID}, {-1, -1, 0, 0, 0, 0}};
  VerifyGet2dOutputSizeBoundaries(pad_struct, error::INVALID_ARGUMENT);
}

TEST_F(OpsUtilTest, Get2dOutputSizeSquareFilterTest) {
  padding_struct pad_struct1 = {{3, 3, 2, 2, 2, 2, SAME}, {2, 2, 0, 0, 0, 0}};
  padding_struct pad_struct2 = {{3, 3, 2, 2, 2, 2, VALID}, {1, 1, 0, 0, 0, 0}};
  VerifyGet2dOutputSizeValues(pad_struct1, error::OK);
  VerifyGet2dOutputSizeValues(pad_struct2, error::OK);
}

TEST_F(OpsUtilTest, Get2dOutputSizeNonSquareFilterTest) {
  padding_struct pad_struct1 = {{4, 5, 1, 2, 1, 1, SAME}, {4, 5, 0, 0, 0, 0}};
  padding_struct pad_struct2 = {{4, 5, 1, 2, 1, 1, VALID}, {4, 4, 0, 0, 0, 0}};
  VerifyGet2dOutputSizeValues(pad_struct1, error::OK);
  VerifyGet2dOutputSizeValues(pad_struct2, error::OK);
}

TEST_F(OpsUtilTest, Get2dOutputSizeUnevenStrideTest) {
  padding_struct pad_struct1 = {{4, 4, 2, 2, 1, 2, VALID}, {3, 2, 0, 0, 0, 0}};
  padding_struct pad_struct2 = {{4, 4, 2, 2, 2, 1, VALID}, {2, 3, 0, 0, 0, 0}};
  VerifyGet2dOutputSizeValues(pad_struct1, error::OK);
  VerifyGet2dOutputSizeValues(pad_struct2, error::OK);
}

TEST_F(OpsUtilTest, Get2dOutputSizeVerbose) {
  padding_struct pad_struct1 = {{3, 3, 2, 2, 2, 2, SAME}, {2, 2, 0, 1, 0, 1}};
  padding_struct pad_struct2 = {{3, 3, 2, 2, 2, 2, VALID}, {1, 1, 0, 0, 0, 0}};
  VerifyGet2dOutputVerboseSizeValues(pad_struct1, error::OK);
  VerifyGet2dOutputVerboseSizeValues(pad_struct2, error::OK);
}

// Test index * stride > in_size fails with INVALID_ARGUMENT.
TEST_F(OpsUtilTest, GetBroadcastTestBadIndex) {
  bcast_struct bcast = {{2, 3, 1, 2, 0}, {0, 3}};
  VerifyBoundaries(bcast, error::INVALID_ARGUMENT);
}

// in_size = 3, ksize = 3, stride = 1, pad_size = 0
TEST_F(OpsUtilTest, GetBroadcastTest3_3_1_0) {
  bcast_struct bcast[] = {
      {{0, 3, 3, 1, 0}, {0, 3}},
      {{1, 3, 3, 1, 0}, {1, 2}},
      {{2, 3, 3, 1, 0}, {2, 1}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 3, stride = 1, pad_size = 1
TEST_F(OpsUtilTest, GetBroadcastTest3_3_1_1) {
  bcast_struct bcast[] = {
      {{0, 3, 3, 1, 1}, {0, 2}},
      {{1, 3, 3, 1, 1}, {0, 3}},
      {{2, 3, 3, 1, 1}, {1, 2}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 3, stride = 1, pad_size = 2
TEST_F(OpsUtilTest, GetBroadcastTest3_3_1_2) {
  bcast_struct bcast[] = {
      {{0, 3, 3, 1, 2}, {0, 1}},
      {{1, 3, 3, 1, 2}, {0, 2}},
      {{2, 3, 3, 1, 2}, {0, 3}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 3, stride = 2, pad_size = 0
TEST_F(OpsUtilTest, GetBroadcastTest3_3_2_0) {
  bcast_struct bcast[] = {
      {{0, 3, 3, 2, 0}, {0, 3}},
      {{1, 3, 3, 2, 0}, {2, 1}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 3, stride = 2, pad_size = 1
TEST_F(OpsUtilTest, GetBroadcastTest3_3_2_1) {
  bcast_struct bcast[] = {
      {{0, 3, 3, 2, 1}, {0, 2}},
      {{1, 3, 3, 2, 1}, {1, 2}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 3, stride = 2, pad_size = 2
TEST_F(OpsUtilTest, GetBroadcastTest3_3_2_2) {
  bcast_struct bcast[] = {
      {{0, 3, 3, 2, 2}, {0, 1}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 3, stride = 3, pad_size = 0
TEST_F(OpsUtilTest, GetBroadcastTest3_3_3_0) {
  bcast_struct bcast[] = {
      {{0, 3, 3, 3, 0}, {0, 3}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 3, stride = 3, pad_size = 1
TEST_F(OpsUtilTest, GetBroadcastTest3_3_3_1) {
  bcast_struct bcast[] = {
      {{0, 3, 3, 3, 1}, {0, 2}},
      {{1, 3, 3, 3, 1}, {2, 1}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 3, stride = 3, pad_size = 2
TEST_F(OpsUtilTest, GetBroadcastTest3_3_3_2) {
  bcast_struct bcast[] = {
      {{0, 3, 3, 3, 2}, {0, 1}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 1, stride = 2, pad_size = 0
TEST_F(OpsUtilTest, GetBroadcastTest3_1_2_0) {
  bcast_struct bcast[] = {
      {{0, 3, 1, 2, 0}, {0, 1}},
      {{1, 3, 1, 2, 0}, {2, 1}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 2, stride = 3, pad_size = 0
TEST_F(OpsUtilTest, GetBroadcastTest3_2_3_0) {
  bcast_struct bcast[] = {
      {{0, 3, 2, 3, 0}, {0, 2}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

// in_size = 3, ksize = 2, stride = 3, pad_size = 1
TEST_F(OpsUtilTest, GetBroadcastTest3_2_3_1) {
  bcast_struct bcast[] = {
      {{0, 3, 2, 3, 1}, {0, 1}},
      {{1, 3, 2, 3, 1}, {2, 1}},
  };
  for (size_t i = 0; i < sizeof(bcast) / sizeof(bcast[0]); ++i) {
    VerifyBcastValues(bcast[i]);
  }
}

TEST_F(OpsUtilTest, SanitizeThreadSuffix) {
  EXPECT_EQ("_aBc123_-___", SanitizeThreadSuffix("/aBc123_-  /"));
}

TEST_F(OpsUtilTest, Aligned1DSlice) {
#if EIGEN_MAX_ALIGN_BYTES == 0
  // When EIGEN_MAX_ALIGN_BYTES is 0, a 1D tensor is always aligned.
  Tensor t(DT_FLOAT, TensorShape({3}));
  int64 start = 0;
  int64 end = 1;
  bool output = IsDim0SliceAligned<float>(t.shape(), start, end);
  EXPECT_EQ(output, true);
#else
  Tensor t(DT_FLOAT, TensorShape({EIGEN_MAX_ALIGN_BYTES * 2}));
  int64_t start = 0;
  int64_t end = EIGEN_MAX_ALIGN_BYTES;
  bool output = IsDim0SliceAligned<float>(t.shape(), start, end);
  EXPECT_EQ(output, true);
  // Checks sliced 1D tensor is aligned for sanity.
  Tensor sliced;
  CHECK(sliced.CopyFrom(t.Slice(start, end), TensorShape({end - start})));
  EXPECT_EQ(sliced.IsAligned(), true);
#endif
}

#if EIGEN_MAX_ALIGN_BYTES > 0
TEST_F(OpsUtilTest, Misaligned1DSlice) {
  Tensor t(DT_FLOAT, TensorShape({EIGEN_MAX_ALIGN_BYTES * 2}));
  int64_t start = 1;
  int64_t end = EIGEN_MAX_ALIGN_BYTES + 1;
  bool output = IsDim0SliceAligned<float>(t.shape(), start, end);
  EXPECT_EQ(output, false);
  // Checks sliced 1D tensor is misaligned for sanity.
  Tensor sliced;
  CHECK(sliced.CopyFrom(t.Slice(start, end), TensorShape({end - start})));
  EXPECT_EQ(sliced.IsAligned(), false);
}
#endif

TEST_F(OpsUtilTest, Aligned2DSliceOfDim0) {
#if EIGEN_MAX_ALIGN_BYTES == 0
  // When EIGEN_MAX_ALIGN_BYTES is 0 and the size of the first dimension is
  // nonzero, a multidimensional tensor is always aligned.
  Tensor t(DT_FLOAT, TensorShape({3, 4}));
  int64 start = 1;
  int64 end = 2;
  bool output = IsDim0SliceAligned<float>(t.shape(), start, end);
  EXPECT_EQ(output, true);
#else
  // For multidimensional tensors, alignment is dictated by inner_dim_size.
  int64_t inner_dim_size = EIGEN_MAX_ALIGN_BYTES;
  Tensor t(DT_FLOAT, TensorShape({3, inner_dim_size}));
  int64_t start = 1;
  int64_t end = 2;
  bool output = IsDim0SliceAligned<float>(t.shape(), start, end);
  EXPECT_EQ(output, true);
  // Checks sliced 2D is aligned, for sanity.
  Tensor sliced;
  CHECK(sliced.CopyFrom(t.Slice(start, end), TensorShape({1, inner_dim_size})));
  EXPECT_EQ(sliced.IsAligned(), true);
#endif
}

#if EIGEN_MAX_ALIGN_BYTES > 0
TEST_F(OpsUtilTest, Misaligned2DSliceOfDim0) {
  // For multidimensional tensors, alignment is dictated by inner_dim_size.
  int64_t inner_dim_size = EIGEN_MAX_ALIGN_BYTES + 1;
  Tensor t(DT_FLOAT, TensorShape({3, inner_dim_size}));
  int64_t start = 1;
  int64_t end = 2;
  bool output = IsDim0SliceAligned<float>(t.shape(), start, end);
  EXPECT_EQ(output, false);
  // Checks sliced 2D is misaligned, for sanity.
  Tensor sliced;
  CHECK(sliced.CopyFrom(t.Slice(start, end), TensorShape({1, inner_dim_size})));
  EXPECT_EQ(sliced.IsAligned(), false);
}
#endif

TEST_F(OpsUtilTest, MisalignedEmptyShape) {
  TensorShape shape({});
  int64_t start = 1;
  int64_t end = 2;
  bool output = IsDim0SliceAligned<float>(shape, start, end);
  EXPECT_EQ(output, false);
}

TEST_F(OpsUtilTest, MisalignedEmptyDim0) {
  TensorShape shape({0, 1, 2});
  int64_t start = 0;
  int64_t end = 1;
  bool output = IsDim0SliceAligned<float>(shape, start, end);
  EXPECT_EQ(output, false);
}

}  // namespace
}  // namespace tensorflow
