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
class MHTracer_DTPStensorflowPScorePSkernelsPSeigen_pooling_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_pooling_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSeigen_pooling_testDTcc() {
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

#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/platform/test.h"

namespace Eigen {

namespace {
void EigenApprox(float a, float b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSeigen_pooling_testDTcc mht_0(mht_0_v, 191, "", "./tensorflow/core/kernels/eigen_pooling_test.cc", "EigenApprox");

  ASSERT_TRUE(std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * 1e-3);
}
}  // namespace

TEST(EigenPoolingTest, Simple) {
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 4;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4> input(depth, input_rows, input_cols, num_batches);
  Tensor<float, 4> result(depth, output_rows, output_cols, num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.f);

  // Max pooling using a 4x4 window and a stride of 1.
  const int stride = 1;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(0), depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);
  EXPECT_EQ(result.dimension(3), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(expected, input(d, r + i, c + j, b));
            }
          }
          if (result(d, i, j, b) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(d, i, j, b) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(d, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenPoolingTest, SimpleRowMajor) {
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 4;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4, RowMajor> input(num_batches, input_cols, input_rows, depth);
  Tensor<float, 4, RowMajor> result(num_batches, output_cols, output_rows,
                                    depth);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.f);

  // Max pooling using a 4x4 window and a stride of 1.
  const int stride = 1;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(3), depth);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(expected, input(b, c + j, r + i, d));
            }
          }
          if (result(b, j, i, d) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(b, j, i, d) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(b, j, i, d), expected);
        }
      }
    }
  }
}

TEST(EigenPoolingTest, Cuboid) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = 2;
  const int output_cols = 3;
  const int output_planes = 4;

  Tensor<float, 5> input(channels, input_planes, input_rows, input_cols,
                         num_batches);
  Tensor<float, 5> result(channels, output_planes, output_rows, output_cols,
                          num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidMaxPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), channels);
  EXPECT_EQ(result.dimension(1), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_cols);
  EXPECT_EQ(result.dimension(4), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected = -10000.f;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected =
                      (std::max)(expected, input(d, p + i, r + j, c + k, b));
                }
              }
            }
            if (result(d, i, j, k, b) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(d, i, j, k, b) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(d, i, j, k, b), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, CuboidRowMajor) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = 2;
  const int output_cols = 3;
  const int output_planes = 4;

  Tensor<float, 5, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_planes, channels);
  Tensor<float, 5, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_planes, channels);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidMaxPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(4), channels);
  EXPECT_EQ(result.dimension(3), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected = -10000.f;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected =
                      (std::max)(expected, input(b, c + k, r + j, p + i, d));
                }
              }
            }
            if (result(b, k, j, i, d) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(b, k, j, i, d) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(b, k, j, i, d), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, ValidCuboid) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = 2;
  const int output_cols = 3;
  const int output_planes = 4;

  Tensor<float, 5> input(channels, input_planes, input_rows, input_cols,
                         num_batches);
  Tensor<float, 5> result(channels, output_planes, output_rows, output_cols,
                          num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidAvgPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), channels);
  EXPECT_EQ(result.dimension(1), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_cols);
  EXPECT_EQ(result.dimension(4), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected_sum = 0.0f;
            int expected_count = 0;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected_sum += input(d, p + i, r + j, c + k, b);
                  expected_count++;
                }
              }
            }
            const float expected = expected_sum / expected_count;
            if (result(d, i, j, k, b) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(d, i, j, k, b) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(d, i, j, k, b), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, ValidCuboidRowMajor) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = 2;
  const int output_cols = 3;
  const int output_planes = 4;

  Tensor<float, 5, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_planes, channels);
  Tensor<float, 5, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_planes, channels);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidAvgPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(4), channels);
  EXPECT_EQ(result.dimension(3), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected_sum = 0.0f;
            int expected_count = 0;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected_sum += input(b, c + k, r + j, p + i, d);
                  expected_count++;
                }
              }
            }
            const float expected = expected_sum / expected_count;
            if (result(b, k, j, i, d) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(b, k, j, i, d) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(b, k, j, i, d), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, SameCuboid) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = input_rows;
  const int output_cols = input_cols;
  const int output_planes = input_planes;

  Tensor<float, 5> input(channels, input_planes, input_rows, input_cols,
                         num_batches);
  Tensor<float, 5> result(channels, output_planes, output_rows, output_cols,
                          num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidAvgPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_SAME);

  EXPECT_EQ(result.dimension(0), channels);
  EXPECT_EQ(result.dimension(1), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_cols);
  EXPECT_EQ(result.dimension(4), num_batches);

  const int pad_p = output_planes - input_planes + patch_planes - 1;
  const int pad_r = output_rows - input_rows + patch_rows - 1;
  const int pad_c = output_cols - input_cols + patch_cols - 1;

  // Number of pixels the input is extended with at the lower end in every
  // dimension.
  const int dp = pad_p / 2;
  const int dr = pad_r / 2;
  const int dc = pad_c / 2;

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected_sum = 0.0f;
            int expected_count = 0;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  const int in_p = p + i - dp;
                  const int in_r = r + j - dr;
                  const int in_c = c + k - dc;
                  if (in_p >= 0 && in_p < input_planes && in_r >= 0 &&
                      in_r < input_rows && in_c >= 0 && in_c < input_cols) {
                    expected_sum += input(d, in_p, in_r, in_c, b);
                    expected_count++;
                  }
                }
              }
            }
            const float expected = expected_sum / expected_count;
            if (result(d, i, j, k, b) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(d, i, j, k, b) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(d, i, j, k, b), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, SameCuboidRowMajor) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = input_rows;
  const int output_cols = input_cols;
  const int output_planes = input_planes;

  Tensor<float, 5, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_planes, channels);
  Tensor<float, 5, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_planes, channels);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidAvgPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_SAME);

  EXPECT_EQ(result.dimension(4), channels);
  EXPECT_EQ(result.dimension(3), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  const int pad_p = output_planes - input_planes + patch_planes - 1;
  const int pad_r = output_rows - input_rows + patch_rows - 1;
  const int pad_c = output_cols - input_cols + patch_cols - 1;

  // Number of pixels the input is extended with at the lower end in every
  // dimension.
  const int dp = pad_p / 2;
  const int dr = pad_r / 2;
  const int dc = pad_c / 2;

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected_sum = 0.0f;
            int expected_count = 0;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  const int in_p = p + i - dp;
                  const int in_r = r + j - dr;
                  const int in_c = c + k - dc;
                  if (in_p >= 0 && in_p < input_planes && in_r >= 0 &&
                      in_r < input_rows && in_c >= 0 && in_c < input_cols) {
                    expected_sum += input(b, in_c, in_r, in_p, d);
                    expected_count++;
                  }
                }
              }
            }
            const float expected = expected_sum / expected_count;
            if (result(b, k, j, i, d) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(b, k, j, i, d) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(b, k, j, i, d), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, Strided) {
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4> input(depth, input_rows, input_cols, num_batches);
  Tensor<float, 4> result(depth, output_rows, output_cols, num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3 window and a stride of 2.
  int stride = 2;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(0), depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);
  EXPECT_EQ(result.dimension(3), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(
                  expected, input(d, r + stride * i, c + stride * j, b));
            }
          }
          if (result(d, i, j, b) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(d, i, j, b) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(d, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenPoolingTest, StridedRowMajor) {
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4, RowMajor> input(num_batches, input_cols, input_rows, depth);
  Tensor<float, 4, RowMajor> result(num_batches, output_cols, output_rows,
                                    depth);
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3 window and a stride of 2.
  int stride = 2;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(3), depth);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(
                  expected, input(b, c + stride * j, r + stride * i, d));
            }
          }
          if (result(b, j, i, d) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(b, j, i, d) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(b, j, i, d), expected);
        }
      }
    }
  }
}

TEST(EigenPoolingTest, StridedCuboid) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_planes = 3;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_planes = 2;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 5> input(channels, input_planes, input_rows, input_cols,
                         num_batches);
  Tensor<float, 5> result(channels, output_planes, output_rows, output_cols,
                          num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3x3 window and a stride of 2.
  int stride = 2;
  result = CuboidMaxPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), channels);
  EXPECT_EQ(result.dimension(1), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_cols);
  EXPECT_EQ(result.dimension(4), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected = -10000.f;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected = (std::max)(expected,
                                        input(d, p + stride * i, r + stride * j,
                                              c + stride * k, b));
                }
              }
            }
            if (result(d, i, j, k, b) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " " << k << " "
                        << result(d, i, j, k, b) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(d, i, j, k, b), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, StridedCuboidRowMajor) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_planes = 3;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_planes = 2;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 5, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_planes, channels);
  Tensor<float, 5, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_planes, channels);
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3x3 window and a stride of 2.
  int stride = 2;
  result = CuboidMaxPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(4), channels);
  EXPECT_EQ(result.dimension(3), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected = -10000.f;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected = (std::max)(expected,
                                        input(b, c + stride * k, r + stride * j,
                                              p + stride * i, d));
                }
              }
            }
            if (result(b, k, j, i, d) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " " << k << " "
                        << result(b, k, j, i, d) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(b, k, j, i, d), expected);
          }
        }
      }
    }
  }
}

}  // namespace Eigen
