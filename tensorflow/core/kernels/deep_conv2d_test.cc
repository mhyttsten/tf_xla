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
class MHTracer_DTPStensorflowPScorePSkernelsPSdeep_conv2d_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdeep_conv2d_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdeep_conv2d_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/winograd_transform.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static void ComputeKroneckerProduct(const int rows, const int cols,
                                    const float* matrix, float* matrix_out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdeep_conv2d_testDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/kernels/deep_conv2d_test.cc", "ComputeKroneckerProduct");

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float v = matrix[i * cols + j];
      const int output_index_base = cols * (i * rows * cols + j);
      for (int k = 0; k < rows; ++k) {
        for (int l = 0; l < cols; ++l) {
          const int input_index = k * cols + l;
          const int output_index = k * cols * cols + l;
          matrix_out[output_index_base + output_index] =
              matrix[input_index] * v;
        }
      }
    }
  }
}

TEST(DeepConv2DTransformTest, Basic) {
  // Tests kronecker product of the following matrix with itself:
  //
  // [1.0 2.0]
  // [3.0 4.0]
  //
  const int rows = 2;
  const int cols = 2;

  float transform_matrix[] = {1, 2, 3, 4};

  const int kron_rows = rows * rows;
  const int kron_cols = cols * cols;
  float transform_matrix_kron[kron_rows * kron_cols];

  ComputeKroneckerProduct(rows, cols, &transform_matrix[0],
                          &transform_matrix_kron[0]);

  float transform_matrix_test[] = {1, 2, 2, 4, 3, 4,  6,  8,
                                   3, 6, 4, 8, 9, 12, 12, 16};

  for (int i = 0; i < kron_rows * kron_cols; ++i) {
    EXPECT_FLOAT_EQ(transform_matrix_kron[i], transform_matrix_test[i]);
  }
}

TEST(DeepConv2DTransformTest, WingradFilterTransformMatrix) {
  // Test that the filter transform matrix returned is the kronecker product of
  // the following matrix with itself:
  //
  //   [ 1    0   0   ]
  //   [ 1/2  1/2 1/2 ]
  //   [ 1/2 -1/2 1/2 ]
  //   [ 0    0   1   ]
  //
  const int rows = 4;
  const int cols = 3;

  float transform_matrix[] = {1, 0, 0, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0, 0, 1};

  const int kron_rows = rows * rows;
  const int kron_cols = cols * cols;

  float transform_matrix_kron[kron_rows * kron_cols];

  ComputeKroneckerProduct(rows, cols, &transform_matrix[0],
                          &transform_matrix_kron[0]);

  float transform_matrix_test[kron_rows * kron_cols];
  WinogradTransform<float> t;
  t.GetFilterTransformMatrix(kron_rows, kron_cols, &transform_matrix_test[0]);

  for (int i = 0; i < kron_rows * kron_cols; ++i) {
    EXPECT_FLOAT_EQ(transform_matrix_kron[i], transform_matrix_test[i]);
  }
}

TEST(DeepConv2DTransformTest, WingradInputTransformMatrix) {
  // Test that the filter transform matrix returned is the kronecker product of
  // the following matrix:
  //
  //   [1   0  -1   0]
  //   [0   1   1   0]
  //   [0  -1   1   0]
  //   [0   1   0  -1]
  //
  const int rows = 4;
  const int cols = 4;

  float transform_matrix[] = {1, 0,  -1, 0, 0, 1, 1, 0,
                              0, -1, 1,  0, 0, 1, 0, -1};

  const int kron_rows = rows * rows;
  const int kron_cols = cols * cols;

  float transform_matrix_kron[kron_rows * kron_cols];

  ComputeKroneckerProduct(rows, cols, &transform_matrix[0],
                          &transform_matrix_kron[0]);

  float transform_matrix_test[kron_rows * kron_cols];
  WinogradTransform<float> t;
  t.GetInputTransformMatrix(kron_rows, kron_cols, &transform_matrix_test[0]);

  for (int i = 0; i < kron_rows * kron_cols; ++i) {
    EXPECT_FLOAT_EQ(transform_matrix_kron[i], transform_matrix_test[i]);
  }
}

TEST(DeepConv2DTransformTest, WingradOutputTransformMatrix) {
  // Test that the filter transform matrix returned is the kronecker product of
  // the following matrix:
  //
  //   [1  1  1  0]
  //   [0  1 -1 -1]
  //
  const int rows = 2;
  const int cols = 4;

  float transform_matrix[] = {1, 1, 1, 0, 0, 1, -1, -1};

  const int kron_rows = rows * rows;
  const int kron_cols = cols * cols;

  float transform_matrix_kron[kron_rows * kron_cols];

  ComputeKroneckerProduct(rows, cols, &transform_matrix[0],
                          &transform_matrix_kron[0]);

  float transform_matrix_test[kron_rows * kron_cols];
  WinogradTransform<float> t;
  t.GetOutputTransformMatrix(kron_rows, kron_cols, &transform_matrix_test[0]);

  for (int i = 0; i < kron_rows * kron_cols; ++i) {
    EXPECT_FLOAT_EQ(transform_matrix_kron[i], transform_matrix_test[i]);
  }
}

}  // namespace
}  // namespace tensorflow
