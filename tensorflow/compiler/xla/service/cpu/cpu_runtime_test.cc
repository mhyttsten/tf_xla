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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_runtime_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_runtime_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_runtime_testDTcc() {
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
#define EIGEN_USE_THREADS
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"

#include <memory>
#include <string>
#include <tuple>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_custom_call_status.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class CpuRuntimeTest : public ::testing::Test {};

template <typename T>
std::unique_ptr<Array2D<float>> MaybeTransposeArray2D(const Array2D<T>& array,
                                                      bool transpose) {
  int64_t output_height = array.height();
  int64_t output_width = array.width();
  if (transpose) {
    std::swap(output_width, output_height);
  }
  auto output = absl::make_unique<Array2D<float>>(output_height, output_width);
  for (int y = 0; y < array.height(); y++) {
    for (int x = 0; x < array.width(); x++) {
      if (transpose) {
        (*output)(x, y) = array(y, x);
      } else {
        (*output)(y, x) = array(y, x);
      }
    }
  }
  return output;
}

// Verifies that matrix 'c' equals the result of matrix 'a' times matrix 'b'.
// Each element is compared to within a small error bound.
void CheckMatrixMultiply(const Array2D<float>& a, const Array2D<float>& b,
                         const Array2D<float>& c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_runtime_testDTcc mht_0(mht_0_v, 235, "", "./tensorflow/compiler/xla/service/cpu/cpu_runtime_test.cc", "CheckMatrixMultiply");

  for (int i = 0; i < a.height(); ++i) {
    for (int j = 0; j < b.width(); ++j) {
      float sum = 0.0;
      for (int k = 0; k < a.width(); ++k) {
        sum += a(i, k) * b(k, j);
      }
      EXPECT_NEAR(sum, c(i, j), 0.01);
    }
  }
}

std::unique_ptr<Array2D<float>> EigenMatrixMultiply(const Array2D<float>& a,
                                                    const Array2D<float>& b,
                                                    bool transpose_lhs,
                                                    bool transpose_rhs,
                                                    bool single_threaded) {
  CHECK_EQ(a.width(), b.height());
  int64_t m = a.height();
  int64_t n = b.width();
  int64_t k = a.width();

  // The Eigen matmul runtime function expects the matrix to be in column major
  // order and array2d is in row-major order. Create transposes of a and b. The
  // 'data' buffer in the transposed array is the original array in column major
  // order.
  auto a_transpose = MaybeTransposeArray2D(a, !transpose_lhs);
  auto b_transpose = MaybeTransposeArray2D(b, !transpose_rhs);

  // Since we're going to transpose c before returning it. Swap the order of the
  // dimension sizes to ensure the returned array is properly dimensioned.
  auto c_transpose = absl::make_unique<Array2D<float>>(n, m);
  if (single_threaded) {
    __xla_cpu_runtime_EigenSingleThreadedMatMulF32(
        nullptr, c_transpose->data(), a_transpose->data(), b_transpose->data(),
        m, n, k, transpose_lhs, transpose_rhs);
  } else {
    tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(), "XLAEigen",
                                        2);
    Eigen::ThreadPoolDevice device(pool.AsEigenThreadPool(), pool.NumThreads());
    ExecutableRunOptions run_options;
    run_options.set_intra_op_thread_pool(&device);

    __xla_cpu_runtime_EigenMatMulF32(&run_options, c_transpose->data(),
                                     a_transpose->data(), b_transpose->data(),
                                     m, n, k, transpose_lhs, transpose_rhs);
  }
  return MaybeTransposeArray2D(*c_transpose, true);
}

struct MatMulShape {
  int64_t m;
  int64_t k;
  int64_t n;
};

MatMulShape MatMulShapes[] = {
    MatMulShape{2, 2, 3},     MatMulShape{256, 512, 1024},
    MatMulShape{128, 128, 1}, MatMulShape{1, 128, 128},
    MatMulShape{1, 32, 128},  MatMulShape{1, 32, 16},
    MatMulShape{32, 16, 1},   MatMulShape{32, 128, 1},
};

// This takes 4 parameters:
// * shape of the matmul
// * transpose_lhs
// * transpose_rhs
// * single_threaded
using MatMulTestParam = std::tuple<MatMulShape, bool, bool, bool>;

class EigenMatMulTest : public CpuRuntimeTest,
                        public ::testing::WithParamInterface<MatMulTestParam> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<MatMulTestParam>& info) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_runtime_testDTcc mht_1(mht_1_v, 312, "", "./tensorflow/compiler/xla/service/cpu/cpu_runtime_test.cc", "Name");

    MatMulShape shape = std::get<0>(info.param);
    bool transpose_lhs = std::get<1>(info.param);
    bool transpose_rhs = std::get<2>(info.param);
    bool single_threaded = std::get<3>(info.param);

    return absl::StrFormat("EigenMatMul_%d_%d_%d_%s%s%s_threaded", shape.m,
                           shape.k, shape.n, transpose_lhs ? "Tlhs_" : "",
                           transpose_rhs ? "Trhs_" : "",
                           single_threaded ? "single" : "multi");
  }
};

TEST_P(EigenMatMulTest, DoIt) {
  MatMulShape shape = std::get<0>(GetParam());
  bool transpose_lhs = std::get<1>(GetParam());
  bool transpose_rhs = std::get<2>(GetParam());
  bool single_threaded = std::get<3>(GetParam());

  auto a = MakeLinspaceArray2D(0.0, 1.0, shape.m, shape.k);
  auto b = MakeLinspaceArray2D(-2.0, 2.0, shape.k, shape.n);
  auto c = EigenMatrixMultiply(*a, *b, transpose_lhs, transpose_rhs,
                               single_threaded);
  CheckMatrixMultiply(*a, *b, *c);
}

INSTANTIATE_TEST_SUITE_P(EigenMatMulTestInstantiaion, EigenMatMulTest,
                         ::testing::Combine(::testing::ValuesIn(MatMulShapes),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()),
                         EigenMatMulTest::Name);

#ifdef ENABLE_MKL
class MKLMatMulTest : public CpuRuntimeTest,
                      public ::testing::WithParamInterface<MatMulTestParam> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<MatMulTestParam>& info) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_runtime_testDTcc mht_2(mht_2_v, 353, "", "./tensorflow/compiler/xla/service/cpu/cpu_runtime_test.cc", "Name");

    MatMulShape shape = std::get<0>(info.param);
    bool transpose_lhs = std::get<1>(info.param);
    bool transpose_rhs = std::get<2>(info.param);
    bool single_threaded = std::get<3>(info.param);

    return absl::StrFormat("MKLMatMul_%d_%d_%d_%s%s%s_threaded", shape.m,
                           shape.k, shape.n, transpose_lhs ? "Tlhs_" : "",
                           transpose_rhs ? "Trhs_" : "",
                           single_threaded ? "single" : "multi");
  }
};

std::unique_ptr<Array2D<float>> MKLMatrixMultiply(const Array2D<float>& a,
                                                  const Array2D<float>& b,
                                                  bool transpose_lhs,
                                                  bool transpose_rhs,
                                                  bool single_threaded) {
  CHECK_EQ(a.width(), b.height());
  int64_t m = a.height();
  int64_t n = b.width();
  int64_t k = a.width();

  // The MKL matmul runtime function expects the matrix to be in column major
  // order and array2d is in row-major order. Create transposes of a and b. The
  // 'data' buffer in the transposed array is the original array in column major
  // order.
  auto a_transpose = MaybeTransposeArray2D(a, !transpose_lhs);
  auto b_transpose = MaybeTransposeArray2D(b, !transpose_rhs);

  // Since we're going to transpose c before returning it, swap the order of the
  // dimension sizes to ensure the returned array is properly dimensioned.
  auto c_transpose = absl::make_unique<Array2D<float>>(n, m);
  if (single_threaded) {
    __xla_cpu_runtime_MKLSingleThreadedMatMulF32(
        nullptr, c_transpose->data(), a_transpose->data(), b_transpose->data(),
        m, n, k, transpose_lhs, transpose_rhs);
  } else {
    __xla_cpu_runtime_MKLMatMulF32(nullptr, c_transpose->data(),
                                   a_transpose->data(), b_transpose->data(), m,
                                   n, k, transpose_lhs, transpose_rhs);
  }
  return MaybeTransposeArray2D(*c_transpose, true);
}

TEST_P(MKLMatMulTest, DoIt) {
  MatMulShape shape = std::get<0>(GetParam());
  bool transpose_lhs = std::get<1>(GetParam());
  bool transpose_rhs = std::get<2>(GetParam());
  bool single_threaded = std::get<3>(GetParam());

  auto a = MakeLinspaceArray2D(0.0, 1.0, shape.m, shape.k);
  auto b = MakeLinspaceArray2D(-2.0, 2.0, shape.k, shape.n);
  auto c =
      MKLMatrixMultiply(*a, *b, transpose_lhs, transpose_rhs, single_threaded);
  CheckMatrixMultiply(*a, *b, *c);
}

INSTANTIATE_TEST_CASE_P(MKLMatMulTestInstantiaion, MKLMatMulTest,
                        ::testing::Combine(::testing::ValuesIn(MatMulShapes),
                                           ::testing::Bool(), ::testing::Bool(),
                                           ::testing::Bool()),
                        MKLMatMulTest::Name);
#endif  // ENABLE_MKL

TEST_F(CpuRuntimeTest, SuccessStatus) {
  XlaCustomCallStatus success_status;
  // Success is the default state.
  ASSERT_TRUE(__xla_cpu_runtime_StatusIsSuccess(&success_status));
}

TEST_F(CpuRuntimeTest, FailureStatus) {
  XlaCustomCallStatus success_status;
  XlaCustomCallStatusSetFailure(&success_status, "Failed", 6);
  ASSERT_FALSE(__xla_cpu_runtime_StatusIsSuccess(&success_status));
}

}  // namespace
}  // namespace xla
