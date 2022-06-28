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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmul_mklDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmul_mklDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmul_mklDTcc() {
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

#if defined(ENABLE_MKL) && !defined(INTEL_MKL_DNN_ONLY)
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.h"

#include "third_party/intel_mkl_ml/include/mkl_cblas.h"
#include "third_party/intel_mkl_ml/include/mkl_service.h"
#include "tensorflow/compiler/xla/executable_run_options.h"

#define EIGEN_USE_THREADS
#include "absl/base/dynamic_annotations.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/ThreadPool"

namespace {
// BLAS GEMM API for 32-bit Matrix Multiplication.

// MatMul function is defined as: c = alpha * op(a) * op(b) + beta * c.
// Since XLA MatMul does not used alpha, beta, we set them to 1.0 and 0.0.
// Matrix lhs, rhs and out are all column-major.
void MatMulF32(const void* run_options_ptr, float* out, float* lhs, float* rhs,
               int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
               int32_t transpose_rhs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmul_mklDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.cc", "MatMulF32");

  const float alpha = 1.0f, beta = 0.0f;
  // lda, ldb, and ldc are the leading dimensions of matrices a, b, and c,
  // respectively. For column-major matrices, the leading dimension is the
  // stride between consecutive columns (which equals the number of rows). If
  // the matrix is transposed, the leading dimension is the stride between
  // consecutive rows (which equals the number of columns).
  int lda = transpose_lhs ? k : m;
  int ldb = transpose_rhs ? n : k;
  int ldc = m;
  cblas_sgemm(CblasColMajor, transpose_lhs ? CblasTrans : CblasNoTrans,
              transpose_rhs ? CblasTrans : CblasNoTrans, m, n, k, alpha, lhs,
              lda, rhs, ldb, beta, out, ldc);
}

// BLAS GEMM API for 64-bit Matrix Multiplication.

// MatMul function is defined as: c = alpha * op(a) * op(b) + beta * c.
// Since XLA MatMul does not used alpha, beta, we set them to 1.0 and 0.0.
// Matrix lhs, rhs and out are all column-major.
void MatMulF64(const void* run_options_ptr, double* out, double* lhs,
               double* rhs, int64_t m, int64_t n, int64_t k,
               int32_t transpose_lhs, int32_t transpose_rhs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmul_mklDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.cc", "MatMulF64");

  const float alpha = 1.0f, beta = 0.0f;
  // lda, ldb, and ldc are the leading dimensions of matrices a, b, and c,
  // respectively. For a column-major matrix, the leading dimension is the
  // stride between consecutive columns (which equals the number of rows). If
  // the matrix is transposed, the leading dimension is the stride between
  // consecutive rows (which equals the number of columns).
  int lda = transpose_lhs ? k : m;
  int ldb = transpose_rhs ? n : k;
  int ldc = m;
  cblas_dgemm(CblasColMajor, transpose_lhs ? CblasTrans : CblasNoTrans,
              transpose_rhs ? CblasTrans : CblasNoTrans, m, n, k, alpha, lhs,
              lda, rhs, ldb, beta, out, ldc);
}

}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_MKLMatMulF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs, int64_t m,
    int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmul_mklDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.cc", "__xla_cpu_runtime_MKLMatMulF32");

  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  // BLAS GEMM MatMul uses OpenMP for parallelization, so we pass the thread
  // number specified in intra_op_thread_pool to MKL.
  int prev_num_threads = mkl_set_num_threads_local(
      run_options->intra_op_thread_pool()->numThreads());
  MatMulF32(nullptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
  // Set thread number back to the previous number.
  mkl_set_num_threads_local(prev_num_threads);
}

// BLAS GEMM API for 64-bit Matrix Multiplication
ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_MKLMatMulF64(
    const void* run_options_ptr, double* out, double* lhs, double* rhs,
    int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
    int32_t transpose_rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmul_mklDTcc mht_3(mht_3_v, 270, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.cc", "__xla_cpu_runtime_MKLMatMulF64");

  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  // BLAS GEMM MatMul uses OpenMP for parallelization, so we pass the thread
  // number specified in intra_op_thread_pool to MKL.
  int prev_num_threads = mkl_set_num_threads_local(
      run_options->intra_op_thread_pool()->numThreads());
  MatMulF64(nullptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
  // Set thread number back to the previous number.
  mkl_set_num_threads_local(prev_num_threads);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_MKLSingleThreadedMatMulF32(const void* run_options_ptr,
                                             float* out, float* lhs, float* rhs,
                                             int64_t m, int64_t n, int64_t k,
                                             int32_t transpose_lhs,
                                             int32_t transpose_rhs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmul_mklDTcc mht_4(mht_4_v, 290, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.cc", "__xla_cpu_runtime_MKLSingleThreadedMatMulF32");

  // Set the thread number to 1 for single threaded execution.
  int prev_num_threads = mkl_set_num_threads_local(1);
  MatMulF32(nullptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
  // Set thread number back to the previous number.
  mkl_set_num_threads_local(prev_num_threads);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_MKLSingleThreadedMatMulF64(const void* run_options_ptr,
                                             double* out, double* lhs,
                                             double* rhs, int64_t m, int64_t n,
                                             int64_t k, int32_t transpose_lhs,
                                             int32_t transpose_rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmul_mklDTcc mht_5(mht_5_v, 306, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.cc", "__xla_cpu_runtime_MKLSingleThreadedMatMulF64");

  // Set the thread number to 1 for single threaded execution.
  int prev_num_threads = mkl_set_num_threads_local(1);
  MatMulF64(nullptr, out, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs);
  // Set thread number back to the previous number.
  mkl_set_num_threads_local(prev_num_threads);
}
#endif  // ENABLE_MKL
