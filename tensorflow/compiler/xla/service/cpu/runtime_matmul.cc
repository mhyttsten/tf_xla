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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"

#define EIGEN_USE_THREADS

#include "absl/base/dynamic_annotations.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_lightweight_check.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace {

bool Is16BytesAligned(void* ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul.cc", "Is16BytesAligned");

  return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

template <typename T, Eigen::AlignmentType Alignment>
void MatMul(const void* run_options_ptr, T* out, T* lhs, T* rhs, int64_t m,
            int64_t n, int64_t k, int32_t transpose_lhs,
            int32_t transpose_rhs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul.cc", "MatMul");

  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);

  int64_t lhs_rows = m;
  int64_t lhs_cols = k;
  if (transpose_lhs) {
    std::swap(lhs_rows, lhs_cols);
  }

  int64_t rhs_rows = k;
  int64_t rhs_cols = n;
  if (transpose_rhs) {
    std::swap(rhs_rows, rhs_cols);
  }

  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, Alignment> A(lhs, lhs_rows,
                                                                 lhs_cols);
  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, Alignment> B(rhs, rhs_rows,
                                                                 rhs_cols);
  Eigen::TensorMap<Eigen::Tensor<T, 2>, Alignment> C(out, m, n);

  typedef typename Eigen::Tensor<T, 2>::DimensionPair DimPair;
  int lhs_contract_dim = transpose_lhs ? 0 : 1;
  int rhs_contract_dim = transpose_rhs ? 1 : 0;
  const Eigen::array<DimPair, 1> dims(
      {DimPair(lhs_contract_dim, rhs_contract_dim)});

  // Matrix multiply is a special case of the "contract" operation where
  // the contraction is performed along dimension 1 of the lhs and dimension
  // 0 of the rhs.
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  C.device(*run_options->intra_op_thread_pool()) = A.contract(B, dims);
}

template <typename T>
void MatMulDispatch(const void* run_options_ptr, T* out, T* lhs, T* rhs,
                    int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
                    int32_t transpose_rhs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul.cc", "MatMulDispatch");

  bool all_buffers_16b_aligned =
      Is16BytesAligned(out) && Is16BytesAligned(lhs) && Is16BytesAligned(rhs);

  if (!all_buffers_16b_aligned) {
    MatMul<T, Eigen::Unaligned>(run_options_ptr, out, lhs, rhs, m, n, k,
                                transpose_lhs, transpose_rhs);
    return;
  }

  MatMul<T, Eigen::Aligned16>(run_options_ptr, out, lhs, rhs, m, n, k,
                              transpose_lhs, transpose_rhs);
}

}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulF16(
    const void* run_options_ptr, Eigen::half* out, Eigen::half* lhs,
    Eigen::half* rhs, int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
    int32_t transpose_rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc mht_3(mht_3_v, 273, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul.cc", "__xla_cpu_runtime_EigenMatMulF16");

  MatMulDispatch<Eigen::half>(run_options_ptr, out, lhs, rhs, m, n, k,
                              transpose_lhs, transpose_rhs);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs, int64_t m,
    int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc mht_4(mht_4_v, 283, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul.cc", "__xla_cpu_runtime_EigenMatMulF32");

  MatMulDispatch<float>(run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs,
                        transpose_rhs);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulF64(
    const void* run_options_ptr, double* out, double* lhs, double* rhs,
    int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
    int32_t transpose_rhs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc mht_5(mht_5_v, 294, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul.cc", "__xla_cpu_runtime_EigenMatMulF64");

  MatMulDispatch<double>(run_options_ptr, out, lhs, rhs, m, n, k, transpose_lhs,
                         transpose_rhs);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulC64(
    const void* run_options_ptr, std::complex<float>* out,
    std::complex<float>* lhs, std::complex<float>* rhs, int64_t m, int64_t n,
    int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc mht_6(mht_6_v, 305, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul.cc", "__xla_cpu_runtime_EigenMatMulC64");

  MatMulDispatch<std::complex<float>>(run_options_ptr, out, lhs, rhs, m, n, k,
                                      transpose_lhs, transpose_rhs);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulC128(
    const void* run_options_ptr, std::complex<double>* out,
    std::complex<double>* lhs, std::complex<double>* rhs, int64_t m, int64_t n,
    int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc mht_7(mht_7_v, 316, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul.cc", "__xla_cpu_runtime_EigenMatMulC128");

  MatMulDispatch<std::complex<double>>(run_options_ptr, out, lhs, rhs, m, n, k,
                                       transpose_lhs, transpose_rhs);
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_EigenMatMulS32(
    const void* run_options_ptr, int32_t* out, int32_t* lhs, int32_t* rhs,
    int64_t m, int64_t n, int64_t k, int32_t transpose_lhs,
    int32_t transpose_rhs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_matmulDTcc mht_8(mht_8_v, 327, "", "./tensorflow/compiler/xla/service/cpu/runtime_matmul.cc", "__xla_cpu_runtime_EigenMatMulS32");

  MatMulDispatch<int32_t>(run_options_ptr, out, lhs, rhs, m, n, k,
                          transpose_lhs, transpose_rhs);
}
