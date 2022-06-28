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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_opDTcc() {
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

// See docs in ../ops/math_ops.cc.

// This file uses MKL CBLAS xGEMM for acceleration of TF Matrix-Matrix
// Multiplication (MatMul) operations.
// We currently register this kernel only for MKL supported data
// types (float, double, complex64, complex128). The macro INTEL_MKL is defined
// by the build system only when MKL is chosen as an option at configure stage
// and when it is undefined at build time, this file becomes an empty
// compilation unit

#if defined(INTEL_MKL)

#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, bool USE_CUBLAS>
class MklMatMulOp : public OpKernel {
 public:
  explicit MklMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_opDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/mkl/mkl_matmul_op.cc", "MklMatMulOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_opDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/mkl/mkl_matmul_op.cc", "Compute");

    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] ndims must be >= 2"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] ndims must be >= 2"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    int d1 = a.dim_size(dim_pair[0].first);
    int d2 = b.dim_size(dim_pair[0].second);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(),
                                        ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    const int m = a.dim_size(1 - dim_pair[0].first);
    const int k = a.dim_size(dim_pair[0].first);
    const int n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;

    auto a_ptr = (a.template flat<T>().data());
    auto b_ptr = (b.template flat<T>().data());
    auto c_ptr = (out->template flat<T>().data());

    MklBlasGemm(ctx, transpose_a, transpose_b, m, n, k, a_ptr,
                transpose_a ? m : k, b_ptr, transpose_b ? k : n, c_ptr, n);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  // --------------------------------------------------------------------------
  //
  // @brief Matrix-Matrix Multiplication with FP32 tensors, a, b, c using CBLAS
  // interface. c = op(a) * op(b)
  //
  // @param transa  Specifies the form of op(a) used in MatMul. If transa is
  // true, then op(a) = a^T, otherwise op(a) = a
  //
  // @param transb  Specifies the form of op(b) used in MatMul. If transb is
  // true, then op(b) = b^T, otherwise op(b) = b
  //
  // @param m       Specifies the number of rows of the matrix op(a) and of the
  // matrix c. The value of m must be at least zero.
  //
  // @param n       Specifies the number of columns of the matrix op(b) and the
  // number of columns of the matrix c. The value of n must be at least zero.
  //
  // @param k       Specifies the number of columns of the matrix op(a) and the
  // number of rows of the matrix op(b)
  //
  // @param a       Address of matrix a
  //
  // @param lda     Leading dimension of 'a' matrix. This is set at calling site
  // depending on transa parameter. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows
  // lda = max(1,k) when transa is false, otherwise lda = max(1,m)
  //
  // @param b       Address of matrix b
  //
  // @param ldb     Leading dimension of 'b' matrix. This is set at calling site
  // depending on transb parameter. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows
  // ldb = max(1,n) when transb is false, otherwise ldb = max(1,k)
  //
  // @param c       Address of matrix c
  //
  // @param ldc     Leading dimension of 'c' matrix. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows, max(1,n)
  //
  // --------------------------------------------------------------------------
  void MklBlasGemm(OpKernelContext* ctx, bool transa, bool transb, const int m,
                   const int n, const int k, const float* a, const int lda,
                   const float* b, const int ldb, float* c, const int ldc) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_opDTcc mht_2(mht_2_v, 323, "", "./tensorflow/core/kernels/mkl/mkl_matmul_op.cc", "MklBlasGemm");

    // BLAS GEMM API defines Matrix Multiplication as c = alpha * op(a) * op(b)
    // + beta * c.
    // Since TF MatMul does not have parameters for alpha, beta, we set them to
    // 1.0 and 0.0 respectively.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    char char_transa = transa ? 'T' : 'N';
    char char_transb = transb ? 'T' : 'N';
    VLOG(2) << "MKL DNN SGEMM called";
#ifndef ENABLE_ONEDNN_OPENMP
    MklDnnThreadPool eigen_tp(ctx);
    // With threadpool , the runtime overhead is comparable to the kernel
    // execution for small kernel sizes. For such sizes, it may be better to run
    // the kernel single threaded. Here we are coming up with a cost model based
    // on L1 sizes. If we find that matrices are small enough, we will execute
    // single threaded. This may need tuning.
    if (ExecuteSingleThreadedGemm(m, n, k, sizeof(float))) {
      // For now, call single-threaded gemm.
      dnnl::threadpool_interop::sgemm(char_transa, char_transb, m, n, k, alpha,
                                      a, lda, b, ldb, beta, c, ldc, nullptr);
    } else {
      dnnl::threadpool_interop::sgemm(char_transa, char_transb, m, n, k, alpha,
                                      a, lda, b, ldb, beta, c, ldc, &eigen_tp);
    }
#else
    dnnl_sgemm(char_transa, char_transb, m, n, k, alpha, a, lda, b, ldb, beta,
               c, ldc);
#endif  // !ENABLE_ONEDNN_OPENMP
  }

  void MklBlasGemm(OpKernelContext* ctx, bool transa, bool transb, const int m,
                   const int n, const int k, const bfloat16* a, const int lda,
                   const bfloat16* b, const int ldb, bfloat16* c,
                   const int ldc) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_matmul_opDTcc mht_3(mht_3_v, 360, "", "./tensorflow/core/kernels/mkl/mkl_matmul_op.cc", "MklBlasGemm");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int index_transa = transa ? 1 : 0;
    const int index_transb = transb ? 1 : 0;

    const char ftrans[] = {'N', 'T', 'C'};
    dnnl_gemm<bfloat16>(ftrans[index_transa], ftrans[index_transb], m, n, k,
                        alpha, a, lda, b, ldb, beta, c, ldc, ctx);
  }
};

#define REGISTER_CPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                \
      Name("_MklMatMul")                                  \
          .Device(DEVICE_CPU)                             \
          .TypeConstraint<T>("T")                         \
          .Label(mkl_op_registry::kMklNameChangeOpLabel), \
      MklMatMulOp<CPUDevice, T, false /* cublas, ignored for CPU */>);

// TODO(intel-tf): Consider template specialization when adding/removing
// additional types
TF_CALL_float(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);
}  // namespace tensorflow
#endif  // INTEL_MKL
