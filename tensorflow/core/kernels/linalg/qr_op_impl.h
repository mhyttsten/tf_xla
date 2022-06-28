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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_QR_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_QR_OP_IMPL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh() {
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


// See docs in ../ops/linalg_ops.cc.
//
// This header file is used by the individual qr_*op*.cc files for registering
// individual kernels. A separate file is used for each instantiated kernel to
// improve compilation times.
#include <algorithm>
#include <numeric>

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/Eigen/QR"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/linalg/eye_functor.h"
#include "tensorflow/core/kernels/linalg/matrix_band_part_op.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

namespace tensorflow {

template <class Scalar>
class QrOp : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit QrOp(OpKernelConstruction* context) : Base(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh mht_0(mht_0_v, 227, "", "./tensorflow/core/kernels/linalg/qr_op_impl.h", "QrOp");

    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  using TensorShapes = typename Base::TensorShapes;

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/linalg/qr_op_impl.h", "ValidateInputMatrixShapes");

    Base::ValidateSingleMatrix(context, input_matrix_shapes);
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/linalg/qr_op_impl.h", "GetOutputMatrixShapes");

    int64_t m = input_matrix_shapes[0].dim_size(0);
    int64_t n = input_matrix_shapes[0].dim_size(1);
    int64_t min_size = std::min(m, n);
    if (full_matrices_) {
      return TensorShapes({TensorShape({m, m}), TensorShape({m, n})});
    } else {
      return TensorShapes(
          {TensorShape({m, min_size}), TensorShape({min_size, n})});
    }
  }

  int64_t GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh mht_3(mht_3_v, 261, "", "./tensorflow/core/kernels/linalg/qr_op_impl.h", "GetCostPerUnit");

    double m = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double n = static_cast<double>(input_matrix_shapes[0].dim_size(1));
    double max_size = std::max(m, n);
    double min_size = std::min(m, n);
    double cost = 2 * max_size * min_size * min_size -
                  2 * min_size * min_size * min_size / 3.;
    // TODO(jpoulson): Increase the cost if full_matrices is true in a manner
    // that reflects the algorithm used for the expansion.
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh mht_4(mht_4_v, 283, "", "./tensorflow/core/kernels/linalg/qr_op_impl.h", "ComputeMatrix");

    Eigen::HouseholderQR<Matrix> qr(inputs[0]);
    const int m = inputs[0].rows();
    const int n = inputs[0].cols();
    const int min_size = std::min(m, n);

    if (full_matrices_) {
      outputs->at(0) = qr.householderQ();
      outputs->at(1) = qr.matrixQR().template triangularView<Eigen::Upper>();
    } else {
      // TODO(jpoulson): Exploit the fact that Householder transformations can
      // be expanded faster than they can be applied to an arbitrary matrix
      // (Cf. LAPACK's DORGQR).
      Matrix tmp = Matrix::Identity(m, min_size);
      outputs->at(0) = qr.householderQ() * tmp;
      auto qr_top = qr.matrixQR().block(0, 0, min_size, n);
      outputs->at(1) = qr_top.template triangularView<Eigen::Upper>();
    }
  }

 private:
  bool full_matrices_;

  TF_DISALLOW_COPY_AND_ASSIGN(QrOp);
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class QrOpGpu : public AsyncOpKernel {
 public:
  explicit QrOpGpu(OpKernelConstruction* context) : AsyncOpKernel(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh mht_5(mht_5_v, 319, "", "./tensorflow/core/kernels/linalg/qr_op_impl.h", "QrOpGpu");

    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSqr_op_implDTh mht_6(mht_6_v, 326, "", "./tensorflow/core/kernels/linalg/qr_op_impl.h", "ComputeAsync");

    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64_t m = input.dim_size(ndims - 2);
    const int64_t n = input.dim_size(ndims - 1);
    const int64_t min_size = std::min(m, n);
    const int64_t batch_size =
        input.template flat_inner_dims<Scalar, 3>().dimension(0);

    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);

    // Allocate output.
    // If full_matrices_ is true then Q is m x m and R is m x n.
    // Otherwise, Q is m x min(m, n), and R is min(m, n) x n.
    Tensor* q;
    TensorShape q_shape = input.shape();
    q_shape.set_dim(ndims - 1, full_matrices_ ? m : min_size);
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, q_shape, &q),
                         done);
    Tensor* r;
    TensorShape r_shape = input.shape();
    r_shape.set_dim(ndims - 2, full_matrices_ ? m : min_size);
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(1, r_shape, &r),
                         done);

    if (input.NumElements() == 0) {
      done();
      return;
    }

    // TODO(rmlarsen): Convert to std::make_unique when available.
    std::unique_ptr<GpuSolver> solver(new GpuSolver(context));

    // Allocate temporaries.
    Tensor input_transposed;
    TensorShape transposed_shape = input.shape();
    transposed_shape.set_dim(ndims - 2, input.dim_size(ndims - 1));
    transposed_shape.set_dim(ndims - 1, input.dim_size(ndims - 2));

    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<Scalar>::value,
                                       transposed_shape, &input_transposed),
        done);

    Tensor tau;
    OP_REQUIRES_OK_ASYNC(context,
                         solver->allocate_scoped_tensor(
                             DataTypeToEnum<Scalar>::value,
                             TensorShape({batch_size, min_size}), &tau),
                         done);

    // Transpose input, since cuSolver uses column-major, while TensorFlow uses
    // row-major storage.
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    OP_REQUIRES_OK_ASYNC(
        context, DoMatrixTranspose(device, input, &input_transposed), done);

    // Compute QR decomposition in-place in input_transposed.
    std::vector<DeviceLapackInfo> dev_info;
    dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "geqrf"));
    auto input_transposed_reshaped =
        input_transposed.flat_inner_dims<Scalar, 3>();
    auto tau_matrix = tau.matrix<Scalar>();
    auto r_reshaped = r->flat_inner_dims<Scalar, 3>();
    for (int batch = 0; batch < batch_size; ++batch) {
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->Geqrf(m, n, &input_transposed_reshaped(batch, 0, 0), m,
                        &tau_matrix(batch, 0),
                        dev_info.back().mutable_data() + batch),
          done);
    }

#if GOOGLE_CUDA
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
#elif TENSORFLOW_USE_ROCM
    rocblas_operation transa = rocblas_operation_transpose;
    rocblas_operation transb = rocblas_operation_none;
    rocblas_side side = rocblas_side_left;
#endif

    // Generate R. R is equal to the upper triangle of the decomposition
    // stored in input_transposed. Crop, transpose (to get back to row-major)
    // and copy it to the output buffer.
    if (full_matrices_ || m == n) {
      OP_REQUIRES_OK_ASYNC(
          context, DoMatrixTranspose(device, input_transposed, r), done);
    } else {
      const Scalar alpha(1);
      const Scalar beta(0);
      const Scalar* dummy = nullptr;
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Geam(transa, transb, n, full_matrices_ ? m : min_size,
                         &alpha, &input_transposed_reshaped(batch, 0, 0), m,
                         &beta, dummy, n, &r_reshaped(batch, 0, 0), n),
            done);
      }
    }
    // Extract the upper triangle of r (i.e. zero out the strictly lower
    // triangle).
    functor::MatrixBandPartFunctor<GPUDevice, Scalar> band_part;
    auto r_reshaped_const =
        const_cast<const Tensor*>(r)->flat_inner_dims<Scalar, 3>();
    band_part(context, device, 0 /* num_lower_diags */,
              -1 /* num_upper_diags */, r_reshaped_const, r_reshaped);

    // Generate Q from the decomposition in input_transposed.
    if (m != n && (full_matrices_ || m < n)) {
      // Generate full m x m matrix Q by computing the product Q^T * I,
      // where the transpose is to get back to row-major form.
      // In the complex case we actually form Q^H * I and conjugate it
      // to get Q in row-major form.
      functor::EyeFunctor<GPUDevice, Scalar> eye;
      auto q_reshaped = q->flat_inner_dims<Scalar, 3>();
      eye(device, q_reshaped);
#if GOOGLE_CUDA
      cublasOperation_t trans = CublasAdjointOp<Scalar>();
#elif TENSORFLOW_USE_ROCM
      rocblas_operation trans = RocblasAdjointOp<Scalar>();
#endif
      for (int batch = 0; batch < batch_size; ++batch) {
        // Notice: It appears that Unmqr does not write a zero into *info upon
        // success (probably a bug), so we simply re-use the info array already
        // zeroed by Geqrf above.
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Unmqr(side, trans, m, m, min_size,
                          &input_transposed_reshaped(batch, 0, 0), m,
                          &tau_matrix(batch, 0), &q_reshaped(batch, 0, 0), m,
                          dev_info.back().mutable_data() + batch),
            done);
      }
      if (Eigen::NumTraits<Scalar>::IsComplex) {
        functor::UnaryFunctor<GPUDevice, functor::conj<Scalar>> conj;
        conj(device, q->flat<Scalar>() /*out*/,
             const_cast<const Tensor*>(q)->flat<Scalar>() /*in*/);
      }
    } else {
      // Generate m x n matrix Q. In this case we can use the more efficient
      // algorithm in Ungqr to generate Q in place.
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "orgqr"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Ungqr(
                m, n, min_size, &input_transposed_reshaped(batch, 0, 0), m,
                &tau_matrix(batch, 0), dev_info.back().mutable_data() + batch),
            done);
      }
      OP_REQUIRES_OK_ASYNC(
          context, DoMatrixTranspose(device, input_transposed, q), done);
    }

    // Asynchronously check return status from cuSolver kernels.
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(done));
  }

 private:
  bool full_matrices_;

  TF_DISALLOW_COPY_AND_ASSIGN(QrOpGpu);
};

#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_QR_OP_IMPL_H_
