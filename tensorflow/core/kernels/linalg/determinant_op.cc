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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc() {
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

// See docs in ../ops/linalg_ops.cc.

#include <cmath>

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/linalg/determinant_op.h"
#endif

#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

namespace tensorflow {

// A helper function to compute the sign and absolute value of the log of the
// determinant of inputs via a partially pivoted LU
// factorization.
//
// Returns the log of the absolute value of the determinant, and its sign in
// 'sign'.
template <class Scalar>
static typename Eigen::NumTraits<Scalar>::Real SLogDet(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& inputs,
    Scalar* sign) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "SLogDet");

  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  RealScalar log_abs_det = 0;
  *sign = 1;
  // An empty matrix' determinant is defined to be 1.
  // (https://en.wikipedia.org/wiki/Determinant)
  if (inputs.size() > 0) {
    // Compute the log determinant through a Partially Pivoted LU decomposition
    using Eigen::Dynamic;
    Eigen::PartialPivLU<Eigen::Matrix<Scalar, Dynamic, Dynamic>> lu(inputs);
    Eigen::Matrix<Scalar, Dynamic, Dynamic> LU = lu.matrixLU();
    *sign = lu.permutationP().determinant();
    auto diag = LU.diagonal().array().eval();
    auto abs_diag = diag.cwiseAbs().eval();
    log_abs_det += abs_diag.log().sum();
    *sign *= (diag / abs_diag).prod();
  }
  if (!Eigen::numext::isfinite(log_abs_det)) {
    *sign = 0;
    log_abs_det =
        log_abs_det > 0 ? -std::log(RealScalar(0)) : std::log(RealScalar(0));
  }
  return log_abs_det;
}

template <class Scalar>
class LogDeterminantOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit LogDeterminantOp(OpKernelConstruction* context) : Base(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_1(mht_1_v, 254, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "LogDeterminantOp");
}

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_2(mht_2_v, 260, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "GetOutputMatrixShapes");

    return TensorShapes({TensorShape({}), TensorShape({})});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_3(mht_3_v, 268, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "ComputeMatrix");

    Scalar sign;
    const RealScalar log_abs_det = SLogDet(
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(inputs[0]),
        &sign);

    outputs->at(0)(0, 0) = sign;
    outputs->at(1)(0, 0) = log_abs_det;
  }
};

template <class Scalar>
class DeterminantOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit DeterminantOp(OpKernelConstruction* context) : Base(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_4(mht_4_v, 287, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "DeterminantOp");
}

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shape) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_5(mht_5_v, 293, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "GetOutputMatrixShapes");

    return TensorShapes({TensorShape({})});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_6(mht_6_v, 301, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "ComputeMatrix");

    Scalar sign;
    const RealScalar log_abs_det = SLogDet(
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(inputs[0]),
        &sign);
    outputs->at(0)(0, 0) = sign * std::exp(log_abs_det);
  }
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class DeterminantOpGpu : public AsyncOpKernel {
 public:
  explicit DeterminantOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_7(mht_7_v, 321, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "DeterminantOpGpu");
}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_8(mht_8_v, 326, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "ComputeAsync");

    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64_t n = input.dim_size(ndims - 1);
    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);
    OP_REQUIRES_ASYNC(
        context, input.dim_size(ndims - 2) == n,
        errors::InvalidArgument("Input matrices must be square, got",
                                input.dim_size(ndims - 2), " != ", n),
        done);

    // Allocate output.
    TensorShape out_shape;
    for (int dim = 0; dim < ndims - 2; ++dim) {
      out_shape.AddDim(input.dim_size(dim));
    }
    out_shape.AppendShape(TensorShape({}));
    Tensor* out;
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, out_shape, &out),
                         done);

    // By definition, the determinant of an empty matrix is equal to one.
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    if (input.NumElements() == 0) {
      functor::SetOneFunctor<GPUDevice, Scalar> f;
      f(d, out->template flat<Scalar>());
      done();
      return;
    }

    // TODO(rmlarsen): Convert to absl::make_unique when available.
    std::unique_ptr<GpuSolver> solver(new GpuSolver(context));

    // Reuse the input buffer or make a copy for the factorization step,
    // depending on whether this ops owns it exclusively.
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->forward_input_or_allocate_scoped_tensor(
            {0}, DataTypeToEnum<Scalar>::value, input.shape(), &input_copy),
        done);
    if (!input.SharesBufferWith(input_copy)) {
      d.memcpy(input_copy.flat<Scalar>().data(), input.flat<Scalar>().data(),
               input.NumElements() * sizeof(Scalar));
    }
    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const int64_t batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<int>::value,
                                       TensorShape{batch_size, n}, &pivots),
        done);
    auto pivots_mat = pivots.template matrix<int>();

    // Prepare pointer arrays for cuBlas' batch interface.
    // TODO(rmlarsen): Find a way to encode pointer arrays in pinned host memory
    // without the ugly casting.
    auto input_copy_ptrs = solver->GetScratchSpace<uint8>(
        sizeof(Scalar*) * batch_size, "input_copy_ptrs",
        /* on_host */ true);
    auto output_reshaped = out->template flat_inner_dims<Scalar, 1>();

    // Compute the partially pivoted LU factorization(s) of the matrix/matrices.
    std::vector<DeviceLapackInfo> dev_info;
    if (n / batch_size <= 128) {
      // For small matrices or large batch sizes, we use the batched interface
      // from cuBlas.
      const Scalar** input_copy_ptrs_base =
          reinterpret_cast<const Scalar**>(input_copy_ptrs.mutable_data());
      for (int batch = 0; batch < batch_size; ++batch) {
        input_copy_ptrs_base[batch] = &input_copy_reshaped(batch, 0, 0);
      }
      dev_info.push_back(
          solver->GetDeviceLapackInfo(batch_size, "getrfBatched"));
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->GetrfBatched(n, input_copy_ptrs_base, n, pivots_mat.data(),
                               &dev_info.back(), batch_size),
          done);
    } else {
      // For small batch sizes we use the non-batched interface from cuSolver,
      // which is much faster for large matrices.
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "getrf"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Getrf(n, n, &input_copy_reshaped(batch, 0, 0), n,
                          &pivots_mat(batch, 0), &dev_info.back()(batch)),
            done);
      }
    }

    // Compute the determinant for each batch as (-1)^s * prod(diag(U)),
    // where s is the order of the permutation encoded in pivots and U is the
    // upper triangular factor of the LU factorization, which is written to
    // input_copy by the Getrf{Batched} kernel.
    functor::DeterminantFromPivotedLUFunctor<GPUDevice, Scalar> functor;
    functor(d,
            const_cast<const Tensor*>(&input_copy)
                ->template flat_inner_dims<Scalar, 3>(),
            pivots_mat.data(), output_reshaped, dev_info.back().mutable_data());

    // Register callback to check info after kernels finish.
    auto info_checker = [context, done](
                            const Status& status,
                            const std::vector<HostLapackInfo>& host_infos) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_9(mht_9_v, 441, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "lambda");

      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        for (int i = 0; i < host_infos[0].size(); ++i) {
          // It is OK for a matrix to be singular (signaled by info > 0),
          // corresponding to determinant of zero, but we do want to catch
          // invalid arguments to Getrf{Batched}.
          OP_REQUIRES_ASYNC(
              context, host_infos[0](i) >= 0,
              errors::InvalidArgument("Invalid input argument no. ",
                                      host_infos[0].data()[i],
                                      " for batch index ", i, "."),
              done);
        }
      }
      done();
    };
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(info_checker));
  }
};

template <class Scalar>
class LogDeterminantOpGpu : public AsyncOpKernel {
 public:
  explicit LogDeterminantOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_10(mht_10_v, 470, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "LogDeterminantOpGpu");
}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_11(mht_11_v, 475, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "ComputeAsync");

    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64_t n = input.dim_size(ndims - 1);
    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);
    OP_REQUIRES_ASYNC(
        context, input.dim_size(ndims - 2) == n,
        errors::InvalidArgument("Input matrices must be square, got",
                                input.dim_size(ndims - 2), " != ", n),
        done);

    // Allocate output.
    TensorShape out_shape;
    for (int dim = 0; dim < ndims - 2; ++dim) {
      out_shape.AddDim(input.dim_size(dim));
    }
    out_shape.AppendShape(TensorShape({}));
    Tensor* sign;
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, out_shape, &sign),
                         done);
    Tensor* log_abs_det;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(1, out_shape, &log_abs_det), done);

    // By definition, the determinant of an empty matrix is equal to one.
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    if (input.NumElements() == 0) {
      functor::SetOneFunctor<GPUDevice, Scalar> one_func;
      one_func(d, sign->template flat<Scalar>());
      functor::SetZeroFunctor<GPUDevice, Scalar> zero_func;
      zero_func(d, log_abs_det->template flat<Scalar>());
      done();
      return;
    }

    // TODO(rmlarsen): Convert to absl::make_unique when available.
    std::unique_ptr<GpuSolver> solver(new GpuSolver(context));

    // Reuse the input buffer or make a copy for the factorization step,
    // depending on whether this ops owns it exclusively.
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->forward_input_or_allocate_scoped_tensor(
            {0}, DataTypeToEnum<Scalar>::value, input.shape(), &input_copy),
        done);
    if (!input.SharesBufferWith(input_copy)) {
      d.memcpy(input_copy.flat<Scalar>().data(), input.flat<Scalar>().data(),
               input.NumElements() * sizeof(Scalar));
    }
    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const int64_t batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<int>::value,
                                       TensorShape{batch_size, n}, &pivots),
        done);
    auto pivots_mat = pivots.template matrix<int>();

    // Prepare pointer arrays for cuBlas' batch interface.
    // TODO(rmlarsen): Find a way to encode pointer arrays in pinned host memory
    // without the ugly casting.
    auto input_copy_ptrs = solver->GetScratchSpace<uint8>(
        sizeof(Scalar*) * batch_size, "input_copy_ptrs",
        /* on_host */ true);

    // Compute the partially pivoted LU factorization(s) of the matrix/matrices.
    std::vector<DeviceLapackInfo> dev_info;
    if (n / batch_size <= 128) {
      // For small matrices or large batch sizes, we use the batched interface
      // from cuBlas.
      const Scalar** input_copy_ptrs_base =
          reinterpret_cast<const Scalar**>(input_copy_ptrs.mutable_data());
      for (int batch = 0; batch < batch_size; ++batch) {
        input_copy_ptrs_base[batch] = &input_copy_reshaped(batch, 0, 0);
      }
      dev_info.push_back(
          solver->GetDeviceLapackInfo(batch_size, "getrfBatched"));
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->GetrfBatched(n, input_copy_ptrs_base, n, pivots_mat.data(),
                               &dev_info.back(), batch_size),
          done);
    } else {
      // For large matrices or small batch sizes we use the non-batched
      // interface from cuSolver, which is much faster for large matrices.
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "getrf"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Getrf(n, n, &input_copy_reshaped(batch, 0, 0), n,
                          &pivots_mat(batch, 0), &dev_info.back()(batch)),
            done);
      }
    }

    auto input_copy_reshaped_const =
        const_cast<const Tensor*>(&input_copy)
            ->template flat_inner_dims<Scalar, 3>();
    auto sign_reshaped = sign->flat<Scalar>();
    auto log_abs_det_reshaped = log_abs_det->flat<Scalar>();
    // Compute the determinant for each batch as (-1)^s * prod(diag(U)),
    // where s is the order of the permutation encoded in pivots and U is the
    // upper triangular factor of the LU factorization, which is written to
    // input_copy by the Getrf{Batched} kernel.
    functor::LogDeterminantFromPivotedLUFunctor<GPUDevice, Scalar> functor;
    functor(d, input_copy_reshaped_const, pivots_mat.data(), sign_reshaped,
            log_abs_det_reshaped);

    // Register callback to check info after kernels finish.
    auto info_checker = [context, done](
                            const Status& status,
                            const std::vector<HostLapackInfo>& host_infos) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSdeterminant_opDTcc mht_12(mht_12_v, 597, "", "./tensorflow/core/kernels/linalg/determinant_op.cc", "lambda");

      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        for (int i = 0; i < host_infos[0].size(); ++i) {
          // It is OK for a matrix to be singular (signaled by info > 0),
          // corresponding to determinant of zero, but we do want to catch
          // invalid arguments to Getrf{Batched}.
          OP_REQUIRES_ASYNC(
              context, host_infos[0](i) >= 0,
              errors::InvalidArgument("Invalid input argument no. ",
                                      host_infos[0].data()[i],
                                      " for batch index ", i, "."),
              done);
        }
      }
      done();
    };
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(info_checker));
  }
};

REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<double>), double);
REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<complex128>),
                       complex128);

REGISTER_LINALG_OP_GPU("LogMatrixDeterminant", (LogDeterminantOpGpu<float>),
                       float);
REGISTER_LINALG_OP_GPU("LogMatrixDeterminant", (LogDeterminantOpGpu<double>),
                       double);
REGISTER_LINALG_OP_GPU("LogMatrixDeterminant", (LogDeterminantOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("LogMatrixDeterminant",
                       (LogDeterminantOpGpu<complex128>), complex128);
#endif  // GOOGLE_CUDA

REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<float>), float);
REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<double>), double);
REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<complex64>), complex64);
REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<complex128>),
                   complex128);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<float>), float);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<double>), double);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<complex64>),
                   complex64);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<complex128>),
                   complex128);

REGISTER_LINALG_OP("LogMatrixDeterminant", (LogDeterminantOp<float>), float);
REGISTER_LINALG_OP("LogMatrixDeterminant", (LogDeterminantOp<double>), double);
REGISTER_LINALG_OP("LogMatrixDeterminant", (LogDeterminantOp<complex64>),
                   complex64);
REGISTER_LINALG_OP("LogMatrixDeterminant", (LogDeterminantOp<complex128>),
                   complex128);
}  // namespace tensorflow
