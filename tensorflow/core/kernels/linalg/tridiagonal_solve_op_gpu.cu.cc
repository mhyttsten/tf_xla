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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

static const char kNotInvertibleMsg[] = "The matrix is not invertible.";

static const char kNotInvertibleScalarMsg[] =
    "The matrix is not invertible: it is a scalar with value zero.";

template <typename Scalar>
__global__ void SolveForSizeOneOrTwoKernel(const int m,
                                           const Scalar* __restrict__ diags,
                                           const Scalar* __restrict__ rhs,
                                           const int num_rhs,
                                           Scalar* __restrict__ x) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "SolveForSizeOneOrTwoKernel");

  const Scalar nan = Eigen::NumTraits<Scalar>::quiet_NaN();
  if (m == 1) {
    bool singular = diags[1] == Scalar(0);
    for (int i : GpuGridRangeX(num_rhs)) {
      x[i] = singular ? nan : rhs[i] / diags[1];
    }
  } else {
    const Scalar det = diags[2] * diags[3] - diags[0] * diags[5];
    bool singular = det == Scalar(0);
    for (int i : GpuGridRangeX(num_rhs)) {
      x[i] = singular ? nan
                      : (diags[3] * rhs[i] - diags[0] * rhs[i + num_rhs]) / det;
      x[i + num_rhs] =
          singular ? nan
                   : (diags[2] * rhs[i + num_rhs] - diags[5] * rhs[i]) / det;
    }
  }
}

template <typename Scalar>
se::DeviceMemory<Scalar> AsDeviceMemory(const Scalar* cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<Scalar*>(cuda_memory));
  se::DeviceMemory<Scalar> typed(wrapped);
  return typed;
}

template <typename Scalar>
void CopyDeviceToDevice(OpKernelContext* context, const Scalar* src,
                        Scalar* dst, const int num_elements) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_1(mht_1_v, 248, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "CopyDeviceToDevice");

  auto src_device_mem = AsDeviceMemory(src);
  auto dst_device_mem = AsDeviceMemory(dst);
  auto* stream = context->op_device_context()->stream();
  bool copy_status = stream
                         ->ThenMemcpyD2D(&dst_device_mem, src_device_mem,
                                         sizeof(Scalar) * num_elements)
                         .ok();

  if (!copy_status) {
    context->SetStatus(errors::Internal("Copying device-to-device failed."));
  }
}

// This implementation is used in cases when the batching mechanism of
// LinearAlgebraOp is suitable. See TridiagonalSolveOpGpu below.
template <class Scalar>
class TridiagonalSolveOpGpuLinalg : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit TridiagonalSolveOpGpuLinalg(OpKernelConstruction* context)
      : Base(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_2(mht_2_v, 273, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "TridiagonalSolveOpGpuLinalg");

    OP_REQUIRES_OK(context, context->GetAttr("partial_pivoting", &pivoting_));
    perturb_singular_ = false;
    if (context->HasAttr("perturb_singular")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("perturb_singular", &perturb_singular_));
    }
    OP_REQUIRES(
        context, perturb_singular_ == false,
        errors::Unimplemented("The solver to support perturb_singular is"
                              " not implemented on GPU."));
  }

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_3(mht_3_v, 291, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "ValidateInputMatrixShapes");

    auto num_inputs = input_matrix_shapes.size();
    OP_REQUIRES(context, num_inputs == 2,
                errors::InvalidArgument("Expected two input matrices, got ",
                                        num_inputs, "."));

    auto num_diags = input_matrix_shapes[0].dim_size(0);
    OP_REQUIRES(
        context, num_diags == 3,
        errors::InvalidArgument("Expected diagonals to be provided as a "
                                "matrix with 3 columns, got ",
                                num_diags, " columns."));

    auto num_rows1 = input_matrix_shapes[0].dim_size(1);
    auto num_rows2 = input_matrix_shapes[1].dim_size(0);
    OP_REQUIRES(context, num_rows1 == num_rows2,
                errors::InvalidArgument("Expected same number of rows in both "
                                        "arguments, got ",
                                        num_rows1, " and ", num_rows2, "."));
  }

  bool EnableInputForwarding() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_4(mht_4_v, 315, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "EnableInputForwarding");
 return false; }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_5(mht_5_v, 321, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "GetOutputMatrixShapes");

    return TensorShapes({input_matrix_shapes[1]});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_6(mht_6_v, 329, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "ComputeMatrix");

    const auto diagonals = inputs[0];
    // Superdiagonal elements, first is ignored.
    const auto& superdiag = diagonals.row(0);
    // Diagonal elements.
    const auto& diag = diagonals.row(1);
    // Subdiagonal elements, last is ignored.
    const auto& subdiag = diagonals.row(2);
    // Right-hand sides.
    const auto& rhs = inputs[1];
    MatrixMap& x = outputs->at(0);
    const int m = diag.size();
    const int k = rhs.cols();

    if (m == 0) {
      return;
    }
    if (m < 3) {
      // Cusparse gtsv routine requires m >= 3. Solving manually for m < 3.
      SolveForSizeOneOrTwo(context, diagonals.data(), rhs.data(), x.data(), m,
                           k);
      return;
    }
    std::unique_ptr<GpuSparse> cusparse_solver(new GpuSparse(context));
    OP_REQUIRES_OK(context, cusparse_solver->Initialize());
    if (k == 1) {
      // rhs is copied into x, then gtsv replaces x with solution.
      CopyDeviceToDevice(context, rhs.data(), x.data(), m);
      SolveWithGtsv(context, cusparse_solver, superdiag.data(), diag.data(),
                    subdiag.data(), x.data(), m, 1);
    } else {
      // Gtsv expects rhs in column-major form, so we have to transpose.
      // rhs is transposed into temp, gtsv replaces temp with solution, then
      // temp is transposed into x.
      std::unique_ptr<GpuSolver> cublas_solver(new GpuSolver(context));
      Tensor temp;
      TensorShape temp_shape({k, m});
      OP_REQUIRES_OK(context,
                     cublas_solver->allocate_scoped_tensor(
                         DataTypeToEnum<Scalar>::value, temp_shape, &temp));
      TransposeWithGeam(context, cublas_solver, rhs.data(),
                        temp.flat<Scalar>().data(), m, k);
      SolveWithGtsv(context, cusparse_solver, superdiag.data(), diag.data(),
                    subdiag.data(), temp.flat<Scalar>().data(), m, k);
      TransposeWithGeam(context, cublas_solver, temp.flat<Scalar>().data(),
                        x.data(), k, m);
    }
  }

 private:
  void TransposeWithGeam(OpKernelContext* context,
                         const std::unique_ptr<GpuSolver>& cublas_solver,
                         const Scalar* src, Scalar* dst, const int src_rows,
                         const int src_cols) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_7(mht_7_v, 385, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "TransposeWithGeam");

    const Scalar zero(0), one(1);
    OP_REQUIRES_OK(context,
                   cublas_solver->Geam(CUBLAS_OP_T, CUBLAS_OP_N, src_rows,
                                       src_cols, &one, src, src_cols, &zero,
                                       static_cast<const Scalar*>(nullptr),
                                       src_rows, dst, src_rows));
  }

  void SolveWithGtsv(OpKernelContext* context,
                     std::unique_ptr<GpuSparse>& cusparse_solver,
                     const Scalar* superdiag, const Scalar* diag,
                     const Scalar* subdiag, Scalar* rhs, const int num_eqs,
                     const int num_rhs) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_8(mht_8_v, 401, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "SolveWithGtsv");

    auto buffer_function = pivoting_
                               ? &GpuSparse::Gtsv2BufferSizeExt<Scalar>
                               : &GpuSparse::Gtsv2NoPivotBufferSizeExt<Scalar>;
    size_t buffer_size;
    OP_REQUIRES_OK(context, (cusparse_solver.get()->*buffer_function)(
                                num_eqs, num_rhs, subdiag, diag, superdiag, rhs,
                                num_eqs, &buffer_size));
    Tensor temp_tensor;
    TensorShape temp_shape({static_cast<int64_t>(buffer_size)});
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_UINT8, temp_shape, &temp_tensor));
    void* buffer = temp_tensor.flat<std::uint8_t>().data();

    auto solver_function = pivoting_ ? &GpuSparse::Gtsv2<Scalar>
                                     : &GpuSparse::Gtsv2NoPivot<Scalar>;
    OP_REQUIRES_OK(context, (cusparse_solver.get()->*solver_function)(
                                num_eqs, num_rhs, subdiag, diag, superdiag, rhs,
                                num_eqs, buffer));
  }

  void SolveForSizeOneOrTwo(OpKernelContext* context, const Scalar* diagonals,
                            const Scalar* rhs, Scalar* output, int m, int k) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_9(mht_9_v, 426, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "SolveForSizeOneOrTwo");

    const Eigen::GpuDevice& device = context->eigen_device<Eigen::GpuDevice>();
    GpuLaunchConfig cfg = GetGpuLaunchConfig(
        /*work_element_count=*/1, device, &SolveForSizeOneOrTwoKernel<Scalar>,
        /*dynamic_shared_memory_size=*/0,
        /*block_size_limit=*/0);
    TF_CHECK_OK(GpuLaunchKernel(SolveForSizeOneOrTwoKernel<Scalar>,
                                cfg.block_count, cfg.thread_per_block,
                                /*shared_memory_size_bytes=*/0, device.stream(),
                                m, diagonals, rhs, k, output));
  }

  bool pivoting_;
  bool perturb_singular_;
};

template <class Scalar>
class TridiagonalSolveOpGpu : public OpKernel {
 public:
  explicit TridiagonalSolveOpGpu(OpKernelConstruction* context)
      : OpKernel(context), linalgOp_(context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_10(mht_10_v, 449, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "TridiagonalSolveOpGpu");

    OP_REQUIRES_OK(context, context->GetAttr("partial_pivoting", &pivoting_));
  }

  void Compute(OpKernelContext* context) final {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_11(mht_11_v, 456, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "Compute");

    const Tensor& lhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const int ndims = lhs.dims();
    const int64 num_rhs = rhs.dim_size(rhs.dims() - 1);
    const int64 matrix_size = lhs.dim_size(ndims - 1);
    int64 batch_size = 1;
    for (int i = 0; i < ndims - 2; i++) {
      batch_size *= lhs.dim_size(i);
    }

    // The batching mechanism of LinearAlgebraOp is used when it's not
    // possible or desirable to use GtsvBatched.
    const bool use_linalg_op =
        pivoting_            // GtsvBatched doesn't do pivoting
        || num_rhs > 1       // GtsvBatched doesn't support multiple rhs
        || matrix_size < 3   // Not supported in cuSparse, use the custom kernel
        || batch_size == 1;  // No point to use GtsvBatched

    if (use_linalg_op) {
      linalgOp_.Compute(context);
    } else {
      ComputeWithGtsvBatched(context, lhs, rhs, batch_size);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TridiagonalSolveOpGpu);

  void ComputeWithGtsvBatched(OpKernelContext* context, const Tensor& lhs,
                              const Tensor& rhs, const int batch_size) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_12(mht_12_v, 489, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "ComputeWithGtsvBatched");

    const Scalar* rhs_data = rhs.flat<Scalar>().data();
    const int ndims = lhs.dims();

    // To use GtsvBatched we need to transpose the left-hand side from shape
    // [..., 3, M] into shape [3, ..., M]. With shape [..., 3, M] the stride
    // between corresponding diagonal elements of consecutive batch components
    // is 3 * M, while for the right-hand side the stride is M. Unfortunately,
    // GtsvBatched requires the strides to be the same. For this reason we
    // transpose into [3, ..., M], so that diagonals, superdiagonals, and
    // and subdiagonals are separated from each other, and have stride M.
    Tensor lhs_transposed;
    TransposeLhsForGtsvBatched(context, lhs, lhs_transposed);
    int matrix_size = lhs.dim_size(ndims - 1);
    const Scalar* lhs_data = lhs_transposed.flat<Scalar>().data();
    const Scalar* superdiag = lhs_data;
    const Scalar* diag = lhs_data + matrix_size * batch_size;
    const Scalar* subdiag = lhs_data + 2 * matrix_size * batch_size;

    // Copy right-hand side into the output. GtsvBatched will replace it with
    // the solution.
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs.shape(), &output));
    CopyDeviceToDevice(context, rhs_data, output->flat<Scalar>().data(),
                       rhs.flat<Scalar>().size());
    Scalar* x = output->flat<Scalar>().data();

    std::unique_ptr<GpuSparse> cusparse_solver(new GpuSparse(context));

    OP_REQUIRES_OK(context, cusparse_solver->Initialize());

    size_t buffer_size;
    OP_REQUIRES_OK(context, cusparse_solver->Gtsv2StridedBatchBufferSizeExt(
                                matrix_size, subdiag, diag, superdiag, x,
                                batch_size, matrix_size, &buffer_size));
    Tensor temp_tensor;
    TensorShape temp_shape({static_cast<int64_t>(buffer_size)});
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_UINT8, temp_shape, &temp_tensor));
    void* buffer = temp_tensor.flat<std::uint8_t>().data();
    OP_REQUIRES_OK(context, cusparse_solver->Gtsv2StridedBatch(
                                matrix_size, subdiag, diag, superdiag, x,
                                batch_size, matrix_size, buffer));
  }

  void TransposeLhsForGtsvBatched(OpKernelContext* context, const Tensor& lhs,
                                  Tensor& lhs_transposed) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_solve_op_gpuDTcuDTcc mht_13(mht_13_v, 538, "", "./tensorflow/core/kernels/linalg/tridiagonal_solve_op_gpu.cu.cc", "TransposeLhsForGtsvBatched");

    const int ndims = lhs.dims();

    // Permutation of indices, transforming [..., 3, M] into [3, ..., M].
    // E.g. for ndims = 6, it is [4, 0, 1, 2, 3, 5].
    std::vector<int> perm(ndims);
    perm[0] = ndims - 2;
    for (int i = 0; i < ndims - 2; ++i) {
      perm[i + 1] = i;
    }
    perm[ndims - 1] = ndims - 1;

    std::vector<int64_t> dims;
    for (int index : perm) {
      dims.push_back(lhs.dim_size(index));
    }
    TensorShape lhs_transposed_shape(
        gtl::ArraySlice<int64_t>(dims.data(), ndims));

    std::unique_ptr<GpuSolver> cublas_solver(new GpuSolver(context));
    OP_REQUIRES_OK(context, cublas_solver->allocate_scoped_tensor(
                                DataTypeToEnum<Scalar>::value,
                                lhs_transposed_shape, &lhs_transposed));
    auto device = context->eigen_device<Eigen::GpuDevice>();
    OP_REQUIRES_OK(
        context,
        DoTranspose(device, lhs, gtl::ArraySlice<int>(perm.data(), ndims),
                    &lhs_transposed));
  }

  TridiagonalSolveOpGpuLinalg<Scalar> linalgOp_;
  bool pivoting_;
};

REGISTER_LINALG_OP_GPU("TridiagonalSolve", (TridiagonalSolveOpGpu<float>),
                       float);
REGISTER_LINALG_OP_GPU("TridiagonalSolve", (TridiagonalSolveOpGpu<double>),
                       double);
REGISTER_LINALG_OP_GPU("TridiagonalSolve", (TridiagonalSolveOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("TridiagonalSolve", (TridiagonalSolveOpGpu<complex128>),
                       complex128);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
