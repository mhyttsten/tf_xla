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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_inverse_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_inverse_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_inverse_opDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/linalg/eye_functor.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

namespace tensorflow {

template <class Scalar>
class MatrixInverseOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit MatrixInverseOp(OpKernelConstruction* context) : Base(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_inverse_opDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/linalg/matrix_inverse_op.cc", "MatrixInverseOp");

    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_inverse_opDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/linalg/matrix_inverse_op.cc", "ComputeMatrix");

    const ConstMatrixMap& input = inputs[0];
    if (input.rows() == 0) {
      // By definition, an empty matrix's inverse is an empty matrix.
      return;
    }
    Eigen::PartialPivLU<Matrix> lu_decomposition;
    if (adjoint_) {
      // TODO(rmlarsen): For Eigen 3.2, this creates a temporary copy.
      // Make sure to backport: https://bitbucket.org/eigen/eigen/commits/
      // bd2219a74c96dfe3f6bc2c23588749e36d2d8173
      lu_decomposition.compute(input.adjoint());
    } else {
      lu_decomposition.compute(input);
    }
    // TODO(rmlarsen): Add check based on condition number estimation.
    // PartialPivLU cannot give strong guarantees on invertibility, but
    // we can at least guard against exact zero pivots. This can occur as
    // a result of basic user mistakes, such as providing integer valued
    // matrices that are exactly singular, or due to underflow if this
    // code is run with denormals being flushed to zero.
    const RealScalar min_abs_pivot =
        lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
    OP_REQUIRES(context, min_abs_pivot > RealScalar(0),
                errors::InvalidArgument("Input is not invertible."));
    outputs->at(0).noalias() = lu_decomposition.inverse();
  }

 private:
  bool adjoint_;

  TF_DISALLOW_COPY_AND_ASSIGN(MatrixInverseOp);
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class MatrixInverseOpGpu : public AsyncOpKernel {
 public:
  explicit MatrixInverseOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_inverse_opDTcc mht_2(mht_2_v, 269, "", "./tensorflow/core/kernels/linalg/matrix_inverse_op.cc", "MatrixInverseOpGpu");

    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_inverse_opDTcc mht_3(mht_3_v, 276, "", "./tensorflow/core/kernels/linalg/matrix_inverse_op.cc", "ComputeAsync");

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
        errors::InvalidArgument("Input matrices must be squares, got",
                                input.dim_size(ndims - 2), " != ", n),
        done);

    // By definition, an empty matrix's inverse is an empty matrix.
    if (input.NumElements() == 0) {
      context->set_output(0, input);
      done();
      return;
    }

    // Allocate output.
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(context,
                         context->forward_input_or_allocate_output(
                             {0}, 0, input.shape(), &output),
                         done);

    // TODO(rmlarsen): Convert to std::make_unique when available.
    std::unique_ptr<GpuSolver> solver(new GpuSolver(context));

    // Make a copy of the (possible adjointed) input that we will use for the
    // factorization step.
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<Scalar>::value,
                                       input.shape(), &input_copy),
        done);
    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    if (!adjoint_) {
      device.memcpy(input_copy.flat<Scalar>().data(),
                    input.flat<Scalar>().data(),
                    input.NumElements() * sizeof(Scalar));
    } else {
      OP_REQUIRES_OK_ASYNC(
          context, DoConjugateMatrixTranspose(device, input, &input_copy),
          done);
    }
    const int64_t batch_size = input_copy_reshaped.dimension(0);

    Tensor pivots;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<int>::value,
                                       TensorShape{batch_size, n}, &pivots),
        done);
    auto pivots_mat = pivots.template matrix<int>();
    auto input_copy_ptr_array = solver->GetScratchSpace<uint8>(
        sizeof(Scalar*) * batch_size, "input_copy_ptr_array",
        /* on_host */ true);
    auto output_ptr_array = solver->GetScratchSpace<uint8>(
        sizeof(Scalar*) * batch_size, "output_copy_ptr_array",
        /* on_host */ true);
    auto output_reshaped = output->template flat_inner_dims<Scalar, 3>();
    std::vector<DeviceLapackInfo> dev_info;
    if (n < 32 || batch_size > n) {
      // For small matrices or very large batch sizes, we use the batched
      // interfaces in cuBlas to avoid being dominated by kernel launch
      // overhead.
      // TODO(rmlarsen): Come up with a better heuristic based on a simple
      // cost model.
      const Scalar** input_copy_ptr_array_base =
          reinterpret_cast<const Scalar**>(input_copy_ptr_array.mutable_data());
      const Scalar** output_ptr_array_base =
          reinterpret_cast<const Scalar**>(output_ptr_array.mutable_data());
      for (int batch = 0; batch < batch_size; ++batch) {
        input_copy_ptr_array_base[batch] = &input_copy_reshaped(batch, 0, 0);
        output_ptr_array_base[batch] = &output_reshaped(batch, 0, 0);
      }

      if (n < 32) {
        // MatInvBatched only supports n < 32.
        dev_info.push_back(
            solver->GetDeviceLapackInfo(batch_size, "MatInvBatched"));
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->MatInvBatched(n, input_copy_ptr_array_base, n,
                                  output_ptr_array_base, n, &dev_info.back(),
                                  batch_size),

            done);
      } else {
        // For larger matrices and large batch size, we used the batched
        // GETRF/GETRI kernels.
        dev_info.push_back(
            solver->GetDeviceLapackInfo(batch_size, "GetrfBatched"));
        OP_REQUIRES_OK_ASYNC(context,
                             solver->GetrfBatched(n, input_copy_ptr_array_base,
                                                  n, pivots_mat.data(),
                                                  &dev_info.back(), batch_size),
                             done);
        // 2. Compute the inverse(s).
        dev_info.push_back(
            solver->GetDeviceLapackInfo(batch_size, "GetriBatched"));
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->GetriBatched(n, input_copy_ptr_array_base, n,
                                 pivots_mat.data(), output_ptr_array_base, n,
                                 &dev_info.back(), batch_size),
            done);
      }
    } else {
      // For large matrices, we compute the inverse of each matrix in the batch
      // sequentially. Here we use the cuSolver methods GETRF/GETRS because they
      // are MUCH faster than their batched cuBlas equivalents for large
      // matrices.
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "getrf"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Getrf(n, n, &input_copy_reshaped(batch, 0, 0), n,
                          &pivots_mat(batch, 0), &dev_info.back()(batch)),
            done);
      }

      // Set all right-hand sides to the identity.
      functor::EyeFunctor<GPUDevice, Scalar> eye;
      eye(device, output_reshaped);

#if GOOGLE_CUDA
      cublasOperation_t trans = CUBLAS_OP_N;
#elif TENSORFLOW_USE_ROCM
      rocblas_operation trans = rocblas_operation_none;
#endif

      // Solve A X = I.
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "getrs"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Getrs(trans, n, n, &input_copy_reshaped(batch, 0, 0), n,
                          &pivots_mat(batch, 0), &output_reshaped(batch, 0, 0),
                          n, &dev_info.back()(batch)),
            done);
      }
    }
    // Callback for checking info after kernels finish.
    auto info_checker = [context, done](
                            const Status& status,
                            const std::vector<HostLapackInfo>& host_infos) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_inverse_opDTcc mht_4(mht_4_v, 431, "", "./tensorflow/core/kernels/linalg/matrix_inverse_op.cc", "lambda");

      if (!status.ok() && errors::IsInvalidArgument(status)) {
        for (const auto& host_info : host_infos) {
          for (int i = 0; i < host_info.size(); ++i) {
            // Match the CPU error message for singular matrices. Otherwise
            // just print the original error message from the call itself
            // below.
            OP_REQUIRES_ASYNC(
                context, host_info(i) <= 0,
                errors::InvalidArgument("Input is not invertible."), done);
          }
        }
      }
      OP_REQUIRES_OK_ASYNC(context, status, done);
      done();
    };
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(info_checker));
  }

 private:
  bool adjoint_;
};

REGISTER_LINALG_OP_GPU("MatrixInverse", (MatrixInverseOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("MatrixInverse", (MatrixInverseOpGpu<double>), double);
REGISTER_LINALG_OP_GPU("MatrixInverse", (MatrixInverseOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("MatrixInverse", (MatrixInverseOpGpu<complex128>),
                       complex128);

#endif  // GOOGLE_CUDA

REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<float>), float);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<double>), double);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<complex64>), complex64);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<complex128>), complex128);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<float>), float);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<double>), double);

}  // namespace tensorflow
