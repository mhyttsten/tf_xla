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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_op_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/kernels/linalg/matrix_band_part_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

namespace functor {
namespace {

template <typename Scalar>
__global__ void MatrixBandFillKernel(const int num_threads,
                                     const int batch_size, const int m,
                                     const int n, const int num_lower_diags,
                                     const int num_upper_diags,
                                     const Scalar value,
                                     Scalar* __restrict__ output_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_op_gpuDTcuDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/linalg/cholesky_op_gpu.cu.cc", "MatrixBandFillKernel");

  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int col = index % n;
    const int row = (index / n) % m;
    const int band_start = (num_lower_diags < 0 ? 0 : row - num_lower_diags);
    const int band_end = (num_upper_diags < 0 ? n : row + num_upper_diags + 1);
    if (col < band_start || col >= band_end) {
      output_ptr[index] = Scalar(0);
    } else {
      output_ptr[index] = value;
    }
  }
}

}  // namespace

// Fills a banded matrix with a constant value.
template <typename Device, typename Scalar>
struct MatrixBandFillFunctor;

typedef Eigen::GpuDevice GPUDevice;

template <typename Scalar>
struct MatrixBandFillFunctor<GPUDevice, Scalar> {
  void operator()(OpKernelContext* context, const GPUDevice& device,
                  int num_lower_diags, int num_upper_diags, const Scalar& value,
                  typename TTypes<Scalar, 3>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int m = output.dimension(1);
    const int n = output.dimension(2);
    GpuLaunchConfig config = GetGpuLaunchConfig(batch_size * m * n, device);
    TF_CHECK_OK(GpuLaunchKernel(MatrixBandFillKernel<Scalar>,
                                config.block_count, config.thread_per_block, 0,
                                device.stream(), config.virtual_thread_count,
                                batch_size, m, n, num_lower_diags,
                                num_upper_diags, value, output.data()));
  }
};

}  // namespace functor

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class CholeskyOpGpu : public AsyncOpKernel {
 public:
  explicit CholeskyOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_op_gpuDTcuDTcc mht_1(mht_1_v, 264, "", "./tensorflow/core/kernels/linalg/cholesky_op_gpu.cu.cc", "CholeskyOpGpu");
}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_op_gpuDTcuDTcc mht_2(mht_2_v, 269, "", "./tensorflow/core/kernels/linalg/cholesky_op_gpu.cu.cc", "ComputeAsync");

    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64_t n = input.dim_size(ndims - 1);
#if GOOGLE_CUDA
    cublasFillMode_t fill = CUBLAS_FILL_MODE_UPPER;
#elif TENSORFLOW_USE_ROCM
#if TF_ROCM_VERSION >= 40500
    hipsolverFillMode_t fill = HIPSOLVER_FILL_MODE_UPPER;
#else
    rocblas_fill fill = rocblas_fill_upper;
#endif
#endif
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

    if (input.NumElements() == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      context->set_output(0, input);
      done();
      return;
    }

    // Allocate output.
    std::unique_ptr<GpuSolver> solver = std::make_unique<GpuSolver>(context);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(context,
                         context->forward_input_or_allocate_output(
                             {0}, 0, input.shape(), &output),
                         done);

    // Copy the lower triangular part of the input matrices to the output and
    // set the strictly upper triangular part to zero. We use a pre-existing
    // kernel MatrixBandPart to do this for all matrices in the batch at once,
    // before we launch each of the Cholesky factorization kernels.
    auto input_reshaped = input.template flat_inner_dims<Scalar, 3>();
    auto output_reshaped = output->template flat_inner_dims<Scalar, 3>();
    functor::MatrixBandPartFunctor<GPUDevice, Scalar> band_part;
    band_part(context, context->eigen_device<GPUDevice>(),
              n /* num_lower_diags */, 0 /* num_upper_diags */, input_reshaped,
              output_reshaped);

    // Launch a Cholesky kernel for each matrix in the batch.
    const int64_t batch_size = input_reshaped.dimension(0);
    std::vector<DeviceLapackInfo> dev_info;

#if CUDA_VERSION >= 9020
    // Decide whether to use the batched API.
    // TODO(rmlarsen): The value 128 was found to be optimal for the equivalent
    // split in matrix_solve_op. Tune this heuristic.
    constexpr int kMaxMatrixSizeToBatchSizeRatio = 128;
    const bool use_batched_solver =
        n <= kMaxMatrixSizeToBatchSizeRatio * batch_size;
    if (use_batched_solver) {
      // For small matrices or large batch sizes, we use the batched interface
      // from cuSolver.
      auto output_reshaped_ptrs = solver->GetScratchSpace<uint8>(
          sizeof(Scalar*) * batch_size, "input_copt_ptrs",
          /* on_host */ true);
      const Scalar** output_reshaped_ptrs_base =
          reinterpret_cast<const Scalar**>(output_reshaped_ptrs.mutable_data());
      for (int batch = 0; batch < batch_size; ++batch) {
        output_reshaped_ptrs_base[batch] = &output_reshaped(batch, 0, 0);
      }
      dev_info.push_back(
          solver->GetDeviceLapackInfo(batch_size, "potrfBatched"));
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->PotrfBatched(fill, n, output_reshaped_ptrs_base, n,
                               &dev_info.back(), batch_size),
          done);
      // TODO(rmlarsen): We have to clear the upper triangle of the output
      // due to a bug in potrfBatched. Remove this workaround once the bug
      // is fixed.
      auto input_reshaped = const_cast<const Tensor*>(output)
                                ->template flat_inner_dims<Scalar, 3>();
      auto output_reshaped = output->template flat_inner_dims<Scalar, 3>();
      functor::MatrixBandPartFunctor<GPUDevice, Scalar> band_part;
      band_part(context, context->eigen_device<GPUDevice>(),
                n /* num_lower_diags */, 0 /* num_upper_diags */,
                input_reshaped, output_reshaped);
    } else {
#endif

      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "potrf"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Potrf(fill, n, &output_reshaped(batch, 0, 0), n,
                          &dev_info.back()(batch)),
            done);
      }

#if CUDA_VERSION >= 9020
    }
#endif

    // Register callback to check info after kernels finish.
    auto info_checker = [context, done, n](
                            const Status& status,
                            const std::vector<HostLapackInfo>& host_infos) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPScholesky_op_gpuDTcuDTcc mht_3(mht_3_v, 381, "", "./tensorflow/core/kernels/linalg/cholesky_op_gpu.cu.cc", "lambda");

      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        Tensor* output = context->mutable_output(0);
        auto output_reshaped = output->template flat_inner_dims<Scalar, 3>();

        for (int i = 0; i < host_infos[0].size(); ++i) {
          if (host_infos[0](i) > 0) {
            LOG(WARNING) << "Cholesky decomposition was not successful for "
                            "batch "
                         << i
                         << ". The input might not be valid. "
                            "Filling lower-triangular output with NaNs.";
            typename TTypes<Scalar, 3>::Tensor output_batch(
                &output_reshaped(i, 0, 0), 1, n, n);
            functor::MatrixBandFillFunctor<GPUDevice, Scalar> band_fill;
            band_fill(context, context->eigen_device<GPUDevice>(),
                      /*num_lower_diags=*/n, /*num_upper_diags=*/0,
                      /*value=*/Eigen::NumTraits<Scalar>::quiet_NaN(),
                      output_batch);
          }
        }
      }
      done();
    };
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(info_checker));
  }
};

REGISTER_LINALG_OP_GPU("Cholesky", (CholeskyOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("Cholesky", (CholeskyOpGpu<double>), double);
REGISTER_LINALG_OP_GPU("Cholesky", (CholeskyOpGpu<complex64>), complex64);
REGISTER_LINALG_OP_GPU("Cholesky", (CholeskyOpGpu<complex128>), complex128);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
