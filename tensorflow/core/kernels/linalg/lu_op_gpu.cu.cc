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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_op_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include <algorithm>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {
template <typename Scalar>
__device__ void ComputePermutationFromTranspositions(
    int64 num_rows, const int* __restrict__ pivots,
    Scalar* __restrict__ permutation_indices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_op_gpuDTcuDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/linalg/lu_op_gpu.cu.cc", "ComputePermutationFromTranspositions");

  // Fill in the output array with the identity permutation.
  for (int i = 0; i < num_rows; ++i) {
    permutation_indices[i] = Scalar(i);
  }

  // Compute the permutation from a sequence of transpositions encoded
  // in the pivot array by applying the transpositions in order on the
  // identity permutation.
  for (int i = 0; i < num_rows; ++i) {
    // Note: Internally, the cuBlas code uses Fortran convention (1-based)
    // indexing so ith row was swapped with (pivots[i]-1)'th row in 0-based
    // indexing.
    Scalar t = permutation_indices[i];
    permutation_indices[i] = permutation_indices[pivots[i] - 1];
    permutation_indices[pivots[i] - 1] = t;
  }
}
}  // namespace

// Kernel to compute the inverse of a permutation from a sequence of
// transpositions.
template <typename Scalar>
__global__ void ComputePermutationFromTranspositionsKernel(
    GpuLaunchConfig config, const int64 num_rows,
    const int* __restrict__ all_pivots,
    Scalar* __restrict__ all_permutation_indices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_op_gpuDTcuDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/kernels/linalg/lu_op_gpu.cu.cc", "ComputePermutationFromTranspositionsKernel");

  // We only parallelize over batches here. Performance is not critical,
  // since this cheap O(num_rows) kernel always follows an O(num_rows^3)
  // LU factorization.
  GPU_1D_KERNEL_LOOP(index, config.virtual_thread_count) {
    ComputePermutationFromTranspositions(
        num_rows, all_pivots + index * num_rows,
        all_permutation_indices + index * num_rows);
  }
}

template <class Scalar, class Tidx>
class LuOpGpu : public AsyncOpKernel {
 public:
  explicit LuOpGpu(OpKernelConstruction* context) : AsyncOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_op_gpuDTcuDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/kernels/linalg/lu_op_gpu.cu.cc", "LuOpGpu");
}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_op_gpuDTcuDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/kernels/linalg/lu_op_gpu.cu.cc", "ComputeAsync");

    const Tensor& input = context->input(0);

    // Analyze shape and validate inputs.
    const int input_rank = input.dims();

    OP_REQUIRES_ASYNC(
        context, input_rank >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", input_rank),
        done);

    const int64 num_rows = input.dim_size(input_rank - 2);
    const int64 num_cols = input.dim_size(input_rank - 1);

    OP_REQUIRES_ASYNC(
        context, num_rows == num_cols,
        errors::InvalidArgument("Input matrices must be squares, got", num_rows,
                                " != ", num_cols),
        done);

    TensorShape batch_shape;
    for (int dim = 0; dim < input_rank - 2; ++dim) {
      batch_shape.AddDim(input.dim_size(dim));
    }
    TensorShape permutation_indices_shape = batch_shape;
    permutation_indices_shape.AddDim(num_rows);

    const GPUDevice& device = context->eigen_device<GPUDevice>();
    auto solver = absl::make_unique<GpuSolver>(context);

    // We output the packed triangular factors in a dense form.
    // The lower triangular factor L corresponds to the strictly lower
    // triangular part of packed_triangular_factors with an implicit unit
    // diagonal. The upper triangular factor U is the upper triangular part of
    // packed_triangular_factors. The triangular factors satisfy the equation
    //     P * input_matrix = L * U
    // where P is the permutation matrix corresponding to the indices in
    // permutation_indices.
    //
    // Reuse the input buffer or make a copy for the factorization step,
    // depending on whether this ops owns it exclusively.
    Tensor* packed_triangular_factors;
    OP_REQUIRES_OK_ASYNC(context,
                         context->forward_input_or_allocate_output(
                             {0}, 0, input.shape(), &packed_triangular_factors),
                         done);
    if (!packed_triangular_factors->SharesBufferWith(input)) {
      device.memcpy(packed_triangular_factors->flat<Scalar>().data(),
                    input.flat<Scalar>().data(),
                    input.NumElements() * sizeof(Scalar));
    }

    // Allocate output permutation.
    Tensor* permutation_indices = nullptr;
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_output(1, permutation_indices_shape,
                                                  &permutation_indices),
                         done);

    if (input.NumElements() == 0) {
      done();
      return;
    }

    // Allocate a temporary Tensor to store the transposed packed triangular
    // factors.
    Tensor packed_triangular_factors_transpose;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<Scalar>::value, input.shape(),
                               &packed_triangular_factors_transpose),
        done);
    auto packed_triangular_factors_transpose_reshaped =
        packed_triangular_factors_transpose
            .template flat_inner_dims<Scalar, 3>();
    const int64 batch_size =
        packed_triangular_factors_transpose_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK_ASYNC(context,
                         solver->allocate_scoped_tensor(
                             DataTypeToEnum<int32>::value,
                             TensorShape{batch_size, num_rows}, &pivots),
                         done);
    auto pivots_mat = pivots.template matrix<int32>();

    // Transpose the input. This is necessary because cuBLAS assumes
    // column-major storage while TensorFlow uses row-major.
    OP_REQUIRES_OK_ASYNC(
        context,
        DoMatrixTranspose(device, *packed_triangular_factors,
                          &packed_triangular_factors_transpose),
        done);

    std::vector<DeviceLapackInfo> dev_info;
    if (num_rows == num_cols && num_rows / batch_size <= 128) {
      // For small matrices or large batch sizes, we use the batched
      // interface from cuBlas.
      auto packed_triangular_factors_ptrs = solver->GetScratchSpace<uint8>(
          sizeof(Scalar*) * batch_size, "packed_triangular_factors_ptrs",
          /* on_host */ true);

      Scalar** packed_triangular_factors_ptrs_base = reinterpret_cast<Scalar**>(
          packed_triangular_factors_ptrs.mutable_data());

      for (int batch = 0; batch < batch_size; ++batch) {
        packed_triangular_factors_ptrs_base[batch] =
            &packed_triangular_factors_transpose_reshaped(batch, 0, 0);
      }
      dev_info.push_back(
          solver->GetDeviceLapackInfo(batch_size, "getrfBatched"));
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->GetrfBatched(num_rows, packed_triangular_factors_ptrs_base,
                               num_rows, pivots_mat.data(), &dev_info.back(),
                               batch_size),
          done);
    } else {
      // For small batch sizes we use the non-batched interface from cuSolver,
      // which is much faster for large matrices.
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "getrf"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Getrf(
                num_rows, num_cols,
                &packed_triangular_factors_transpose_reshaped(batch, 0, 0),
                num_rows, &pivots_mat(batch, 0), &dev_info.back()(batch)),
            done);
      }
    }

    // Transpose the result since we had transposed the input.
    OP_REQUIRES_OK_ASYNC(
        context,
        DoMatrixTranspose(device, packed_triangular_factors_transpose,
                          packed_triangular_factors),
        done);

    // Pivots encode the permutation of the rows as a sequences of row swaps.
    // For each index i, row i is swapped with row pivots[i].
    int* pivots_ptr = pivots.flat<int>().data();
    Tidx* permutation_indices_ptr =
        permutation_indices->template flat<Tidx>().data();
    GpuLaunchConfig cfgPivots = GetGpuLaunchConfig(batch_size, device);
    TF_CHECK_OK(GpuLaunchKernel(
        ComputePermutationFromTranspositionsKernel<Tidx>, cfgPivots.block_count,
        cfgPivots.thread_per_block, 0, device.stream(), cfgPivots, num_rows,
        pivots_ptr, permutation_indices_ptr));

    // Callback for checking info after kernels finish. Also capture the
    // temporary Tensors/ScratchSpace so they don't get deallocated before the
    // kernels run.
    // TODO(rmlarsen): Use move capture once C++14 becomes available.
    auto info_checker = [context, done, dev_info](
                            const Status& status,
                            const std::vector<HostLapackInfo>& host_infos) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSlu_op_gpuDTcuDTcc mht_4(mht_4_v, 421, "", "./tensorflow/core/kernels/linalg/lu_op_gpu.cu.cc", "lambda");

      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        for (int i = 0; i < host_infos[0].size(); ++i) {
          // Match the CPU error message for singular matrices. Otherwise
          // just print the original error message from the status below.
          OP_REQUIRES_ASYNC(context, host_infos[0].data()[i] <= 0,
                            errors::InvalidArgument("Input is not invertible."),
                            done);
        }
      }
      OP_REQUIRES_OK_ASYNC(context, status, done);
      done();
    };

    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(info_checker));
  }
};

#define REGISTER_LU_GPU(type, idx_type)                                     \
  REGISTER_KERNEL_BUILDER(Name("Lu")                                        \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<type>("T")                    \
                              .TypeConstraint<idx_type>("output_idx_type"), \
                          LuOpGpu<type, idx_type>);

REGISTER_LU_GPU(float, int32);
REGISTER_LU_GPU(double, int32);
REGISTER_LU_GPU(complex64, int32);
REGISTER_LU_GPU(complex128, int32);

REGISTER_LU_GPU(float, int64);
REGISTER_LU_GPU(double, int64);
REGISTER_LU_GPU(complex64, int64);
REGISTER_LU_GPU(complex128, int64);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
