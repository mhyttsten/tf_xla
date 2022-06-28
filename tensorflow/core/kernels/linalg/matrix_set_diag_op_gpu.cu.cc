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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_set_diag_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_set_diag_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_set_diag_op_gpuDTcuDTcc() {
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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/linalg/matrix_set_diag_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// TODO(penporn): Merge this file with matrix_diag_op_gpu.cu.cc.
__device__ inline int ComputeContentOffset(const int diag_index,
                                           const int max_diag_len,
                                           const int num_rows,
                                           const int num_cols,
                                           const bool left_align_superdiagonal,
                                           const bool left_align_subdiagonal) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_set_diag_op_gpuDTcuDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/linalg/matrix_set_diag_op_gpu.cu.cc", "ComputeContentOffset");

  const bool left_align = (diag_index >= 0 && left_align_superdiagonal) ||
                          (diag_index <= 0 && left_align_subdiagonal);
  if (left_align) return 0;
  const int y_offset = min(0, diag_index);
  const int x_offset = max(0, diag_index);
  const int diag_len = min(num_rows + y_offset, num_cols - x_offset);
  return max_diag_len - diag_len;
}

template <typename Scalar>
__global__ void MatrixSetDiagKernel(
    const int num_threads, const int m, const int n, const int num_diags,
    const int max_diag_len, const int upper_diag_index,
    const bool left_align_superdiagonal, const bool left_align_subdiagonal,
    const Scalar* __restrict__ diag_ptr, Scalar* __restrict__ output_ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_set_diag_op_gpuDTcuDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/linalg/matrix_set_diag_op_gpu.cu.cc", "MatrixSetDiagKernel");

  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int batch_and_diag_index = index / max_diag_len;
    int index_in_the_diagonal = index - batch_and_diag_index * max_diag_len;
    const int batch = batch_and_diag_index / num_diags;
    const int diag_index_in_input = batch_and_diag_index - batch * num_diags;
    const int diag_index = upper_diag_index - diag_index_in_input;
    index_in_the_diagonal -=
        ComputeContentOffset(diag_index, max_diag_len, m, n,
                             left_align_superdiagonal, left_align_subdiagonal);
    const int y_index = index_in_the_diagonal - min(0, diag_index);
    const int x_index = index_in_the_diagonal + max(0, diag_index);

    // Upper-bound checks for diagonals shorter than max_diag_len.
    if (index_in_the_diagonal >= 0 && y_index < m && x_index < n) {
      const int out_index = batch * m * n + y_index * n + x_index;
      output_ptr[out_index] = diag_ptr[index];
    }
  }
}

template <typename Scalar>
__global__ void MatrixCopyInputAndSetDiagKernel(
    const int num_threads, const int m, const int n, const int num_diags,
    const int max_diag_len, const int lower_diag_index,
    const int upper_diag_index, const bool left_align_superdiagonal,
    const bool left_align_subdiagonal, const Scalar* __restrict__ input_ptr,
    const Scalar* __restrict__ diag_ptr, Scalar* __restrict__ output_ptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_set_diag_op_gpuDTcuDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/kernels/linalg/matrix_set_diag_op_gpu.cu.cc", "MatrixCopyInputAndSetDiagKernel");

  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int batch_and_row_index = index / n;
    const int col = index - batch_and_row_index * n;
    const int batch = batch_and_row_index / m;
    const int row = batch_and_row_index - batch * m;
    const int diag_index = col - row;
    const int diag_index_in_input = upper_diag_index - diag_index;
    const int index_in_the_diagonal =
        col - max(0, diag_index) +
        ComputeContentOffset(diag_index, max_diag_len, m, n,
                             left_align_superdiagonal, left_align_subdiagonal);
    if (lower_diag_index <= diag_index && diag_index <= upper_diag_index) {
      output_ptr[index] =
          diag_ptr[batch * num_diags * max_diag_len +
                   diag_index_in_input * max_diag_len + index_in_the_diagonal];
    } else {
      output_ptr[index] = input_ptr[index];
    }
  }
}

template <typename Scalar>
struct MatrixSetDiag<GPUDevice, Scalar> {
  static void Compute(OpKernelContext* context, const GPUDevice& device,
                      typename TTypes<Scalar, 3>::ConstTensor& input,
                      typename TTypes<Scalar>::ConstTensor& diag,
                      typename TTypes<Scalar, 3>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_set_diag_op_gpuDTcuDTcc mht_3(mht_3_v, 287, "", "./tensorflow/core/kernels/linalg/matrix_set_diag_op_gpu.cu.cc", "Compute");

    const int batch_size = input.dimension(0);
    const int m = input.dimension(1);
    const int n = input.dimension(2);
    const int num_diags = upper_diag_index - lower_diag_index + 1;

    if (batch_size == 0 || max_diag_len == 0 || m == 0 || n == 0) return;
    if (input.data() == output.data()) {
      GpuLaunchConfig config =
          GetGpuLaunchConfig(batch_size * num_diags * max_diag_len, device);
      TF_CHECK_OK(GpuLaunchKernel(
          MatrixSetDiagKernel<Scalar>, config.block_count,
          config.thread_per_block, 0, device.stream(),
          config.virtual_thread_count, m, n, num_diags, max_diag_len,
          upper_diag_index, left_align_superdiagonal, left_align_subdiagonal,
          diag.data(), output.data()));
    } else {
      GpuLaunchConfig config = GetGpuLaunchConfig(batch_size * m * n, device);
      TF_CHECK_OK(GpuLaunchKernel(
          MatrixCopyInputAndSetDiagKernel<Scalar>, config.block_count,
          config.thread_per_block, 0, device.stream(),
          config.virtual_thread_count, m, n, num_diags, max_diag_len,
          lower_diag_index, upper_diag_index, left_align_superdiagonal,
          left_align_subdiagonal, input.data(), diag.data(), output.data()));
    }
  }
};

#define DEFINE_GPU_SPEC(T) template struct MatrixSetDiag<GPUDevice, T>;

TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_SPEC);

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
