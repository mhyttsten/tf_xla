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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_op_gpuDTcuDTcc() {
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
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

template <typename Scalar>
__global__ void TridiagonalMatMulKernel(int batch_size, int m, int n,
                                        const Scalar* __restrict__ superdiag,
                                        const Scalar* __restrict__ maindiag,
                                        const Scalar* __restrict__ subdiag,
                                        const Scalar* __restrict__ rhs,
                                        Scalar* __restrict__ product) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_op_gpuDTcuDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op_gpu.cu.cc", "TridiagonalMatMulKernel");

  for (int i : CudaGridRangeX(batch_size * m * n)) {
    int row_id = i / n;
    Scalar result = maindiag[row_id] * rhs[i];
    if (row_id % m != 0) {
      result = result + subdiag[row_id] * rhs[i - n];
    }
    if ((row_id + 1) % m != 0) {
      result = result + superdiag[row_id] * rhs[i + n];
    }
    product[i] = result;
  }
}

template <typename Scalar>
class TridiagonalMatMulOpGpu : public OpKernel {
 public:
  explicit TridiagonalMatMulOpGpu(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_op_gpuDTcuDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op_gpu.cu.cc", "TridiagonalMatMulOpGpu");
}

  void Compute(OpKernelContext* context) final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_op_gpuDTcuDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op_gpu.cu.cc", "Compute");

    const Tensor& superdiag = context->input(0);
    const Tensor& maindiag = context->input(1);
    const Tensor& subdiag = context->input(2);
    const Tensor& rhs = context->input(3);

    const int ndims = rhs.dims();
    OP_REQUIRES(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, but got ", ndims));
    OP_REQUIRES_OK(context, ValidateInputTensor(superdiag, "superdiag", rhs));
    OP_REQUIRES_OK(context, ValidateInputTensor(maindiag, "maindiag", rhs));
    OP_REQUIRES_OK(context, ValidateInputTensor(subdiag, "subdiag", rhs));
    int64 batch_size = 1;
    for (int i = 0; i < ndims - 2; i++) {
      batch_size *= rhs.dim_size(i);
    }
    const int m = rhs.dim_size(ndims - 2);
    const int n = rhs.dim_size(ndims - 1);

    // Allocate output.
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs.shape(), &output));

    const Eigen::GpuDevice& device = context->eigen_device<Eigen::GpuDevice>();
    GpuLaunchConfig cfg = GetGpuLaunchConfig(1, device);
    TF_CHECK_OK(GpuLaunchKernel(
        TridiagonalMatMulKernel<Scalar>, cfg.block_count, cfg.thread_per_block,
        0, device.stream(), batch_size, m, n, superdiag.flat<Scalar>().data(),
        maindiag.flat<Scalar>().data(), subdiag.flat<Scalar>().data(),
        rhs.flat<Scalar>().data(), output->flat<Scalar>().data()));
  }

 private:
  Status ValidateInputTensor(const Tensor& tensor,
                             const std::string& tensor_name,
                             const Tensor& rhs) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("tensor_name: \"" + tensor_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPStridiagonal_matmul_op_gpuDTcuDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/kernels/linalg/tridiagonal_matmul_op_gpu.cu.cc", "ValidateInputTensor");

    const int ndims = rhs.dims();
    if (tensor.dims() != ndims) {
      return errors::InvalidArgument(tensor_name,
                                     " must have same rank as rhs, but got ",
                                     tensor.dims(), " and ", ndims);
    }
    for (int i = 0; i < ndims - 2; i++) {
      if (tensor.dim_size(i) != rhs.dim_size(i)) {
        return errors::InvalidArgument(
            tensor_name,
            " must have same outer dimensions as rhs, but for index ", i,
            ", got ", tensor.dim_size(i), " and ", rhs.dim_size(i));
      }
    }
    if (tensor.dim_size(ndims - 2) != 1) {
      return errors::InvalidArgument(
          tensor_name, "'s second-to-last dimension must be 1, but got ",
          tensor.dim_size(ndims - 2));
    }
    if (tensor.dim_size(ndims - 1) != rhs.dim_size(ndims - 2)) {
      return errors::InvalidArgument(tensor_name,
                                     "'s last dimension size must be rhs's "
                                     "second-to-last dimension size, but got ",
                                     tensor.dim_size(ndims - 1), " and ",
                                     rhs.dim_size(ndims - 2));
    }
    return Status::OK();
  }
};

REGISTER_LINALG_OP_GPU("TridiagonalMatMul", (TridiagonalMatMulOpGpu<float>),
                       float);
REGISTER_LINALG_OP_GPU("TridiagonalMatMul", (TridiagonalMatMulOpGpu<double>),
                       double);
REGISTER_LINALG_OP_GPU("TridiagonalMatMul", (TridiagonalMatMulOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("TridiagonalMatMul",
                       (TridiagonalMatMulOpGpu<complex128>), complex128);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
