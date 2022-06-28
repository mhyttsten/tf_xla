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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSself_adjoint_eig_v2_op_gpuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSself_adjoint_eig_v2_op_gpuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSself_adjoint_eig_v2_op_gpuDTcc() {
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

// See docs in ../ops/linalg_ops.cc.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM && TF_ROCM_VERSION >= 40500

#include <numeric>
#include <type_traits>

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class SelfAdjointEigV2OpGpu : public AsyncOpKernel {
 public:
  explicit SelfAdjointEigV2OpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSself_adjoint_eig_v2_op_gpuDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/linalg/self_adjoint_eig_v2_op_gpu.cc", "SelfAdjointEigV2OpGpu");

    OP_REQUIRES_OK(context, context->GetAttr("compute_v", &compute_v_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSself_adjoint_eig_v2_op_gpuDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/linalg/self_adjoint_eig_v2_op_gpu.cc", "ComputeAsync");

    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);
    const int64_t n = input.dim_size(ndims - 1);
    OP_REQUIRES_ASYNC(
        context, input.dim_size(ndims - 2) == n,
        errors::InvalidArgument("Input matrices must be squares, got",
                                input.dim_size(ndims - 2), " != ", n),
        done);
    const int64_t batch_size =
        input.template flat_inner_dims<Scalar, 3>().dimension(0);

    // Allocate outputs.
    Tensor* eigenvalues;
    TensorShape eigenvalues_shape = input.shape();
    eigenvalues_shape.RemoveLastDims(1);
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, eigenvalues_shape, &eigenvalues),
        done);
    Tensor* eigenvectors;
    TensorShape eigenvectors_shape =
        compute_v_ ? input.shape() : TensorShape({});
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(1, eigenvectors_shape, &eigenvectors),
        done);

    if (input.NumElements() == 0) {
      done();
      return;
    }

    // Allocate workspace.
    // TODO(rmlarsen): Convert to std::make_unique when available.
    std::unique_ptr<GpuSolver> solver(new GpuSolver(context));
    Tensor eigenvalues_real;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    if (std::is_same<Scalar, RealScalar>::value) {
      eigenvalues_real = *eigenvalues;
    } else {
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->allocate_scoped_tensor(DataTypeToEnum<RealScalar>::value,
                                         eigenvalues_shape, &eigenvalues_real),
          done);
    }

    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->forward_input_or_allocate_scoped_tensor(
            {0}, DataTypeToEnum<Scalar>::value, input.shape(), &input_copy),
        done);
    // For real symmetric matrices, row-major and column-major are the same. For
    // complex Hermitian, row-major and column-major differ by a conjugation,
    // which is still cheaper than a transpose.
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    if (!input.SharesBufferWith(input_copy)) {
      if (Eigen::NumTraits<Scalar>::IsComplex) {
        functor::UnaryFunctor<GPUDevice, functor::conj<Scalar>> conj;
        conj(device, input_copy.flat<Scalar>() /*out*/,
             input.flat<Scalar>() /*in*/);
      } else {
        device.memcpy(input_copy.flat<Scalar>().data(),
                      input.flat<Scalar>().data(),
                      input.NumElements() * sizeof(Scalar));
      }
    } else if (Eigen::NumTraits<Scalar>::IsComplex) {
      functor::UnaryFunctor<GPUDevice, functor::conj<Scalar>> conj;
      conj(device, const_cast<Tensor*>(&input)->flat<Scalar>() /*out*/,
           input.flat<Scalar>() /*in*/);
    }

#if GOOGLE_CUDA
    cublasFillMode_t fill = CUBLAS_FILL_MODE_UPPER;
    cusolverEigMode_t jobz =
        compute_v_ ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
#elif TENSORFLOW_USE_ROCM
    hipsolverFillMode_t fill = HIPSOLVER_FILL_MODE_UPPER;
    hipsolverEigMode_t jobz =
        compute_v_ ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR;
#endif

    // Compute eigen decomposition in-place in input_copy.
    std::vector<DeviceLapackInfo> dev_info;
    dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "heevd"));
    auto input_copy_reshaped = input_copy.flat_inner_dims<Scalar, 3>();
    auto eigenvalues_real_reshaped =
        eigenvalues_real.flat_inner_dims<RealScalar, 2>();
    for (int batch = 0; batch < batch_size; ++batch) {
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->Heevd(jobz, fill, n, &input_copy_reshaped(batch, 0, 0), n,
                        &eigenvalues_real_reshaped(batch, 0),
                        dev_info.back().mutable_data() + batch),
          done);
    }

    if (!std::is_same<Scalar, RealScalar>::value) {
      functor::CastFunctor<GPUDevice, Scalar, RealScalar> cast;
      cast(device, eigenvalues->flat<Scalar>(),
           const_cast<const Tensor*>(&eigenvalues_real)->flat<RealScalar>());
    }

    if (compute_v_) {
      // Transpose eigenvectors now stored in input_copy in column-major form to
      // output in row-major form.
      OP_REQUIRES_OK_ASYNC(
          context, DoMatrixTranspose(device, input_copy, eigenvectors), done);
    }

    // Asynchronously check return status from cuSolver kernels.
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(done));
  }

 private:
  bool compute_v_;

  TF_DISALLOW_COPY_AND_ASSIGN(SelfAdjointEigV2OpGpu);
};

#define REGISTER(Scalar)                                                       \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SelfAdjointEigV2").Device(DEVICE_GPU).TypeConstraint<Scalar>("T"), \
      (SelfAdjointEigV2OpGpu<Scalar>))

REGISTER(float);
REGISTER(double);
REGISTER(complex64);
REGISTER(complex128);

#undef REGISTER

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
