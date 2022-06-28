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
class MHTracer_DTPStensorflowPScorePSkernelsPSdiag_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdiag_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdiag_opDTcc() {
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

// See docs in ../ops/array_ops.cc

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/diag_op.h"

#include <algorithm>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Generate the diagonal tensor with the diagonal set to the input tensor.
template <typename Device, typename T>
class DiagOp : public OpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdiag_opDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/diag_op.cc", "DiagOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdiag_opDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/diag_op.cc", "Compute");

    const Tensor& diagonal = context->input(0);
    const int num_dims = diagonal.dims();
    OP_REQUIRES(
        context, 0 != num_dims,
        errors::InvalidArgument("Input must be at least rank 1, got 0"));
    TensorShape out_shape;
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));
    functor::DiagFunctor<Device, T> diagFunc;
    Status s =
        diagFunc(context, diagonal.NumElements(), diagonal.flat<T>().data(),
                 output_tensor->flat<T>().data());
    OP_REQUIRES_OK(context, s);
  }
};

// Extract the diagonal tensor with the diagonal set to the input tensor.
template <typename Device, typename T>
class DiagPartOp : public OpKernel {
 public:
  explicit DiagPartOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdiag_opDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/kernels/diag_op.cc", "DiagPartOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdiag_opDTcc mht_3(mht_3_v, 255, "", "./tensorflow/core/kernels/diag_op.cc", "Compute");

    const Tensor& tensor = context->input(0);
    const int num_dims = tensor.dims();
    const int out_dims = num_dims / 2;
    OP_REQUIRES(context, 0 == num_dims % 2,
                errors::InvalidArgument("The rank of the tensor should be \
                                         even and positive, got shape ",
                                        tensor.shape().DebugString()));
    for (int i = 0; i < out_dims; i++) {
      OP_REQUIRES(
          context, tensor.dim_size(i) == tensor.dim_size(i + out_dims),
          errors::InvalidArgument("Invalid shape ",
                                  tensor.shape().DebugString(), ": dimensions ",
                                  i, " and ", i + out_dims, " do not match."));
    }

    TensorShape out_shape;
    for (int i = 0; i < out_dims; ++i) {
      out_shape.AddDim(tensor.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    functor::DiagPartFunctor<Device, T> diagPartFunc;
    Status s = diagPartFunc(context, out_shape.num_elements(),
                            tensor.flat<T>().data(), output->flat<T>().data());
    OP_REQUIRES_OK(context, s);
  }
};

// Implementation of the functor specialization for CPU.
//
// According to the diagonal definition,
// `output[i1,..., ik, i1,..., ik] = input[i1,..., ik]`,
//
// Let the rank of input is [s1,..., sk], then any offset of input's
// pointer can be represent by coordinate [i1,..., ik],
// where `index = i1*(s2*...*sk) + i2*(s3*...*sk) +... + ik`
//
// Let new_index is the offset of output's pointer with coordinate
// [i1,..., ik, i1,..., ik], then we have
// `new_index = i1*(s2*...sk*s1*...*sk) + i2*(s3*...*sk*s1*...*sk) +... + \
//              ik*(s1*...*sk) + i1*(s2*...*sk) + i2*(s3*...*sk) +... + ik
//            = (i1*(s2*...*sk) + i2*(s3*...*sk) +... + ik) * (1 + s1*...*sk)
//            = index * (1 + s1*...*sk)
//
// Let `size = s1*...*sk`, we finally have `new_index = index * (1 + size)`,
// which is the transfer function we use below.
// This trick make our implementations clear and easy to be parallel.
namespace functor {
template <typename T>
struct DiagFunctor<CPUDevice, T> {
  EIGEN_ALWAYS_INLINE Status operator()(OpKernelContext* context,
                                        const int64_t size, const T* in,
                                        T* out) {
    // This subprocess is responsible for writing values in index range
    // [start*size, limit*size)
    auto subDiag = [in, out, size](int64_t start, int64_t limit) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdiag_opDTcc mht_4(mht_4_v, 315, "", "./tensorflow/core/kernels/diag_op.cc", "lambda");

      std::fill(out + size * start, out + size * limit, T());
      for (int64_t index = start; index < limit; ++index) {
        out[(1 + size) * index] = in[index];
      }
    };

    // Here, 5 is a empirical factor of cost_per_unit.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, size, 5 * size,
          subDiag);
    return Status::OK();
  }
};

template <typename T>
struct DiagPartFunctor<CPUDevice, T> {
  EIGEN_ALWAYS_INLINE Status operator()(OpKernelContext* context,
                                        const int64_t size, const T* in,
                                        T* out) {
    // This subprocess is responsible for extracting values in index range
    // [start, limit)
    auto subDiagPart = [in, out, size](int64_t start, int64_t limit) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdiag_opDTcc mht_5(mht_5_v, 340, "", "./tensorflow/core/kernels/diag_op.cc", "lambda");

      for (int64_t index = start; index < limit; ++index) {
        out[index] = in[(1 + size) * index];
      }
    };

    // Here, 5 is a empirical factor of cost_per_unit.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, size, 5,
          subDiagPart);
    return Status::OK();
  }
};
}  // namespace functor

// Register the CPU kernels.
#define REGISTER_DIAGOP(T)                                    \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Diag").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DiagOp<CPUDevice, T>)

TF_CALL_double(REGISTER_DIAGOP);
TF_CALL_float(REGISTER_DIAGOP);
TF_CALL_int32(REGISTER_DIAGOP);
TF_CALL_int64(REGISTER_DIAGOP);
TF_CALL_COMPLEX_TYPES(REGISTER_DIAGOP);
TF_CALL_half(REGISTER_DIAGOP);
#undef REGISTER_DIAGOP

#define REGISTER_DIAGPARTOP(T)                                    \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("DiagPart").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DiagPartOp<CPUDevice, T>)

TF_CALL_double(REGISTER_DIAGPARTOP);
TF_CALL_float(REGISTER_DIAGPARTOP);
TF_CALL_int32(REGISTER_DIAGPARTOP);
TF_CALL_int64(REGISTER_DIAGPARTOP);
TF_CALL_COMPLEX_TYPES(REGISTER_DIAGPARTOP);
TF_CALL_half(REGISTER_DIAGPARTOP);
#undef REGISTER_DIAGPARTOP

// Register the GPU kernels.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
extern template struct DiagFunctor<GPUDevice, double>;
extern template struct DiagFunctor<GPUDevice, float>;
extern template struct DiagFunctor<GPUDevice, int32>;
extern template struct DiagFunctor<GPUDevice, int64_t>;
extern template struct DiagFunctor<GPUDevice, complex64>;
extern template struct DiagFunctor<GPUDevice, complex128>;
}  // namespace functor

#define REGISTER_DIAGOP_GPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Diag").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DiagOp<GPUDevice, T>)

TF_CALL_double(REGISTER_DIAGOP_GPU);
TF_CALL_float(REGISTER_DIAGOP_GPU);
TF_CALL_int32(REGISTER_DIAGOP_GPU);
TF_CALL_int64(REGISTER_DIAGOP_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_DIAGOP_GPU);
TF_CALL_half(REGISTER_DIAGOP_GPU);
#undef REGISTER_DIAGOP_GPU

// Forward declarations of the functor specializations for GPU.
namespace functor {
extern template struct DiagPartFunctor<GPUDevice, double>;
extern template struct DiagPartFunctor<GPUDevice, float>;
extern template struct DiagPartFunctor<GPUDevice, int32>;
extern template struct DiagPartFunctor<GPUDevice, int64_t>;
extern template struct DiagPartFunctor<GPUDevice, complex64>;
extern template struct DiagPartFunctor<GPUDevice, complex128>;
extern template struct DiagPartFunctor<GPUDevice, Eigen::half>;
}  // namespace functor

#define REGISTER_DIAGPARTOP_GPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("DiagPart").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DiagPartOp<GPUDevice, T>)

TF_CALL_double(REGISTER_DIAGPARTOP_GPU);
TF_CALL_float(REGISTER_DIAGPARTOP_GPU);
TF_CALL_int32(REGISTER_DIAGPARTOP_GPU);
TF_CALL_int64(REGISTER_DIAGPARTOP_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_DIAGPARTOP_GPU);
TF_CALL_half(REGISTER_DIAGPARTOP_GPU);
#undef REGISTER_DIAGPARTOP_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
