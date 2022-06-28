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
class MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_band_part_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_band_part_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_band_part_opDTcc() {
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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/linalg/matrix_band_part_op.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class MatrixBandPartOp : public OpKernel {
 public:
  explicit MatrixBandPartOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_band_part_opDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/kernels/linalg/matrix_band_part_op.cc", "MatrixBandPartOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_band_part_opDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/linalg/matrix_band_part_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));
    auto input_reshaped = input.flat_inner_dims<T, 3>();

    const Tensor& num_lower_in = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_lower_in.shape()),
                errors::InvalidArgument("num_lower must be scalar, got shape ",
                                        num_lower_in.shape().DebugString()));

    auto as_int64_scalar = [](const Tensor& tensor) -> int64 {
      if (tensor.dtype() == DT_INT32) {
        return tensor.scalar<int32>()();
      } else {
        return tensor.scalar<int64_t>()();
      }
    };
    const int64_t num_lower = as_int64_scalar(num_lower_in);
    OP_REQUIRES(
        context, num_lower <= input_reshaped.dimension(1),
        errors::InvalidArgument(
            "num_lower must be negative or less or equal to number of rows (",
            input_reshaped.dimension(1), ") got: ", num_lower));

    const Tensor& num_upper_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_upper_in.shape()),
                errors::InvalidArgument("num_upper must be scalar, got shape ",
                                        num_upper_in.shape().DebugString()));
    const int64_t num_upper = as_int64_scalar(num_upper_in);
    OP_REQUIRES(context, num_upper <= input_reshaped.dimension(2),
                errors::InvalidArgument("num_upper must be negative or less or "
                                        "equal to number of columns (",
                                        input_reshaped.dimension(2),
                                        ") got: ", num_upper));

    if (input.NumElements() == 0 ||
        ((num_lower < 0 || num_lower == input_reshaped.dimension(1)) &&
         (num_upper < 0 || num_upper == input_reshaped.dimension(2)))) {
      // This is a no-op.
      context->set_output(0, input);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input_shape, &output));
    auto output_reshaped = output->flat_inner_dims<T, 3>();
    functor::MatrixBandPartFunctor<Device, T> fn;
    fn(context, context->eigen_device<Device>(), num_lower, num_upper,
       input_reshaped, output_reshaped);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixBandPartOp);
};

#define REGISTER_MATRIX_BAND_PART(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixBandPart").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixBandPartOp<CPUDevice, type>);
TF_CALL_POD_TYPES(REGISTER_MATRIX_BAND_PART);
#undef REGISTER_MATRIX_BAND_PART

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_BAND_PART(type)             \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixBandPart")     \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          MatrixBandPartOp<CPUDevice, type>);
TF_CALL_NUMBER_TYPES(REGISTER_BATCH_MATRIX_BAND_PART);
#undef REGISTER_BATCH_MATRIX_BAND_PART

// Implementation of the functor specialization for CPU.
namespace functor {

// CPU implementation of BandPartFunctor.
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Scalar>
struct MatrixBandPartFunctor<CPUDevice, Scalar> {
  void operator()(OpKernelContext* context, const CPUDevice& device,
                  int num_lower_diags, int num_upper_diags,
                  typename TTypes<Scalar, 3>::ConstTensor input,
                  typename TTypes<Scalar, 3>::Tensor output) {
    const int64_t b = input.dimension(0);
    const int64_t m = input.dimension(1);
    const int64_t n = input.dimension(2);
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int64_t total_rows = b * m;
    const int64_t row_cost = 10 * n;
    const bool in_place = input.data() == output.data();
    auto compute_shard = [=, &input, &output](int64_t begin, int64_t end) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlinalgPSmatrix_band_part_opDTcc mht_2(mht_2_v, 325, "", "./tensorflow/core/kernels/linalg/matrix_band_part_op.cc", "lambda");

      if (!in_place) {
        std::fill(output.data() + begin * n, output.data() + end * n,
                  Scalar(0));
      }
      const int64_t batch_begin = begin / m;
      const int64_t batch_end = (end + m - 1) / m;
      for (int64_t batch = batch_begin; batch < batch_end; ++batch) {
        const int64_t row_begin = begin > batch * m ? begin % m : 0;
        const int64_t row_end = end < (batch + 1) * m ? end % m : m;
        for (int64_t row = row_begin; row < row_end; ++row) {
          const int64_t band_start =
              num_lower_diags < 0
                  ? 0
                  : std::min(n, std::max(int64{0}, row - num_lower_diags));
          const int64_t band_end = num_upper_diags < 0
                                       ? n
                                       : std::min(static_cast<int64_t>(n),
                                                  row + num_upper_diags + 1);
          if (in_place) {
            if (band_start > 0) {
              std::fill(&output(batch, row, 0), &output(batch, row, band_start),
                        Scalar(0));
            }
            if (band_end < n) {
              std::fill(&output(batch, row, band_end), &output(batch, row, n),
                        Scalar(0));
            }
          } else {
            if (band_start < band_end) {
              const Eigen::DSizes<Eigen::DenseIndex, 3> indices(batch, row,
                                                                band_start);
              const Eigen::DSizes<Eigen::DenseIndex, 3> sizes(
                  1, 1, band_end - band_start);
              output.slice(indices, sizes) = input.slice(indices, sizes);
            }
          }
        }
      }
    };
    thread_pool->ParallelFor(total_rows, row_cost, std::move(compute_shard));
  }
};

#define DEFINE_CPU_SPEC(T) template struct MatrixBandPartFunctor<CPUDevice, T>;
TF_CALL_POD_TYPES(DEFINE_CPU_SPEC);
#undef DEFINE_CPU_SPEC

}  // namespace functor

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                            \
  template <>                                                          \
  struct MatrixBandPartFunctor<GPUDevice, T> {                         \
    void operator()(OpKernelContext* context, const GPUDevice& device, \
                    int num_upper_diags, int num_lower_diags,          \
                    typename TTypes<T, 3>::ConstTensor input,          \
                    typename TTypes<T, 3>::Tensor output);             \
  };                                                                   \
  extern template struct MatrixBandPartFunctor<GPUDevice, T>;

TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_MATRIX_BAND_PART_GPU(type)              \
  REGISTER_KERNEL_BUILDER(Name("MatrixBandPart")         \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("num_lower")   \
                              .HostMemory("num_upper"),  \
                          MatrixBandPartOp<GPUDevice, type>);
TF_CALL_GPU_ALL_TYPES(REGISTER_MATRIX_BAND_PART_GPU);
#undef REGISTER_MATRIX_BAND_PART_GPU

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_BAND_PART_GPU(type)        \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixBandPart")    \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("num_lower")   \
                              .HostMemory("num_upper"),  \
                          MatrixBandPartOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_BATCH_MATRIX_BAND_PART_GPU);
#undef REGISTER_BATCH_MATRIX_BAND_PART_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
