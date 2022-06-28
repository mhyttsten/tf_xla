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
class MHTracer_DTPStensorflowPScorePSkernelsPSreverse_sequence_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_sequence_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSreverse_sequence_opDTcc() {
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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/reverse_sequence_op.h"

#include <memory>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Tlen>
void CheckErrors(OpKernelContext* context, int batch_dim, int seq_dim) {
  const Tensor& input = context->input(0);
  const Tensor& seq_lengths = context->input(1);

  auto seq_lens_t = seq_lengths.vec<Tlen>();

  std::vector<Tlen> seq_lens_vec(seq_lens_t.size());

  // Copy seq_len info down for validity checks
  context->eigen_device<Device>().memcpyDeviceToHost(
      seq_lens_vec.data(), seq_lens_t.data(), sizeof(Tlen) * seq_lens_t.size());

  OP_REQUIRES(context, batch_dim != seq_dim,
              errors::InvalidArgument("batch_dim == seq_dim == ", seq_dim));
  OP_REQUIRES(context, seq_dim < input.dims(),
              errors::InvalidArgument("seq_dim must be < input rank", " ( ",
                                      seq_dim, " vs. ", input.dims(), ")"));
  OP_REQUIRES(context, batch_dim < input.dims(),
              errors::InvalidArgument("batch_dim must be < input rank", " ( ",
                                      batch_dim, " vs. ", input.dims(), ")"));
  OP_REQUIRES(
      context, seq_lengths.NumElements() == input.dim_size(batch_dim),
      errors::InvalidArgument("Length of seq_lengths != input.dims(", batch_dim,
                              "), ", "(", seq_lengths.NumElements(), " vs. ",
                              input.dim_size(batch_dim), ")"));

  for (size_t d = 0; d < seq_lens_vec.size(); ++d) {
    OP_REQUIRES(context, seq_lens_vec[d] >= 0,
                errors::InvalidArgument("seq_lens(", d, ") < 0"));
    OP_REQUIRES(context, seq_lens_vec[d] <= input.dim_size(seq_dim),
                errors::InvalidArgument("seq_lens(", d, ") > input.dims(",
                                        seq_dim, ")"));
  }
}

void CheckErrorsGPU(OpKernelContext* context, int batch_dim, int seq_dim) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_sequence_opDTcc mht_0(mht_0_v, 248, "", "./tensorflow/core/kernels/reverse_sequence_op.cc", "CheckErrorsGPU");

  const Tensor& input = context->input(0);
  const Tensor& seq_lengths = context->input(1);

  OP_REQUIRES(context, batch_dim != seq_dim,
              errors::InvalidArgument("batch_dim == seq_dim == ", seq_dim));
  OP_REQUIRES(context, seq_dim < input.dims(),
              errors::InvalidArgument("seq_dim must be < input rank", " ( ",
                                      seq_dim, " vs. ", input.dims(), ")"));
  OP_REQUIRES(context, batch_dim < input.dims(),
              errors::InvalidArgument("batch_dim must be < input rank", " ( ",
                                      batch_dim, " vs. ", input.dims(), ")"));

  OP_REQUIRES(
      context, seq_lengths.NumElements() == input.dim_size(batch_dim),
      errors::InvalidArgument("Length of seq_lengths != input.dims(", batch_dim,
                              "), ", "(", seq_lengths.NumElements(), " vs. ",
                              input.dim_size(batch_dim), ")"));
}

template <>
void CheckErrors<GPUDevice, int32>(OpKernelContext* context, int batch_dim,
                                   int seq_dim) {
  CheckErrorsGPU(context, batch_dim, seq_dim);
}

template <>
void CheckErrors<GPUDevice, int64_t>(OpKernelContext* context, int batch_dim,
                                     int seq_dim) {
  CheckErrorsGPU(context, batch_dim, seq_dim);
}

template <typename Device, typename T, typename Tlen>
class ReverseSequenceOp : public OpKernel {
 public:
  explicit ReverseSequenceOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_sequence_opDTcc mht_1(mht_1_v, 287, "", "./tensorflow/core/kernels/reverse_sequence_op.cc", "ReverseSequenceOp");

    OP_REQUIRES_OK(context, context->GetAttr("batch_dim", &batch_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("seq_dim", &seq_dim_));
    OP_REQUIRES(context, batch_dim_ >= 0,
                errors::InvalidArgument("Invalid batch_dim ", batch_dim_));
    OP_REQUIRES(context, seq_dim_ >= 0,
                errors::InvalidArgument("Invalid seq_dim ", seq_dim_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreverse_sequence_opDTcc mht_2(mht_2_v, 299, "", "./tensorflow/core/kernels/reverse_sequence_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& seq_lengths = context->input(1);

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsVector(seq_lengths.shape()),
                errors::InvalidArgument("seq_lengths must be 1-dim, not ",
                                        seq_lengths.dims()));

    auto seq_lens_t = seq_lengths.vec<Tlen>();

    CheckErrors<Device, Tlen>(context, batch_dim_, seq_dim_);
    if (!context->status().ok()) return;

    const int input_dims = input.dims();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

#define HANDLE_DIM(NDIM)                                                      \
  case NDIM:                                                                  \
    functor::ReverseSequence<Device, T, Tlen, NDIM>::Compute(                 \
        context->eigen_device<Device>(), input.tensor<T, NDIM>(), batch_dim_, \
        seq_dim_, seq_lens_t, output->tensor<T, NDIM>());                     \
    break;

    switch (input_dims) {
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "ReverseSequenceOp : Unhandled input dimensions: ",
                        input_dims));
    }
  }

 private:
  int32 batch_dim_;
  int32 seq_dim_;

  TF_DISALLOW_COPY_AND_ASSIGN(ReverseSequenceOp);
};

#define REGISTER_REVERSE_SEQUENCE(type, len_type)                \
  REGISTER_KERNEL_BUILDER(Name("ReverseSequence")                \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<len_type>("Tlen"), \
                          ReverseSequenceOp<CPUDevice, type, len_type>);

#define REGISTER_REVERSE_SEQUENCE_LEN(type) \
  REGISTER_REVERSE_SEQUENCE(type, int32);   \
  REGISTER_REVERSE_SEQUENCE(type, int64_t);

TF_CALL_NUMBER_TYPES(REGISTER_REVERSE_SEQUENCE_LEN);
TF_CALL_bool(REGISTER_REVERSE_SEQUENCE_LEN);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tlen, Dims)                                \
  template <>                                                          \
  void ReverseSequence<GPUDevice, T, Tlen, Dims>::Compute(             \
      const GPUDevice& d, typename TTypes<T, Dims>::ConstTensor input, \
      int32 batch_dim, int32 seq_dim,                                  \
      typename TTypes<Tlen>::ConstVec seq_lengths,                     \
      typename TTypes<T, Dims>::Tensor output);                        \
  extern template struct ReverseSequence<GPUDevice, T, Tlen, Dims>;

#define DECLARE_GPU_SPEC_LEN(T, Dims) \
  DECLARE_GPU_SPEC(T, int32, Dims);   \
  DECLARE_GPU_SPEC(T, int64_t, Dims);

#define DECLARE_GPU_SPECS(T)  \
  DECLARE_GPU_SPEC_LEN(T, 2); \
  DECLARE_GPU_SPEC_LEN(T, 3); \
  DECLARE_GPU_SPEC_LEN(T, 4); \
  DECLARE_GPU_SPEC_LEN(T, 5);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_bool(DECLARE_GPU_SPECS);

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_REVERSE_SEQUENCE_GPU(type, len_type)            \
  REGISTER_KERNEL_BUILDER(Name("ReverseSequence")                \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<len_type>("Tlen"), \
                          ReverseSequenceOp<GPUDevice, type, len_type>);

#define REGISTER_REVERSE_SEQUENCE_GPU_LEN(type) \
  REGISTER_REVERSE_SEQUENCE_GPU(type, int32);   \
  REGISTER_REVERSE_SEQUENCE_GPU(type, int64_t);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_REVERSE_SEQUENCE_GPU_LEN);
TF_CALL_bool(REGISTER_REVERSE_SEQUENCE_GPU_LEN);

#undef REGISTER_REVERSE_SEQUENCE_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
