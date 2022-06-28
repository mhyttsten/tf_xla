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
class MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc() {
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

#include "tensorflow/core/platform/errors.h"
#define EIGEN_USE_THREADS

// See docs in ../ops/fft_ops.cc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/work_sharder.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

class FFTBase : public OpKernel {
 public:
  explicit FFTBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/fft_ops.cc", "FFTBase");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/fft_ops.cc", "Compute");

    const Tensor& in = ctx->input(0);
    const TensorShape& input_shape = in.shape();
    const int fft_rank = Rank();
    OP_REQUIRES(
        ctx, input_shape.dims() >= fft_rank,
        errors::InvalidArgument("Input must have rank of at least ", fft_rank,
                                " but got: ", input_shape.DebugString()));

    Tensor* out;
    TensorShape output_shape = input_shape;
    uint64 fft_shape[3] = {0, 0, 0};

    // In R2C or C2R mode, we use a second input to specify the FFT length
    // instead of inferring it from the input shape.
    if (IsReal()) {
      const Tensor& fft_length = ctx->input(1);
      OP_REQUIRES(ctx,
                  fft_length.shape().dims() == 1 &&
                      fft_length.shape().dim_size(0) == fft_rank,
                  errors::InvalidArgument("fft_length must have shape [",
                                          fft_rank, "]"));

      auto fft_length_as_vec = fft_length.vec<int32>();
      for (int i = 0; i < fft_rank; ++i) {
        OP_REQUIRES(ctx, fft_length_as_vec(i) >= 0,
                    errors::InvalidArgument(
                        "fft_length[", i,
                        "] must >= 0, but got: ", fft_length_as_vec(i)));
        fft_shape[i] = fft_length_as_vec(i);
        // Each input dimension must have length of at least fft_shape[i]. For
        // IRFFTs, the inner-most input dimension must have length of at least
        // fft_shape[i] / 2 + 1.
        bool inner_most = (i == fft_rank - 1);
        uint64 min_input_dim_length =
            !IsForward() && inner_most ? fft_shape[i] / 2 + 1 : fft_shape[i];
        auto input_index = input_shape.dims() - fft_rank + i;
        OP_REQUIRES(
            ctx,
            // We pass through empty tensors, so special case them here.
            input_shape.dim_size(input_index) == 0 ||
                input_shape.dim_size(input_index) >= min_input_dim_length,
            errors::InvalidArgument(
                "Input dimension ", input_index,
                " must have length of at least ", min_input_dim_length,
                " but got: ", input_shape.dim_size(input_index)));
        uint64 dim = IsForward() && inner_most && fft_shape[i] != 0
                         ? fft_shape[i] / 2 + 1
                         : fft_shape[i];
        output_shape.set_dim(output_shape.dims() - fft_rank + i, dim);
      }
    } else {
      for (int i = 0; i < fft_rank; ++i) {
        fft_shape[i] =
            output_shape.dim_size(output_shape.dims() - fft_rank + i);
      }
    }

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));

    if (IsReal()) {
      if (IsForward()) {
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_FLOAT && out->dtype() == DT_COMPLEX64) ||
                (in.dtype() == DT_DOUBLE && out->dtype() == DT_COMPLEX128),
            errors::InvalidArgument("Wrong types for forward real FFT: in=",
                                    in.dtype(), " out=", out->dtype()));
      } else {
        OP_REQUIRES(
            ctx,
            (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_FLOAT) ||
                (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_DOUBLE),
            errors::InvalidArgument("Wrong types for backward real FFT: in=",
                                    in.dtype(), " out=", out->dtype()));
      }
    } else {
      OP_REQUIRES(
          ctx,
          (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_COMPLEX64) ||
              (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_COMPLEX128),
          errors::InvalidArgument("Wrong types for FFT: in=", in.dtype(),
                                  " out=", out->dtype()));
    }

    if (input_shape.num_elements() == 0) {
      DCHECK_EQ(0, output_shape.num_elements());
      return;
    }

    DoFFT(ctx, in, fft_shape, out);
  }

 protected:
  virtual int Rank() const = 0;
  virtual bool IsForward() const = 0;
  virtual bool IsReal() const = 0;

  // The function that actually computes the FFT.
  virtual void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
                     Tensor* out) = 0;
};

typedef Eigen::ThreadPoolDevice CPUDevice;

template <bool Forward, bool _Real, int FFTRank>
class FFTCPU : public FFTBase {
 public:
  using FFTBase::FFTBase;

 protected:
  int Rank() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_2(mht_2_v, 330, "", "./tensorflow/core/kernels/fft_ops.cc", "Rank");
 return FFTRank; }
  bool IsForward() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_3(mht_3_v, 334, "", "./tensorflow/core/kernels/fft_ops.cc", "IsForward");
 return Forward; }
  bool IsReal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_4(mht_4_v, 338, "", "./tensorflow/core/kernels/fft_ops.cc", "IsReal");
 return _Real; }

  void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
             Tensor* out) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_5(mht_5_v, 344, "", "./tensorflow/core/kernels/fft_ops.cc", "DoFFT");

    // Create the axes (which are always trailing).
    const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);
    auto device = ctx->eigen_device<CPUDevice>();

    const bool is_complex128 =
        in.dtype() == DT_COMPLEX128 || out->dtype() == DT_COMPLEX128;

    if (!IsReal()) {
      // Compute the FFT using Eigen.
      constexpr auto direction =
          Forward ? Eigen::FFT_FORWARD : Eigen::FFT_REVERSE;
      if (is_complex128) {
        DCHECK_EQ(in.dtype(), DT_COMPLEX128);
        DCHECK_EQ(out->dtype(), DT_COMPLEX128);
        auto input = Tensor(in).flat_inner_dims<complex128, FFTRank + 1>();
        auto output = out->flat_inner_dims<complex128, FFTRank + 1>();
        output.device(device) =
            input.template fft<Eigen::BothParts, direction>(axes);
      } else {
        DCHECK_EQ(in.dtype(), DT_COMPLEX64);
        DCHECK_EQ(out->dtype(), DT_COMPLEX64);
        auto input = Tensor(in).flat_inner_dims<complex64, FFTRank + 1>();
        auto output = out->flat_inner_dims<complex64, FFTRank + 1>();
        output.device(device) =
            input.template fft<Eigen::BothParts, direction>(axes);
      }
    } else {
      if (IsForward()) {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_DOUBLE);
          DCHECK_EQ(out->dtype(), DT_COMPLEX128);
          DoRealForwardFFT<double, complex128>(ctx, fft_shape, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_FLOAT);
          DCHECK_EQ(out->dtype(), DT_COMPLEX64);
          DoRealForwardFFT<float, complex64>(ctx, fft_shape, in, out);
        }
      } else {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_COMPLEX128);
          DCHECK_EQ(out->dtype(), DT_DOUBLE);
          DoRealBackwardFFT<complex128, double>(ctx, fft_shape, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_COMPLEX64);
          DCHECK_EQ(out->dtype(), DT_FLOAT);
          DoRealBackwardFFT<complex64, float>(ctx, fft_shape, in, out);
        }
      }
    }
  }

  template <typename RealT, typename ComplexT>
  void DoRealForwardFFT(OpKernelContext* ctx, uint64* fft_shape,
                        const Tensor& in, Tensor* out) {
    // Create the axes (which are always trailing).
    const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);
    auto device = ctx->eigen_device<CPUDevice>();
    auto input = Tensor(in).flat_inner_dims<RealT, FFTRank + 1>();
    const auto input_dims = input.dimensions();

    // Slice input to fft_shape on its inner-most dimensions.
    Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> input_slice_sizes;
    input_slice_sizes[0] = input_dims[0];
    TensorShape temp_shape{input_dims[0]};
    for (int i = 1; i <= FFTRank; ++i) {
      input_slice_sizes[i] = fft_shape[i - 1];
      temp_shape.AddDim(fft_shape[i - 1]);
    }
    OP_REQUIRES(ctx, temp_shape.num_elements() > 0,
                errors::InvalidArgument("Obtained a FFT shape of 0 elements: ",
                                        temp_shape.DebugString()));

    auto output = out->flat_inner_dims<ComplexT, FFTRank + 1>();
    const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> zero_start_indices;

    // Compute the full FFT using a temporary tensor.
    Tensor temp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<ComplexT>::v(),
                                           temp_shape, &temp));
    auto full_fft = temp.flat_inner_dims<ComplexT, FFTRank + 1>();
    full_fft.device(device) =
        input.slice(zero_start_indices, input_slice_sizes)
            .template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);

    // Slice away the negative frequency components.
    output.device(device) =
        full_fft.slice(zero_start_indices, output.dimensions());
  }

  template <typename ComplexT, typename RealT>
  void DoRealBackwardFFT(OpKernelContext* ctx, uint64* fft_shape,
                         const Tensor& in, Tensor* out) {
    auto device = ctx->eigen_device<CPUDevice>();
    // Reconstruct the full FFT and take the inverse.
    auto input = Tensor(in).flat_inner_dims<ComplexT, FFTRank + 1>();
    auto output = out->flat_inner_dims<RealT, FFTRank + 1>();
    const auto input_dims = input.dimensions();

    // Calculate the shape of the temporary tensor for the full FFT and the
    // region we will slice from input given fft_shape. We slice input to
    // fft_shape on its inner-most dimensions, except the last (which we
    // slice to fft_shape[-1] / 2 + 1).
    Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> input_slice_sizes;
    input_slice_sizes[0] = input_dims[0];
    TensorShape full_fft_shape;
    full_fft_shape.AddDim(input_dims[0]);
    for (auto i = 1; i <= FFTRank; i++) {
      input_slice_sizes[i] =
          i == FFTRank ? fft_shape[i - 1] / 2 + 1 : fft_shape[i - 1];
      full_fft_shape.AddDim(fft_shape[i - 1]);
    }
    OP_REQUIRES(ctx, full_fft_shape.num_elements() > 0,
                errors::InvalidArgument("Obtained a FFT shape of 0 elements: ",
                                        full_fft_shape.DebugString()));

    Tensor temp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<ComplexT>::v(),
                                           full_fft_shape, &temp));
    auto full_fft = temp.flat_inner_dims<ComplexT, FFTRank + 1>();

    // Calculate the starting point and range of the source of
    // negative frequency part.
    auto neg_sizes = input_slice_sizes;
    neg_sizes[FFTRank] = fft_shape[FFTRank - 1] - input_slice_sizes[FFTRank];
    Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_target_indices;
    neg_target_indices[FFTRank] = input_slice_sizes[FFTRank];

    const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> start_indices;
    Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_start_indices;
    neg_start_indices[FFTRank] = 1;

    full_fft.slice(start_indices, input_slice_sizes).device(device) =
        input.slice(start_indices, input_slice_sizes);

    // First, conduct IFFTs on outer dimensions. We save computation (and
    // avoid touching uninitialized memory) by slicing full_fft to the
    // subregion we wrote input to.
    if (FFTRank > 1) {
      const auto outer_axes =
          Eigen::ArrayXi::LinSpaced(FFTRank - 1, 1, FFTRank - 1);
      full_fft.slice(start_indices, input_slice_sizes).device(device) =
          full_fft.slice(start_indices, input_slice_sizes)
              .template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(outer_axes);
    }

    // Reconstruct the full FFT by appending reversed and conjugated
    // spectrum as the negative frequency part.
    Eigen::array<bool, FFTRank + 1> reverse_last_axis;
    for (auto i = 0; i <= FFTRank; i++) {
      reverse_last_axis[i] = i == FFTRank;
    }

    if (neg_sizes[FFTRank] != 0) {
      full_fft.slice(neg_target_indices, neg_sizes).device(device) =
          full_fft.slice(neg_start_indices, neg_sizes)
              .reverse(reverse_last_axis)
              .conjugate();
    }

    auto inner_axis = Eigen::array<int, 1>{FFTRank};
    output.device(device) =
        full_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(inner_axis);
  }
};

REGISTER_KERNEL_BUILDER(Name("FFT").Device(DEVICE_CPU), FFTCPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("IFFT").Device(DEVICE_CPU),
                        FFTCPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("FFT2D").Device(DEVICE_CPU),
                        FFTCPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("IFFT2D").Device(DEVICE_CPU),
                        FFTCPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("FFT3D").Device(DEVICE_CPU),
                        FFTCPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("IFFT3D").Device(DEVICE_CPU),
                        FFTCPU<false, false, 3>);

REGISTER_KERNEL_BUILDER(Name("RFFT").Device(DEVICE_CPU), FFTCPU<true, true, 1>);
REGISTER_KERNEL_BUILDER(Name("IRFFT").Device(DEVICE_CPU),
                        FFTCPU<false, true, 1>);
REGISTER_KERNEL_BUILDER(Name("RFFT2D").Device(DEVICE_CPU),
                        FFTCPU<true, true, 2>);
REGISTER_KERNEL_BUILDER(Name("IRFFT2D").Device(DEVICE_CPU),
                        FFTCPU<false, true, 2>);
REGISTER_KERNEL_BUILDER(Name("RFFT3D").Device(DEVICE_CPU),
                        FFTCPU<true, true, 3>);
REGISTER_KERNEL_BUILDER(Name("IRFFT3D").Device(DEVICE_CPU),
                        FFTCPU<false, true, 3>);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

namespace {
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory, uint64 size) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory), size * sizeof(T));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

// A class to provide scratch-space allocator for Stream-Executor Cufft
// callback. Tensorflow is responsible for releasing the temporary buffers after
// the kernel finishes.
// TODO(yangzihao): Refactor redundant code in subclasses of ScratchAllocator
// into base class.
class CufftScratchAllocator : public se::ScratchAllocator {
 public:
  ~CufftScratchAllocator() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_6(mht_6_v, 562, "", "./tensorflow/core/kernels/fft_ops.cc", "~CufftScratchAllocator");
}
  CufftScratchAllocator(int64_t memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_7(mht_7_v, 567, "", "./tensorflow/core/kernels/fft_ops.cc", "CufftScratchAllocator");
}
  int64_t GetMemoryLimitInBytes() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_8(mht_8_v, 571, "", "./tensorflow/core/kernels/fft_ops.cc", "GetMemoryLimitInBytes");
 return memory_limit_; }
  se::port::StatusOr<se::DeviceMemory<uint8>> AllocateBytes(
      int64_t byte_size) override {
    Tensor temporary_memory;
    if (byte_size > memory_limit_) {
      return se::port::StatusOr<se::DeviceMemory<uint8>>();
    }
    AllocationAttributes allocation_attr;
    allocation_attr.retry_on_failure = false;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory,
        AllocatorAttributes(), allocation_attr));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<se::DeviceMemory<uint8>>();
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return se::port::StatusOr<se::DeviceMemory<uint8>>(
        AsDeviceMemory(temporary_memory.flat<uint8>().data(),
                       temporary_memory.flat<uint8>().size()));
  }
  int64_t TotalByteSize() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_9(mht_9_v, 597, "", "./tensorflow/core/kernels/fft_ops.cc", "TotalByteSize");
 return total_byte_size_; }

 private:
  int64_t memory_limit_;
  int64_t total_byte_size_;
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};

}  // end namespace

int64_t GetCufftWorkspaceLimit(const string& envvar_in_mb,
                               int64_t default_value_in_bytes) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("envvar_in_mb: \"" + envvar_in_mb + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_10(mht_10_v, 613, "", "./tensorflow/core/kernels/fft_ops.cc", "GetCufftWorkspaceLimit");

  const char* workspace_limit_in_mb_str = getenv(envvar_in_mb.c_str());
  if (workspace_limit_in_mb_str != nullptr &&
      strcmp(workspace_limit_in_mb_str, "") != 0) {
    int64_t scratch_limit_in_mb = -1;
    Status status = ReadInt64FromEnvVar(envvar_in_mb, default_value_in_bytes,
                                        &scratch_limit_in_mb);
    if (!status.ok()) {
      LOG(WARNING) << "Invalid value for env-var " << envvar_in_mb << ": "
                   << workspace_limit_in_mb_str;
    } else {
      return scratch_limit_in_mb * (1 << 20);
    }
  }
  return default_value_in_bytes;
}

class FFTGPUBase : public FFTBase {
 public:
  using FFTBase::FFTBase;

 protected:
  static int64_t CufftScratchSize;
  void DoFFT(OpKernelContext* ctx, const Tensor& in, uint64* fft_shape,
             Tensor* out) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_11(mht_11_v, 640, "", "./tensorflow/core/kernels/fft_ops.cc", "DoFFT");

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    const TensorShape& input_shape = in.shape();
    const TensorShape& output_shape = out->shape();

    const int fft_rank = Rank();
    int batch_size = 1;
    for (int i = 0; i < input_shape.dims() - fft_rank; ++i) {
      batch_size *= input_shape.dim_size(i);
    }
    uint64 input_embed[3];
    const uint64 input_stride = 1;
    uint64 input_distance = 1;
    uint64 output_embed[3];
    const uint64 output_stride = 1;
    uint64 output_distance = 1;

    for (int i = 0; i < fft_rank; ++i) {
      auto dim_offset = input_shape.dims() - fft_rank + i;
      input_embed[i] = input_shape.dim_size(dim_offset);
      input_distance *= input_shape.dim_size(dim_offset);
      output_embed[i] = output_shape.dim_size(dim_offset);
      output_distance *= output_shape.dim_size(dim_offset);
    }

    constexpr bool kInPlaceFft = false;
    const bool is_complex128 =
        in.dtype() == DT_COMPLEX128 || out->dtype() == DT_COMPLEX128;

    const auto kFftType =
        IsReal()
            ? (IsForward()
                   ? (is_complex128 ? se::fft::Type::kD2Z : se::fft::Type::kR2C)
                   : (is_complex128 ? se::fft::Type::kZ2D
                                    : se::fft::Type::kC2R))
            : (IsForward() ? (is_complex128 ? se::fft::Type::kZ2ZForward
                                            : se::fft::Type::kC2CForward)
                           : (is_complex128 ? se::fft::Type::kZ2ZInverse
                                            : se::fft::Type::kC2CInverse));

    CufftScratchAllocator scratch_allocator(CufftScratchSize, ctx);
    auto plan =
        stream->parent()->AsFft()->CreateBatchedPlanWithScratchAllocator(
            stream, fft_rank, fft_shape, input_embed, input_stride,
            input_distance, output_embed, output_stride, output_distance,
            kFftType, kInPlaceFft, batch_size, &scratch_allocator);

    if (IsReal()) {
      if (IsForward()) {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_DOUBLE);
          DCHECK_EQ(out->dtype(), DT_COMPLEX128);
          DoFFTInternal<double, complex128>(ctx, stream, plan.get(), kFftType,
                                            output_distance, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_FLOAT);
          DCHECK_EQ(out->dtype(), DT_COMPLEX64);
          DoFFTInternal<float, complex64>(ctx, stream, plan.get(), kFftType,
                                          output_distance, in, out);
        }
      } else {
        if (is_complex128) {
          DCHECK_EQ(in.dtype(), DT_COMPLEX128);
          DCHECK_EQ(out->dtype(), DT_DOUBLE);
          DoFFTInternal<complex128, double>(ctx, stream, plan.get(), kFftType,
                                            output_distance, in, out);
        } else {
          DCHECK_EQ(in.dtype(), DT_COMPLEX64);
          DCHECK_EQ(out->dtype(), DT_FLOAT);
          DoFFTInternal<complex64, float>(ctx, stream, plan.get(), kFftType,
                                          output_distance, in, out);
        }
      }
    } else {
      if (is_complex128) {
        DCHECK_EQ(in.dtype(), DT_COMPLEX128);
        DCHECK_EQ(out->dtype(), DT_COMPLEX128);
        DoFFTInternal<complex128, complex128>(ctx, stream, plan.get(), kFftType,
                                              output_distance, in, out);
      } else {
        DCHECK_EQ(in.dtype(), DT_COMPLEX64);
        DCHECK_EQ(out->dtype(), DT_COMPLEX64);
        DoFFTInternal<complex64, complex64>(ctx, stream, plan.get(), kFftType,
                                            output_distance, in, out);
      }
    }
  }

 private:
  template <typename T>
  struct RealTypeFromComplexType {
    typedef T RealT;
  };

  template <typename T>
  struct RealTypeFromComplexType<std::complex<T>> {
    typedef T RealT;
  };

  template <typename InT, typename OutT>
  void DoFFTInternal(OpKernelContext* ctx, se::Stream* stream,
                     se::fft::Plan* plan, const se::fft::Type fft_type,
                     const uint64 output_distance, const Tensor& in,
                     Tensor* out) {
    const TensorShape& input_shape = in.shape();
    const TensorShape& output_shape = out->shape();
    auto src =
        AsDeviceMemory<InT>(in.flat<InT>().data(), input_shape.num_elements());
    auto dst = AsDeviceMemory<OutT>(out->flat<OutT>().data(),
                                    output_shape.num_elements());
    OP_REQUIRES(
        ctx, stream->ThenFft(plan, src, &dst).ok(),
        errors::Internal("fft failed : type=", static_cast<int>(fft_type),
                         " in.shape=", input_shape.DebugString()));
    if (!IsForward()) {
      typedef typename RealTypeFromComplexType<OutT>::RealT RealT;
      RealT alpha = 1.0 / output_distance;
      OP_REQUIRES(
          ctx,
          stream->ThenBlasScal(output_shape.num_elements(), alpha, &dst, 1)
              .ok(),
          errors::Internal("BlasScal failed : in.shape=",
                           input_shape.DebugString()));
    }
  }
};

int64_t FFTGPUBase::CufftScratchSize = GetCufftWorkspaceLimit(
    // default value is in bytes despite the name of the environment variable
    "TF_CUFFT_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
);

template <bool Forward, bool _Real, int FFTRank>
class FFTGPU : public FFTGPUBase {
 public:
  static_assert(FFTRank >= 1 && FFTRank <= 3,
                "Only 1D, 2D and 3D FFTs supported.");
  explicit FFTGPU(OpKernelConstruction* ctx) : FFTGPUBase(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_12(mht_12_v, 782, "", "./tensorflow/core/kernels/fft_ops.cc", "FFTGPU");
}

 protected:
  int Rank() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_13(mht_13_v, 788, "", "./tensorflow/core/kernels/fft_ops.cc", "Rank");
 return FFTRank; }
  bool IsForward() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_14(mht_14_v, 792, "", "./tensorflow/core/kernels/fft_ops.cc", "IsForward");
 return Forward; }
  bool IsReal() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfft_opsDTcc mht_15(mht_15_v, 796, "", "./tensorflow/core/kernels/fft_ops.cc", "IsReal");
 return _Real; }
};

// Register GPU kernels with priority 1 so that if a custom FFT CPU kernel is
// registered with priority 1 (to override the default Eigen CPU kernel), the
// CPU kernel does not outrank the GPU kernel.
REGISTER_KERNEL_BUILDER(Name("FFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("IFFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("FFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("IFFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("FFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("IFFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 3>);

REGISTER_KERNEL_BUILDER(
    Name("RFFT").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<true, true, 1>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<false, true, 1>);
REGISTER_KERNEL_BUILDER(
    Name("RFFT2D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<true, true, 2>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT2D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<false, true, 2>);
REGISTER_KERNEL_BUILDER(
    Name("RFFT3D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<true, true, 3>);
REGISTER_KERNEL_BUILDER(
    Name("IRFFT3D").Device(DEVICE_GPU).HostMemory("fft_length").Priority(1),
    FFTGPU<false, true, 3>);

// Deprecated kernels.
REGISTER_KERNEL_BUILDER(Name("BatchFFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 1>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT2D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 2>);
REGISTER_KERNEL_BUILDER(Name("BatchFFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<true, false, 3>);
REGISTER_KERNEL_BUILDER(Name("BatchIFFT3D").Device(DEVICE_GPU).Priority(1),
                        FFTGPU<false, false, 3>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow
