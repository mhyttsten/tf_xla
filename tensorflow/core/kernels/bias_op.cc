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
class MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc() {
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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/bias_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/redux_functor.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/tensor_format.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/bias_op_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

void GetBiasValueDims(const Tensor& value_tensor, TensorFormat data_format,
                      int32* batch, int32* height, int32* width, int32* depth,
                      int32* channel) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/kernels/bias_op.cc", "GetBiasValueDims");

  *batch = 1;
  *height = 1;
  *width = 1;
  *depth = 1;
  *channel = 1;
  if (data_format == FORMAT_NHWC) {
    int32_t channel_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    for (int32_t i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  } else if (data_format == FORMAT_NCHW) {
    *batch = static_cast<int32>(value_tensor.dim_size(0));
    *channel = static_cast<int32>(value_tensor.dim_size(1));
    *height = static_cast<int32>(value_tensor.dim_size(2));
    if (value_tensor.dims() > 3) {
      *width = static_cast<int32>(value_tensor.dim_size(3));
    }
    if (value_tensor.dims() > 4) {
      *depth = static_cast<int32>(value_tensor.dim_size(4));
    }
  }
}

template <class T>
struct AccumulatorType {
  typedef T type;
};

// float is faster on the CPU than half, and also more precise,
// so use float for the temporary accumulators.
template <>
struct AccumulatorType<Eigen::half> {
  typedef float type;
};

}  // namespace

template <typename Device, typename T>
class BiasOp : public BinaryOp<T> {
 public:
  explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_1(mht_1_v, 264, "", "./tensorflow/core/kernels/bias_op.cc", "BiasOp");

    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_2(mht_2_v, 277, "", "./tensorflow/core/kernels/bias_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ", bias.shape()));

    // Added by intel_tf to support NCHW on CPU regardless of MKL used or not.
    int channel_dim;
    if (data_format_ == FORMAT_NCHW) {
      channel_dim = 1;  // NCHW always have channel dim in 1 (with 3, 4, 5
                        // dimensions data).
    } else {
      channel_dim = input.shape().dims() - 1;  // End of code by intel_tf.
    }

    OP_REQUIRES(context,
                bias.shape().dim_size(0) == input.shape().dim_size(channel_dim),
                errors::InvalidArgument(
                    "Must provide as many biases as the last dimension "
                    "of the input tensor: ",
                    bias.shape(), " vs. ", input.shape()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    functor::Bias<Device, T> functor;
    const Device& d = context->eigen_device<Device>();
    if (data_format_ == FORMAT_NCHW && input.shape().dims() > 2) {
      functor(d, input.flat_inner_outer_dims<T, 2>(1),
              bias.flat_outer_dims<T, 2>(),
              output->flat_inner_outer_dims<T, 2>(1));
    } else {
      functor(d, input.flat<T>(), bias.vec<T>(), output->flat<T>());
    }
  }

 private:
  TensorFormat data_format_;
};

#define REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      BiasOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAddV1").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BiasOp<CPUDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

template <typename Device, typename T>
class BiasGradOp : public OpKernel {
 public:
  explicit BiasGradOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_3(mht_3_v, 340, "", "./tensorflow/core/kernels/bias_op.cc", "BiasGradOp");

    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_4(mht_4_v, 353, "", "./tensorflow/core/kernels/bias_op.cc", "Compute");

    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape()));

    OP_REQUIRES(
        context,
        FastBoundsCheck(output_backprop.NumElements(),
                        std::numeric_limits<int32>::max()),
        errors::InvalidArgument("BiasGrad requires tensor size <= int32 max"));

    int channel_dim;
    if (data_format_ == FORMAT_NCHW) {
      channel_dim = 1;
    } else {
      channel_dim = output_backprop.shape().dims() - 1;
    }
    Tensor* output = nullptr;
    TensorShape output_shape{output_backprop.shape().dim_size(channel_dim)};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (output_backprop.NumElements() == 0) {
      // Eigen often crashes by design on empty tensors, but setZero is safe
      output->template flat<T>().setZero();
    } else {
      // Added by intel_tf to support NCHW on CPU regardless of MKL used or not.
      using AccumT = typename AccumulatorType<T>::type;
      if (data_format_ == FORMAT_NCHW) {
        const functor::ReduceMiddleDimensions<
            T, AccumT, T, Eigen::internal::scalar_sum_op<AccumT>,
            Eigen::internal::SumReducer<T>>
            redux;

        auto flat_outer = output_backprop.flat_outer_dims<T, 3>();
        redux(context->eigen_device<Device>(), flat_outer.dimensions(),
              output_backprop, output, 1);
      } else {
        const functor::ReduceOuterDimensions<
            T, AccumT, T, Eigen::internal::scalar_sum_op<AccumT>>
            redux;

        auto flat_inner = output_backprop.flat_inner_dims<T, 2>();
        redux(context->eigen_device<Device>(), flat_inner.dimensions(),
              output_backprop, output);
      }
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BiasGradOp<CPUDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
class BiasOp<GPUDevice, T> : public BinaryOp<T> {
 public:
  typedef GPUDevice Device;
  explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_5(mht_5_v, 425, "", "./tensorflow/core/kernels/bias_op.cc", "BiasOp");

    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_6(mht_6_v, 438, "", "./tensorflow/core/kernels/bias_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias.shape().DebugString()));
    int32_t batch, height, width, depth, channel;
    GetBiasValueDims(input, data_format_, &batch, &height, &width, &depth,
                     &channel);
    OP_REQUIRES(context, bias.shape().dim_size(0) == channel,
                errors::InvalidArgument(
                    "Must provide as many biases as the channel dimension "
                    "of the input tensor: ",
                    bias.shape().DebugString(), " vs. ", channel, " in ",
                    input.shape().DebugString()));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (input.NumElements() > 0) {
      BiasGPU<T>::compute(context->template eigen_device<Device>(),
                          input.flat<T>().data(), bias.flat<T>().data(),
                          output->flat<T>().data(), batch, width, height, depth,
                          channel, data_format_);
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      BiasOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAddV1").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
REGISTER_GPU_KERNEL(int32);
#undef REGISTER_GPU_KERNEL

struct BiasGradAutotuneGroup {
  static string name() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_7(mht_7_v, 489, "", "./tensorflow/core/kernels/bias_op.cc", "name");
 return "BiasGrad"; }
};

class BiasAddGradGPUConfig {
 public:
  BiasAddGradGPUConfig() : mode_(BiasAddGradGPUMode::kReduction) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_8(mht_8_v, 497, "", "./tensorflow/core/kernels/bias_op.cc", "BiasAddGradGPUConfig");
}
  string ToString() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_9(mht_9_v, 501, "", "./tensorflow/core/kernels/bias_op.cc", "ToString");

    if (mode_ == BiasAddGradGPUMode::kNative) {
      return "native CUDA kernel.";
    }
    if (mode_ == BiasAddGradGPUMode::kReduction) {
      return "cub reduction kernel.";
    }
    return "unknown kernel.";
  }
  BiasAddGradGPUMode get_mode() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_10(mht_10_v, 513, "", "./tensorflow/core/kernels/bias_op.cc", "get_mode");
 return mode_; }
  void set_mode(BiasAddGradGPUMode val) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_11(mht_11_v, 517, "", "./tensorflow/core/kernels/bias_op.cc", "set_mode");
 mode_ = val; }

  bool operator==(const BiasAddGradGPUConfig& other) const {
    return this->mode_ == other.get_mode();
  }

  bool operator!=(const BiasAddGradGPUConfig& other) const {
    return !(*this == other);
  }

 private:
  BiasAddGradGPUMode mode_;
};

// Encapsulate all the shape information that is used in bias add grad
// operations.
class BiasAddParams {
 public:
  // We use a list to maintain both the shape value and the order (data format).
  using SpatialArray = gtl::InlinedVector<int64_t, 4>;
  BiasAddParams(const SpatialArray& in_shape, TensorFormat data_format,
                DataType dtype, int device_id)
      : in_shape_(in_shape),
        data_format_(data_format),
        dtype_(dtype),
        device_id_(device_id) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_12(mht_12_v, 545, "", "./tensorflow/core/kernels/bias_op.cc", "BiasAddParams");

    for (int64_t val : in_shape_) {
      hash_code_ = Hash64Combine(hash_code_, val);
    }
    hash_code_ = Hash64Combine(hash_code_, data_format);
    hash_code_ = Hash64Combine(hash_code_, dtype);
    hash_code_ = Hash64Combine(hash_code_, device_id);
  }
  bool operator==(const BiasAddParams& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const BiasAddParams& other) const {
    return !(*this == other);
  }
  uint64 hash() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_13(mht_13_v, 563, "", "./tensorflow/core/kernels/bias_op.cc", "hash");
 return hash_code_; }

  string ToString() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_14(mht_14_v, 568, "", "./tensorflow/core/kernels/bias_op.cc", "ToString");

    // clang-format off
    return strings::StrCat(
        "(", absl::StrJoin(in_shape_, ", "), "), ",
        data_format_, ", ", dtype_, ", ", device_id_);
    // clang-format on
  }

 protected:
  using ParamsDataType = std::tuple<SpatialArray, TensorFormat, DataType, int>;

  ParamsDataType get_data_as_tuple() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_15(mht_15_v, 582, "", "./tensorflow/core/kernels/bias_op.cc", "get_data_as_tuple");

    return std::make_tuple(in_shape_, data_format_, dtype_, device_id_);
  }

  uint64 hash_code_ = 0;

 private:
  SpatialArray in_shape_;
  TensorFormat data_format_;
  DataType dtype_;
  int device_id_;
};

typedef AutotuneSingleton<BiasGradAutotuneGroup, BiasAddParams,
                          BiasAddGradGPUConfig>
    AutotuneBiasGrad;

template <typename T>
class BiasGradOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit BiasGradOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_16(mht_16_v, 606, "", "./tensorflow/core/kernels/bias_op.cc", "BiasGradOp");

    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NCHW;
    }
  }

  void ComputeWithCustomKernel(OpKernelContext* context,
                               const Tensor& output_backprop, int32_t batch,
                               int32_t width, int32_t height, int32_t depth,
                               int32_t channel, Tensor* output) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_17(mht_17_v, 622, "", "./tensorflow/core/kernels/bias_op.cc", "ComputeWithCustomKernel");

    BiasGradGPU<T>::compute(context->template eigen_device<Device>(),
                            output_backprop.template flat<T>().data(),
                            output->flat<T>().data(), batch, width, height,
                            depth, channel, data_format_);
  }

  void ComputeWithReduceSum(OpKernelContext* context,
                            const Tensor& output_backprop, int32_t batch,
                            int32_t width, int32_t height, int32_t depth,
                            int32_t channel, Tensor* output) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_18(mht_18_v, 635, "", "./tensorflow/core/kernels/bias_op.cc", "ComputeWithReduceSum");

    if (data_format_ == FORMAT_NCHW) {
      int32_t row_count = batch * channel;
      int32_t col_count = height * width * depth;
      Tensor temp_grad_outputs;
      // For 'NCHW' format, we perform reduction twice: first HW, then N.
      TensorShape temp_grad_output_shape{row_count, col_count};
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                     temp_grad_output_shape,
                                                     &temp_grad_outputs));
      BiasGradGPU<T>::DoRowReduction(
          context, temp_grad_outputs.flat<T>().data(),
          output_backprop.template flat<T>().data(), row_count, col_count);

      row_count = batch;
      col_count = channel;
      BiasGradGPU<T>::DoColReduction(context, output->flat<T>().data(),
                                     temp_grad_outputs.flat<T>().data(),
                                     row_count, col_count);
    } else {
      // For 'NHWC', we simply apply reduction once on NHW.
      int32_t row_count = batch * height * width * depth;
      int32_t col_count = channel;
      BiasGradGPU<T>::DoColReduction(
          context, const_cast<T*>(output->flat<T>().data()),
          reinterpret_cast<const T*>(output_backprop.template flat<T>().data()),
          row_count, col_count);
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbias_opDTcc mht_19(mht_19_v, 668, "", "./tensorflow/core/kernels/bias_op.cc", "Compute");

    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));
    int32_t batch, height, width, depth, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &depth, &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (channel == 0) return;
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    se::DeviceMemoryBase output_ptr(output->flat<T>().data(),
                                    output->NumElements() * sizeof(T));
    stream->ThenMemZero(&output_ptr, output->NumElements() * sizeof(T));
    if (output_backprop.NumElements() <= 0) return;
    if (OpDeterminismRequired()) {
      // ComputeWithReduceSum is the only deterministic algorithm.
      ComputeWithReduceSum(context, output_backprop, batch, width, height,
                           depth, channel, output);
      return;
    }

    int device_id = stream->parent()->device_ordinal();
    DataType dtype = output_backprop.dtype();
    BiasAddParams bias_parameters = {
        {batch, height * width * depth, channel},
        data_format_,
        dtype,
        device_id,
    };

    // Autotune two algorithm: customized
    BiasAddGradGPUConfig algo_config;
    if (!AutotuneBiasGrad::GetInstance()->Find(bias_parameters, &algo_config)) {
      profiler::ScopedAnnotation trace("bias_grad_autotuning");

      BiasGradGPUProfileResult best_result;
      // Initialize the timer.
      perftools::gputools::Timer timer(stream->parent());
      stream->InitTimer(&timer);
      stream->ThenStartTimer(&timer);
      ComputeWithCustomKernel(context, output_backprop, batch, width, height,
                              depth, channel, output);
      stream->ThenStopTimer(&timer);
      uint64 elapsed_microseconds = timer.Microseconds();
      VLOG(1) << "BiasAddGrad " << bias_parameters.ToString()
              << " Native algo latency: " << elapsed_microseconds;
      if (elapsed_microseconds < best_result.elapsed_time()) {
        best_result.set_algorithm(BiasAddGradGPUMode::kNative);
        best_result.set_elapsed_time(elapsed_microseconds);
      }

      // Try reduction and profile.
      stream->ThenStartTimer(&timer);
      ComputeWithReduceSum(context, output_backprop, batch, width, height,
                           depth, channel, output);
      stream->ThenStopTimer(&timer);

      elapsed_microseconds = timer.Microseconds();
      VLOG(1) << "BiasAddGrad " << bias_parameters.ToString()
              << " Reduction algo latency: " << elapsed_microseconds;
      if (elapsed_microseconds < best_result.elapsed_time()) {
        best_result.set_algorithm(BiasAddGradGPUMode::kReduction);
        best_result.set_elapsed_time(elapsed_microseconds);
      }

      algo_config.set_mode(best_result.algorithm());
      AutotuneBiasGrad::GetInstance()->Insert(bias_parameters, algo_config);

      // Results are already available during autotune, so no need to continue.
      return;
    }

    // Choose the best algorithm based on autotune results.
    if (algo_config.get_mode() == BiasAddGradGPUMode::kReduction) {
      ComputeWithReduceSum(context, output_backprop, batch, width, height,
                           depth, channel, output);
    } else {
      // Default to the customized kernel.
      ComputeWithCustomKernel(context, output_backprop, batch, width, height,
                              depth, channel, output);
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasGradOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
