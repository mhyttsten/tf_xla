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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc() {
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

// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/image/adjust_contrast_op.h"

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/determinism.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// AdjustContrastOp is deprecated as of GraphDef version >= 2

template <typename Device, typename T>
class AdjustContrastOp : public OpKernel {
 public:
  explicit AdjustContrastOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "AdjustContrastOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& factor = context->input(1);
    const Tensor& min_value = context->input(2);
    const Tensor& max_value = context->input(3);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    const int64_t height = input.dim_size(input.dims() - 3);
    const int64_t width = input.dim_size(input.dims() - 2);
    const int64_t channels = input.dim_size(input.dims() - 1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(factor.shape()),
                errors::InvalidArgument("contrast_factor must be scalar: ",
                                        factor.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(min_value.shape()),
                errors::InvalidArgument("min_value must be scalar: ",
                                        min_value.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(max_value.shape()),
                errors::InvalidArgument("max_value must be scalar: ",
                                        max_value.shape().DebugString()));

    if (std::is_same<Device, GPUDevice>::value) {
      OP_REQUIRES(
          context, !OpDeterminismRequired(),
          errors::Unimplemented(
              "A deterministic GPU implementation of AdjustContrast is not"
              " currently available."));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    Tensor mean_values;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                   TensorShape(input.shape()),
                                                   &mean_values));

    if (input.NumElements() > 0) {
      const int64_t batch = input.NumElements() / (height * width * channels);
      const int64_t shape[4] = {batch, height, width, channels};
      functor::AdjustContrast<Device, T>()(
          context->eigen_device<Device>(), input.shaped<T, 4>(shape),
          factor.scalar<float>(), min_value.scalar<float>(),
          max_value.scalar<float>(), mean_values.shaped<float, 4>(shape),
          output->shaped<float, 4>(shape));
    }
  }
};

#define REGISTER_KERNEL(T)                                              \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AdjustContrast").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      AdjustContrastOp<CPUDevice, T>);

REGISTER_KERNEL(uint8);
REGISTER_KERNEL(int8);
REGISTER_KERNEL(int16);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
// Forward declarations of the function specializations for GPU (to prevent
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
namespace functor {
#define DECLARE_GPU_SPEC(T)                                         \
  template <>                                                       \
  void AdjustContrast<GPUDevice, T>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input, \
      typename TTypes<float>::ConstScalar contrast_factor,          \
      typename TTypes<float>::ConstScalar min_value,                \
      typename TTypes<float>::ConstScalar max_value,                \
      typename TTypes<float, 4>::Tensor mean_values,                \
      typename TTypes<float, 4>::Tensor output);                    \
  extern template struct AdjustContrast<GPUDevice, T>;

DECLARE_GPU_SPEC(uint8);
DECLARE_GPU_SPEC(int8);
DECLARE_GPU_SPEC(int16);
DECLARE_GPU_SPEC(int32);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AdjustContrast").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustContrastOp<GPUDevice, T>);
REGISTER_GPU_KERNEL(uint8);
REGISTER_GPU_KERNEL(int8);
REGISTER_GPU_KERNEL(int16);
REGISTER_GPU_KERNEL(int32);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

class AdjustContrastOpV2Base : public OpKernel {
 protected:
  explicit AdjustContrastOpV2Base(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_2(mht_2_v, 328, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "AdjustContrastOpV2Base");
}

  struct ComputeOptions {
    const Tensor* input = nullptr;
    const Tensor* factor = nullptr;
    Tensor* output = nullptr;
    int64_t batch = 0;
    int64_t height = 0;
    int64_t width = 0;
    int64_t channels = 0;
  };

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_3(mht_3_v, 343, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& factor = context->input(1);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    const int64_t height = input.dim_size(input.dims() - 3);
    const int64_t width = input.dim_size(input.dims() - 2);
    const int64_t channels = input.dim_size(input.dims() - 1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(factor.shape()),
                errors::InvalidArgument("contrast_factor must be scalar: ",
                                        factor.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    if (input.NumElements() > 0) {
      const int64_t batch = input.NumElements() / (height * width * channels);
      ComputeOptions options;
      options.input = &input;
      options.factor = &factor;
      options.output = output;
      options.batch = batch;
      options.height = height;
      options.width = width;
      options.channels = channels;
      DoCompute(context, options);
    }
  }

  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;
};

template <typename Device, typename T>
class AdjustContrastOpv2;

template <>
class AdjustContrastOpv2<CPUDevice, float> : public AdjustContrastOpV2Base {
 public:
  explicit AdjustContrastOpv2(OpKernelConstruction* context)
      : AdjustContrastOpV2Base(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_4(mht_4_v, 389, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "AdjustContrastOpv2");
}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_5(mht_5_v, 395, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "DoCompute");

    const int64_t batch = options.batch;
    const int64_t height = options.height;
    const int64_t width = options.width;
    const int64_t channels = options.channels;
    const int64_t image_size = height * width;
    const Tensor* input = options.input;
    const Tensor* factor = options.factor;
    Tensor* output = options.output;
    Tensor mean_values;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<float>::value,
                                TensorShape({batch, channels}), &mean_values));
    // TODO(zhengxq): for multiple batches, shard them into different batches.
    auto input_data = input->shaped<float, 3>({batch, image_size, channels});
    auto mean_data = mean_values.tensor<float, 2>();
    auto output_data = output->shaped<float, 3>({batch, image_size, channels});

    // Calculate the mean of the inputs.
    ReduceMeanAcrossImage(input_data, mean_data, output_data);
    // Broadcast the mean into the outputs.
    BroadcastAcrossImage(mean_data, output_data);
    // Increment the outputs with the scaled difference through their flat
    // structure.
    IncrementWithScaling(input_data, factor->scalar<float>(), output_data);
  }

 private:
  // Reduce the mean of the inputs along the image dimension, i.e. dim_1, in a
  // 3D tensor. Effectively means(i, k) = inputs(i, :, k).mean().
  void ReduceMeanAcrossImage(typename TTypes<float, 3>::ConstTensor input,
                             typename TTypes<float, 2>::Tensor mean,
                             typename TTypes<float, 3>::Tensor scratch) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_6(mht_6_v, 430, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "ReduceMeanAcrossImage");

    const int64_t batch = input.dimension(0);
    const int64_t image_size = input.dimension(1);
    const int64_t channels = input.dimension(2);
    TTypes<float, 1>::ConstTensor input_flat(&input(0, 0, 0), input.size());
    TTypes<float, 1>::Tensor mean_flat(&mean(0, 0), mean.size());
    TTypes<float, 1>::Tensor summation_scratch(&scratch(0, 0, 0),
                                               scratch.size());
    typedef Eigen::array<Eigen::DenseIndex, 1> Index;
    const int64_t plane_size = image_size * channels;
    // Since the number of channels in the early layers is often small, a
    // straightforward loop for summing cannot utilize vectorization.
    // This algorithm repeatedly folds each image plane by half, until
    // only one set of channels remains.
    for (int64_t i = 0; i < batch; i++) {
      auto input_plane =
          input_flat.slice(Index(i * plane_size), Index(plane_size));
      auto summation_plane =
          summation_scratch.slice(Index(i * plane_size), Index(plane_size));
      int64_t remaining_size = image_size;
      int round = 0;
      // Sum the input(i, :, k) into mean(i, k). Repeatedly splits the input
      // array into half and sums the two halves, until only one set of channels
      // is left, which holds the sum. Since each half is large enough, this
      // leads to much better vectorizations between components. An example of
      // how this works:
      //
      //   x = float[4096, 3]
      //   round 0
      //     y[:2048, :] = x[:2048, :] + x[2048:, :]
      //   round 1
      //     y[:1024, :] += y[1024:2048, :]
      //   round 2
      //     y[:512, :] += y[512:1024, :]
      //   ...
      //   round 11
      //     y[:1, :] += y[1:2, :]
      //   At this point y[0, :] holds the sum of all x[:, :]
      //
      // The algorithm itself can handle size that is not power-of-two. Note
      // that in each round we sum up elements that are contiguous. So we can
      // use their flattened structure to gain vectorization efficiency.
      do {
        int64_t right_size = remaining_size / 2;
        int64_t left_size = remaining_size - right_size;
        DCHECK(left_size == right_size || left_size == right_size + 1);
        if (round == 0) {
          // In the first round, sum the left side and right side of the input
          // array into the summation area.
          summation_plane.slice(Index(0), Index(right_size * channels)) =
              input_plane.slice(Index(left_size * channels),
                                Index(right_size * channels)) +
              input_plane.slice(Index(0), Index(right_size * channels));
          if (left_size > right_size) {
            DCHECK_EQ(left_size - right_size, 1);
            // Copy over the remaining column if the remaining_size is odd.
            // This also handles the case where image_size == 1.
            summation_plane.slice(Index(right_size * channels),
                                  Index(channels)) =
                input_plane.slice(Index(right_size * channels),
                                  Index(channels));
          }
        } else {
          // For all the remaining rounds, add the second half of the inputs
          // into the first half of the inputs. With the flat structure and
          // large size, this utilizes vectorization between components.
          summation_plane.slice(Index(0), Index(right_size * channels)) +=
              summation_plane.slice(Index(left_size * channels),
                                    Index(right_size * channels));
        }
        remaining_size = left_size;
        round++;
      } while (remaining_size > 1);
      const float mean_scaling = 1.0f / image_size;
      // The first channels elements in summation_plane now holds the summation.
      // Scale it with image_size and copy over to the means.
      auto mean_plane = mean_flat.slice(Index(i * channels), Index(channels));
      mean_plane =
          summation_plane.slice(Index(0), Index(channels)) * mean_scaling;
    }
  }

  // Broadcast a 2D inputs into a 3D outputs across the image dimension, i.e.,
  // dim-1.
  void BroadcastAcrossImage(typename TTypes<float, 2>::Tensor inputs,
                            typename TTypes<float, 3>::Tensor outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_7(mht_7_v, 518, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "BroadcastAcrossImage");

    int64_t batch = outputs.dimension(0);
    int64_t image_size = outputs.dimension(1);
    int64_t channels = outputs.dimension(2);
    // Similar to the reduction case, a straightforward implementation of this
    // does not utilize vectorization well because of the small channel size.
    // This algorithm repeatedly increases the area to be copied, and leads to
    // much better vectorizations in the copy.
    for (int64_t i = 0; i < batch; i++) {
      // Copy over the inputs into outputs in this batch. Effectively:
      // outputs(i, :, k) = inputs(i, k). An example of how this algorithm
      // works:
      //
      //    x = float[1, 3], y = float[2048, 3]
      //    round 0
      //      y[:1, :] = x[:, :]
      //    round 1
      //      y[1:2, :] = y[:1, :]
      //    round 2
      //      y[2:4, :] = y[:2, :]
      //    round 3
      //      y[4:8, :] = y[:4, :]
      //    ...
      //    round 11
      //      y[1024:2048, :] = y[:1024, :]
      //    At this point y[:, k] == x[k]
      //
      // The algorithm works for size that is not power-of-two. For each round,
      // the elements that are copied are continuous, so it benefits from the
      // vectorized copy via memcpy.
      const float* mean_p = &inputs(i, 0);
      // Copy the first set of channels.
      float* output_p = &outputs(i, 0, 0);
      memcpy(output_p, mean_p, sizeof(float) * channels);
      int64_t copied = 1;
      while (copied < image_size) {
        // Repeatedly increases the number of elements to copy so they have
        // better vectorizations. However, the source of the copy has to be
        // not too large to stay in the cache.
        const int64_t kMaxToCopy = 1024;
        int64_t to_copy = std::min({copied, image_size - copied, kMaxToCopy});
        memcpy(output_p + channels * copied, output_p,
               to_copy * channels * sizeof(float));
        copied += to_copy;
      }
    }
  }

  // Increment the outputs with the scaled difference between inputs and
  // outputs. Effectively: outputs += factor * (inputs - outputs).
  void IncrementWithScaling(typename TTypes<float, 3>::ConstTensor input,
                            typename TTypes<float>::ConstScalar factor,
                            typename TTypes<float, 3>::Tensor output) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_8(mht_8_v, 573, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "IncrementWithScaling");

    const float factor_value = factor();
    float* p = output.data();
    const float* q = input.data();
    for (int64_t n = 0; n < input.size(); ++n) {
      p[n] += factor_value * (q[n] - p[n]);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("AdjustContrastv2").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    AdjustContrastOpv2<CPUDevice, float>);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
// Forward declarations of the function specializations for GPU (to prevent
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
namespace functor {

#define DECLARE_GPU_SPEC(T)                                         \
  template <>                                                       \
  void AdjustContrastv2<GPUDevice, T>::operator()(                  \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input, \
      typename TTypes<float>::ConstScalar contrast_factor,          \
      typename TTypes<T, 4>::Tensor output);                        \
  extern template struct AdjustContrastv2<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);

#undef DECLARE_GPU_SPEC

}  // namespace functor

template <typename T>
class AdjustContrastOpv2<GPUDevice, T> : public AdjustContrastOpV2Base {
 public:
  explicit AdjustContrastOpv2(OpKernelConstruction* context)
      : AdjustContrastOpV2Base(context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_9(mht_9_v, 615, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "AdjustContrastOpv2");
}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSadjust_contrast_opDTcc mht_10(mht_10_v, 621, "", "./tensorflow/core/kernels/image/adjust_contrast_op.cc", "DoCompute");

    const int64_t shape[4] = {options.batch, options.height, options.width,
                              options.channels};
    OP_REQUIRES(
        context, !OpDeterminismRequired(),
        errors::Unimplemented(
            "A deterministic GPU implementation of AdjustContrastv2 is not"
            " currently available."));

    functor::AdjustContrastv2<GPUDevice, T>()(
        context->eigen_device<GPUDevice>(), options.input->shaped<T, 4>(shape),
        options.factor->scalar<float>(), options.output->shaped<T, 4>(shape));
  }
};

#define REGISTER_GPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("AdjustContrastv2").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustContrastOpv2<GPUDevice, T>);

REGISTER_GPU(float)
REGISTER_GPU(Eigen::half)

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


}  // namespace tensorflow
