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
class MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc() {
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

// LRN = Local Response Normalization
// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/util/work_sharder.h"
#endif

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#if TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#endif
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/stream_executor_util.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

namespace {

// When the depth is large and beta_ is 0.5 or 1.0, Single-threaded
// LRN is faster than the main band matrix approach used
// below. Benchmarks suggest switching to SingleThreadedLRN when depth > 384.
const int kSingleThreadedLRNDepthCutoff = 384;

// Create a depth-by-depth band matrix with 1s along a swath of size (2 *
// depth_radius + 1) around the diagonal.
template <typename T>
void GetBandMatrix(int depth, int depth_radius,
                   Eigen::Tensor<T, 2, Eigen::RowMajor>* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_0(mht_0_v, 233, "", "./tensorflow/core/kernels/lrn_op.cc", "GetBandMatrix");

  result->setZero();
  for (int row = 0; row < depth; ++row) {
    const int begin = std::max<int>(0, row - depth_radius);
    const int end = std::min<int>(depth, row + depth_radius + 1);
    Eigen::DSizes<Eigen::DenseIndex, 2> start(row, begin);
    Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, end - begin);
    result->slice(start, sizes).setConstant(T(1));
  }
}

}  // namespace

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchLRN;

template <typename T>
struct LaunchLRN<CPUDevice, T> {
  LaunchLRN(int depth_radius, T bias, T alpha, T beta)
      : depth_radius_(depth_radius), bias_(bias), alpha_(alpha), beta_(beta) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_1(mht_1_v, 258, "", "./tensorflow/core/kernels/lrn_op.cc", "LaunchLRN");
}

  void launch(OpKernelContext* context, OpKernel* kernel, const Tensor& in,
              Tensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_2(mht_2_v, 264, "", "./tensorflow/core/kernels/lrn_op.cc", "launch");

    const int batch = static_cast<int>(in.dim_size(0));
    const int rows = static_cast<int>(in.dim_size(1));
    const int cols = static_cast<int>(in.dim_size(2));
    const int depth = static_cast<int>(in.dim_size(3));

#if defined(IS_MOBILE_PLATFORM)
    SingleThreadedLRN(in, batch, rows, cols, depth, output);
#else
    const int nodes = cols * rows;
    if (depth > kSingleThreadedLRNDepthCutoff &&
        (beta_ == T(0.5) || beta_ == T(1))) {
      SingleThreadedLRN(in, batch, rows, cols, depth, output);
      return;
    }

    auto in_shaped = in.shaped<T, 2>({nodes * batch, depth});

    // Multiplying the input with the band matrix has the effect of reducing the
    // correct patch along the depth.
    Eigen::Tensor<T, 2, Eigen::RowMajor> multiplier(depth, depth);
    GetBandMatrix<T>(depth, depth_radius_, &multiplier);

    auto out_shaped = output->shaped<T, 2>({nodes * batch, depth});
    Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
    auto tmp = in_shaped.square().contract(multiplier, dims) * alpha_ + bias_;
    if (beta_ == T(1)) {
      out_shaped.device(context->eigen_cpu_device()) =
          in_shaped * tmp.inverse();
    } else if (beta_ == T(0.5)) {
      out_shaped.device(context->eigen_cpu_device()) = in_shaped * tmp.rsqrt();
    } else {
      out_shaped.device(context->eigen_cpu_device()) =
          in_shaped * (tmp.log() * -beta_).exp();
    }
#endif
  }

 private:
  typedef typename Eigen::Tensor<T, 1, Eigen::RowMajor>::DimensionPair DimPair;

  void SingleThreadedLRN(const Tensor& in, const int batch, const int rows,
                         const int cols, const int depth, Tensor* out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_3(mht_3_v, 309, "", "./tensorflow/core/kernels/lrn_op.cc", "SingleThreadedLRN");

    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> data_in(
        in.flat<T>().data(), depth, batch * rows * cols);

    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> data_out(
        out->flat<T>().data(), depth, batch * rows * cols);

    const int double_depth_radius = depth_radius_ * 2;
    Eigen::Matrix<T, Eigen::Dynamic, 1> padded_square(data_in.rows() +
                                                      double_depth_radius);
    padded_square.setZero();
    for (int r = 0; r < data_in.cols(); ++r) {
      // Do local response normalization for data_in(:, r). First, compute the
      // square and store them in buffer for repeated use.
      padded_square.block(depth_radius_, 0, data_out.rows(), 1) =
          data_in.col(r).cwiseProduct(data_in.col(r)) * alpha_;
      // Then, compute the scale and write it to data_out.
      T accumulated_scale(0);
      for (int i = 0; i < double_depth_radius; ++i) {
        accumulated_scale += padded_square(i);
      }
      for (int i = 0; i < data_in.rows(); ++i) {
        accumulated_scale += padded_square(i + double_depth_radius);
        data_out(i, r) = bias_ + accumulated_scale;
        accumulated_scale -= padded_square(i);
      }
    }

    if (beta_ == T(1)) {
      data_out.array() = data_in.array() * data_out.array().inverse();
    } else if (beta_ == T(0.5)) {
      data_out.array() = data_in.array() * data_out.array().rsqrt();
    } else {
      data_out.array() =
          data_in.array() * (data_out.array().log() * -beta_).exp();
    }
  }

  int depth_radius_;
  T bias_;
  T alpha_;
  T beta_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
struct LaunchLRN<GPUDevice, T> {
  LaunchLRN(int depth_radius, T bias, T alpha, T beta)
      : depth_radius_(depth_radius), bias_(bias), alpha_(alpha), beta_(beta) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_4(mht_4_v, 361, "", "./tensorflow/core/kernels/lrn_op.cc", "LaunchLRN");
}

  void launch(OpKernelContext* context, OpKernel* kernel, const Tensor& in,
              Tensor* output) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_5(mht_5_v, 367, "", "./tensorflow/core/kernels/lrn_op.cc", "launch");

#if GOOGLE_CUDA
    OP_REQUIRES(
        context, beta_ >= 0.01,
        errors::InvalidArgument("cuDNN requires beta >= 0.01, got: ", beta_));

    OP_REQUIRES(
        context, depth_radius_ > 0 && depth_radius_ <= 7,
        errors::InvalidArgument("cuDNN requires depth_radius in [1, 7], got: ",
                                depth_radius_));
    OP_REQUIRES(
        context, bias_ >= 1e-5,
        errors::InvalidArgument("cuDNN requires bias >= 1e-5, got: ", bias_));

    // Cast to platform-specific int to avoid conversion warnings.
    const int batch = static_cast<int>(in.dim_size(0));
    const int rows = static_cast<int>(in.dim_size(1));
    const int cols = static_cast<int>(in.dim_size(2));
    const int depth = static_cast<int>(in.dim_size(3));

    se::dnn::BatchDescriptor dimensions_desc;
    dimensions_desc.set_count(batch)
        .set_height(rows)
        .set_width(cols)
        .set_feature_map_count(depth)
        .set_layout(se::dnn::DataLayout::kBatchYXDepth);

    se::dnn::NormalizeDescriptor normalize_desc;
    normalize_desc.set_bias(bias_)
        .set_range(depth_radius_)
        .set_alpha(alpha_)
        .set_beta(beta_);

    auto input_data = StreamExecutorUtil::AsDeviceMemory<T>(in);
    auto output_data = StreamExecutorUtil::AsDeviceMemory<T>(*output);

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    bool status =
        stream
            ->ThenNormalizeWithDimensions(normalize_desc, dimensions_desc,
                                          input_data, &output_data)
            .ok();
    OP_REQUIRES(context, status,
                errors::Internal("NormalizeWithDimensions launch failed"));
#elif TENSORFLOW_USE_ROCM
    // For NHWC input/output tensors, convert to NCHW because it's the only
    // supported format in MIOpen for now.

    // Cast to platform-specific int to avoid conversion warnings.
    const int batch = static_cast<int>(in.dim_size(0));
    const int rows = static_cast<int>(in.dim_size(1));
    const int cols = static_cast<int>(in.dim_size(2));
    const int depth = static_cast<int>(in.dim_size(3));

    Tensor transformed_input;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       ShapeFromFormat(FORMAT_NCHW, in.shape(), FORMAT_NHWC),
                       &transformed_input));
    functor::NHWCToNCHW<GPUDevice, T, 4>()(context->eigen_device<GPUDevice>(),
                                           in.tensor<T, 4>(),
                                           transformed_input.tensor<T, 4>());

    Tensor transformed_output;
    OP_REQUIRES_OK(
        context, context->allocate_temp(
                     DataTypeToEnum<T>::value,
                     ShapeFromFormat(FORMAT_NCHW, output->shape(), FORMAT_NHWC),
                     &transformed_output));

    perftools::gputools::dnn::BatchDescriptor dimensions_desc;
    dimensions_desc.set_count(batch)
        .set_height(rows)
        .set_width(cols)
        .set_feature_map_count(depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    perftools::gputools::dnn::NormalizeDescriptor normalize_desc;
    normalize_desc.set_bias(bias_)
        .set_range(depth_radius_)
        .set_alpha(alpha_)
        .set_beta(beta_);

    auto input_data =
        AsDeviceMemory(transformed_input.template flat<T>().data(),
                       transformed_input.template flat<T>().size());
    auto output_data =
        AsDeviceMemory(transformed_output.template flat<T>().data(),
                       transformed_output.template flat<T>().size());

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    bool status =
        stream
            ->ThenNormalizeWithDimensions(normalize_desc, dimensions_desc,
                                          input_data, &output_data)
            .ok();
    OP_REQUIRES(context, status,
                errors::Internal("NormalizeWithDimensions launch failed"));

    // Need to convert it back to NHWC once MIOpen kernels finishes.
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        context->eigen_device<GPUDevice>(),
        toConstTensor(transformed_output).template tensor<T, 4>(),
        output->tensor<T, 4>());
#endif
  }

  int depth_radius_;
  T bias_;
  T alpha_;
  T beta_;
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T>
class LRNOp : public OpKernel {
 public:
  explicit LRNOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_6(mht_6_v, 494, "", "./tensorflow/core/kernels/lrn_op.cc", "LRNOp");

    int64_t depth_radius64;
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius64));
    OP_REQUIRES(
        context,
        FastBoundsCheck(depth_radius64, std::numeric_limits<int>::max()),
        errors::InvalidArgument("depth_radius = ", depth_radius64,
                                " larger than int max"));
    depth_radius_ = static_cast<int>(depth_radius64);
    float tmp;
    OP_REQUIRES_OK(context, context->GetAttr("bias", &tmp));
    bias_ = T(tmp);
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &tmp));
    alpha_ = T(tmp);
    OP_REQUIRES_OK(context, context->GetAttr("beta", &tmp));
    beta_ = T(tmp);
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_7(mht_7_v, 515, "", "./tensorflow/core/kernels/lrn_op.cc", "Compute");

    const Tensor& in = context->input(0);
    OP_REQUIRES(context, in.dims() == 4,
                errors::InvalidArgument("in must be 4-dimensional"));
    OP_REQUIRES(
        context,
        FastBoundsCheck(in.NumElements(), std::numeric_limits<int>::max()),
        errors::InvalidArgument("argument to LRN too large"));
    // Cast to platform-specific int to avoid conversion warnings.
    const int batch = static_cast<int>(in.dim_size(0));
    const int rows = static_cast<int>(in.dim_size(1));
    const int cols = static_cast<int>(in.dim_size(2));
    const int depth = static_cast<int>(in.dim_size(3));

    OP_REQUIRES(context,
                (depth + depth_radius_) <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("depth ", depth, " + depth_radius ",
                                        depth_radius_, " exceeds int max."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({batch, rows, cols, depth}), &output));

    LaunchLRN<Device, T> launcher(depth_radius_, bias_, alpha_, beta_);
    launcher.launch(context, this, in, output);
  }

 private:
  int depth_radius_;
  T bias_;
  T alpha_;
  T beta_;
};

#define REGISTER_CPU(T)                                      \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("LRN").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LRNOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU);
TF_CALL_half(REGISTER_CPU);

#undef REGISTER_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(T)                                      \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("LRN").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LRNOp<GPUDevice, T>);
TF_CALL_float(REGISTER_GPU);

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if !defined(IS_MOBILE_PLATFORM)

template <typename Device, typename T>
struct LaunchLRNGrad;

template <typename T>
struct LaunchLRNGrad<CPUDevice, T> {
  LaunchLRNGrad(int depth_radius, T bias, T alpha, T beta)
      : depth_radius_(depth_radius),
        bias_(bias),
        alpha_(alpha),
        beta_(beta),
        alpha_beta_2_(T(-2) * alpha * beta) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_8(mht_8_v, 586, "", "./tensorflow/core/kernels/lrn_op.cc", "LaunchLRNGrad");
}

  void launch(OpKernelContext* context, OpKernel* kernel,
              const Tensor& in_grads, const Tensor& in_image,
              const Tensor& out_image, Tensor* output) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_9(mht_9_v, 593, "", "./tensorflow/core/kernels/lrn_op.cc", "launch");

    const int64_t batch = in_grads.dim_size(0);
    const int64_t rows = in_grads.dim_size(1);
    const int64_t cols = in_grads.dim_size(2);
    const int64_t depth = in_grads.dim_size(3);
    const auto nodes = cols * rows;
    auto grads_shaped = in_grads.shaped<T, 2>({nodes * batch, depth});
    auto in_shaped = in_image.shaped<T, 2>({nodes * batch, depth});
    auto activations = out_image.shaped<T, 2>({nodes * batch, depth});

    auto out_shaped = output->shaped<T, 2>({nodes * batch, depth});
    out_shaped.setZero();

    auto shard = [this, activations, in_shaped, grads_shaped, out_shaped,
                  depth](int64_t begin, int64_t end) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_10(mht_10_v, 610, "", "./tensorflow/core/kernels/lrn_op.cc", "lambda");

      for (int64_t i = begin; i < end; ++i) {
        for (int64_t j = 0; j < depth; ++j) {
          // Let y be the LRN activations and x be the inputs along the depth
          // dimension. (LRN operates independently along rows, cols, and
          // batch).
          // We have
          // yi = xi / (bias + alpha(sum_j_{i - depth_radius}^{i + depth_radius}
          //      x_j^2))^beta
          //
          // Let N = (bias + alpha(sum_j_{i - depth_radius}^{i + depth_radius}
          //           x_j^2))
          // dy_i/dx_i = (N^beta - xi. beta*N^(beta-1)*2*alpha*xi)/N^(2*beta)
          // dy_i/dx_j = (       - xi. beta*N^(beta-1)*2*alpha*xj)/N^(2*beta)
          //
          // NOTE(keveman) : We can compute N by doing (yi/xi) ^ (1/beta).
          // However, this is numerically unstable for small values of xi. We
          // compute N explicitly here to avoid that.

          T gs = grads_shaped(i, j);
          if (gs == T(0)) continue;

          int64_t depth_begin = std::max<int64_t>(0, j - depth_radius_);
          int64_t depth_end = std::min<int64_t>(depth, j + depth_radius_ + 1);

          T norm(0);
          for (int64_t k = depth_begin; k < depth_end; ++k) {
            norm += in_shaped(i, k) * in_shaped(i, k);
          }
          norm = alpha_ * norm + bias_;
          DCHECK_GT(norm, T(1e-6));
          T pre_computed_pow = Eigen::numext::pow(norm, -beta_);
          T activations_ab2 = alpha_beta_2_ * activations(i, j);
          for (int64_t k = depth_begin; k < depth_end; ++k) {
            T dyi = in_shaped(i, k) * activations_ab2 / norm;
            if (k == j) {
              dyi += pre_computed_pow;
            }
            dyi *= gs;
            const_cast<typename TTypes<T, 2>::Tensor&>(out_shaped)(i, k) += dyi;
          }
        }
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, nodes * batch,
          depth * depth, shard);
  }

  int depth_radius_;
  T bias_;
  T alpha_;
  T beta_;
  T alpha_beta_2_;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
struct LaunchLRNGrad<GPUDevice, T> {
  LaunchLRNGrad(int depth_radius, T bias, T alpha, T beta)
      : depth_radius_(depth_radius), bias_(bias), alpha_(alpha), beta_(beta) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_11(mht_11_v, 674, "", "./tensorflow/core/kernels/lrn_op.cc", "LaunchLRNGrad");
}

  void launch(OpKernelContext* context, OpKernel* kernel,
              const Tensor& in_grads, const Tensor& in_image,
              const Tensor& out_image, Tensor* output) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_12(mht_12_v, 681, "", "./tensorflow/core/kernels/lrn_op.cc", "launch");

#if GOOGLE_CUDA
    OP_REQUIRES(
        context, beta_ >= 0.01,
        errors::InvalidArgument("cuDNN requires beta >= 0.01, got: ", beta_));

    OP_REQUIRES(
        context, depth_radius_ > 0 && depth_radius_ <= 7,
        errors::InvalidArgument("cuDNN requires depth_radius in [1, 7], got: ",
                                depth_radius_));
    OP_REQUIRES(
        context, bias_ >= 1e-5,
        errors::InvalidArgument("cuDNN requires bias >= 1e-5, got: ", bias_));

    const int64_t batch = in_grads.dim_size(0);
    const int64_t rows = in_grads.dim_size(1);
    const int64_t cols = in_grads.dim_size(2);
    const int64_t depth = in_grads.dim_size(3);

    se::dnn::BatchDescriptor dimensions_desc;
    dimensions_desc.set_count(batch)
        .set_height(rows)
        .set_width(cols)
        .set_feature_map_count(depth)
        .set_layout(se::dnn::DataLayout::kBatchYXDepth);

    se::dnn::NormalizeDescriptor normalize_desc;
    normalize_desc.set_bias(bias_)
        .set_range(depth_radius_)
        .set_alpha(alpha_)
        .set_beta(beta_);

    auto input_grads_data = StreamExecutorUtil::AsDeviceMemory<T>(in_grads);
    auto input_image_data = StreamExecutorUtil::AsDeviceMemory<T>(in_image);
    auto output_image_data = StreamExecutorUtil::AsDeviceMemory<T>(out_image);
    auto output_grads_data = StreamExecutorUtil::AsDeviceMemory<T>(*output);

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    bool status =
        stream
            ->ThenNormalizeBackwardWithDimensions(
                normalize_desc, dimensions_desc, input_image_data,
                output_image_data, input_grads_data, &output_grads_data)
            .ok();
    OP_REQUIRES(
        context, status,
        errors::Internal("NormalizeBackwardWithDimensions launch failed"));
#elif TENSORFLOW_USE_ROCM
    // For NHWC input/output tensors, convert to NCHW because it's the only
    // supported format in MIOpen for now.
    const int64 batch = in_grads.dim_size(0);
    const int64 rows = in_grads.dim_size(1);
    const int64 cols = in_grads.dim_size(2);
    const int64 depth = in_grads.dim_size(3);

    Tensor transformed_in_grads;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                ShapeFromFormat(FORMAT_NCHW, in_grads.shape(),
                                                FORMAT_NHWC),
                                &transformed_in_grads));
    functor::NHWCToNCHW<GPUDevice, T, 4>()(context->eigen_device<GPUDevice>(),
                                           in_grads.tensor<T, 4>(),
                                           transformed_in_grads.tensor<T, 4>());

    Tensor transformed_in_image;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                ShapeFromFormat(FORMAT_NCHW, in_image.shape(),
                                                FORMAT_NHWC),
                                &transformed_in_image));
    functor::NHWCToNCHW<GPUDevice, T, 4>()(context->eigen_device<GPUDevice>(),
                                           in_image.tensor<T, 4>(),
                                           transformed_in_image.tensor<T, 4>());

    Tensor transformed_out_image;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                ShapeFromFormat(FORMAT_NCHW, out_image.shape(),
                                                FORMAT_NHWC),
                                &transformed_out_image));
    functor::NHWCToNCHW<GPUDevice, T, 4>()(
        context->eigen_device<GPUDevice>(), out_image.tensor<T, 4>(),
        transformed_out_image.tensor<T, 4>());

    Tensor transformed_output;
    OP_REQUIRES_OK(
        context, context->allocate_temp(
                     DataTypeToEnum<T>::value,
                     ShapeFromFormat(FORMAT_NCHW, output->shape(), FORMAT_NHWC),
                     &transformed_output));

    perftools::gputools::dnn::BatchDescriptor dimensions_desc;
    dimensions_desc.set_count(batch)
        .set_height(rows)
        .set_width(cols)
        .set_feature_map_count(depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    perftools::gputools::dnn::NormalizeDescriptor normalize_desc;
    normalize_desc.set_bias(bias_)
        .set_range(depth_radius_)
        .set_alpha(alpha_)
        .set_beta(beta_);

    auto input_grads_data =
        AsDeviceMemory(transformed_in_grads.template flat<T>().data(),
                       transformed_in_grads.template flat<T>().size());
    auto input_image_data =
        AsDeviceMemory(transformed_in_image.template flat<T>().data(),
                       transformed_in_image.template flat<T>().size());
    auto output_image_data =
        AsDeviceMemory(transformed_out_image.template flat<T>().data(),
                       transformed_out_image.template flat<T>().size());
    auto output_grads_data =
        AsDeviceMemory(transformed_output.template flat<T>().data(),
                       transformed_output.template flat<T>().size());

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    static int64 NormalizeBackwardScratchSize = GetDnnWorkspaceLimit(
        // default value is in bytes despite the name of the environment
        // variable
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
    );

    DnnScratchAllocator scratch_allocator(NormalizeBackwardScratchSize,
                                          context);
    bool status = stream
                      ->ThenNormalizeBackwardWithDimensions(
                          normalize_desc, dimensions_desc, input_image_data,
                          output_image_data, input_grads_data,
                          &output_grads_data, &scratch_allocator)
                      .ok();
    OP_REQUIRES(
        context, status,
        errors::Internal("NormalizeBackwardWithDimensions launch failed"));

    // Need to convert it back to NHWC once MIOpen kernels finishes.
    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        context->eigen_device<GPUDevice>(),
        toConstTensor(transformed_output).template tensor<T, 4>(),
        output->tensor<T, 4>());
#endif
  }

  int depth_radius_;
  T bias_;
  T alpha_;
  T beta_;
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T>
class LRNGradOp : public OpKernel {
 public:
  explicit LRNGradOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_13(mht_13_v, 845, "", "./tensorflow/core/kernels/lrn_op.cc", "LRNGradOp");

    int64_t depth_radius64;
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius64));
    OP_REQUIRES(
        context,
        FastBoundsCheck(depth_radius64, std::numeric_limits<int>::max()),
        errors::InvalidArgument("depth_radius = ", depth_radius64,
                                " larger than int max"));
    depth_radius_ = static_cast<int>(depth_radius64);
    float tmp;
    OP_REQUIRES_OK(context, context->GetAttr("bias", &tmp));
    bias_ = T(tmp);
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &tmp));
    alpha_ = T(tmp);
    OP_REQUIRES_OK(context, context->GetAttr("beta", &tmp));
    beta_ = T(tmp);
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSlrn_opDTcc mht_14(mht_14_v, 866, "", "./tensorflow/core/kernels/lrn_op.cc", "Compute");

    const Tensor& in_grads = context->input(0);
    const Tensor& in_image = context->input(1);
    const Tensor& out_image = context->input(2);

    OP_REQUIRES(context, in_grads.dims() == 4 && in_image.dims() == 4,
                errors::InvalidArgument("inputs must be 4-dimensional"));
    const int64_t batch = in_grads.dim_size(0);
    const int64_t rows = in_grads.dim_size(1);
    const int64_t cols = in_grads.dim_size(2);
    const int64_t depth = in_grads.dim_size(3);
    OP_REQUIRES(
        context,
        in_image.dim_size(0) == batch && in_image.dim_size(1) == rows &&
            in_image.dim_size(2) == cols && in_image.dim_size(3) == depth &&
            out_image.dim_size(0) == batch && out_image.dim_size(1) == rows &&
            out_image.dim_size(2) == cols && out_image.dim_size(3) == depth,
        errors::InvalidArgument(
            "input_grads, input_image, and out_image should have the same "
            "shape"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({batch, rows, cols, depth}), &output));

    LaunchLRNGrad<Device, T> launcher(depth_radius_, bias_, alpha_, beta_);
    launcher.launch(context, this, in_grads, in_image, out_image, output);
  }

 private:
  int depth_radius_;
  T bias_;
  T alpha_;
  T beta_;
};

#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("LRNGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LRNGradOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU);
TF_CALL_half(REGISTER_CPU);

#undef REGISTER_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("LRNGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LRNGradOp<GPUDevice, T>);
TF_CALL_float(REGISTER_GPU);

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // !defined(IS_MOBILE_PLATFORM)

}  // namespace tensorflow
