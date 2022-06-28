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
class MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc() {
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

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/fake_quant_ops_functor.h"
// Above is the related header but clang tidy doesn't recognize it.
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/determinism.h"

using tensorflow::BinaryElementWiseOp;
using tensorflow::DEVICE_CPU;
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
using tensorflow::DEVICE_GPU;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TTypes;  // NOLINT This is needed in CUDA mode, do not remove.
using tensorflow::UnaryElementWiseOp;
using tensorflow::errors::InvalidArgument;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

auto* using_fake_quant = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/api/op/using_fake_quantization",
    "True if a fake_quant op is created.");

#define SET_USING_FAKE_QUANT() using_fake_quant->GetCell()->Set(true)

namespace {
bool IsNumBitsValid(int num_bits) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_0(mht_0_v, 227, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "IsNumBitsValid");
 return num_bits >= 2 && num_bits <= 16; }
}  // namespace

// -----------------------------------------------------------------------------
// Implementation of FakeQuantWithMinMaxArgsOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxArgsOp
    : public UnaryElementWiseOp<float, FakeQuantWithMinMaxArgsOp<Device>> {
 public:
  typedef UnaryElementWiseOp<float, FakeQuantWithMinMaxArgsOp<Device>> Base;
  explicit FakeQuantWithMinMaxArgsOp(OpKernelConstruction* context)
      : Base::UnaryElementWiseOp(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "FakeQuantWithMinMaxArgsOp");

    OP_REQUIRES_OK(context, context->GetAttr("min", &min_));
    OP_REQUIRES_OK(context, context->GetAttr("max", &max_));
    OP_REQUIRES(context, min_ < max_,
                InvalidArgument("min has to be smaller than max, was: ", min_,
                                " >= ", max_));
    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
    SET_USING_FAKE_QUANT();
  }

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_2(mht_2_v, 263, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "Operate");

    FakeQuantWithMinMaxArgsFunctor<Device> functor;
    functor(context->eigen_device<Device>(), input.flat<float>(), min_, max_,
            quant_min_, quant_max_, output->flat<float>());
  }

 private:
  float min_;
  float max_;
  int quant_min_;
  int quant_max_;
};

// Implementation of FakeQuantWithMinMaxArgsGradientOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxArgsGradientOp
    : public BinaryElementWiseOp<float,
                                 FakeQuantWithMinMaxArgsGradientOp<Device>> {
 public:
  typedef BinaryElementWiseOp<float, FakeQuantWithMinMaxArgsGradientOp<Device>>
      Base;
  explicit FakeQuantWithMinMaxArgsGradientOp(OpKernelConstruction* context)
      : Base::BinaryElementWiseOp(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_3(mht_3_v, 289, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "FakeQuantWithMinMaxArgsGradientOp");

    OP_REQUIRES_OK(context, context->GetAttr("min", &min_));
    OP_REQUIRES_OK(context, context->GetAttr("max", &max_));
    OP_REQUIRES(context, min_ < max_,
                InvalidArgument("min has to be smaller than max, was: ", min_,
                                " >= ", max_));
    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
  }

  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& gradient,
               const Tensor& input, Tensor* output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_4(mht_4_v, 311, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "Operate");

    OperateNoTemplate(context, gradient, input, output);
  }

  void OperateNoTemplate(OpKernelContext* context, const Tensor& gradient,
                         const Tensor& input, Tensor* output) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_5(mht_5_v, 319, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "OperateNoTemplate");

    OP_REQUIRES(context, input.IsSameSize(gradient),
                InvalidArgument("gradient and input must be the same size"));
    FakeQuantWithMinMaxArgsGradientFunctor<Device> functor;
    functor(context->eigen_device<Device>(), gradient.flat<float>(),
            input.flat<float>(), min_, max_, quant_min_, quant_max_,
            output->flat<float>());
  }

 private:
  float min_;
  float max_;
  int quant_min_;
  int quant_max_;
};

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxArgs").Device(DEVICE_CPU),
                        FakeQuantWithMinMaxArgsOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxArgsGradient").Device(DEVICE_CPU),
    FakeQuantWithMinMaxArgsGradientOp<CPUDevice>);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
typedef Eigen::GpuDevice GPUDevice;

// Forward declarations for functor specializations for GPU.
template <>
void FakeQuantWithMinMaxArgsFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstFlat inputs,
    const float min, const float max, const int quant_min, const int quant_max,
    typename TTypes<float>::Flat outputs);
extern template struct FakeQuantWithMinMaxArgsFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxArgs").Device(DEVICE_GPU),
                        FakeQuantWithMinMaxArgsOp<GPUDevice>);

template <>
void FakeQuantWithMinMaxArgsGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstFlat gradients,
    typename TTypes<float>::ConstFlat inputs, const float min, const float max,
    const int quant_min, const int quant_max,
    typename TTypes<float>::Flat backprops);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxArgsGradient").Device(DEVICE_GPU),
    FakeQuantWithMinMaxArgsGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// -----------------------------------------------------------------------------
// Implementation of FakeQuantWithMinMaxVarsOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsOp(OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_6(mht_6_v, 376, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "FakeQuantWithMinMaxVarsOp");

    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
    SET_USING_FAKE_QUANT();
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_7(mht_7_v, 392, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "Compute");

    CHECK_EQ(3, context->num_inputs());
    const Tensor& input = context->input(0);
    const Tensor& min = context->input(1);
    const Tensor& max = context->input(2);

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    FakeQuantWithMinMaxVarsFunctor<Device> functor;
    functor(context->eigen_device<Device>(), input.flat<float>(),
            min.scalar<float>(), max.scalar<float>(), quant_min_, quant_max_,
            output->flat<float>());
  }

 private:
  int quant_min_;
  int quant_max_;
};

// Implementation of FakeQuantWithMinMaxVarsGradientOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsGradientOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsGradientOp(OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_8(mht_8_v, 422, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "FakeQuantWithMinMaxVarsGradientOp");

    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
    if (std::is_same<Device, Eigen::GpuDevice>::value) {
      OP_REQUIRES(
          context, !OpDeterminismRequired(),
          errors::Unimplemented(
              "Determinism is not yet supported in GPU implementation of "
              "FakeQuantWithMinMaxVarsGradient."));
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_9(mht_9_v, 444, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "Compute");

    CHECK_EQ(4, context->num_inputs());
    const Tensor& gradient = context->input(0);
    const Tensor& input = context->input(1);
    OP_REQUIRES(context, input.IsSameSize(gradient),
                InvalidArgument("gradient and input must be the same size"));
    const Tensor& min = context->input(2);
    const Tensor& max = context->input(3);

    Tensor* grad_wrt_input;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &grad_wrt_input));

    TensorShape scalar_shape;
    Tensor* grad_wrt_min;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, scalar_shape, &grad_wrt_min));

    Tensor* grad_wrt_max;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, scalar_shape, &grad_wrt_max));

    FakeQuantWithMinMaxVarsGradientFunctor<Device> functor;
    functor(context->eigen_device<Device>(), gradient.flat<float>(),
            input.flat<float>(), min.scalar<float>(), max.scalar<float>(),
            quant_min_, quant_max_, grad_wrt_input->flat<float>(),
            grad_wrt_min->scalar<float>(), grad_wrt_max->scalar<float>());
  }

 private:
  int quant_min_;
  int quant_max_;
};

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVars").Device(DEVICE_CPU),
                        FakeQuantWithMinMaxVarsOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxVarsGradient").Device(DEVICE_CPU),
    FakeQuantWithMinMaxVarsGradientOp<CPUDevice>);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
template <>
void FakeQuantWithMinMaxVarsFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstScalar min,
    typename TTypes<float>::ConstScalar max, const int quant_min,
    const int quant_max, typename TTypes<float>::Flat output);
extern template struct FakeQuantWithMinMaxVarsFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVars")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsOp<GPUDevice>);

template <>
void FakeQuantWithMinMaxVarsGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstFlat gradients,
    typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstScalar min,
    typename TTypes<float>::ConstScalar max, const int quant_min,
    const int quant_max, typename TTypes<float>::Flat backprops_wrt_input,
    typename TTypes<float>::Scalar backprop_wrt_min,
    typename TTypes<float>::Scalar backprop_wrt_max);
extern template struct FakeQuantWithMinMaxVarsGradientFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsGradient")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// -----------------------------------------------------------------------------
// Implementation of FakeQuantWithMinMaxVarsPerChannelOp, see its documentation
// in core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsPerChannelOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelOp(OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_10(mht_10_v, 526, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "FakeQuantWithMinMaxVarsPerChannelOp");

    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
    SET_USING_FAKE_QUANT();
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_11(mht_11_v, 542, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "Compute");

    CHECK_EQ(3, context->num_inputs());
    const Tensor& input = context->input(0);
    const int depth = input.dim_size(input.dims() - 1);  // last dimension size.
    const Tensor& min = context->input(1);
    OP_REQUIRES(context, min.dim_size(0) == depth,
                InvalidArgument("min has incorrect size, expected ", depth,
                                " was ", min.dim_size(0)));
    const Tensor& max = context->input(2);
    OP_REQUIRES(context, max.dim_size(0) == depth,
                InvalidArgument("max has incorrect size, expected ", depth,
                                " was ", max.dim_size(0)));

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    FakeQuantWithMinMaxVarsPerChannelFunctor<Device> functor;
    functor(context->eigen_device<Device>(), input.flat_inner_dims<float, 2>(),
            min.vec<float>(), max.vec<float>(), quant_min_, quant_max_,
            output->flat_inner_dims<float, 2>());
  }

 private:
  int quant_min_;
  int quant_max_;
};

// Implementation of FakeQuantWithMinMaxVarsPerChannelGradientOp, see its
// documentation in core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsPerChannelGradientOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelGradientOp(
      OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_12(mht_12_v, 580, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "FakeQuantWithMinMaxVarsPerChannelGradientOp");

    int num_bits;
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits));
    OP_REQUIRES(
        context, IsNumBitsValid(num_bits),
        InvalidArgument("num_bits must be between 2 and 16, inclusive"));
    bool narrow_range;
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range));
    quant_min_ = narrow_range ? 1 : 0;
    quant_max_ = (1 << num_bits) - 1;
    if (std::is_same<Device, Eigen::GpuDevice>::value) {
      OP_REQUIRES(
          context, !OpDeterminismRequired(),
          errors::Unimplemented(
              "Determinism is not yet supported in GPU implementation of "
              "FakeQuantWithMinMaxVarsPerChannelGradient."));
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfake_quant_opsDTcc mht_13(mht_13_v, 602, "", "./tensorflow/core/kernels/fake_quant_ops.cc", "Compute");

    CHECK_EQ(4, context->num_inputs());
    const Tensor& gradient = context->input(0);
    const Tensor& input = context->input(1);
    OP_REQUIRES(context, input.IsSameSize(gradient),
                InvalidArgument("gradient and input must be the same size"));
    const int depth = input.dim_size(input.dims() - 1);  // last dimension size.
    const Tensor& min = context->input(2);
    OP_REQUIRES(context, min.dim_size(0) == depth,
                InvalidArgument("min has incorrect size, expected ", depth,
                                " was ", min.dim_size(0)));
    const Tensor& max = context->input(3);
    OP_REQUIRES(context, max.dim_size(0) == depth,
                InvalidArgument("max has incorrect size, expected ", depth,
                                " was ", max.dim_size(0)));

    Tensor* grad_wrt_input;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &grad_wrt_input));

    TensorShape min_max_shape({input.dim_size(input.dims() - 1)});
    Tensor* grad_wrt_min;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, min_max_shape, &grad_wrt_min));

    Tensor* grad_wrt_max;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, min_max_shape, &grad_wrt_max));

    FakeQuantWithMinMaxVarsPerChannelGradientFunctor<Device> functor;
    functor(
        context->eigen_device<Device>(), gradient.flat_inner_dims<float, 2>(),
        input.flat_inner_dims<float, 2>(), min.vec<float>(), max.vec<float>(),
        quant_min_, quant_max_, grad_wrt_input->flat_inner_dims<float, 2>(),
        grad_wrt_min->vec<float>(), grad_wrt_max->vec<float>());
  }

 private:
  int quant_min_;
  int quant_max_;
};

REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxVarsPerChannel").Device(DEVICE_CPU),
    FakeQuantWithMinMaxVarsPerChannelOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxVarsPerChannelGradient").Device(DEVICE_CPU),
    FakeQuantWithMinMaxVarsPerChannelGradientOp<CPUDevice>);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
template <>
void FakeQuantWithMinMaxVarsPerChannelFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstMatrix inputs,
    typename TTypes<float>::ConstFlat min,
    typename TTypes<float>::ConstFlat max, const int quant_min,
    const int quant_max, typename TTypes<float>::Matrix outputs);
extern template struct FakeQuantWithMinMaxVarsPerChannelFunctor<GPUDevice>;

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsPerChannel")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsPerChannelOp<GPUDevice>);

template <>
void FakeQuantWithMinMaxVarsPerChannelGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<float>::ConstMatrix gradients,
    typename TTypes<float>::ConstMatrix inputs,
    typename TTypes<float>::ConstVec min, typename TTypes<float>::ConstVec max,
    const int quant_min, const int quant_max,
    typename TTypes<float>::Matrix backprops_wrt_input,
    typename TTypes<float>::Vec backprop_wrt_min,
    typename TTypes<float>::Vec backprop_wrt_max);
extern template struct FakeQuantWithMinMaxVarsPerChannelGradientFunctor<
    GPUDevice>;

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsPerChannelGradient")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsPerChannelGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
