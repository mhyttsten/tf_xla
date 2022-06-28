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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/image/image_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace functor {

// Explicit instantiation of the CPU functor.
typedef Eigen::ThreadPoolDevice CPUDevice;

template struct FillProjectiveTransform<CPUDevice, uint8>;
template struct FillProjectiveTransform<CPUDevice, int32>;
template struct FillProjectiveTransform<CPUDevice, int64_t>;
template struct FillProjectiveTransform<CPUDevice, Eigen::half>;
template struct FillProjectiveTransform<CPUDevice, float>;
template struct FillProjectiveTransform<CPUDevice, double>;

}  // end namespace functor

typedef Eigen::ThreadPoolDevice CPUDevice;

using functor::FillProjectiveTransform;
using generator::Interpolation;
using generator::Mode;

template <typename Device, typename T>
void DoImageProjectiveTransformOp(OpKernelContext* ctx,
                                  const Interpolation& interpolation,
                                  const Mode& fill_mode) {
  const Tensor& images_t = ctx->input(0);
  const Tensor& transform_t = ctx->input(1);
  OP_REQUIRES(ctx, images_t.shape().dims() == 4,
              errors::InvalidArgument("Input images must have rank 4"));
  OP_REQUIRES(ctx,
              (TensorShapeUtils::IsMatrix(transform_t.shape()) &&
               (transform_t.dim_size(0) == images_t.dim_size(0) ||
                transform_t.dim_size(0) == 1) &&
               transform_t.dim_size(1) == 8),
              errors::InvalidArgument(
                  "Input transform should be num_images x 8 or 1 x 8"));

  int32_t out_height, out_width;
  // Kernel is shared by legacy "ImageProjectiveTransform" op with 2 args.
  if (ctx->num_inputs() >= 3) {
    const Tensor& shape_t = ctx->input(2);
    OP_REQUIRES(ctx, shape_t.dims() == 1,
                errors::InvalidArgument("output shape must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(ctx, shape_t.NumElements() == 2,
                errors::InvalidArgument("output shape must have two elements",
                                        shape_t.shape().DebugString()));
    auto shape_vec = shape_t.vec<int32>();
    out_height = shape_vec(0);
    out_width = shape_vec(1);
    OP_REQUIRES(ctx, out_height > 0 && out_width > 0,
                errors::InvalidArgument("output dimensions must be positive"));
  } else {
    // Shape is N (batch size), H (height), W (width), C (channels).
    out_height = images_t.shape().dim_size(1);
    out_width = images_t.shape().dim_size(2);
  }

  T fill_value(0);
  // Kernel is shared by "ImageProjectiveTransformV2" with 3 args.
  if (ctx->num_inputs() >= 4) {
    const Tensor& fill_value_t = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(fill_value_t.shape()),
                errors::InvalidArgument("fill_value must be a scalar",
                                        fill_value_t.shape().DebugString()));
    fill_value = static_cast<T>(*(fill_value_t.scalar<float>().data()));
  }

  Tensor* output_t;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_output(0,
                                TensorShape({images_t.dim_size(0), out_height,
                                             out_width, images_t.dim_size(3)}),
                                &output_t));
  auto output = output_t->tensor<T, 4>();
  auto images = images_t.tensor<T, 4>();
  auto transform = transform_t.matrix<float>();

  (FillProjectiveTransform<Device, T>(interpolation))(
      ctx->eigen_device<Device>(), &output, images, transform, fill_mode,
      fill_value);
}

template <typename Device, typename T>
class ImageProjectiveTransformV2 : public OpKernel {
 private:
  Interpolation interpolation_;
  Mode fill_mode_;

 public:
  explicit ImageProjectiveTransformV2(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTcc mht_0(mht_0_v, 290, "", "./tensorflow/core/kernels/image/image_ops.cc", "ImageProjectiveTransformV2");

    string interpolation_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("interpolation", &interpolation_str));
    if (interpolation_str == "NEAREST") {
      interpolation_ = Interpolation::NEAREST;
    } else if (interpolation_str == "BILINEAR") {
      interpolation_ = Interpolation::BILINEAR;
    } else {
      LOG(ERROR) << "Invalid interpolation " << interpolation_str
                 << ". Supported types: NEAREST, BILINEAR";
    }
    string mode_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_mode", &mode_str));
    if (mode_str == "REFLECT") {
      fill_mode_ = Mode::FILL_REFLECT;
    } else if (mode_str == "WRAP") {
      fill_mode_ = Mode::FILL_WRAP;
    } else if (mode_str == "CONSTANT") {
      fill_mode_ = Mode::FILL_CONSTANT;
    } else if (mode_str == "NEAREST") {
      fill_mode_ = Mode::FILL_NEAREST;
    } else {
      LOG(ERROR) << "Invalid mode " << mode_str
                 << ". Supported types: REFLECT, WRAP, CONSTANT, NEAREST";
    }
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTcc mht_1(mht_1_v, 320, "", "./tensorflow/core/kernels/image/image_ops.cc", "Compute");

    DoImageProjectiveTransformOp<Device, T>(ctx, interpolation_, fill_mode_);
  }
};

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransformV2")  \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          ImageProjectiveTransformV2<CPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

template <typename Device, typename T>
class ImageProjectiveTransformV3
    : public ImageProjectiveTransformV2<Device, T> {
 public:
  explicit ImageProjectiveTransformV3(OpKernelConstruction* ctx)
      : ImageProjectiveTransformV2<Device, T>(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSimage_opsDTcc mht_2(mht_2_v, 348, "", "./tensorflow/core/kernels/image/image_ops.cc", "ImageProjectiveTransformV3");
}
};

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransformV3")  \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          ImageProjectiveTransformV3<CPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;
typedef generator::Mode Mode;

namespace functor {

// NOTE(ringwalt): We get an undefined symbol error if we don't explicitly
// instantiate the operator() in GCC'd code.
#define DECLARE_PROJECT_FUNCTOR(TYPE)                                       \
  template <>                                                               \
  void FillProjectiveTransform<GPUDevice, TYPE>::operator()(                \
      const GPUDevice& device, OutputType* output, const InputType& images, \
      const TransformsType& transform, const Mode fill_mode,                \
      const TYPE fill_value) const;                                         \
  extern template struct FillProjectiveTransform<GPUDevice, TYPE>

TF_CALL_uint8(DECLARE_PROJECT_FUNCTOR);
TF_CALL_int32(DECLARE_PROJECT_FUNCTOR);
TF_CALL_int64(DECLARE_PROJECT_FUNCTOR);
TF_CALL_half(DECLARE_PROJECT_FUNCTOR);
TF_CALL_float(DECLARE_PROJECT_FUNCTOR);
TF_CALL_double(DECLARE_PROJECT_FUNCTOR);

}  // end namespace functor

namespace generator {

#define DECLARE_MAP_FUNCTOR(Mode)                                         \
  template <>                                                             \
  float MapCoordinate<GPUDevice, Mode>::operator()(const float out_coord, \
                                                   const DenseIndex len); \
  extern template struct MapCoordinate<GPUDevice, Mode>

DECLARE_MAP_FUNCTOR(Mode::FILL_REFLECT);
DECLARE_MAP_FUNCTOR(Mode::FILL_WRAP);
DECLARE_MAP_FUNCTOR(Mode::FILL_CONSTANT);
DECLARE_MAP_FUNCTOR(Mode::FILL_NEAREST);

}  // end namespace generator

#define REGISTER(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransformV2") \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<TYPE>("dtype") \
                              .HostMemory("output_shape"),   \
                          ImageProjectiveTransformV2<GPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#define REGISTER(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransformV3") \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<TYPE>("dtype") \
                              .HostMemory("output_shape")    \
                              .HostMemory("fill_value"),     \
                          ImageProjectiveTransformV3<GPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
