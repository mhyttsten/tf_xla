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
class MHTracer_DTPStensorflowPScorePSkernelsPSdepthtospace_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdepthtospace_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdepthtospace_opDTcc() {
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

#include "tensorflow/core/kernels/depthtospace_op.h"

#include <memory>
#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class DepthToSpaceOp : public OpKernel {
 public:
  explicit DepthToSpaceOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdepthtospace_opDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/depthtospace_op.cc", "DepthToSpaceOp");

    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, context->GetAttr("block_size", &block_size_));
    OP_REQUIRES(context, block_size_ > 1,
                errors::InvalidArgument("Block size should be > 1, but was: ",
                                        block_size_));

    if (std::is_same<Device, CPUDevice>::value) {
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Only NHWC data_format supported on CPU. Got ", data_format_str));
    }
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdepthtospace_opDTcc mht_1(mht_1_v, 237, "", "./tensorflow/core/kernels/depthtospace_op.cc", "Compute");

    const Tensor& input = context->input(0);
    const int dims = input.dims();

    // Assuming qint8 <--> NCHW_VECT_C, OIHW_VECT_I (int8x4) here.
    constexpr bool is_int8x4 = std::is_same<T, qint8>::value;
    OP_REQUIRES(context, (is_int8x4 == (data_format_ == FORMAT_NCHW_VECT_C)),
                errors::InvalidArgument(
                    "qint8 should be used with data_format NCHW_VECT_C."));

    constexpr int kVect = is_int8x4 ? 4 : 1;
    constexpr int kDims = is_int8x4 ? 5 : 4;
    OP_REQUIRES(context, kDims == dims,
                errors::InvalidArgument("Input rank should be: ", kDims,
                                        " instead of: ", dims));

    constexpr int kNumSpatialDims = 2;
    const int batch_size =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'N'));
    const int input_height =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'H'));
    const int input_width =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'W'));
    const int input_depth =
        input.dim_size(GetTensorDimIndex<kNumSpatialDims>(data_format_, 'C')) *
        kVect;

    const int block_size_sq = block_size_ * block_size_;

    // The depth must be divisible by block_size_ * block_size_
    OP_REQUIRES(
        context, input_depth % block_size_sq == 0,
        errors::InvalidArgument("Input depth dimension ", input_depth,
                                " should be divisible by: ", block_size_sq));

    const int output_depth = input_depth / block_size_sq;
    const int output_width = input_width * block_size_;
    const int output_height = input_height * block_size_;

    // Allocate output tensor.
    Tensor* outputs_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0,
                       ShapeFromFormat(data_format_, batch_size, output_height,
                                       output_width, output_depth),
                       &outputs_tensor));
    auto Tinput = input.tensor<T, kDims>();
    auto Toutput = outputs_tensor->tensor<T, kDims>();

    if (std::is_same<Device, GPUDevice>::value) {
      if (is_int8x4) {
        // NCHW_VECT_C with 4 x qint8 can be treated as NCHW int32.
        auto Tinput_v = input.template reinterpret_last_dimension<int32, 4>();
        auto Toutput_v = outputs_tensor->reinterpret_last_dimension<int32, 4>();
        functor::DepthToSpaceOpFunctor<Device, int32, FORMAT_NCHW> functor;
        functor(context->eigen_device<Device>(), Tinput_v, block_size_,
                Toutput_v);
        return;
      } else if (data_format_ == FORMAT_NCHW) {
        functor::DepthToSpaceOpFunctor<Device, T, FORMAT_NCHW> functor;
        functor(context->eigen_device<Device>(), Tinput, block_size_, Toutput);
        return;
      }
    }

    // NOTE: Assumes data_format_ == FORMAT_NHWC here, since we have rejected
    // (CPU && data_format_ != FORMAT_NHWC) in the constructor.

    if (!is_int8x4) {
      functor::DepthToSpaceOpFunctor<Device, T, FORMAT_NHWC> functor;
      functor(context->eigen_device<Device>(), Tinput, block_size_, Toutput);
    }
  };

 private:
  int block_size_;
  TensorFormat data_format_;
};

// Partial specialization of DepthToSpaceOpFunctor for a CPUDevice
// with FORMAT_NHWC.
namespace functor {
template <typename T>
struct DepthToSpaceOpFunctor<CPUDevice, T, FORMAT_NHWC> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int output_height = output.dimension(1);
    const int output_width = output.dimension(2);
    const int output_depth = output.dimension(3);

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < output_height; ++h) {
        const int in_h = h / block_size;
        const int offset_h = (h % block_size);
        for (int w = 0; w < output_width; ++w) {
          const int in_w = w / block_size;
          const int offset_w = (w % block_size);
          const int offset_d =
              (offset_h * block_size + offset_w) * output_depth;
          for (int d = 0; d < output_depth; ++d) {
            const int in_d = d + offset_d;
            output(b, h, w, d) = input(b, in_h, in_w, in_d);
          }
        }
      }
    }
  }
};
}  // namespace functor

#define REGISTER(type)                                                \
  REGISTER_KERNEL_BUILDER(Name("DepthToSpace")                        \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T")              \
                              .AttrConstraint("data_format", "NHWC"), \
                          DepthToSpaceOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNEL_BUILDER(
    Name("DepthToSpace").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    DepthToSpaceOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("DepthToSpace").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    DepthToSpaceOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("DepthToSpace").Device(DEVICE_GPU).TypeConstraint<qint8>("T"),
    DepthToSpaceOp<GPUDevice, qint8>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow
