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
class MHTracer_DTPStensorflowPScorePSkernelsPSunpack_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSunpack_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSunpack_opDTcc() {
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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class UnpackOp : public OpKernel {
 public:
  explicit UnpackOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunpack_opDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/unpack_op.cc", "UnpackOp");

    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunpack_opDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/kernels/unpack_op.cc", "Compute");

    const int32_t num = num_outputs();
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();

    int axis = axis_;
    if (axis < 0) axis += input_shape.dims();

    OP_REQUIRES(context, 0 <= axis && axis < input_shape.dims(),
                errors::InvalidArgument("axis = ", axis_, " not in [",
                                        -input_shape.dims(), ", ",
                                        input_shape.dims(), ")"));

    OP_REQUIRES(
        context, input_shape.dims() > 0 && input_shape.dim_size(axis) == num,
        errors::InvalidArgument("Input shape axis ", axis, " must equal ", num,
                                ", got shape ", input_shape.DebugString()));

    auto output_shape = input_shape;
    output_shape.RemoveDim(axis);
    const int64_t output_size = output_shape.num_elements();
    OP_REQUIRES(
        context,
        FastBoundsCheck(output_size,
                        std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("output size must fit in Eigen DenseIndex"));

    // Special case: Aligned, so we can share the underlying buffer.
    //
    // Apply this optimization conservatively: if input is aligned,
    // the resulting tensors must be aligned. It's conservative
    // because if the immediate consumer of the resulting tensors are
    // not using eigen for computation, its perfectly fine to avoid
    // the copying.
    if (axis == 0 &&
        (output_size == 0 || IsInnerDimsSizeAligned<T>(input_shape))) {
      for (int i = 0; i < num; ++i) {
        Tensor output;
        CHECK(output.CopyFrom(input.Slice(i, i + 1), output_shape));
        context->set_output(i, output);
      }
      return;
    }

    Eigen::DenseIndex before_dim = 1;
    for (int i = 0; i < axis; ++i) {
      before_dim *= input_shape.dim_size(i);
    }

    Eigen::DenseIndex after_dim = 1;
    for (int i = axis + 1; i < input_shape.dims(); ++i) {
      after_dim *= input_shape.dim_size(i);
    }
    const Eigen::DenseIndex axis_dim = input_shape.dim_size(axis);

    // Except for shape, unpack is a special case of split, so we reuse the
    // same computational kernels.
    auto input_reshaped =
        input.shaped<T, 2>({before_dim, axis_dim * after_dim});

    for (int i = 0; i < num; ++i) {
      if (!context->output_required(i)) continue;

      Tensor* output;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, output_shape, &output));

      if (output_shape.num_elements() > 0) {
        auto output_shaped = output->shaped<T, 2>({before_dim, after_dim});
        Eigen::DSizes<Eigen::DenseIndex, 2> indices{0, i * after_dim};
        Eigen::DSizes<Eigen::DenseIndex, 2> sizes{before_dim, after_dim};
        functor::Split<Device, T, 2>()(context->eigen_device<Device>(),
                                       output_shaped, input_reshaped, indices,
                                       sizes);
      }
    }
  }

 private:
  int axis_;
};

#define REGISTER_UNPACK(type)                                      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Unpack").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      UnpackOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_UNPACK);

#undef REGISTER_UNPACK

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Unpack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      UnpackOp<GPUDevice, type>)

TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_uint8(REGISTER_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// A special DEVICE_DEFAULT kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Unpack")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("value")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        UnpackOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("Unpack")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("value")
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("T"),
                        UnpackOp<CPUDevice, int64>);

}  // end namespace tensorflow
