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
class MHTracer_DTPStensorflowPScorePSkernelsPSsequence_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsequence_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsequence_opsDTcc() {
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

// See docs in ../ops/math_ops.cc.

#include "tensorflow/core/kernels/sequence_ops.h"

#include <cmath>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
struct RangeFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, int64_t size, T start, T delta,
                  typename TTypes<T>::Flat output) const {
    (void)context;
    T val = start;
    for (int64_t i = 0; i < size; ++i) {
      output(i) = T(val);
      val += delta;
    }
  }
};

}  // namespace functor

template <typename Device, typename T>
class RangeOp : public OpKernel {
 public:
  explicit RangeOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsequence_opsDTcc mht_0(mht_0_v, 222, "", "./tensorflow/core/kernels/sequence_ops.cc", "RangeOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsequence_opsDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/kernels/sequence_ops.cc", "Compute");

    const Tensor& start_in = context->input(0);
    const Tensor& limit_in = context->input(1);
    const Tensor& delta_in = context->input(2);
    // TODO(rmlarsen): Disallow legacy use of length-1 vectors as scalars.
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(start_in.shape()) ||
                    (TensorShapeUtils::IsVector(start_in.shape()) &&
                     start_in.shape().dim_size(0) == 1),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in.shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(limit_in.shape()) ||
                    (TensorShapeUtils::IsVector(limit_in.shape()) &&
                     limit_in.shape().dim_size(0) == 1),
                errors::InvalidArgument("limit must be a scalar, not shape ",
                                        limit_in.shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(delta_in.shape()) ||
                    (TensorShapeUtils::IsVector(delta_in.shape()) &&
                     delta_in.shape().dim_size(0) == 1),
                errors::InvalidArgument("delta must be a scalar, not shape ",
                                        delta_in.shape().DebugString()));
    const T start = start_in.scalar<T>()();
    const T limit = limit_in.scalar<T>()();
    const T delta = delta_in.scalar<T>()();
    OP_REQUIRES(context, delta != 0,
                errors::InvalidArgument("Requires delta != 0: ", delta));
    if (delta > 0) {
      OP_REQUIRES(
          context, start <= limit,
          errors::InvalidArgument(
              "Requires start <= limit when delta > 0: ", start, "/", limit));
    } else {
      OP_REQUIRES(
          context, start >= limit,
          errors::InvalidArgument(
              "Requires start >= limit when delta < 0: ", start, "/", limit));
    }
    auto size_auto = (std::is_integral<T>::value
                          ? (Eigen::numext::abs(limit - start) +
                             Eigen::numext::abs(delta) - T(1)) /
                                Eigen::numext::abs(delta)
                          : Eigen::numext::ceil(
                                Eigen::numext::abs((limit - start) / delta)));
    OP_REQUIRES(
        context, size_auto <= std::numeric_limits<int64_t>::max(),
        errors::InvalidArgument("Requires ((limit - start) / delta) <= ",
                                std::numeric_limits<int64_t>::max()));

    int64_t size = static_cast<int64_t>(size_auto);

    TensorShape shape;
    OP_REQUIRES_OK(context, shape.AddDimWithStatus(size));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &out));
    if (size == 0) return;
    auto flat = out->flat<T>();
    functor::RangeFunctor<Device, T>()(context, size, start, delta, flat);
  }
};

#define REGISTER_KERNEL(DEV, DEV_TYPE, TYPE)                 \
  REGISTER_KERNEL_BUILDER(Name("Range")                      \
                              .Device(DEV)                   \
                              .HostMemory("start")           \
                              .HostMemory("limit")           \
                              .HostMemory("delta")           \
                              .TypeConstraint<TYPE>("Tidx"), \
                          RangeOp<DEV_TYPE, TYPE>);

#define REGISTER_CPU_KERNEL(T) REGISTER_KERNEL(DEVICE_CPU, CPUDevice, T)
#define REGISTER_GPU_KERNEL(T) REGISTER_KERNEL(DEVICE_GPU, GPUDevice, T)

TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
TF_CALL_int32(REGISTER_CPU_KERNEL);
TF_CALL_int64(REGISTER_CPU_KERNEL);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Special case to execute int32 on the host with host output.
REGISTER_KERNEL_BUILDER(Name("Range")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("start")
                            .HostMemory("limit")
                            .HostMemory("delta")
                            .HostMemory("output")
                            .TypeConstraint<int32_t>("Tidx"),
                        RangeOp<CPUDevice, int32_t>);

#undef REGISTER_KERNEL
#undef REGISTER_CPU_KERNEL
#undef REGISTER_GPU_KERNEL

template <typename T, typename Tnum>
class LinSpaceOp : public OpKernel {
 public:
  explicit LinSpaceOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsequence_opsDTcc mht_2(mht_2_v, 334, "", "./tensorflow/core/kernels/sequence_ops.cc", "LinSpaceOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsequence_opsDTcc mht_3(mht_3_v, 339, "", "./tensorflow/core/kernels/sequence_ops.cc", "Compute");

    const Tensor& start_in = context->input(0);
    const Tensor& stop_in = context->input(1);
    const Tensor& num_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(start_in.shape()),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(stop_in.shape()),
                errors::InvalidArgument("stop must be a scalar, not shape ",
                                        stop_in.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_in.shape()),
                errors::InvalidArgument("num must be a scalar, not shape ",
                                        num_in.shape().DebugString()));
    const T start = start_in.scalar<T>()();
    const T stop = stop_in.scalar<T>()();
    const Tnum num = num_in.scalar<Tnum>()();
    OP_REQUIRES(context, num > 0,
                errors::InvalidArgument("Requires num > 0: ", num));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num}), &out));
    auto flat = out->flat<T>();
    flat(0) = start;
    if (num > 1) {
      const T step = (stop - start) / (num - 1);
      for (Tnum i = 1; i < num - 1; ++i) flat(i) = start + step * i;
      // Ensure final value == stop; float arithmetic won't guarantee this.
      flat(num - 1) = stop;
    }
  }
};

#define REGISTER_KERNEL(DEV, T, Tidx)                       \
  REGISTER_KERNEL_BUILDER(Name("LinSpace")                  \
                              .Device(DEV)                  \
                              .TypeConstraint<T>("T")       \
                              .TypeConstraint<Tidx>("Tidx") \
                              .HostMemory("start")          \
                              .HostMemory("stop")           \
                              .HostMemory("num")            \
                              .HostMemory("output"),        \
                          LinSpaceOp<T, Tidx>);

#define REGISTER_KERNEL_ALL_NUMS(dev, T) \
  REGISTER_KERNEL(dev, T, int32);        \
  REGISTER_KERNEL(dev, T, int64_t)

#define REGISTER_CPU_KERNEL(T) REGISTER_KERNEL_ALL_NUMS(DEVICE_CPU, T)
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);

#define REGISTER_DEFAULT_KERNEL(T) REGISTER_KERNEL_ALL_NUMS(DEVICE_DEFAULT, T)
TF_CALL_float(REGISTER_DEFAULT_KERNEL);
TF_CALL_double(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

#undef REGISTER_CPU_KERNEL
#undef REGISTER_KERNEL_ALL_NUMS
#undef REGISTER_KERNEL

}  // namespace tensorflow
