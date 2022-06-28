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
class MHTracer_DTPStensorflowPScorePSkernelsPShistogram_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPShistogram_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPShistogram_opDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/histogram_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T, typename Tout>
struct HistogramFixedWidthFunctor<CPUDevice, T, Tout> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<T, 1>::ConstTensor& values,
                        const typename TTypes<T, 1>::ConstTensor& value_range,
                        int32_t nbins, typename TTypes<Tout, 1>::Tensor& out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPShistogram_opDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/histogram_op.cc", "Compute");

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    Tensor index_to_bin_tensor;

    TF_RETURN_IF_ERROR(context->forward_input_or_allocate_temp(
        {0}, DataTypeToEnum<int32>::value, TensorShape({values.size()}),
        &index_to_bin_tensor));
    auto index_to_bin = index_to_bin_tensor.flat<int32>();

    const double step = static_cast<double>(value_range(1) - value_range(0)) /
                        static_cast<double>(nbins);
    const double nbins_minus_1 = static_cast<double>(nbins - 1);

    // We cannot handle NANs in the algorithm below (due to the case to int32)
    const Eigen::Tensor<int32, 1, 1> nans_tensor =
        values.isnan().template cast<int32>();
    const Eigen::Tensor<int32, 0, 1> reduced_tensor = nans_tensor.sum();
    const int num_nans = reduced_tensor(0);
    if (num_nans > 0) {
      return errors::InvalidArgument("Histogram values must not contain NaN");
    }

    // The calculation is done by finding the slot of each value in `values`.
    // With [a, b]:
    //   step = (b - a) / nbins
    //   (x - a) / step
    // , then the entries are mapped to output.

    // Bug fix: Switch the order of cwiseMin and int32-casting to avoid
    // producing a negative index when casting an big int64 number to int32
    index_to_bin.device(d) =
        ((values.cwiseMax(value_range(0)) - values.constant(value_range(0)))
             .template cast<double>() /
         step)
            .cwiseMin(nbins_minus_1)
            .template cast<int32>();

    out.setZero();
    for (int32_t i = 0; i < index_to_bin.size(); i++) {
      out(index_to_bin(i)) += Tout(1);
    }
    return Status::OK();
  }
};

}  // namespace functor

template <typename Device, typename T, typename Tout>
class HistogramFixedWidthOp : public OpKernel {
 public:
  explicit HistogramFixedWidthOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPShistogram_opDTcc mht_1(mht_1_v, 262, "", "./tensorflow/core/kernels/histogram_op.cc", "HistogramFixedWidthOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPShistogram_opDTcc mht_2(mht_2_v, 267, "", "./tensorflow/core/kernels/histogram_op.cc", "Compute");

    const Tensor& values_tensor = ctx->input(0);
    const Tensor& value_range_tensor = ctx->input(1);
    const Tensor& nbins_tensor = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(value_range_tensor.shape()),
                errors::InvalidArgument("value_range should be a vector."));
    OP_REQUIRES(ctx, (value_range_tensor.shape().num_elements() == 2),
                errors::InvalidArgument(
                    "value_range should be a vector of 2 elements."));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(nbins_tensor.shape()),
                errors::InvalidArgument("nbins should be a scalar."));

    const auto values = values_tensor.flat<T>();
    const auto value_range = value_range_tensor.flat<T>();
    const auto nbins = nbins_tensor.scalar<int32>()();

    OP_REQUIRES(
        ctx, value_range(0) < value_range(1),
        errors::InvalidArgument("value_range should satisfy value_range[0] < "
                                "value_range[1], but got '[",
                                value_range(0), ", ", value_range(1), "]'"));
    OP_REQUIRES(
        ctx, nbins > 0,
        errors::InvalidArgument("nbins should be a positive number, but got '",
                                nbins, "'"));

    Tensor* out_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({nbins}), &out_tensor));
    auto out = out_tensor->flat<Tout>();

    OP_REQUIRES_OK(
        ctx, functor::HistogramFixedWidthFunctor<Device, T, Tout>::Compute(
                 ctx, values, value_range, nbins, out));
  }
};

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(Name("HistogramFixedWidth")                    \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<int32>("dtype"),           \
                          HistogramFixedWidthOp<CPUDevice, type, int32>) \
  REGISTER_KERNEL_BUILDER(Name("HistogramFixedWidth")                    \
                              .Device(DEVICE_CPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<int64_t>("dtype"),         \
                          HistogramFixedWidthOp<CPUDevice, type, int64>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("HistogramFixedWidth")          \
                              .Device(DEVICE_GPU)              \
                              .HostMemory("value_range")       \
                              .HostMemory("nbins")             \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int32>("dtype"), \
                          HistogramFixedWidthOp<GPUDevice, type, int32>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow
