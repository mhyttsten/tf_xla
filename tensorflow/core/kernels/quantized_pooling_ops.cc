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
class MHTracer_DTPStensorflowPScorePSkernelsPSquantized_pooling_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_pooling_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSquantized_pooling_opsDTcc() {
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

#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/errors.h"
#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class QuantizedAvgPoolingOp : public OpKernel {
 public:
  explicit QuantizedAvgPoolingOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_pooling_opsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/quantized_pooling_ops.cc", "QuantizedAvgPoolingOp");

    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_pooling_opsDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/kernels/quantized_pooling_ops.cc", "Compute");

    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context,
                          ksize_,
                          stride_,
                          padding_,
                          /*explicit_paddings=*/{},
                          FORMAT_NHWC,
                          tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    const float min_input = context->input(1).flat<float>()(0);
    const float max_input = context->input(2).flat<float>()(0);

    OP_REQUIRES(context, params.depth_window == 1,
                errors::Unimplemented("Non-spatial pooling is not "
                                      "yet supported. Volunteers? :)"));

    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));
    const int32_t highest = static_cast<int32>(Eigen::NumTraits<T>::highest());
    const int32_t lowest = static_cast<int32>(Eigen::NumTraits<T>::lowest());

    // TODO(vrv): Switch this to the Eigen::Tensor version of
    // SpatialAvgPooling once that version is running quickly.
    Tensor int32_output(DT_INT32, params.forward_output_shape());
    // Cast input to int32 tensor and call SpatialAvgPool.
    Tensor int32_input(DT_INT32, tensor_in.shape());
    int32_input.flat<int32>() = tensor_in.flat<T>().template cast<int32>();
    SpatialAvgPool<Device, int32>(context, &int32_output, int32_input, params,
                                  padding_);

    // Clamp the int32 output back into quantized space.
    output->flat<T>() = int32_output.flat<int32>()
                            .cwiseMax(lowest)
                            .cwiseMin(highest)
                            .template cast<T>();

    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &output_min));
    output_min->flat<float>()(0) = min_input;
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_max));
    output_max->flat<float>()(0) = max_input;
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

template <typename Device, typename T>
class QuantizedMaxPoolingOp : public MaxPoolingOp<Device, T> {
 public:
  explicit QuantizedMaxPoolingOp(OpKernelConstruction* context)
      : MaxPoolingOp<Device, T>(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_pooling_opsDTcc mht_2(mht_2_v, 294, "", "./tensorflow/core/kernels/quantized_pooling_ops.cc", "QuantizedMaxPoolingOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantized_pooling_opsDTcc mht_3(mht_3_v, 299, "", "./tensorflow/core/kernels/quantized_pooling_ops.cc", "Compute");

    auto min_input_tensor = context->input(1);
    auto max_input_tensor = context->input(2);
    OP_REQUIRES(
        context, min_input_tensor.NumElements() == 1,
        errors::InvalidArgument(
            "min_input must be a scalar float value, got tensor with shape ",
            min_input_tensor.shape()));
    OP_REQUIRES(
        context, max_input_tensor.NumElements() == 1,
        errors::InvalidArgument(
            "max_input must be a scalar float value, got tensor with shape ",
            max_input_tensor.shape()));
    const float min_input = context->input(1).flat<float>()(0);
    const float max_input = context->input(2).flat<float>()(0);
    MaxPoolingOp<Device, T>::Compute(context);
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &output_min));
    output_min->flat<float>()(0) = min_input;
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_max));
    output_max->flat<float>()(0) = max_input;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("QuantizedAvgPool").Device(DEVICE_CPU).TypeConstraint<quint8>("T"),
    QuantizedAvgPoolingOp<CPUDevice, quint8>);

REGISTER_KERNEL_BUILDER(
    Name("QuantizedMaxPool").Device(DEVICE_CPU).TypeConstraint<quint8>("T"),
    QuantizedMaxPoolingOp<CPUDevice, quint8>);

#ifdef INTEL_MKL
REGISTER_KERNEL_BUILDER(
    Name("QuantizedAvgPool").Device(DEVICE_CPU).TypeConstraint<qint8>("T"),
    QuantizedAvgPoolingOp<CPUDevice, qint8>);

REGISTER_KERNEL_BUILDER(
    Name("QuantizedMaxPool").Device(DEVICE_CPU).TypeConstraint<qint8>("T"),
    QuantizedMaxPoolingOp<CPUDevice, qint8>);
#endif

}  // namespace tensorflow
