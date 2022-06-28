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
class MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_opDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/bfloat16.h"

namespace {
enum {
  QUANTIZE_MODE_MIN_COMBINED,
  QUANTIZE_MODE_MIN_FIRST,
  QUANTIZE_MODE_SCALED,
};
}  // namespace

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
T Cast(float v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_opDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/dequantize_op.cc", "Cast");

  return v;
}

template <>
bfloat16 Cast<bfloat16>(float v) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_opDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/dequantize_op.cc", "Cast<bfloat16>");

  return bfloat16(v);
}

template <typename Device, typename T, typename S>
class DequantizeOp : public OpKernel {
 public:
  explicit DequantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_opDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/kernels/dequantize_op.cc", "DequantizeOp");

    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(
        ctx,
        (ctx->output_type(0) == DT_FLOAT || ctx->output_type(0) == DT_BFLOAT16),
        errors::InvalidArgument("Output type must be bfloat16 or float,"
                                " is '" +
                                DataTypeString(ctx->output_type(0)) + "'"));

    need_cast_ = true;
    if (ctx->output_type(0) == DT_FLOAT) {
      need_cast_ = false;
      OP_REQUIRES(ctx,
                  (mode_string == "MIN_COMBINED" ||
                   mode_string == "MIN_FIRST" || mode_string == "SCALED"),
                  errors::InvalidArgument("Mode string must be 'MIN_COMBINED',"
                                          " 'MIN_FIRST', or 'SCALED', is '" +
                                          mode_string + "'"));
    } else {
      OP_REQUIRES(
          ctx, (mode_string == "MIN_COMBINED"),
          errors::InvalidArgument("When output type is bfloat16, Mode"
                                  " string must be 'MIN_COMBINED', is '" +
                                  mode_string + "'"));
    }

    if (mode_string == "MIN_COMBINED") {
      mode_ = QUANTIZE_MODE_MIN_COMBINED;
    } else if (mode_string == "MIN_FIRST") {
      mode_ = QUANTIZE_MODE_MIN_FIRST;
    } else if (mode_string == "SCALED") {
      mode_ = QUANTIZE_MODE_SCALED;
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_opDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/kernels/dequantize_op.cc", "Compute");

    const Tensor& input = ctx->input(0);
    const Tensor& input_min_tensor = ctx->input(1);
    const Tensor& input_max_tensor = ctx->input(2);

    OP_REQUIRES(
        ctx, axis_ < input.dims(),
        errors::InvalidArgument("Axis must be less than input dimension(",
                                input.dims(), "), got ", axis_));

    int num_slices = 1;
    if (axis_ > -1) {
      num_slices = input.dim_size(axis_);
    }
    OP_REQUIRES(ctx, input_min_tensor.NumElements() == num_slices,
                errors::InvalidArgument(
                    "input_min_tensor must have as many elements as input on "
                    "the dequantization axis (",
                    axis_, "), got ", input_min_tensor.NumElements(),
                    ", expected ", num_slices));
    OP_REQUIRES(ctx, input_max_tensor.NumElements() == num_slices,
                errors::InvalidArgument(
                    "input_max_tensor must have as many elements as input on "
                    "the dequantization axis (",
                    axis_, "), got ", input_max_tensor.NumElements(),
                    ", expected ", num_slices));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    Tensor float_output =
        need_cast_ ? tensorflow::Tensor(DT_FLOAT, input.shape()) : *output;
    if (num_slices == 1) {
      const float min_range = input_min_tensor.flat<float>()(0);
      const float max_range = input_max_tensor.flat<float>()(0);
      DequantizeTensor(ctx, input, min_range, max_range, &float_output);
    } else {
      OP_REQUIRES(ctx, mode_ != QUANTIZE_MODE_MIN_FIRST,
                  errors::Unimplemented("MIN_FIRST mode is not implemented for "
                                        "Dequantize with axis != -1."));

      int64_t pre_dim = 1, post_dim = 1;
      for (int i = 0; i < axis_; ++i) {
        pre_dim *= float_output.dim_size(i);
      }
      for (int i = axis_ + 1; i < float_output.dims(); ++i) {
        post_dim *= float_output.dim_size(i);
      }
      auto input_tensor = input.template bit_casted_shaped<T, 3>(
          {pre_dim, num_slices, post_dim});
      auto output_tensor =
          float_output.flat_inner_outer_dims<float, 3>(axis_ - 1);
      auto min_ranges = input_min_tensor.vec<float>();
      auto max_ranges = input_max_tensor.vec<float>();
      for (int i = 0; i < num_slices; ++i) {
        DequantizeSlice(ctx->eigen_device<Device>(), ctx,
                        input_tensor.template chip<1>(i), min_ranges(i),
                        max_ranges(i), output_tensor.template chip<1>(i));
      }
    }
    if (need_cast_) {
      S* out_ptr = output->flat<S>().data();
      float* in_ptr = float_output.flat<float>().data();
      for (int64_t i = 0; i < float_output.NumElements(); ++i) {
        out_ptr[i] = static_cast<S>(in_ptr[i]);
      }
    }
  }

  void DequantizeTensor(OpKernelContext* ctx, const Tensor& input,
                        const float min_range, const float max_range,
                        Tensor* output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_opDTcc mht_4(mht_4_v, 343, "", "./tensorflow/core/kernels/dequantize_op.cc", "DequantizeTensor");

    const float half_range =
        !std::is_signed<T>::value
            ? 0.0f
            : (static_cast<float>(std::numeric_limits<T>::max()) -
               std::numeric_limits<T>::min() + 1) /
                  2.0f;

    if (mode_ == QUANTIZE_MODE_MIN_COMBINED) {
      const float scale_factor =
          (max_range - min_range) /
          (static_cast<float>(std::numeric_limits<T>::max()) -
           std::numeric_limits<T>::min());

      const auto& input_tensor = input.flat<T>();
      output->flat<float>() =
          ((input_tensor.template cast<float>() + half_range) * scale_factor) +
          min_range;

    } else if (mode_ == QUANTIZE_MODE_MIN_FIRST) {
      if (meta::IsSupportedAndEnabled() && std::is_same<T, quint8>()) {
        auto input_ui8_array = input.flat<quint8>();
        meta::Dequantize(ctx, input_ui8_array.data(), input_ui8_array.size(),
                         min_range, max_range, output->flat<float>().data());
      } else {
        QuantizedTensorToFloatInPlaceUsingEigen<T>(
            ctx->template eigen_device<Device>(), input, min_range, max_range,
            output);
      }
    } else if (mode_ == QUANTIZE_MODE_SCALED) {
      const int min_output_value =
          std::numeric_limits<T>::min() + (narrow_range_ ? 1 : 0);
      const float scale_factor =
          std::numeric_limits<T>::min() == 0
              ? (max_range / std::numeric_limits<T>::max())
              : std::max(min_range / min_output_value,
                         max_range / std::numeric_limits<T>::max());
      const auto& input_tensor = input.flat<T>();
      output->flat<float>() =
          input_tensor.template cast<int>().template cast<float>() *
          scale_factor;
    }
  }

  template <typename ConstVec, typename Vec>
  void DequantizeSlice(const Device& d, OpKernelContext* ctx,
                       const ConstVec& input, float min_range, float max_range,
                       Vec output) {
    // TODO(pauldonnelly): Factor out the similar calculations in quantize,
    //   dequantize and quantize_and_dequantize ops.
    const float half_range =
        !std::is_signed<T>::value
            ? 0.0f
            : (static_cast<float>(std::numeric_limits<T>::max()) -
               std::numeric_limits<T>::min() + 1) /
                  2.0f;

    if (mode_ == QUANTIZE_MODE_MIN_COMBINED) {
      const float scale_factor =
          (max_range - min_range) /
          (static_cast<float>(std::numeric_limits<T>::max()) -
           std::numeric_limits<T>::min());

      output.device(d) =
          ((input.template cast<float>() + half_range) * scale_factor) +
          min_range;
    } else if (mode_ == QUANTIZE_MODE_SCALED) {
      const int min_output_value =
          std::numeric_limits<T>::min() + (narrow_range_ ? 1 : 0);
      const float scale_factor =
          std::numeric_limits<T>::min() == 0
              ? (max_range / std::numeric_limits<T>::max())
              : std::max(min_range / min_output_value,
                         max_range / std::numeric_limits<T>::max());
      output.device(d) = input.template cast<float>() * scale_factor;
    }
  }

 private:
  int mode_;
  int axis_;
  bool narrow_range_;
  bool need_cast_;
};

REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, quint8, float>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, qint8, float>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint16>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, quint16, float>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint16>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, qint16, float>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T")
                            .TypeConstraint<float>("dtype"),
                        DequantizeOp<CPUDevice, qint32, float>);

REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, quint8, bfloat16>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, qint8, bfloat16>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint16>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, quint16, bfloat16>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint16>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, qint16, bfloat16>);
REGISTER_KERNEL_BUILDER(Name("Dequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T")
                            .TypeConstraint<bfloat16>("dtype"),
                        DequantizeOp<CPUDevice, qint32, bfloat16>);
}  // namespace tensorflow
