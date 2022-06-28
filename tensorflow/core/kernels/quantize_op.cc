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
class MHTracer_DTPStensorflowPScorePSkernelsPSquantize_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSquantize_opDTcc() {
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
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace {
enum {
  QUANTIZE_MODE_MIN_COMBINED,
  QUANTIZE_MODE_MIN_FIRST,
  QUANTIZE_MODE_SCALED,
};
enum {
  // Round half away from zero: if the fraction of y is exactly 0.5, then
  // round(y) = y + 0.5 if y > 0
  // round(y) = y - 0.5 if y < 0
  // E.g., -5.5 gets rounded to -6, -5.4 goes to -5,
  // 5.4 goes to 5, and 5.5 goes to 6.
  ROUND_HALF_AWAY_FROM_ZERO,
  // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
  // the nearest even integer to y.
  // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5 becomes
  // -24, and -24.5 gets rounded to 24.
  ROUND_HALF_TO_EVEN,
};
}  // namespace

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Quantize a tensor from float to T, with user-specified min_range and
// max_range.
// TODO(xbing): Add a new QuantizeOp just taking scale,
//              rather than min_range and max_range.
template <typename Device, typename T>
class QuantizeV2Op : public OpKernel {
 public:
  explicit QuantizeV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_opDTcc mht_0(mht_0_v, 230, "", "./tensorflow/core/kernels/quantize_op.cc", "QuantizeV2Op");

    half_range_ =
        !std::is_signed<T>::value
            ? 0.0f
            : (static_cast<double>(std::numeric_limits<T>::max()) -
               static_cast<double>(std::numeric_limits<T>::min()) + 1) /
                  2.0f;
    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(ctx,
                (mode_string == "MIN_COMBINED" || mode_string == "MIN_FIRST" ||
                 mode_string == "SCALED"),
                errors::InvalidArgument("Mode string must be 'MIN_COMBINED',"
                                        " 'MIN_FIRST', or 'SCALED', is '" +
                                        mode_string + "'"));
    if (mode_string == "MIN_COMBINED") {
      mode_ = QUANTIZE_MODE_MIN_COMBINED;
    } else if (mode_string == "MIN_FIRST") {
      mode_ = QUANTIZE_MODE_MIN_FIRST;
    } else if (mode_string == "SCALED") {
      mode_ = QUANTIZE_MODE_SCALED;
    }

    string round_mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("round_mode", &round_mode_string));
    OP_REQUIRES(ctx,
                (round_mode_string == "HALF_AWAY_FROM_ZERO" ||
                 round_mode_string == "HALF_TO_EVEN"),
                errors::InvalidArgument("Round mode string must be "
                                        "'HALF_AWAY_FROM_ZERO' or "
                                        "'HALF_TO_EVEN', is '" +
                                        round_mode_string + "'"));
    if (round_mode_string == "HALF_AWAY_FROM_ZERO") {
      round_mode_ = ROUND_HALF_AWAY_FROM_ZERO;
    } else if (round_mode_string == "HALF_TO_EVEN") {
      OP_REQUIRES(ctx, mode_string == "SCALED",
                  errors::InvalidArgument("Round mode 'HALF_TO_EVEN' "
                                          "only supported for mode 'SCALED', "
                                          "b  ut mode is '" +
                                          mode_string + "'."));
      round_mode_ = ROUND_HALF_TO_EVEN;
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("ensure_minimum_range", &ensure_minimum_range_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_opDTcc mht_1(mht_1_v, 281, "", "./tensorflow/core/kernels/quantize_op.cc", "Compute");

    const Tensor& input = ctx->input(0);
    const Tensor& input_min_range = ctx->input(1);
    const Tensor& input_max_range = ctx->input(2);

    int num_slices = 1;
    if (axis_ > -1) {
      OP_REQUIRES(
          ctx, input.dims() > axis_,
          errors::InvalidArgument(
              "Axis is on a zero-based index, so its value must always be less "
              "than number of input's dims, but given axis value was ",
              axis_, " and input's dims was ", input.dims()));
      num_slices = input.dim_size(axis_);
      OP_REQUIRES(ctx, input_min_range.dims() == 1,
                  errors::InvalidArgument(
                      "If axis is specified, min_range must be a 1-D tensor "
                      "whose size matches the axis dimension of the input and "
                      "output tensors, but min_range dims are ",
                      input_min_range.dims()));
      OP_REQUIRES(ctx, input_min_range.dim_size(0) == num_slices,
                  errors::InvalidArgument(
                      "If axis is specified, min_range must be a 1-D tensor "
                      "whose size matches the axis dimension of the input and "
                      "output tensors, but min_range is a 1-D tensor of size ",
                      input_min_range.dim_size(0),
                      " and input's axis dimension is of size ", num_slices));
      OP_REQUIRES(ctx, input_max_range.dims() == 1,
                  errors::InvalidArgument(
                      "If axis is specified, max_range must be a 1-D tensor "
                      "whose size matches the axis dimension of the input and "
                      "output tensors, but max_range dims are ",
                      input_max_range.dims()));
      OP_REQUIRES(ctx, input_max_range.dim_size(0) == num_slices,
                  errors::InvalidArgument(
                      "If axis is specified, max_range must be a 1-D tensor "
                      "whose size matches the axis dimension of the input and "
                      "output tensors, but max_range is a 1-D tensor of size ",
                      input_max_range.dim_size(0),
                      " and input's axis dimension is of size ", num_slices));
    } else {
      OP_REQUIRES(ctx, input_min_range.NumElements() == 1,
                  errors::InvalidArgument(
                      "If axis is not specified, min_range must contain a "
                      "single float element, but it contains ",
                      input_min_range.NumElements(), " elements"));
      OP_REQUIRES(ctx, input_max_range.NumElements() == 1,
                  errors::InvalidArgument(
                      "If axis is not specified, max_range must contain a "
                      "single float element, but it contains ",
                      input_max_range.NumElements(), " elements"));
    }

    const TensorShape& minmax_shape = ctx->input(1).shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    Tensor* output_min_tensor = nullptr;
    Tensor* output_max_tensor = nullptr;

    if (num_slices == 1) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {}, &output_min_tensor));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {}, &output_max_tensor));
      const float min_range = input_min_range.template flat<float>()(0);
      const float max_range = input_max_range.template flat<float>()(0);
      QuantizeTensor(ctx, input, min_range, max_range, output,
                     output_min_tensor, output_max_tensor);
      return;
    }

    OP_REQUIRES(ctx, mode_ != QUANTIZE_MODE_MIN_FIRST,
                errors::Unimplemented("MIN_FIRST mode is not implemented for "
                                      "Quantize with axis != -1."));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, minmax_shape, &output_min_tensor));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(2, minmax_shape, &output_max_tensor));

    auto input_tensor =
        input.template flat_inner_outer_dims<float, 3>(axis_ - 1);
    int64_t pre_dim = 1, post_dim = 1;
    for (int i = 0; i < axis_; ++i) {
      pre_dim *= output->dim_size(i);
    }
    for (int i = axis_ + 1; i < output->dims(); ++i) {
      post_dim *= output->dim_size(i);
    }
    auto output_tensor = output->template bit_casted_shaped<T, 3>(
        {pre_dim, num_slices, post_dim});
    auto min_ranges = input_min_range.template vec<float>();
    auto max_ranges = input_max_range.template vec<float>();
    for (int i = 0; i < num_slices; ++i) {
      QuantizeSlice(ctx->eigen_device<Device>(), ctx,
                    input_tensor.template chip<1>(i), min_ranges(i),
                    max_ranges(i), output_tensor.template chip<1>(i),
                    &output_min_tensor->flat<float>()(i),
                    &output_max_tensor->flat<float>()(i));
    }
  }

  void QuantizeTensor(OpKernelContext* ctx, const Tensor& input,
                      const float input_min_range, const float input_max_range,
                      Tensor* output, Tensor* output_min_tensor,
                      Tensor* output_max_tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSquantize_opDTcc mht_2(mht_2_v, 387, "", "./tensorflow/core/kernels/quantize_op.cc", "QuantizeTensor");

    OP_REQUIRES(ctx, !(input_max_range < input_min_range),
                errors::InvalidArgument(
                    "input_max_range must be larger than input_min_range."));

    // When the minimum and maximum ranges are too close together, nudge them
    // apart by a small value so that they are slightly different. This helps
    // us avoid creating ill-formed buffers where all quantized values map to
    // the same float number. These kinds of buffers cause problems for
    // downstream ops when they need to do calculations on them.
    // We pick the value by making sure that zero is not more than 100x the
    // overall range from the maximum, so that the value can be easily
    // represented when we promote the quantized value to a higher
    // intermediate bit depth, since that's a common requirement.
    float min_range = std::min(0.0f, input_min_range);
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                  fabsf(input_max_range))) *
                          ensure_minimum_range_;
    float max_range =
        std::max(0.0f, std::max(input_max_range, min_range + epsilon));

    if (mode_ == QUANTIZE_MODE_MIN_FIRST) {
      if (meta::IsSupportedAndEnabled() && std::is_same<T, quint8>()) {
        TTypes<const float>::Vec input_array = input.flat<float>();

        meta::Quantize(ctx, input_array.data(), input_array.size(), min_range,
                       max_range, output->flat<quint8>().data());
      } else {
        FloatTensorToQuantizedInPlaceUsingEigen<T>(
            ctx->template eigen_device<Device>(), input, min_range, max_range,
            output);
      }
      output_min_tensor->flat<float>()(0) = min_range;
      output_max_tensor->flat<float>()(0) = max_range;
    } else {
      QuantizeSlice(ctx->eigen_device<Device>(), ctx, input.flat<float>(),
                    input_min_range, input_max_range,
                    output->template flat<T>(),
                    &output_min_tensor->flat<float>()(0),
                    &output_max_tensor->flat<float>()(0));
    }
  }

  template <typename ConstVec, typename Vec>
  void QuantizeSlice(const Device& d, OpKernelContext* ctx,
                     const ConstVec& input, float input_min_range,
                     float input_max_range, Vec output, float* output_min_range,
                     float* output_max_range) {
    OP_REQUIRES(ctx, !(input_max_range < input_min_range),
                errors::InvalidArgument(
                    "input_max_range must be larger than input_min_range."));
    float min_range = std::min(0.0f, input_min_range);
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                  fabsf(input_max_range))) *
                          ensure_minimum_range_;
    float max_range =
        std::max(0.0f, std::max(input_max_range, min_range + epsilon));

    if (mode_ == QUANTIZE_MODE_MIN_COMBINED) {
      const float scale_factor =
          (static_cast<double>(std::numeric_limits<T>::max()) -
           static_cast<double>(std::numeric_limits<T>::min())) /
          (max_range - min_range);

      // Quantize:
      // Make input in range of [min_range, max_range], then
      // subtract min_range to be in range of [0, max_range - min_range]
      // Divide by (max_range - min_range) to get to [0, 1.0]
      // Multiply by range of T, after that shift left 1/2 range of T if
      // T is signed.
      // Note that the number is rounded before the cast. Rounding follows the
      // semantic of std::round, which implements "round-half-away-zero",
      // e.g., -5.5 gets rounded to -6, -5.4 goes to -5, 5.4 goes to 5,
      // and 5.5 goes to 6.
      bool is_signed = std::is_signed<T>::value;
      if (is_signed) {
        // The slow path.
        // TODO(xbing,yonghui): Speedup this path as well.
        output.device(d) =
            ((input.cwiseMin(max_range).cwiseMax(min_range) - min_range) *
                 scale_factor -
             half_range_)
                .round()
                .template cast<T>();
      } else {
        // The fast path that avoids unaryExpr
        // According to the micro-benchmark, adding device here doesn't help.
        output.device(d) =
            ((input.cwiseMin(max_range).cwiseMax(min_range) - min_range) *
                 scale_factor +
             0.5f)
                .template cast<T>();
      }
    } else if (mode_ == QUANTIZE_MODE_SCALED) {
      const int min_output_value =
          std::numeric_limits<T>::min() + (narrow_range_ ? 1 : 0);
      const int max_output_value = std::numeric_limits<T>::max();
      const float scale_factor_from_min_side =
          (min_output_value * min_range > 0)
              ? min_output_value / min_range
              : std::numeric_limits<float>::max();
      const float scale_factor_from_max_side =
          (max_output_value * max_range > 0)
              ? max_output_value / max_range
              : std::numeric_limits<float>::max();
      const float scale_factor =
          std::min(scale_factor_from_min_side, scale_factor_from_max_side);
      min_range = min_output_value / scale_factor;
      max_range = max_output_value / scale_factor;
      if (round_mode_ == ROUND_HALF_TO_EVEN) {
        output.device(d) =
            (input.cwiseMin(max_range).cwiseMax(min_range) * scale_factor)
                .unaryExpr(
                    Eigen::internal::scalar_round_half_to_even_op<float>())
                .template cast<T>();
      } else if (round_mode_ == ROUND_HALF_AWAY_FROM_ZERO) {
        output.device(d) =
            (input.cwiseMin(max_range).cwiseMax(min_range) * scale_factor)
                .round()
                .template cast<T>();
      }
    }

    *output_min_range = min_range;
    *output_max_range = max_range;
  }

 private:
  float half_range_;
  float ensure_minimum_range_;
  int mode_;
  int round_mode_;
  int axis_;
  bool narrow_range_;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantizeV2").Device(DEVICE_CPU).TypeConstraint<quint8>("T"),
    QuantizeV2Op<CPUDevice, quint8>);
REGISTER_KERNEL_BUILDER(
    Name("QuantizeV2").Device(DEVICE_CPU).TypeConstraint<qint8>("T"),
    QuantizeV2Op<CPUDevice, qint8>);
REGISTER_KERNEL_BUILDER(
    Name("QuantizeV2").Device(DEVICE_CPU).TypeConstraint<quint16>("T"),
    QuantizeV2Op<CPUDevice, quint16>);
REGISTER_KERNEL_BUILDER(
    Name("QuantizeV2").Device(DEVICE_CPU).TypeConstraint<qint16>("T"),
    QuantizeV2Op<CPUDevice, qint16>);
REGISTER_KERNEL_BUILDER(
    Name("QuantizeV2").Device(DEVICE_CPU).TypeConstraint<qint32>("T"),
    QuantizeV2Op<CPUDevice, qint32>);
}  // namespace tensorflow
