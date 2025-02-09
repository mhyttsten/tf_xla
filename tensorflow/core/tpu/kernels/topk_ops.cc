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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc() {
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

#include "absl/numeric/bits.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

using ::tensorflow::errors::InvalidArgument;

// Computes the Kth order statistic of a data set. The current
// implementation uses a binary search requiring exactly 32 passes
// over the input data. The running time is linear with respect to
// input size. The median-of-medians algorithm is probably faster, but
// is difficult to implement efficiently in XLA. The implementation
// imposes a total ordering on floats. The ordering is consistent with
// the usual partial order.  Positive NaNs are greater than positive
// infinity. Negative NaNs are less than negative infinity. NaNs with
// distinct payloads are treated as distinct. Subnormal numbers are
// preserved (not flushed to zero). Positive infinity is greater than
// all numbers. Negative infinity is less than all numbers. Positive
// is greater than negative zero. There are less than k values greater
// than the kth order statistic. There are at least k values greater
// than or equal to the Kth order statistic. The semantics are not the
// same as TopKUnique.
xla::XlaOp CreateKthOrderStatisticComputation(xla::XlaBuilder* builder,
                                              const TensorShape& input_shape,
                                              const xla::XlaOp input,
                                              const xla::XlaOp k) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "CreateKthOrderStatisticComputation");

  const int64_t height = input_shape.dim_size(0);
  const int64_t width = input_shape.dim_size(1);

  xla::XlaOp input_sm32 = xla::BitcastConvertType(input, xla::S32);
  xla::XlaOp zero_r0 = xla::ConstantR0<int32>(builder, 0);
  xla::XlaOp zero_r1 = xla::Broadcast(zero_r0, {height});
  xla::XlaOp zero_r2 = xla::Broadcast(zero_r0, {height, width});

  xla::XlaOp max_r0 = xla::ConstantR0<int32>(builder, 0x7FFFFFFF);
  xla::XlaOp max_r1 = xla::Broadcast(max_r0, {height});

  // Start at positive zero, so that pivot is always less than top.
  xla::XlaOp negative_zero_r0 = xla::ConstantR0<int32>(builder, 0x80000000);
  xla::XlaOp negative_zero_r1 = xla::Broadcast(negative_zero_r0, {height});
  xla::XlaOp top_r1 = zero_r1;

  for (uint32 mask = 1U << 31; mask; mask >>= 1) {
    xla::XlaOp broadcast_mask_r1 =
        xla::Broadcast(xla::ConstantR0<int32>(builder, mask), {height});

    // The first iteration of the loop determines if the kth element
    // is positive or negative. If the kth element is negative, we
    // start the search from +QNAN (0x7FFFFFF). If k is negative, we
    // start from -0 (0x8000000). The pivot is less than the top and
    // is always half way between the top and the implicit bottom in
    // IEEE754 space.
    xla::XlaOp pivot_r1 = xla::Xor(top_r1, broadcast_mask_r1);
    xla::XlaOp pivot_r2 = xla::Add(pivot_r1, zero_r2, {0});
    xla::XlaOp both_negative_r2 =
        xla::Lt(xla::And(input_sm32, pivot_r2), zero_r0);
    xla::XlaOp left_r2 = xla::Select(both_negative_r2, pivot_r2, input_sm32);
    xla::XlaOp right_r2 = xla::Select(both_negative_r2, input_sm32, pivot_r2);
    xla::XlaOp pred_r2 = xla::Gt(left_r2, right_r2);
    xla::XlaOp conv_r2 = xla::ConvertElementType(pred_r2, xla::S32);

    xla::XlaComputation add = CreateScalarAddComputation(xla::S32, builder);
    xla::XlaOp sum_r1 = xla::Reduce(conv_r2, zero_r0, add, {1});

    xla::XlaOp pivot_too_low_r1 = xla::Le(k, sum_r1, {});

    if (mask == (1U << 31)) {
      top_r1 = xla::Select(pivot_too_low_r1, max_r1, negative_zero_r1);
    } else {
      top_r1 = xla::Select(pivot_too_low_r1, top_r1, pivot_r1);
    }
  }
  return xla::BitcastConvertType(top_r1, xla::F32);
}

class KthOrderStatistic : public XlaOpKernel {
 public:
  explicit KthOrderStatistic(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_1(mht_1_v, 270, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "KthOrderStatistic");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &k_));
    OP_REQUIRES(ctx, k_ >= 0, errors::InvalidArgument("Need k >= 0, got ", k_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_2(mht_2_v, 278, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "Compile");

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    const TensorShape& input_shape = ctx->InputShape(0);
    OP_REQUIRES(
        ctx, input_shape.dims() == 2,
        InvalidArgument("input must be rank-2: ", input_shape.DebugString()));

    xla::XlaOp k = xla::ConstantR0<int32>(builder, k_);
    xla::XlaOp kth_order_statistics =
        CreateKthOrderStatisticComputation(builder, input_shape, input, k);
    ctx->SetOutput(0, kth_order_statistics);
  }

 private:
  int32 k_;
};

REGISTER_XLA_OP(Name("KthOrderStatistic"), KthOrderStatistic);

// Returns the TopK unique values in the array in sorted order and the
// indices of those elements. The running time is proportional to the
// product of K and the input size. Sorting the whole array is more
// efficient for sufficiently large values of K. The median-of-medians
// algorithm is probably faster, but difficult to implement
// efficiently in XLA. If there are fewer than K unique values, the
// results are padded with negative infinity. NaNs are never
// returned. Subnormal numbers are flushed to zero.
//
// If an element appears at multiple indices, the highest index is
// returned. If a TopK element never appears in the input due to
// padding values, the indices are padded with negative one. If a
// padding value appears in the input and padding is needed, the
// highest index of the padding value will be returned.
//
// The semantics are not the same as KthOrderStatistic.
//
// If masked_with_iota is true, the index is already encoded in the lower bits
// of the mantissa, which will be extracted as the index in the output.
// Otherwise, every iteration will use the following algorithm to get the index:
//   index = max([i if data[i] == max else -1 for i in size])
//
// TODO(b/74994968): Replace TopKUnique with an LLO implementation of
// TopK with reasonable semantics.
std::pair<xla::XlaOp, xla::XlaOp> CreateTopKUnique(
    xla::XlaBuilder* builder, const xla::XlaOp input,
    const TensorShape& input_shape, int64_t k, bool masked_with_iota) {
  const int64_t height = input_shape.dim_size(0);
  const int64_t width = input_shape.dim_size(1);

  xla::XlaOp iota_r1 = xla::Iota(builder, xla::S32, width);
  xla::XlaOp iota_r2 = xla::Broadcast(iota_r1, {height});

  xla::XlaOp negative_one_r0 = xla::ConstantR0<int>(builder, -1);
  xla::XlaOp negative_one_r2 = xla::Broadcast(negative_one_r0, {height, width});

  xla::XlaOp negative_infinity_r0 = xla::ConstantR0<float>(builder, -INFINITY);
  xla::XlaOp negative_infinity_r2 =
      xla::Broadcast(negative_infinity_r0, {height, width});

  xla::XlaOp scratch_pad_r2 = input;
  std::vector<xla::XlaOp> topk_r1s;
  std::vector<xla::XlaOp> topk_indices;
  for (int i = 0; i < k; ++i) {
    xla::XlaOp kth_order_statistic_r1 =
        xla::Reduce(scratch_pad_r2, negative_infinity_r0,
                    CreateScalarMaxComputation(xla::F32, builder), {1});
    topk_r1s.push_back(kth_order_statistic_r1);

    xla::XlaOp ge_r2 = xla::Ge(input, kth_order_statistic_r1, {0});
    scratch_pad_r2 = xla::Select(ge_r2, negative_infinity_r2, input);

    if (!masked_with_iota) {
      xla::XlaOp eq_r2 = xla::Eq(input, kth_order_statistic_r1, {0});
      xla::XlaOp indices_r2 = xla::Select(eq_r2, iota_r2, negative_one_r2);
      xla::XlaOp topk_index_r1 =
          xla::Reduce(indices_r2, negative_one_r0,
                      CreateScalarMaxComputation(xla::S32, builder), {1});
      topk_indices.push_back(topk_index_r1);
    }
  }
  xla::XlaOp topk_r1_concat = xla::ConcatInDim(builder, topk_r1s, 0);
  xla::XlaOp topk_r2 =
      xla::Transpose(xla::Reshape(topk_r1_concat, {k, height}), {1, 0});

  xla::XlaOp topk_indices_r2;
  if (masked_with_iota) {
    int32_t next_power_of_two = absl::bit_ceil<uint64_t>(width);
    int32_t count_mask = next_power_of_two - 1;
    xla::XlaOp mask_r0 = xla::ConstantR0(builder, count_mask);
    xla::XlaOp mask_r2 = xla::Broadcast(mask_r0, {height, k});
    xla::XlaOp topk_r2_s32 = xla::BitcastConvertType(topk_r2, xla::S32);
    topk_indices_r2 = xla::And(topk_r2_s32, mask_r2);
  } else {
    xla::XlaOp topk_indices_concat = xla::ConcatInDim(builder, topk_indices, 0);
    topk_indices_r2 =
        xla::Transpose(xla::Reshape(topk_indices_concat, {k, height}), {1, 0});
  }
  return std::make_pair(topk_r2, topk_indices_r2);
}

class TopKUnique : public XlaOpKernel {
 public:
  explicit TopKUnique(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_3(mht_3_v, 384, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "TopKUnique");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &k_));
    OP_REQUIRES(ctx, k_ >= 0, errors::InvalidArgument("Need k >= 0, got ", k_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_4(mht_4_v, 392, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "Compile");

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    const TensorShape& input_shape = ctx->InputShape(0);
    OP_REQUIRES(
        ctx, input_shape.dims() == 2,
        InvalidArgument("input must be rank-2: ", input_shape.DebugString()));

    auto topk = CreateTopKUnique(builder, input, input_shape, k_, false);
    ctx->SetOutput(0, topk.first);
    ctx->SetOutput(1, topk.second);
  }

 private:
  int k_;
};
REGISTER_XLA_OP(Name("TopKUnique"), TopKUnique);

// Make all elements in the non-Batch dimension unique and close to
// their initial value on a relative scale, but potential far from
// their initial value in an absolute scale.
//
// This operation is meant to be combined with TopKUnique to avoid
// suppressing identical elements. For most TopK users, the indices of
// the TopK elements are important but the relative order of the TopK
// elements and their exact values is not so important. Ideally, the
// the indices of the TopK elements of the output of MakeUnique are
// the same as the indices of the TopK elements of the inputs.
//
// Its an open question whether it is better to accept the risk of two
// elements in the input to TopK have exactly the same value or the
// risk that MakeUnique will alter the indices of the TopK
// elements. Model owners are encouraged to experiment!
//
// Never returns a sub-normal number. Never returns zero. The sign of
// each input element is always identical to the sign of the
// corresponding output element. Behavior for infinite elements is
// undefined. Behavior for subnormal elements is undefined.
//
// Algorithm:
// 1. Replace zeros with the smallest representable normal floating
// point number with the same sign.
// 2. Mask away enough low order bits that every value can be distinct.
// 3. Replace the low order bits with iota.
//
// TODO(b/74994968): Replace MakeUnique with an LLO implementation of
// TopK with reasonable semantics.
xla::XlaOp CreateMakeUnique(xla::XlaBuilder* builder, const xla::XlaOp input,
                            const TensorShape& input_shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_5(mht_5_v, 443, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "CreateMakeUnique");

  const int64_t height = input_shape.dim_size(0);
  const int64_t width = input_shape.dim_size(1);

  xla::XlaOp zero_r0 = xla::ConstantR0(builder, 0U);
  xla::XlaOp zero_r2 = xla::Broadcast(zero_r0, {height, width});

  // count_mask is used to mask away the low order bits to ensure
  // that every element is distinct.
  uint32_t next_power_of_two = absl::bit_ceil<uint64_t>(width);
  uint32 count_mask = ~(next_power_of_two - 1);
  xla::XlaOp count_mask_r0 = xla::ConstantR0(builder, count_mask);
  xla::XlaOp count_mask_r2 = xla::Broadcast(count_mask_r0, {height, width});

  // smallest_normal is the bit representation of the smallest
  // positive normal floating point number. The sign is zero,
  // exponent is one, and the fraction is zero.
  uint32 smallest_normal = 1U << 23;
  xla::XlaOp smallest_normal_r0 = xla::ConstantR0(builder, smallest_normal);
  xla::XlaOp smallest_normal_r2 =
      xla::Broadcast(smallest_normal_r0, {height, width});

  // Used to mask away the sign bit when computing the absolute
  // value.
  uint32 low_bit_mask = ~(1U << 31);
  xla::XlaOp low_bit_mask_r0 = xla::ConstantR0(builder, low_bit_mask);
  xla::XlaOp low_bit_mask_r2 = xla::Broadcast(low_bit_mask_r0, {height, width});

  xla::XlaOp iota_r1 = xla::Iota(builder, xla::U32, width);
  xla::XlaOp iota_r2 = xla::Broadcast(iota_r1, {height});

  // Compare the absolute value with positive zero to handle
  // negative zero.
  //
  // Pseudocode: input_no_zeros = abs(input) == 0 ? FLT_MIN : input
  xla::XlaOp input_u32_r2 = xla::BitcastConvertType(input, xla::U32);
  xla::XlaOp abs_r2 = xla::And(input_u32_r2, low_bit_mask_r2);
  xla::XlaOp if_zero_r2 = xla::Eq(abs_r2, zero_r2);
  xla::XlaOp smallest_normal_preserving_sign_r2 =
      xla::Or(input_u32_r2, smallest_normal_r2);
  xla::XlaOp input_no_zeros_r2 =
      xla::Select(if_zero_r2, smallest_normal_preserving_sign_r2, input_u32_r2);

  // Discard the low-order bits and replace with iota.
  xla::XlaOp and_r2 = xla::And(input_no_zeros_r2, count_mask_r2);
  xla::XlaOp or_r2 = xla::Or(and_r2, iota_r2);
  return xla::BitcastConvertType(or_r2, xla::F32);
}

class MakeUnique : public XlaOpKernel {
 public:
  explicit MakeUnique(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_6(mht_6_v, 497, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "MakeUnique");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_7(mht_7_v, 502, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "Compile");

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    const TensorShape& input_shape = ctx->InputShape(0);
    OP_REQUIRES(
        ctx, input_shape.dims() == 2,
        InvalidArgument("input must be rank-2: ", input_shape.DebugString()));

    ctx->SetOutput(0, CreateMakeUnique(builder, input, input_shape));
  }
};
REGISTER_XLA_OP(Name("MakeUnique"), MakeUnique);

// Returns the TopK approximate values in the array in sorted order and the
// indices of those elements. The running time is proportional to the
// product of K and the input size.
//
// The algorithm first updates the lower bits of each element with iota,
// which is used to derive the index. The iota also serves the purpose to
// make each element unique so that each iteration, we are guaranteed to
// get one and only one unique top-1 element.
class TopKWithUnique : public XlaOpKernel {
 public:
  explicit TopKWithUnique(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_8(mht_8_v, 528, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "TopKWithUnique");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &k_));
    OP_REQUIRES(ctx, k_ >= 0, errors::InvalidArgument("Need k >= 0, got ", k_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStopk_opsDTcc mht_9(mht_9_v, 536, "", "./tensorflow/core/tpu/kernels/topk_ops.cc", "Compile");

    xla::XlaBuilder* builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    const TensorShape& input_shape = ctx->InputShape(0);
    OP_REQUIRES(
        ctx, input_shape.dims() == 2,
        InvalidArgument("input must be rank-2: ", input_shape.DebugString()));

    xla::XlaOp unique = CreateMakeUnique(builder, input, input_shape);
    auto topk = CreateTopKUnique(builder, unique, input_shape, k_, true);
    ctx->SetOutput(0, topk.first);
    ctx->SetOutput(1, topk.second);
  }

 private:
  int k_;
};
REGISTER_XLA_OP(Name("TopKWithUnique"), TopKWithUnique);
}  // namespace
}  // namespace tensorflow
