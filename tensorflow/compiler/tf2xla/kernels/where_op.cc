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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhere_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhere_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhere_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/dynamic_shaped_ops.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

using xla::S32;
using xla::XlaOp;

// "Shifts" a rank-1 array one element to the right, inserting a 0 at the
// beginning and cutting off the last element of the array.
//
// That is, transforms [x0, x1, ..., xn] into [0, x0, ..., xn-1].
StatusOr<XlaOp> ShiftElemsRight(XlaOp x) {
  xla::XlaBuilder* b = x.builder();
  StatusOr<xla::Shape> shape = b->GetShape(x);
  TF_RETURN_IF_ERROR(shape.status());
  TF_RET_CHECK(shape->dimensions_size() == 1);
  int64_t n = shape->dimensions(0);

  XlaOp padded = xla::PadInDim(x, xla::Zero(b, shape->element_type()),
                               /*dimno=*/0, /*pad_lo=*/1, /*pad_hi=*/0);
  return xla::SliceInDim(padded, /*start_index=*/0, /*limit_index=*/n,
                         /*stride=*/1, /*dimno=*/0);
}

// Recursive prefix-sum algorithm.
//
// - Let the input be an array x.
// - Let evens be [x0, x2, ...].
// - Let odds be  [x1, x3, ...].
// - Let combined be evens + odds.
// - Let psum = prefix-sum(combined), recursively.
//
// Then the prefix-sum of x is the interleaving of psum - odds and psum.
// Written out, this is:
//
//   [psum[0] - odds[0], psum[0], psum[1] - odds[1], psum[1], ...].
//
// Requires: `arr` is a 1D S32 array whose length is padded to a power of 2.
//
// Optimization: Rather than split the input into two slices (evens/odds), we
// split it into kNumSlices pieces.  The basic algorithm is the same, but this
// reduces the number of GPU kernels we have to launch.
//
// There are much more efficient algorithms to be had!  In particular, on GPU
// this launches O(log4 n) kernels, but there are efficient algorithms that use
// just one kernel, see
// https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
//
// Nonetheless, this is much simpler than the algorithm in the paper above, but
// also much faster than implementing tf.where by sorting the input.
StatusOr<XlaOp> PrefixSum(XlaOp arr) {
  xla::XlaBuilder* b = arr.builder();
  StatusOr<xla::Shape> input_shape = b->GetShape(arr);
  TF_RETURN_IF_ERROR(input_shape.status());

  TF_RET_CHECK(input_shape->dimensions_size() == 1);
  int64_t n = input_shape->dimensions(0);

  // The original input length must be a power of 2, but we recursively divide
  // it into kNumSlices chunks.  Assuming kNumSlices == 4, this means our
  // base-case needs to handle n == 1 (original length was a power of 4) or
  // n == 2 (original size was a power of 2).
  constexpr int kNumSlices = 4;
  if (n <= 1) {
    return arr;
  }
  if (n == 2) {
    TF_ASSIGN_OR_RETURN(XlaOp shifted, ShiftElemsRight(arr));
    return arr + shifted;
  }
  TF_RET_CHECK(n % kNumSlices == 0);

  std::array<XlaOp, kNumSlices> slices;
  for (int i = 0; i < slices.size(); i++) {
    slices[i] = xla::Slice(arr, /*start_indices=*/{i}, /*limit_indices=*/{n},
                           /*strides=*/{kNumSlices});
  }

  XlaOp combined = slices[0];
  for (int i = 1; i < kNumSlices; ++i) {
    combined = combined + slices[i];
  }

  TF_ASSIGN_OR_RETURN(XlaOp psum, PrefixSum(combined));

  std::array<XlaOp, kNumSlices> slices_psummed;
  slices_psummed[kNumSlices - 1] = psum;
  for (int i = kNumSlices - 2; i >= 0; --i) {
    slices_psummed[i] = slices_psummed[i + 1] - slices[i + 1];
  }

  // Interleave the slices.
  std::array<XlaOp, kNumSlices> slices_padded;
  for (int i = 0; i < kNumSlices; ++i) {
    xla::PaddingConfig padding_config;
    auto* dim = padding_config.add_dimensions();
    dim->set_edge_padding_low(i);
    dim->set_edge_padding_high(kNumSlices - i - 1);
    dim->set_interior_padding(kNumSlices - 1);
    slices_padded[i] =
        xla::Pad(slices_psummed[i], xla::Zero(b, S32), padding_config);
  }

  XlaOp ret = slices_padded[0];
  for (int i = 1; i < kNumSlices; ++i) {
    ret = ret + slices_padded[i];
  }

  return ret;
}

// prefix-sum works better on CPU/GPU, whereas sort works better on TPU.
bool ShouldUsePrefixSumImpl(const DeviceType& dt) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhere_opDTcc mht_0(mht_0_v, 315, "", "./tensorflow/compiler/tf2xla/kernels/where_op.cc", "ShouldUsePrefixSumImpl");

  absl::string_view t = dt.type_string();
  return t == DEVICE_CPU_XLA_JIT || t == DEVICE_GPU_XLA_JIT ||
         t == DEVICE_XLA_CPU || t == DEVICE_XLA_GPU;
}

StatusOr<XlaOp> CompileWhereWithSort(XlaOpKernelContext* ctx) {
  XlaOp condition = ctx->Input(0);
  TF_ASSIGN_OR_RETURN(xla::Shape input_shape,
                      ctx->builder()->GetShape(condition));
  auto iota_shape = input_shape;
  iota_shape.set_element_type(xla::S32);

  int64_t flattened_size = xla::Product(iota_shape.dimensions());
  XlaOp reshaped_condition = xla::Reshape(condition, {flattened_size});
  XlaOp zeros = xla::ZerosLike(reshaped_condition);
  XlaOp compared = xla::Ne(reshaped_condition, zeros);

  std::vector<XlaOp> to_sort = {compared};
  std::vector<xla::PrimitiveType> types_to_sort = {xla::PRED};
  // Generate iota for each dimension, which after combining becomes
  // indices of each element.
  for (int64_t axis = 0; axis < iota_shape.rank(); ++axis) {
    XlaOp iota = xla::Iota(ctx->builder(), iota_shape, axis);
    XlaOp reshaped = xla::Reshape(iota, {flattened_size});
    to_sort.push_back(reshaped);
    types_to_sort.push_back(xla::S32);
  }

  XlaOp sorted = xla::Sort(
      to_sort, xla::CreateScalarGtComputation(types_to_sort, ctx->builder()),
      /*dimension=*/0, /*is_stable=*/true);
  std::vector<XlaOp> to_concat;
  for (int64_t i = 0; i < iota_shape.rank(); ++i) {
    XlaOp index_single_dim = xla::GetTupleElement(sorted, i + 1);
    to_concat.push_back(xla::Reshape(index_single_dim, {flattened_size, 1}));
  }

  XlaOp result = xla::ConcatInDim(ctx->builder(), to_concat, 1);
  result = xla::ConvertElementType(result, ctx->output_xla_type(0));

  // Dynamic padder will handle the dynamic dimension.
  XlaOp compared_int = xla::ConvertElementType(compared, xla::S32);
  XlaOp length =
      xla::ReduceAll(compared_int, xla::Zero(ctx->builder(), xla::S32),
                     xla::CreateScalarAddComputation(xla::S32, ctx->builder()));
  StatusOr<XlaOp> rebounded_result = xla::SetDimensionSizeWithRebound(
      &ctx->value_inference(), result, length, 0);
  if (rebounded_result.ok()) {
    return rebounded_result;
  }
  // TODO(b/207187072): Remove special handling once dynamic reshape can also
  // be handled.
  return xla::SetDimensionSize(result, length, 0);
}

StatusOr<XlaOp> CompileWhereWithPrefixSum(XlaOpKernelContext* ctx) {
  xla::XlaBuilder* b = ctx->builder();
  XlaOp condition = ctx->Input(0);

  TF_ASSIGN_OR_RETURN(xla::Shape input_shape, b->GetShape(condition));

  int64_t flattened_size = xla::Product(input_shape.dimensions());
  XlaOp reshaped_condition = xla::Reshape(condition, {flattened_size});
  XlaOp zeros = xla::ZerosLike(reshaped_condition);
  XlaOp preds =
      xla::ConvertElementType(xla::Ne(reshaped_condition, zeros), S32);

  // Given preds, we compute prefix_sum and out_idx as in the following
  // example.
  //
  //   preds =      [T, F, F, T, F, T], therefore
  //   prefix_sum = [1, 1, 1, 2, 2, 3], and
  //   out_idxs   = [0, ⊥, ⊥, 1, ⊥, 2], where ⊥ is an OOB index.
  //
  // We then scatter out_idxs into the result.
  TF_ASSIGN_OR_RETURN(
      XlaOp padded_prefix_sum,
      PrefixSum(xla::PadInDim(
          preds, xla::Zero(b, S32), /*dimno=*/0, /*pad_lo=*/0,
          /*pad_hi=*/NextPowerOfTwo(flattened_size) - flattened_size)));
  XlaOp prefix_sum = xla::SliceInDim(padded_prefix_sum, /*start_index=*/0,
                                     /*limit_index=*/flattened_size,
                                     /*stride=*/1, /*dimno=*/0);

  // We could compute out_idxs as
  //
  //   out_idxs[i] = preds[i] ? prefix_sum[i] - 1 : ⊥,
  //
  // but it's faster to compute it as
  //
  //   let ps = prefix_sum in
  //   out_idxs[i] =
  //     if i == 0: ps[i] != 0       ? ps[i] - 1 : ⊥
  //     else:      ps[i] != ps[i-1] ? ps[i] - 1 : ⊥
  //
  // because we read less memory.
  XlaOp oob_idx = xla::ConstantR0WithType(b, S32, flattened_size);  // ⊥
  TF_ASSIGN_OR_RETURN(XlaOp prefix_sum_shifted, ShiftElemsRight(prefix_sum));
  XlaOp out_idxs = xla::Select(xla::Ne(prefix_sum, prefix_sum_shifted),
                               /*on_true=*/prefix_sum - xla::One(b, S32),
                               /*on_false=*/oob_idx);
  out_idxs = xla::Reshape(out_idxs, {flattened_size, 1});

  // tf.where returns an array of multidimensional indices where the condition
  // is true.  For example:
  //
  //    input =  [
  //      [F, T],
  //      [T, F],
  //      [F, F],
  //    ]
  //
  //  results in
  //
  //    output = [
  //      [0,0], [1,0],
  //    ]
  //
  // Generate the list
  //
  //   iotas = [[0,...,0], [0,...,1], ..., [limit_0,...,limit_n]],
  //
  // and then scatter iotas[out_idxs] into the output.
  std::vector<XlaOp> iotas_to_concat;
  auto iota_shape = input_shape;
  iota_shape.set_element_type(S32);
  for (int64_t axis = 0; axis < iota_shape.rank(); ++axis) {
    iotas_to_concat.push_back(
        xla::Reshape(xla::Iota(b, iota_shape, axis), {flattened_size, 1}));
  }
  XlaOp iotas = xla::ConcatInDim(b, iotas_to_concat, /*dimension=*/1);

  // Scatter subcomputation.  Instead of the usual `return p0 + p1`, simply
  // does `return p1`, because we just want to overwrite whatever was in the
  // scatter dest.
  xla::XlaComputation assn_computation = [&] {
    std::unique_ptr<xla::XlaBuilder> subb =
        b->CreateSubBuilder("where_op_scatter_assn");
    xla::Shape param_shape = xla::ShapeUtil::MakeShape(S32, {});
    xla::Parameter(subb.get(), 0, param_shape, "p0");
    xla::Parameter(subb.get(), 1, param_shape, "p1");
    // Simply return p1, the last op we created.
    return subb->BuildAndNoteError();
  }();

  xla::ScatterDimensionNumbers scatter_dnums;
  scatter_dnums.set_index_vector_dim(1);
  scatter_dnums.add_inserted_window_dims(0);
  scatter_dnums.add_scatter_dims_to_operand_dims(0);
  scatter_dnums.add_update_window_dims(1);
  XlaOp scattered = xla::Scatter(
      /*input=*/xla::Zeros(b, /*shape=*/xla::ShapeUtil::MakeShape(
                               S32, {flattened_size, iota_shape.rank()})),
      /*scatter_indices=*/out_idxs, /*updates=*/iotas,
      /*update_computation=*/assn_computation, scatter_dnums,
      /*indices_are_sorted=*/true, /*unique_indices=*/true);
  scattered = xla::ConvertElementType(scattered, ctx->output_xla_type(0));

  // Now count how many valid elements there are and slice off the tail of
  // `scattered`.
  XlaOp num_valid =
      xla::ReduceAll(xla::ConvertElementType(preds, S32), xla::Zero(b, S32),
                     xla::CreateScalarAddComputation(S32, b));
  StatusOr<XlaOp> rebounded_result = xla::SetDimensionSizeWithRebound(
      &ctx->value_inference(), scattered, num_valid, 0);
  if (rebounded_result.ok()) {
    return *rebounded_result;
  }
  // TODO(b/207187072): Remove special handling once dynamic reshape can also
  // be handled.
  return xla::SetDimensionSize(scattered, num_valid, 0);
}

class WhereOp : public XlaOpKernel {
 public:
  explicit WhereOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        use_prefix_sum_(ShouldUsePrefixSumImpl(ctx->device_type())) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhere_opDTcc mht_1(mht_1_v, 496, "", "./tensorflow/compiler/tf2xla/kernels/where_op.cc", "WhereOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSwhere_opDTcc mht_2(mht_2_v, 501, "", "./tensorflow/compiler/tf2xla/kernels/where_op.cc", "Compile");

    StatusOr<XlaOp> ret;
    if (use_prefix_sum_) {
      ret = CompileWhereWithPrefixSum(ctx);
    } else {
      ret = CompileWhereWithSort(ctx);
    }
    OP_REQUIRES_OK(ctx, ret.status());
    ctx->SetOutput(0, *ret);
  }

 private:
  bool use_prefix_sum_;
};

REGISTER_XLA_OP(Name("Where"), WhereOp);

}  // namespace
}  // namespace tensorflow
