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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc() {
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

#include <sys/types.h>

#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

class UniqueOpBase : public XlaOpKernel {
 public:
  explicit UniqueOpBase(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/tf2xla/kernels/unique_op.cc", "UniqueOpBase");

    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_idx", &dtype));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &idx_type_));
  }

  // Transpose a tensor by moving axis `from` into `to`.
  xla::XlaOp MoveAxis(xla::XlaOp a, int64_t from, int64_t to,
                      const xla::Shape& input_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/tf2xla/kernels/unique_op.cc", "MoveAxis");

    std::vector<int64_t> permutation;
    permutation.reserve(input_shape.rank());
    for (int64_t i = 0; i < input_shape.rank(); ++i) {
      permutation.push_back(i);
    }
    std::swap(permutation[from], permutation[to]);
    return xla::Transpose(a, permutation);
  }

  xla::XlaOp CumSumR1(XlaOpKernelContext* ctx, xla::XlaOp input, int64_t size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc mht_2(mht_2_v, 242, "", "./tensorflow/compiler/tf2xla/kernels/unique_op.cc", "CumSumR1");

    auto init = xla::Zero(ctx->builder(), xla::S32);
    auto reducer = xla::CreateScalarAddComputation(xla::S32, ctx->builder());

    return xla::ReduceWindowWithGeneralPadding(
        input, init, reducer, {size}, {1},
        /*base_dilations=*/{}, /*window_dilations=*/{}, {{size - 1, 0}});
  }

  // RollingSelectR1 takes two arrays: `data` and `mask`. It scans this two
  // arrays in parallel and accumulates outputs into `accum`.
  //
  // For each position i, accum[i] = data[i]
  // if mask[i] = 1 or accum[i - 1] if mask[i] = 0.
  //
  // Requires mask[0] = 1, meaning that accum[i - 1] will never be accessed.
  //
  // This is implemented as an hlo while loop.
  xla::XlaOp RollingSelectR1(XlaOpKernelContext* ctx, xla::XlaOp data,
                             xla::XlaOp mask, int64_t size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/tf2xla/kernels/unique_op.cc", "RollingSelectR1");

    xla::XlaComputation cond, body;
    const xla::Shape r1_shape = xla::ShapeUtil::MakeShape(xla::S32, {size});
    const xla::Shape counter_shape = xla::ShapeUtil::MakeScalarShape(xla::S32);
    const xla::Shape& single_element_shape = counter_shape;

    auto loop_shape = xla::ShapeUtil::MakeTupleShape(
        {counter_shape, r1_shape, r1_shape, r1_shape});
    {
      std::unique_ptr<xla::XlaBuilder> builder =
          ctx->builder()->CreateSubBuilder("loop_cond");
      auto param = xla::Parameter(builder.get(), 0, loop_shape, "param");
      auto counter = xla::GetTupleElement(param, 0);
      auto limit = xla::ConstantR0<int32_t>(builder.get(), size);
      xla::Lt(counter, limit);

      cond = builder->Build().ConsumeValueOrDie();
    }

    {
      std::unique_ptr<xla::XlaBuilder> builder =
          ctx->builder()->CreateSubBuilder("loop_body");
      auto param = xla::Parameter(builder.get(), 0, loop_shape, "param");
      auto counter = xla::GetTupleElement(param, 0);

      auto data_stack = xla::GetTupleElement(param, 1);
      auto data = xla::DynamicSlice(data_stack, {counter}, {1});
      data = xla::Reshape(single_element_shape, data);

      auto mask_stack = xla::GetTupleElement(param, 2);
      auto mask = xla::DynamicSlice(mask_stack, {counter}, {1});
      mask = xla::Reshape(single_element_shape, mask);

      auto counter_minus = counter - xla::One(builder.get(), xla::S32);
      // If counter = 0, then counter_minus = 0.
      auto zero = xla::Zero(builder.get(), xla::S32);
      counter_minus = xla::Select(xla::Eq(counter, zero), zero, counter_minus);

      auto accum_stack = xla::GetTupleElement(param, 3);
      auto accum_minus = xla::DynamicSlice(accum_stack, {counter_minus}, {1});
      accum_minus = xla::Reshape(single_element_shape, accum_minus);

      auto accum = xla::Select(xla::ConvertElementType(mask, xla::PRED), data,
                               accum_minus);
      accum_stack = xla::DynamicUpdateSlice(
          accum_stack, xla::Reshape(accum, {1}), {counter});
      counter = counter + xla::One(builder.get(), xla::S32);

      xla::Tuple(builder.get(), {counter, data_stack, mask_stack, accum_stack});
      body = builder->Build().ConsumeValueOrDie();
    }

    auto zero = xla::Zero(ctx->builder(), xla::S32);
    auto zero_broadcast = xla::Broadcast(zero, {size});
    auto init = xla::Tuple(ctx->builder(), {zero, data, mask, zero_broadcast});
    return xla::GetTupleElement(xla::While(cond, body, init), 3);
  }

  void CompileWithAxis(XlaOpKernelContext* ctx, int64_t axis) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc mht_4(mht_4_v, 325, "", "./tensorflow/compiler/tf2xla/kernels/unique_op.cc", "CompileWithAxis");

    xla::XlaOp input = ctx->Input(0);
    StatusOr<xla::Shape> input_shape_or = ctx->builder()->GetShape(input);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    auto input_shape = input_shape_or.ValueOrDie();
    auto aux = MoveAxis(input, axis, 0, input_shape);
    auto aux_shape = ctx->builder()->GetShape(aux).ValueOrDie();
    int64_t leading_size = aux_shape.dimensions(0);
    int64_t product = 1;
    for (int64_t i = 1; i < aux_shape.rank(); ++i) {
      product *= aux_shape.dimensions(i);
    }
    aux = xla::Reshape(aux, {leading_size, product});
    if (leading_size == 0) {
      auto result_data = xla::Reshape(aux, aux_shape.dimensions());
      result_data = MoveAxis(result_data, 0, axis, aux_shape);
      ctx->SetOutput(0, result_data);
      ctx->SetOutput(1, xla::Iota(ctx->builder(), xla::S32, leading_size));
      return;
    }
    std::vector<xla::XlaOp> sort_keys;
    sort_keys.reserve(product + 1);
    std::vector<xla::PrimitiveType> sort_types;
    sort_types.reserve(product + 1);
    for (int64_t i = 0; i < product; ++i) {
      xla::XlaOp slice = xla::SliceInDim(aux, i, i + 1, 1, 1);
      sort_keys.push_back(xla::Reshape(slice, {leading_size}));
      sort_types.push_back(input_shape.element_type());
    }
    auto iota = xla::Iota(ctx->builder(), xla::S32, leading_size);
    sort_keys.push_back(iota);
    sort_types.push_back(xla::S32);

    std::vector<absl::optional<xla::XlaOp (*)(xla::XlaOp, xla::XlaOp,
                                              absl::Span<const int64_t>)>>
        generators(sort_types.size(), xla::LtTotalOrder);
    auto lt_chain = xla::CreateScalarComparisonComputation(
        "UniqueV2Lt", sort_types, generators, ctx->builder());

    auto sorted = xla::Sort(sort_keys, lt_chain, 0, /*is_stable=*/true);
    // Last element is permutation.
    xla::XlaOp perm;
    if (sort_keys.size() == 1) {
      perm = sorted;
    } else {
      perm = xla::GetTupleElement(sorted, sort_keys.size() - 1);
    }

    // Use gather to rearrange minor dimension.
    xla::GatherDimensionNumbers gather_dim_numbers;
    gather_dim_numbers.add_offset_dims(1);
    // The dimension to rewrite is the index dim.
    gather_dim_numbers.add_start_index_map(0);
    gather_dim_numbers.set_index_vector_dim(1);
    gather_dim_numbers.add_collapsed_slice_dims(0);
    auto permuted = xla::Gather(aux, perm, gather_dim_numbers, {1, product});
    // Tail is everything except for first element.
    auto tail = xla::SliceInDim(permuted, 1, leading_size, 1, 0);
    // Init is everything except for last element.
    auto init = xla::SliceInDim(permuted, 0, leading_size - 1, 1, 0);
    auto ne = xla::Compare(tail, init, xla::ComparisonDirection::kNe);
    auto reduce =
        xla::Reduce(ne, xla::ConstantR0(ctx->builder(), false),
                    CreateScalarOrComputation(xla::PRED, ctx->builder()), {1});
    auto mask = xla::ConvertElementType(reduce, xla::S32);
    mask = xla::PadInDim(mask, xla::One(ctx->builder(), xla::S32), 0, 1, 0);
    auto iperm = RollingSelectR1(ctx, perm, mask, leading_size);

    auto sort_by_iperm =
        xla::Sort({iperm, mask, perm},
                  xla::CreateScalarLtComputation({xla::S32, xla::S32, xla::S32},
                                                 ctx->builder()),
                  0,
                  /*is_stable=*/true);
    mask = xla::GetTupleElement(sort_by_iperm, 1);
    // perm_sort is used later to revert the indices back to input order.
    auto perm_sort = xla::GetTupleElement(sort_by_iperm, 2);

    auto dynamic_size = xla::ReduceAll(
        mask, xla::Zero(ctx->builder(), xla::S32),
        xla::CreateScalarAddComputation(xla::S32, ctx->builder()));
    auto mask_sort = xla::Sort(
        {mask, perm_sort},
        xla::CreateScalarGtComputation({xla::S32, xla::S32}, ctx->builder()), 0,
        /*is_stable=*/true);
    auto mask_permute = xla::GetTupleElement(mask_sort, 1);
    permuted = xla::Gather(aux, mask_permute, gather_dim_numbers, {1, product});
    auto result_data = xla::Reshape(permuted, aux_shape.dimensions());
    result_data = MoveAxis(result_data, 0, axis, aux_shape);
    result_data = xla::SetDimensionSize(result_data, dynamic_size, axis);
    ctx->SetOutput(0, result_data);
    auto imask = CumSumR1(ctx, mask, leading_size);
    imask = xla::Sub(imask, xla::One(ctx->builder(), xla::S32), {});
    auto idx = xla::GetTupleElement(
        xla::Sort({perm_sort, imask},
                  xla::CreateScalarLtComputation({xla::S32, xla::S32},
                                                 ctx->builder())),
        1);
    idx = xla::ConvertElementType(idx, idx_type_);
    ctx->SetOutput(1, idx);
  }

 private:
  xla::PrimitiveType idx_type_;
};

class UniqueOp : public UniqueOpBase {
 public:
  explicit UniqueOp(OpKernelConstruction* ctx) : UniqueOpBase(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc mht_5(mht_5_v, 436, "", "./tensorflow/compiler/tf2xla/kernels/unique_op.cc", "UniqueOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc mht_6(mht_6_v, 441, "", "./tensorflow/compiler/tf2xla/kernels/unique_op.cc", "Compile");

    CompileWithAxis(ctx, /*axis=*/0);
  }
};

REGISTER_XLA_OP(Name("Unique"), UniqueOp);

class UniqueV2Op : public UniqueOpBase {
 public:
  explicit UniqueV2Op(OpKernelConstruction* ctx) : UniqueOpBase(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc mht_7(mht_7_v, 453, "", "./tensorflow/compiler/tf2xla/kernels/unique_op.cc", "UniqueV2Op");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunique_opDTcc mht_8(mht_8_v, 458, "", "./tensorflow/compiler/tf2xla/kernels/unique_op.cc", "Compile");

    std::vector<int64_t> axises;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &axises));
    OP_REQUIRES(
        ctx, axises.size() <= 1,
        xla::InvalidArgument("Only single axis unique op is supported"));
    int64_t axis;
    if (axises.empty()) {
      axis = 0;
    } else {
      axis = axises.front();
    }
    CompileWithAxis(ctx, /*axis=*/axis);
  }
};

REGISTER_XLA_OP(Name("UniqueV2").CompileTimeConstantInput("axis"), UniqueV2Op);

}  // namespace
}  // namespace tensorflow
