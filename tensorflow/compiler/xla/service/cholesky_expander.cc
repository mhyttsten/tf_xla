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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScholesky_expanderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScholesky_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScholesky_expanderDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cholesky_expander.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

// The Cholesky–Banachiewicz algorithm. See
// https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky–Banachiewicz_and_Cholesky–Crout_algorithms
// for a description.
//
// def cholesky_unblocked(a):
//   assert len(a.shape) == 2 and a.shape[-2] == a.shape[-1]
//   n = a.shape[-2]
//   l = np.zeros_like(a)
//   for j in xrange(n):
//     mask = np.zeros_like(a)
//     mask[i, k] == 1 when i >= k and k == j
//     l_square = np.dot(l, l_t)
//     temp = a - l_square
//     l[..., j, j] = temp(j, j)
//     l = temp / l[..., j, j) * mask + l
//   return l
// Returns a (result, error) pair.
StatusOr<std::pair<XlaOp, XlaOp>> CholeskyExpander::CholeskyUnblocked(
    XlaOp a, PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
  const int ndims = a_shape.rank();
  const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
  std::vector<int64_t> error_dims(a_shape.dimensions().begin(),
                                  a_shape.dimensions().end());
  error_dims.back() = error_dims.at(ndims - 2) = 1;

  auto major_dims = a_shape.dimensions().subspan(
      /*pos=*/0,
      /*len=*/ndims - 2);

  auto matrix_dims = a_shape.dimensions().subspan(
      /*pos=*/0,
      /*len=*/ndims);

  XlaOp l = ZerosLike(a);

  // Construct the for loop body to iterate over rows.
  auto body_fn = [&](XlaOp i, absl::Span<const XlaOp> loop_vars,
                     XlaBuilder* body_builder) -> StatusOr<std::vector<XlaOp>> {
    std::vector<int64_t> row_shape_dims(major_dims.begin(), major_dims.end());
    std::vector<int64_t> col_shape_dims(major_dims.begin(), major_dims.end());
    auto body_a = loop_vars[0];
    auto body_l = loop_vars[1];
    auto seen_error = loop_vars[2];
    auto iota_row =
        Iota(body_builder, ShapeUtil::MakeShape(S32, matrix_dims), ndims - 1);
    auto iota_col =
        Iota(body_builder, ShapeUtil::MakeShape(S32, matrix_dims), ndims - 2);

    auto mask_pred = Ge(iota_col, iota_row);
    mask_pred = And(mask_pred, Eq(iota_row, i));
    auto mask_zeros =
        Zeros(body_builder,
              ShapeUtil::MakeShape(a_shape.element_type(), matrix_dims));
    // L * L.T, This matrix has of a lot of multiplying with zero
    // (namely, L[:, j:] = 0) and redundant computation, but it is faster
    // than slice.
    auto l_square =
        BatchDot(body_l, false, MaybeConjugate(body_l, true), true, precision);

    // A - L*L.T
    l_square = body_a - l_square;
    auto l_ii = DynamicSliceInMinorDims(l_square, {i, i}, {1, 1});
    if (ShapeUtil::ElementIsComplex(a_shape)) {
      auto sqrt = Sqrt(Real(l_ii));
      l_ii = Complex(sqrt, ZerosLike(sqrt));
      seen_error = Or(seen_error, IsNan(sqrt));
    } else {
      l_ii = Sqrt(l_ii);
      seen_error = Or(seen_error, IsNan(l_ii));
    }
    // L = (A - L*L.T) / l_ii * mask + L
    body_l = Select(mask_pred, l_square / l_ii, mask_zeros) + body_l;

    return std::vector<XlaOp>{body_a, body_l, seen_error};
  };

  TF_ASSIGN_OR_RETURN(
      auto cholesky_while,
      ForEachIndex(
          n, S32, body_fn,
          {a, l, Zeros(builder, ShapeUtil::MakeShape(PRED, error_dims))},
          "unblocked", builder));

  return std::make_pair(cholesky_while[1], cholesky_while[2]);
}

XlaOp CholeskyExpander::BuildCholesky(XlaOp a, int64_t block_size,
                                      PrecisionConfig::Precision precision) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScholesky_expanderDTcc mht_0(mht_0_v, 296, "", "./tensorflow/compiler/xla/service/cholesky_expander.cc", "CholeskyExpander::BuildCholesky");

  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int ndims = a_shape.rank();
    if (ndims < 2) {
      return InvalidArgument(
          "Argument to Cholesky must have rank >= 2; shape was %s",
          a_shape.ToString());
    }

    const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
    if (n != ShapeUtil::GetDimension(a_shape, -2)) {
      return InvalidArgument(
          "Argument to Cholesky must be batched square matrices; got shape %s",
          ShapeUtil::HumanString(a_shape));
    }

    if (block_size < 1) {
      return InvalidArgument(
          "block_size argument to Cholesky must be >= 1; got %d", block_size);
    }

    std::vector<int64_t> error_dims(a_shape.dimensions().begin(),
                                    a_shape.dimensions().end());
    error_dims.back() = error_dims.at(ndims - 2) = 1;
    std::vector<int64_t> error_dim_indices(ndims);
    absl::c_iota(error_dim_indices, 0);

    // Blocked left-looking Cholesky factorization.
    // Algorithm 1 from
    // Haidar, Azzam, et al. "High-performance Cholesky factorization for
    // GPU-only execution." Proceedings of General Purpose GPUs. ACM, 2017.
    XlaOp l = ZerosLike(a);
    XlaOp seen_error = Zeros(builder, ShapeUtil::MakeShape(PRED, error_dims));
    for (int64_t i = 0; i < n; i += block_size) {
      int64_t k = std::min(block_size, n - i);
      auto panel = SliceInMinorDims(a, {i, i}, {n, i + k});
      if (i > 0) {
        // TODO(phawkins): consider implementing SYRK for the diagonal part of
        // the panel.
        // a[i:, i:i+k] -= np.dot(l[i:, :i], np.transpose(l[i:i+k, :i]))
        auto lhs = SliceInMinorDims(l, {i, 0}, {n, i});
        auto rhs = SliceInMinorDims(l, {i, 0}, {i + k, i});
        auto delta =
            BatchDot(lhs, false, MaybeConjugate(rhs, true), true, precision);
        panel = panel - delta;
      }

      // l[i:i+k, i:i+k] = cholesky_unblocked(a[i:i+k, i:i+k])
      auto x = SliceInMinorDims(panel, {0, 0}, {k, k});
      XlaOp factorized;
      // TODO(b/167896062): A failure in one element of a batch shouldn't fail
      // other elements.
      XlaOp factorized_error;
      if (k == 1) {
        if (ShapeUtil::ElementIsComplex(a_shape)) {
          auto sqrt = Sqrt(Real(x));
          factorized = Complex(sqrt, ZerosLike(sqrt));
          factorized_error = IsNan(sqrt);
        } else {
          factorized = Sqrt(x);
          factorized_error = IsNan(factorized);
        }
      } else {
        TF_ASSIGN_OR_RETURN(auto tile_output, CholeskyUnblocked(x, precision));
        std::tie(factorized, factorized_error) = tile_output;
      }
      seen_error = Or(seen_error, factorized_error);
      l = UpdateSliceInMinorDims(l, factorized, {i, i});

      if (i + k < n) {
        // l[i+k:, i:i+k] =
        //     trsm_right_transpose(l[i:i+k, i:i+k], a[i+k:, i:i+k])
        auto update = TriangularSolve(
            factorized, SliceInMinorDims(panel, {k, 0}, {n - i, k}),
            /*left_side=*/false,
            /*lower=*/true,
            /*unit_diagonal=*/false,
            /*transpose_a=*/TriangularSolveOptions::ADJOINT);
        l = UpdateSliceInMinorDims(l, update, {i + k, i});
      }
    }
    return Select(
        BroadcastInDim(seen_error, a_shape.dimensions(), error_dim_indices),
        FullLike(l, std::numeric_limits<float>::quiet_NaN()), l);
  });
}

bool CholeskyExpander::InstructionMatchesPattern(HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScholesky_expanderDTcc mht_1(mht_1_v, 388, "", "./tensorflow/compiler/xla/service/cholesky_expander.cc", "CholeskyExpander::InstructionMatchesPattern");

  return instruction->opcode() == HloOpcode::kCholesky;
}

StatusOr<HloInstruction*> CholeskyExpander::ExpandInstruction(
    HloInstruction* instruction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScholesky_expanderDTcc mht_2(mht_2_v, 396, "", "./tensorflow/compiler/xla/service/cholesky_expander.cc", "CholeskyExpander::ExpandInstruction");

  const CholeskyOptions& options = instruction->cholesky_options();
  const std::string name = absl::StrFormat(
      "xla.cholesky_%s_%s", instruction->operand(0)->shape().ToString(),
      options.lower() ? "lower" : "upper");

  HloModule* module = instruction->parent()->parent();

  HloComputation*& computation =
      computation_cache_.emplace(name, nullptr).first->second;
  if (!computation) {
    // Builds a new expansion.
    //
    // TODO(b/62327888): We do something unusual here: we build the computation
    // using the XlaBuilder API, which is nominally an XLA client API. We do
    // this because the external APIs for building complicated computations
    // (XlaBuilder) are much more ergonomic than the internal ones. As it turns
    // out, XlaBuilder isn't really a client API—what it does is build a
    // HloModuleProto protocol buffer, that we can then deserialize and clone
    // into our HloModule. Ideally we would avoid the protocol buffer step;
    // that is left as an exercise for future work.
    XlaBuilder builder(name);
    XlaOp a = Parameter(&builder, 0, instruction->operand(0)->shape(), "a");
    XlaOp l = BuildCholesky(MaybeTransposeInMinorDims(a, !options.lower()),
                            /*block_size=*/128,
                            /*precision=*/PrecisionConfig::HIGHEST);
    MaybeTransposeInMinorDims(l, !options.lower());

    TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());

    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        xla_computation.GetProgramShape());
    HloModuleConfig config(program_shape);
    TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                             xla_computation.proto(), config));
    HloCloneContext context(module);
    computation =
        module->DeepCloneComputation(new_module->entry_computation(), &context);
  }

  return instruction->parent()->AddInstruction(HloInstruction::CreateCall(
      instruction->shape(), instruction->operands(), computation));
}

}  // namespace xla
