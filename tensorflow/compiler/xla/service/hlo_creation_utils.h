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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CREATION_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CREATION_UTILS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_creation_utilsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_creation_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_creation_utilsDTh() {
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


#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Some lightweight utilities intended to make HLO instruction creation more
// ergonomic.  We don't have a complete set of helpers yet -- I expect we'll
// expand this interface as needed on an ad-hoc basis.

// Creates a unary HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> MakeUnaryHlo(HloOpcode opcode,
                                       HloInstruction* operand);

// Creates a binary HLO instruction and adds it to the computation containing
// `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
StatusOr<HloInstruction*> MakeBinaryHlo(HloOpcode opcode, HloInstruction* lhs,
                                        HloInstruction* rhs);

// Creates a compare HLO instruction and adds it to the computation containing
// `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
StatusOr<HloInstruction*> MakeCompareHlo(Comparison::Direction direction,
                                         HloInstruction* lhs,
                                         HloInstruction* rhs);

// Creates a pad HLO instruction and adds it to the computation containing
// `operand` and `padding_value` (`operand` and `padding_value` must be in the
// same computation).
StatusOr<HloInstruction*> MakePadHlo(HloInstruction* operand,
                                     HloInstruction* padding_value,
                                     const PaddingConfig& padding_config);

// Creates a slice HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> MakeSliceHlo(HloInstruction* operand,
                                       absl::Span<const int64_t> start_indices,
                                       absl::Span<const int64_t> limit_indices,
                                       absl::Span<const int64_t> strides);

// Creates a convolution HLO instruction and adds it to the computation
// containing `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
// If the result shape has integral element type, an optional
// preferred_element_type can be specified to override the element type.
StatusOr<HloInstruction*> MakeConvolveHlo(
    HloInstruction* lhs, HloInstruction* rhs, int64_t feature_group_count,
    int64_t batch_group_count, const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config,
    absl::optional<PrimitiveType> preferred_element_type);

// Creates a transpose HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> MakeTransposeHlo(
    HloInstruction* operand, absl::Span<const int64_t> dimensions);

// Creates a reshape HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> MakeReshapeHlo(const Shape& result_shape,
                                         HloInstruction* operand);

StatusOr<HloInstruction*> MakeReshapeHlo(
    absl::Span<const int64_t> result_shape_dim_bounds, HloInstruction* operand);

// Creates a dynamic-slice HLO instruction and adds it to the computation
// containing `operand` and `start_indices` (`operand` and `start_indices` must
// be in the same computation).
StatusOr<HloInstruction*> MakeDynamicSliceHlo(
    HloInstruction* operand, absl::Span<HloInstruction* const> start_indices,
    absl::Span<const int64_t> slice_sizes);
StatusOr<HloInstruction*> MakeDynamicSliceHlo(
    HloInstruction* operand, HloInstruction* start_indices,
    absl::Span<const int64_t> slice_sizes);

// Creates a dynamic-update-slice HLO instruction and adds it to the computation
// containing `operand`, `update` and `start_indices` (`operand`, `update` and
// `start_indices` must be in the same computation).
StatusOr<HloInstruction*> MakeDynamicUpdateSliceHlo(
    HloInstruction* operand, HloInstruction* update,
    HloInstruction* start_indices);

// Creates a broadcast HLO instruction and adds it to the computation containing
// `operand`.
HloInstruction* MakeBroadcastHlo(HloInstruction* operand,
                                 absl::Span<const int64_t> broadcast_dimensions,
                                 absl::Span<const int64_t> result_shape_bounds);
HloInstruction* MakeBroadcastHlo(HloInstruction* operand,
                                 absl::Span<const int64_t> broadcast_dimensions,
                                 const Shape& shape);

// Creates a GetTupleElement HLO instruction and adds it to the computation
// containing `operand`.
StatusOr<HloInstruction*> MakeGetTupleElementHlo(HloInstruction* operand,
                                                 int64_t index);

// Creates a Concatenate HLO instruction and adds it to the computation
// containing `operands` (`operands` must be non-empty and every element must be
// contained in the same computation).
StatusOr<HloInstruction*> MakeConcatHlo(
    absl::Span<HloInstruction* const> operands, int64_t dimension);

// Creates a Convert HLO instruction that converts the given instruction to have
// the given primitive type.
HloInstruction* MakeConvertToHlo(HloInstruction* hlo, PrimitiveType type);

// Creates a BitcastConvert HLO instruction.
HloInstruction* MakeBitcastConvertToHlo(HloInstruction* hlo,
                                        PrimitiveType type);

// Creates an Iota HLO instruction.
HloInstruction* MakeIotaHlo(HloComputation* computation, const Shape& shape,
                            int64_t iota_dimension);

// Creates a Dot HLO instruction and adds it to the computation containing `lhs`
// and `rhs` (both must be in the same computation). If the result shape has
// integral element type, an optional preferred_element_type can be specified to
// override the element type.
StatusOr<HloInstruction*> MakeDotHlo(
    HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dim_numbers,
    const PrecisionConfig& precision_config,
    absl::optional<PrimitiveType> preferred_element_type);

// Creates a Map HLO instruction and adds it to the computation containing the
// operands. All operands must be in the same computation.
StatusOr<HloInstruction*> MakeMapHlo(absl::Span<HloInstruction* const> operands,
                                     HloComputation* map_computation);

// Creates a Reduce HLO instruction and adds it to the computation containing
// the operand. This will create the sub-computation needed for the reduction in
// the given module. binary_opcode should represent a binary operation.
StatusOr<HloInstruction*> MakeReduceHlo(HloInstruction* operand,
                                        HloInstruction* init_value,
                                        absl::Span<const int64_t> dimensions,
                                        HloOpcode binary_opcode);

StatusOr<HloInstruction*> MakeReduceHlo(HloInstruction* operand,
                                        HloInstruction* init_value,
                                        absl::Span<const int64_t> dimensions,
                                        HloComputation* reduce_computation);

StatusOr<HloInstruction*> MakeReduceHlo(HloInstruction* operand,
                                        HloInstruction* init_value,
                                        HloOpcode binary_opcode,
                                        HloModule* module);

// Creates a Reverse HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> MakeReverseHlo(HloInstruction* operand,
                                         absl::Span<const int64_t> dimensions);

// Creates a Select HLO instruction and adds it to the computation containing
// the predicate. The on_true and on_false instructions must also be contained
// in the same computation. If on_true and on_false are tuples, create a tuple
// select instead. `pred` is broadcasted up from a scalar if necessary.
StatusOr<HloInstruction*> MakeSelectHlo(HloInstruction* pred,
                                        HloInstruction* on_true,
                                        HloInstruction* on_false,
                                        HloInstruction* derived_from = nullptr);

// Creates a Sort HLO instruction and adds it to the computation containing the
// operands. All operands must be in the same computation. Also creates a
// default compare sub-computation which sorts the first operand into ascending
// order. 'is_stable' specifies whether the sorting should be stable.
StatusOr<HloInstruction*> MakeSortHlo(
    const Shape& sort_shape, absl::Span<HloInstruction* const> operands,
    int64_t dimension_to_sort, bool is_stable, HloComputation::Builder* builder,
    HloModule* module);

// Creates an R1 Constant HLO instruction of the given PrimitiveType with the
// given values and adds it to the given computation.
template <typename NativeT>
StatusOr<HloInstruction*> MakeR1ConstantHlo(HloComputation* computation,
                                            PrimitiveType type,
                                            absl::Span<const NativeT> values) {
  Literal literal = LiteralUtil::CreateR1<NativeT>(values);
  if (literal.shape().element_type() != type) {
    TF_ASSIGN_OR_RETURN(literal, literal.Convert(type));
  }
  return computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
}

// Creates an R0 Constant HLO instruction of the PrimitiveType corresponding to
// `NativeT` with the given value and adds it to the given computation.
template <class NativeT>
HloInstruction* MakeR0ConstantHlo(HloComputation* computation, NativeT value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_creation_utilsDTh mht_0(mht_0_v, 375, "", "./tensorflow/compiler/xla/service/hlo_creation_utils.h", "MakeR0ConstantHlo");

  return computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<NativeT>(value)));
}

// Makes a scalar that is elementwise compatible with the shape of the base
// instruction.
template <class NativeT>
HloInstruction* MakeScalarLike(HloInstruction* base, NativeT value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_creation_utilsDTh mht_1(mht_1_v, 386, "", "./tensorflow/compiler/xla/service/hlo_creation_utils.h", "MakeScalarLike");

  auto scalar = base->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<NativeT>(value)
                                         .Convert(base->shape().element_type())
                                         .ValueOrDie()));
  if (base->shape().rank() == 0) {
    *scalar->mutable_shape() = base->shape();
    return scalar;
  }
  return base->AddInstruction(
      HloInstruction::CreateBroadcast(base->shape(), scalar, {}));
}

// Creates a fusion instruction and fuses `fused` into the created fusion
// instruction.
StatusOr<HloInstruction*> MakeFusionInstruction(
    HloInstruction* fused, HloInstruction::FusionKind kind);

// -----------------------------------------------------------------------------
// Some other miscellaneous helpers to generate common HLO patterns.  All of
// these add all the instructions they generate into the computation containing
// their operand(s).

// Collapses (via reshape) the first N (logical) dimensions of `operand` into a
// single leading dimension.  `operand` must have rank > `n` and `n` must not be
// 0.
//
// For instance if `operand` has shape f32[7,8,9] and n is 2 then the output is
// the `operand` reshaped to [56,9].
StatusOr<HloInstruction*> CollapseFirstNDims(HloInstruction* operand,
                                             int64_t n);

// Prepends `n` degenerate dimensions (dimensions with bound = 1) to `operand`
// using a reshape.
//
// For instance if operand has shape f32[3,4,5] then this returns the operand
// reshaped to f32[1,3,4,5].  If the operand is a f32 scalar (i.e. has shape
// f32[]) then this returns the operand reshaped to f32[1].
StatusOr<HloInstruction*> PrependDegenerateDims(HloInstruction* operand,
                                                int64_t n);

// Expands (via reshape) the first (logical) dimension of `operand` into a
// sequence of `expanded_dims` dimensions.  `operand` must at least be of rank 1
// and the number of elements in its first dimension must be equal to the
// product of `expanded_dims`.
//
// For instance if `operand` has shape f32[200,9,7] and expanded_dims is
// {2,5,20} the result is `operand` reshaped to [2,5,20,9,7].
StatusOr<HloInstruction*> ExpandFirstDimIntoNDims(
    HloInstruction* operand, absl::Span<const int64_t> expanded_dims);

// Elides (via reshape) a set of degenerate dimensions (dimensions containing
// exactly one element), `dims_to_elide` from `operand`.  Every dimension in
// `dims_to_elide` must be a degenerate dimension.  `dims_to_elide` must be
// sorted and not contain duplicates.
//
// For example if `operand` is of shape f32[19,1,20,1,7,1,9] and dims_to_elide
// is {1,5} then the result is `operand` reshaped to [19,20,1,7,9].
StatusOr<HloInstruction*> ElideDegenerateDims(
    HloInstruction* operand, absl::Span<const int64_t> dims_to_elide);

// Inserts (via reshape) a set of degenerate dimensions (dimensions containing
// exactly one element), `dims_to_insert` into `operand`. The dimensions in
// `dims_to_insert` refer to the dimensions in the result, and hence should be
// less than the rank of the result. Also, `dims_to_insert` must be sorted.
//
// For example, if `operand` is of shape f32[12,21,8,34] and dims_to_insert is
// {0, 2}, then the result is `operand` reshaped to [1,12,1,21,8,34].
StatusOr<HloInstruction*> InsertDegenerateDims(
    HloInstruction* operand, absl::Span<const int64_t> dims_to_insert);

// Pads `operand` (which must have rank 1) with `zeros_to_prepend` zeros in the
// front and `zeros_to_append` zeros in the back.
StatusOr<HloInstruction*> PadVectorWithZeros(HloInstruction* operand,
                                             int64_t zeros_to_prepend,
                                             int64_t zeros_to_append);

// Broadcasts a zero value of type `element_type` into a tensor with element
// type `element_type` and dimension bounds `broadcast_dimensions`.  The
// broadcast instruction is emitted into `computation`.
HloInstruction* BroadcastZeros(HloComputation* computation,
                               PrimitiveType element_type,
                               absl::Span<const int64_t> broadcast_dimensions);

// Same as above, but fill the tensor with ones.
HloInstruction* BroadcastOnes(HloComputation* computation,
                              PrimitiveType element_type,
                              absl::Span<const int64_t> broadcast_dimensions);

// Creates a HLO computation that takes arguments of type `domain` and produces
// a value of type `range`.
StatusOr<std::unique_ptr<HloComputation>> CreateComputationWithSignature(
    absl::Span<const Shape* const> domain, const Shape& range,
    absl::string_view name);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CREATION_UTILS_H_
