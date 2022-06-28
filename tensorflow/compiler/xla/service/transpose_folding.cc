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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc() {
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

#include "tensorflow/compiler/xla/service/transpose_folding.h"

#include <vector>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

TransposeFolding::OperandIndices CanFoldOperandsIntoDot(
    const HloInstruction& dot,
    const TransposeFolding::TransposableGemmOperandsFn&
        transposable_gemm_operands) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/transpose_folding.cc", "CanFoldOperandsIntoDot");

  if (HloOpcode::kDot != dot.opcode()) {
    return {};
  }

  if (!absl::c_equal(dot.dot_dimension_numbers().lhs_batch_dimensions(),
                     dot.dot_dimension_numbers().rhs_batch_dimensions())) {
    return {};
  }

  int64_t num_batch_dims =
      dot.dot_dimension_numbers().lhs_batch_dimensions_size();
  int64_t expected_rank = 2 + num_batch_dims;
  auto is_r2_transpose = [&](const HloInstruction& transpose) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/xla/service/transpose_folding.cc", "lambda");

    if (transpose.opcode() != HloOpcode::kTranspose) {
      return false;
    }
    const auto& transpose_dims = transpose.dimensions();
    if (transpose_dims.size() != expected_rank) {
      return false;
    }

    // Check that the transpose doesn't touch any batch dimensions, but does
    // transpose the non-batch ones.
    for (int64_t i = 0; i != expected_rank; ++i) {
      bool is_batch = absl::c_linear_search(
          dot.dot_dimension_numbers().lhs_batch_dimensions(),
          transpose_dims[i]);
      if ((transpose_dims[i] == i) != is_batch) {
        return false;
      }
    }
    return true;
  };

  TransposeFolding::OperandIndices operand_set;
  for (int64_t i = 0; i < dot.operand_count(); ++i) {
    auto& operand = *dot.operand(i);
    if (is_r2_transpose(operand)) {
      operand_set.push_back(i);
    } else if (operand.shape().rank() != expected_rank) {
      return {};
    }
  }

  return transposable_gemm_operands(dot, operand_set);
}

TransposeFolding::OperandIndices CanFoldOperandsIntoConvolution(
    const HloInstruction& convolution,
    const TransposeFolding::TransposableConvOperandsFn&
        transposable_conv_operands) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc mht_2(mht_2_v, 263, "", "./tensorflow/compiler/xla/service/transpose_folding.cc", "CanFoldOperandsIntoConvolution");

  if (HloOpcode::kConvolution != convolution.opcode()) {
    return {};
  }

  TransposeFolding::OperandIndices operand_set;
  for (int64_t i = 0; i < convolution.operand_count(); ++i) {
    auto& operand = *convolution.operand(i);
    if (operand.opcode() == HloOpcode::kTranspose) {
      operand_set.push_back(i);
    }
  }

  return transposable_conv_operands(convolution, operand_set);
}

using InstructionOperandsPair =
    std::pair<HloInstruction*, TransposeFolding::OperandIndices>;

// Folds the operands of `dot` that are foldable transposes. `computation` is
// the parent HLO computation of `dot`.
Status FoldTransposeIntoDot(InstructionOperandsPair pair) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc mht_3(mht_3_v, 287, "", "./tensorflow/compiler/xla/service/transpose_folding.cc", "FoldTransposeIntoDot");

  HloInstruction* dot = pair.first;

  DotDimensionNumbers new_dim_numbers = dot->dot_dimension_numbers();
  HloInstruction* new_lhs = dot->mutable_operand(0);
  HloInstruction* new_rhs = dot->mutable_operand(1);

  CHECK_EQ(new_dim_numbers.lhs_contracting_dimensions_size(), 1);
  CHECK_EQ(new_dim_numbers.rhs_contracting_dimensions_size(), 1);

  for (int64_t operand_index : pair.second) {
    // We checked that the batch dimensions are not touched by the transpose,
    // and shape inference guarantees that there is exactly one contracting
    // dimension.
    if (operand_index == 0) {
      CHECK_EQ(new_lhs->opcode(), HloOpcode::kTranspose);
      new_dim_numbers.set_lhs_contracting_dimensions(
          0,
          new_lhs->dimensions(new_dim_numbers.lhs_contracting_dimensions(0)));
      new_lhs = new_lhs->mutable_operand(0);
    } else {
      CHECK_EQ(operand_index, 1);
      CHECK_EQ(new_rhs->opcode(), HloOpcode::kTranspose);
      new_dim_numbers.set_rhs_contracting_dimensions(
          0,
          new_rhs->dimensions(new_dim_numbers.rhs_contracting_dimensions(0)));
      new_rhs = new_rhs->mutable_operand(0);
    }
  }

  std::unique_ptr<HloInstruction> new_dot = HloInstruction::CreateDot(
      dot->shape(), new_lhs, new_rhs, new_dim_numbers, dot->precision_config());
  return dot->parent()->ReplaceWithNewInstruction(dot, std::move(new_dot));
}

// Folds the operands of `convolution` that are foldable transposes.
// `computation` is the parent HLO computation of `convolution`.
//
// Returns whether the module is changed.
bool FoldTransposeIntoConvolution(InstructionOperandsPair pair) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc mht_4(mht_4_v, 329, "", "./tensorflow/compiler/xla/service/transpose_folding.cc", "FoldTransposeIntoConvolution");

  auto& convolution = *pair.first;
  auto& operand_indices = pair.second;

  if (operand_indices.empty()) {
    return false;
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  ConvolutionDimensionNumbers new_dnums = dnums;

  HloInstruction* new_lhs;
  const int64_t kLhsIdx = 0;
  if (absl::c_linear_search(operand_indices, kLhsIdx)) {
    HloInstruction& transpose = *convolution.mutable_operand(kLhsIdx);
    const auto& transpose_dimensions = transpose.dimensions();
    HloInstruction& transpose_operand = *transpose.mutable_operand(0);

    // Everything remains the same except for the input/output dimension
    // numbers. We need to apply the transpose permutation to the original shape
    // to figure out what the new logical dimensions are.
    new_dnums.set_input_batch_dimension(
        transpose_dimensions[dnums.input_batch_dimension()]);
    new_dnums.set_input_feature_dimension(
        transpose_dimensions[dnums.input_feature_dimension()]);
    for (auto& input_spatial_dimension :
         *new_dnums.mutable_input_spatial_dimensions()) {
      input_spatial_dimension = transpose_dimensions[input_spatial_dimension];
    }
    new_lhs = &transpose_operand;
  } else {
    new_lhs = convolution.mutable_operand(kLhsIdx);
  }

  HloInstruction* new_rhs;
  const int64_t kRhsIdx = 1;
  if (absl::c_linear_search(operand_indices, kRhsIdx)) {
    HloInstruction& transpose = *convolution.mutable_operand(kRhsIdx);
    const auto& transpose_dimensions = transpose.dimensions();
    HloInstruction& transpose_operand = *transpose.mutable_operand(0);

    // Everything remains the same except for the kernel dimension numbers. We
    // need to apply the transpose permutation to the original shape to figure
    // out what the new logical dimensions are.
    new_dnums.set_kernel_input_feature_dimension(
        transpose_dimensions[dnums.kernel_input_feature_dimension()]);
    new_dnums.set_kernel_output_feature_dimension(
        transpose_dimensions[dnums.kernel_output_feature_dimension()]);
    for (auto& kernel_spatial_dimension :
         *new_dnums.mutable_kernel_spatial_dimensions()) {
      kernel_spatial_dimension = transpose_dimensions[kernel_spatial_dimension];
    }
    new_rhs = &transpose_operand;
  } else {
    new_rhs = convolution.mutable_operand(kRhsIdx);
  }

  auto new_conv = HloInstruction::CreateConvolve(
      convolution.shape(), new_lhs, new_rhs, convolution.feature_group_count(),
      convolution.batch_group_count(), convolution.window(), new_dnums,
      convolution.precision_config());
  TF_CHECK_OK(convolution.parent()->ReplaceWithNewInstruction(
      &convolution, std::move(new_conv)));

  return true;
}

}  // namespace

TransposeFolding::TransposeFolding(
    TransposableGemmOperandsFn transposable_gemm_operands,
    TransposableConvOperandsFn transposable_conv_operands)
    : transposable_gemm_operands_(std::move(transposable_gemm_operands)),
      transposable_conv_operands_(std::move(transposable_conv_operands)) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc mht_5(mht_5_v, 406, "", "./tensorflow/compiler/xla/service/transpose_folding.cc", "TransposeFolding::TransposeFolding");
}

StatusOr<bool> TransposeFolding::Run(HloModule* module) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStranspose_foldingDTcc mht_6(mht_6_v, 411, "", "./tensorflow/compiler/xla/service/transpose_folding.cc", "TransposeFolding::Run");

  // Modifying the graph while traversing is dangerous, so we find all folding
  // opportunities before actually folding them.
  std::vector<std::pair<HloInstruction*, OperandIndices>> foldable_dots;
  std::vector<std::pair<HloInstruction*, OperandIndices>> foldable_convolutions;
  FunctionVisitor visit_fn([this, &foldable_dots, &foldable_convolutions](
                               HloInstruction* instruction) {
    {
      OperandIndices operand_indices =
          CanFoldOperandsIntoDot(*instruction, transposable_gemm_operands_);
      if (!operand_indices.empty()) {
        foldable_dots.emplace_back(instruction, operand_indices);
      }
    }
    {
      OperandIndices operand_indices = CanFoldOperandsIntoConvolution(
          *instruction, transposable_conv_operands_);
      if (!operand_indices.empty()) {
        foldable_convolutions.emplace_back(
            std::make_pair(instruction, operand_indices));
      }
    }
    return Status::OK();
  });

  for (auto* comp : module->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(comp->Accept(&visit_fn));
  }

  bool changed = false;
  for (InstructionOperandsPair& pair : foldable_dots) {
    TF_RETURN_IF_ERROR(FoldTransposeIntoDot(pair));
    changed = true;
  }
  for (InstructionOperandsPair& pair : foldable_convolutions) {
    changed |= FoldTransposeIntoConvolution(pair);
  }
  return changed;
}

}  // namespace xla
