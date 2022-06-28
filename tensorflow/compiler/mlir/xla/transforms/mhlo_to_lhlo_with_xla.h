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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MHLO_TO_LHLO_WITH_XLA_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MHLO_TO_LHLO_WITH_XLA_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSmhlo_to_lhlo_with_xlaDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSmhlo_to_lhlo_with_xlaDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSmhlo_to_lhlo_with_xlaDTh() {
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


#include "absl/types/optional.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace mlir {

// This class will process an HloModule with the supplied BufferAssignment and
// populate the MLIR ModuleOp with the computation converted in the LHLO
// dialect.
class LhloDialectEmitter : public xla::ConstDfsHloVisitorWithDefault {
 public:
  // Initializes internal data structures. It must be called before calling any
  // of the visitors.
  tensorflow::Status Initialize();

  LhloDialectEmitter(const xla::BufferAssignment& assignment,
                     const xla::HloComputation& computation, ModuleOp module)
      : assignment_(std::move(assignment)),
        computation_(computation),
        module_(module),
        builder_(module.getContext()),
        i8_type_(builder_.getIntegerType(8)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSmhlo_to_lhlo_with_xlaDTh mht_0(mht_0_v, 220, "", "./tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h", "LhloDialectEmitter");
}

  xla::StatusOr<mlir::Operation*> EmitOp(const xla::HloInstruction* instr);

  static xla::StatusOr<mhlo::ScatterDimensionNumbersAttr>
  GetScatterDimensionNumbers(const xla::HloInstruction* instr,
                             mlir::MLIRContext* context);

 private:
  xla::StatusOr<lmhlo::SortOp> EmitSortOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::FusionOp> EmitFusionOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::ScatterOp> EmitScatterOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::SelectAndScatterOp> EmitSelectAndScatterOp(
      const xla::HloInstruction* instr);

  xla::StatusOr<Operation*> EmitCustomCallOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo_gpu::CholeskyOp> EmitCholesky(
      const xla::HloCustomCallInstruction* custom_call);
  xla::StatusOr<Operation*> EmitGemm(
      const xla::HloCustomCallInstruction* custom_call);
  xla::StatusOr<Operation*> EmitDnnConvolution(
      const xla::HloCustomCallInstruction* custom_call);
  xla::StatusOr<Operation*> EmitDnnBatchNorm(
      const xla::HloCustomCallInstruction* custom_call);

  xla::StatusOr<memref::GetGlobalOp> EmitConstant(
      const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::InfeedOp> EmitInfeedOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::OutfeedOp> EmitOutfeedOp(
      const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::AllToAllOp> EmitAllToAllOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::AllGatherOp> EmitAllGatherOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::AllReduceOp> EmitAllReduceOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo_gpu::AllReduceStartOp> EmitAllReduceStartOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo_gpu::AllReduceDoneOp> EmitAllReduceDoneOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::ReduceScatterOp> EmitReduceScatterOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::CollectivePermuteOp> EmitCollectivePermuteOp(
      const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::RngGetAndUpdateStateOp> EmitRngGetAndUpdateStateOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::FftOp> EmitFftOp(const xla::HloInstruction* instr);
  xla::StatusOr<lmhlo::TriangularSolveOp> EmitTriangularSolveOp(
      const xla::HloInstruction* instr);
  xla::StatusOr<Operation*> EmitBitcast(const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::CaseOp> EmitCaseOp(const xla::HloInstruction* instr);

  xla::StatusOr<lmhlo::WhileOp> EmitWhileOp(const xla::HloInstruction* instr);

  xla::Status ImportAsLmhloRegion(xla::HloComputation* computation,
                                  mlir::Region* region);

  // Since LMHLO dialect does not define token types, this enum controls how
  // token operand/results from XLA:HLO are lowered to MLIR.
  enum class TokenLoweringMode {
    kFailToLower,  // Fail lowering if token inputs are encountered.
    kUseNull,      // Use a null Value in the operand list for each token.
    // kSkip,        // Skip any token inputs or outputs (not yet needed)
  };

  // Create LHLO operation operands given an XLA HLO instruction. By default,
  // all XLA HLO operands and results are converted to MLIR and appended to
  // `operands`. If `num_operands` is specified, only the first `num_operand`
  // operands of the instruction are converted to MLIR. The function returns the
  // actual number of operands and results generated for MLIR in `num_arguments`
  // and `num_results`.
  xla::Status CreateOperands(const xla::HloInstruction* instr,
                             absl::optional<int64_t> num_operands,
                             TokenLoweringMode token_mode,
                             SmallVectorImpl<Value>& operands,
                             size_t& num_arguments, size_t& num_results);

  template <typename OpType>
  xla::StatusOr<OpType> CreateOpWithoutAttrs(
      const xla::HloInstruction* instr,
      absl::optional<int64_t> num_operands = absl::nullopt) {
    size_t unused;
    return CreateOpWithoutAttrs<OpType>(instr, unused, unused, num_operands);
  }

  template <typename OpType>
  xla::StatusOr<OpType> CreateOpWithoutAttrs(
      const xla::HloInstruction* instr, size_t& num_arguments,
      size_t& num_results,
      absl::optional<int64_t> num_operands = absl::nullopt);

  template <typename OpType>
  OpType CreateOpWithoutAttrs(const xla::HloInstruction* instr,
                              ValueRange operands);

  xla::StatusOr<mlir::Operation*> CreateOpInFusion(
      const xla::HloInstruction* instr, ValueRange buffer_operands,
      size_t num_arguments, size_t num_results);

  xla::StatusOr<mlir::Operation*> CreateOpInFusion(
      const xla::HloInstruction* instr);

  template <typename T>
  DenseIntElementsAttr GetI64DenseElementsAttr(const T& container) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSmhlo_to_lhlo_with_xlaDTh mht_1(mht_1_v, 331, "", "./tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h", "GetI64DenseElementsAttr");

    return builder_.getI64TensorAttr(
        {container.data(), static_cast<size_t>(container.size())});
  }

  DenseIntElementsAttr GetWindowElements(
      const xla::Window& window,
      std::function<int64_t(const xla::WindowDimension& dim)> getter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSmhlo_to_lhlo_with_xlaDTh mht_2(mht_2_v, 341, "", "./tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h", "GetWindowElements");

    llvm::SmallVector<int64_t, 4> elements;
    elements.reserve(window.dimensions_size());
    for (const xla::WindowDimension& dim : window.dimensions()) {
      elements.push_back(getter(dim));
    }
    return GetI64DenseElementsAttr(elements);
  }

  static mlir::DenseIntElementsAttr GetLayoutAttribute(
      const xla::Layout& layout, Builder* builder);

  tensorflow::Status DefaultAction(const xla::HloInstruction* instr) final;

  // Computation parameters don't need any specific handling when they are
  // visited, they are already processed when we enter a new computation.
  tensorflow::Status HandleParameter(const xla::HloInstruction* instr) final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSmhlo_to_lhlo_with_xlaDTh mht_3(mht_3_v, 360, "", "./tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h", "HandleParameter");

    return tensorflow::Status::OK();
  }

  // Helper function that recursively visits the tuple structure in
  // `current_shape`, and reconstruct a matching lmhlo::TupleOp.
  // Each leaf node is converted to an std.view op with corresponding offsets.
  // If no tuple presents, it simply returns a view of the buffer.
  tensorflow::Status GetOrCreateViewImpl(const xla::HloInstruction* instr,
                                         const xla::Shape& current_shape,
                                         xla::ShapeIndex* current_shape_index,
                                         SmallVectorImpl<Value>* values,
                                         TokenLoweringMode token_mode);

  // Helper function to create view/tuple of views to a buffer for a given
  // instruction result. `result_subset` can be used to for instructions that
  // have a tuple result and MLIR conversion needs to convert only one of the
  // tuple elements. Note that if needed, this can be extended to take a list of
  // ShapeIndex values in case we need finer control on what elements of the
  // output tuple to be converted to MLIR.
  tensorflow::Status GetOrCreateView(
      const xla::HloInstruction* instr, SmallVectorImpl<Value>* values,
      const xla::ShapeIndex& result_subset = {},
      TokenLoweringMode token_mode = TokenLoweringMode::kFailToLower);

  xla::StatusOr<Value> GetOrCreateArrayView(
      const xla::HloInstruction* instr, const xla::Shape& current_shape,
      const xla::ShapeIndex& current_shape_index);

  xla::StatusOr<Value> RewriteFusionOperand(const xla::HloInstruction* root,
                                            const xla::Shape& shape,
                                            xla::ShapeIndex* shape_index,
                                            OpBuilder* b, Location loc);

  // Return an MLIR location for an HLO instruction.
  Location getLocation(const xla::HloInstruction* inst) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSmhlo_to_lhlo_with_xlaDTh mht_4(mht_4_v, 398, "", "./tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h", "getLocation");

    return NameLoc::get(builder_.getStringAttr(inst->name()));
  }

  // This map provides access to MLIR buffers for each HLO buffer allocation.
  // The MLIR buffers are all `memref<{size}xi8>` and correspond to function
  // parameters. It is populated at the beginning of the processing with all
  // the buffer allocations and is unchanged afterward. Every HLOInstruction
  // is using a "slice" of the buffer allocation and providing shape, layout,
  // and Dtype. An MLIR view is used separately to model slices into the
  // allocations (see below).
  llvm::DenseMap<const xla::BufferAllocation*, Value> allocations_;

  // This map provides access to MLIR buffers for each HLO instruction, keyed
  // instruction identity. A slice is contained in a BufferAllocation, and has
  // an offset and a size.
  //
  // As for why we don't use HloInstruction*, see GetOrCreateView(), but
  // mostly we want to leverage better of the aliased buffers.
  //
  // If the HloInstruction is a tuple, all leaf nodes are stored flattened.
  // Otherwise, there will be a single buffer.
  //
  // An MLIR buffer is either an input parameter, or a ViewOp in the case
  // where the slice is only part of its allocation.
  //
  // `slices_` is populated lazily in the `GetOrCreateView()` helper as we
  // process every instruction.
  absl::flat_hash_map<std::pair<const xla::HloInstruction*, xla::ShapeIndex>,
                      Value>
      slices_;

  // The BufferAssignment computed by XLA ahead of time.
  const xla::BufferAssignment& assignment_;

  // The HLO module that will be converted.
  const xla::HloComputation& computation_;

  // This is the MLIR module in which a function will be created for every HLO
  // computation.
  ModuleOp module_;

  // The builder keeps track of the current insertion point in the MLIR
  // module.
  OpBuilder builder_;
  // Convenient "cached" access to this widely used MLIR type (i8).
  Type i8_type_;

  // Map all-reduce-start ops to their LHLO op, so we can connect the
  // all-reduce-done op with the correct token.
  absl::flat_hash_map<const xla::HloInstruction*, lmhlo_gpu::AllReduceStartOp>
      all_reduce_start_ops_;
};

// Populate the MLIR `module` with the computation from the `hlo_module` using
// the provided buffer `assignment`. The returned `Status` indicates success
// or failure in the conversion.
tensorflow::Status HloToLhloModule(const xla::BufferAssignment& assignment,
                                   const xla::HloModule& hlo_module,
                                   ModuleOp module);

tensorflow::Status OptimizeAndConvertHloToLmhlo(
    std::unique_ptr<xla::HloModule> hlo_module, ModuleOp module,
    StringRef platform_name, bool optimize_xla_hlo);
OwningOpRef<mlir::ModuleOp> HloTextToLhloTranslateFunction(
    llvm::StringRef input, MLIRContext* context, bool optimize_xla_hlo);

// This register the MLIR pass with the command line.
void RegisterMhloToLhloWithXlaPass();

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MHLO_TO_LHLO_WITH_XLA_H_
