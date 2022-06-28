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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering TensorFlow dialect's control flow to
// the XLA dialect.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes_detail.h"

using mlir::PassRegistration;

namespace mlir {
namespace mhlo {
namespace {
class LegalizeTFControlFlow
    : public LegalizeTFControlFlowBase<LegalizeTFControlFlow> {
 public:
  void runOnOperation() override;
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLegalizeTFControlFlowPass() {
  return std::make_unique<LegalizeTFControlFlow>();
}

namespace {

void Detuple(Value tuple, ValueRange replace, OpBuilder* builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_0(mht_0_v, 230, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "Detuple");

  // De-tuple the results of the xla hlo if result.
  for (auto result_it : llvm::enumerate(replace)) {
    auto get_tuple_value = builder->create<mhlo::GetTupleElementOp>(
        result_it.value().getLoc(), tuple, result_it.index());
    result_it.value().replaceAllUsesWith(get_tuple_value);
  }
}

// For mlir::IfOp or mlir::CaseOp, replace the uses of their region's block
// arguments with 'implicit_operands'. Here | 'implicit_operands' | == Number of
// arguments in any of the regions in IfOp or CaseOp.
void ReplaceBlockArgumentsWithImplicitOperands(
    mlir::Operation* op, llvm::ArrayRef<mlir::Value> implicit_operands) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "ReplaceBlockArgumentsWithImplicitOperands");

  assert((mlir::dyn_cast<mlir::mhlo::IfOp>(*op) ||
          mlir::dyn_cast<mlir::mhlo::CaseOp>(*op)) &&
         "Unexpected mlir op in ReplaceBlockArgumentsWithImplicitOperands!");

  for (auto& region : op->getRegions()) {
    int implicit_operand_index = 0;
    for (auto arg : region.getArguments()) {
      assert(implicit_operand_index < implicit_operands.size());
      arg.replaceAllUsesWith(implicit_operands[implicit_operand_index++]);
    }

    region.front().eraseArguments(
        llvm::to_vector(llvm::seq<unsigned>(0, region.getNumArguments())));
  }
}

// Imports the source region into the destination region. MHLO supports
// multiple arguments per branch and multiple returns which are individually
// tupled together during export to XLA. This tupling is needed as XLA if/while
// operation only supports one argument per branch and a single return value.
// `tuple_arg` allows any branch that requires additional arguments to have
// their values be tupled together. Similarly, `tuple_return` allows the results
// of the if/while operation to be tupled together.
void ImportXlaRegion(mlir::func::FuncOp func, Region* dest_region, Location loc,
                     bool tuple_return = true, bool tuple_arg = true) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_2(mht_2_v, 274, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "ImportXlaRegion");

  OpBuilder builder(dest_region);

  auto entry_block = builder.createBlock(dest_region);
  func::CallOp result;
  if (!tuple_arg) {
    auto inputs = func.getFunctionType().getInputs();
    auto args = entry_block->addArguments(
        inputs, SmallVector<Location>(inputs.size(), loc));
    ArrayRef<Value> callop_args(args.begin(), args.end());
    result = builder.create<func::CallOp>(loc, func, callop_args);
  } else {
    auto tuple_arg = entry_block->addArgument(
        builder.getTupleType(func.getFunctionType().getInputs()), loc);
    llvm::SmallVector<Value, 4> detupled_args;
    detupled_args.reserve(func.getNumArguments());

    for (int64_t i = 0, s = func.getNumArguments(); i < s; i++) {
      auto extract = builder.create<GetTupleElementOp>(loc, tuple_arg, i);
      detupled_args.push_back(extract);
    }

    result = builder.create<func::CallOp>(loc, func, detupled_args);
  }

  if (!tuple_return) {
    builder.create<mhlo::ReturnOp>(loc, result.getResults());
  } else {
    auto tuple_op = builder.create<TupleOp>(loc, result.getResults());
    builder.create<mhlo::ReturnOp>(loc, tuple_op.getResult());
  }
}

void LowerIf(TF::IfOp op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_3(mht_3_v, 310, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "LowerIf");

  Location loc = op.getLoc();
  OpBuilder builder(op);

  SmallVector<Value, 3> inputs(op.input());

  // Create the new `mhlo.if` op.
  auto if_op = builder.create<mhlo::IfOp>(loc, op.getResultTypes(), op.cond());

  // Import the regions for both the true and false cases. These regions
  // must be updated to tuple the return results together and use the xla hlo
  // return op.
  ImportXlaRegion(op.then_function(), &if_op.true_branch(), loc,
                  /*tuple_return=*/false, /*tuple_arg=*/false);
  ImportXlaRegion(op.else_function(), &if_op.false_branch(), loc,
                  /*tuple_return=*/false, /*tuple_arg=*/false);

  // Replace the uses of block-arguments of the IfOp with the
  // implicit_operands.
  ReplaceBlockArgumentsWithImplicitOperands(if_op.getOperation(), inputs);

  op->replaceAllUsesWith(if_op);
  op.erase();
}

void LowerCase(TF::CaseOp op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_4(mht_4_v, 338, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "LowerCase");

  Location loc = op.getLoc();
  OpBuilder builder(op);

  SmallVector<Value, 4> inputs(op.input());

  // Create the new `mhlo.case` op.
  auto case_op = builder.create<mhlo::CaseOp>(
      loc, op.getResultTypes(), op.branch_index(), op.branches().size());

  // Import the regions for all branches.
  for (unsigned i = 0; i < op.num_branches(); ++i) {
    mlir::func::FuncOp branch_func = op.branch_function(i);
    ImportXlaRegion(branch_func, &case_op.branches()[i], loc,
                    /*tuple_return=*/false, /*tuple_arg=*/false);
  }

  // Replace the uses of block-arguments of the IfOp with the
  // implicit_operands.
  ReplaceBlockArgumentsWithImplicitOperands(case_op.getOperation(), inputs);

  op.replaceAllUsesWith(case_op);
  op.erase();
}

void LowerWhile(TF::WhileOp op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_5(mht_5_v, 366, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "LowerWhile");

  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA prefers tuple arguments for control flow due to XLA not supporting
  // multiple return values.
  SmallVector<Value, 3> inputs(op.input());
  builder.setInsertionPoint(op);

  // Create the new `mhlo.while` op with inputs.
  auto while_op =
      builder.create<mhlo::WhileOp>(loc, op.getResultTypes(), inputs);

  // Import the regions for both the cond and body.
  ImportXlaRegion(op.body_function(), &while_op.body(), loc,
                  /*tuple_return=*/false, /*tuple_arg=*/false);
  ImportXlaRegion(op.cond_function(), &while_op.cond(), loc,
                  /*tuple_return=*/false, /*tuple_arg=*/false);

  op->replaceAllUsesWith(while_op);
  op.erase();
}

// Replaces all block arguments of a block with a single block arg of Tuple
// type `tuple_type`. Single block arguments are removed and remapped to
// get_tuple_element(tuple_arg, index).
void ReplaceBlockArgs(Block* block, Type tuple_type, OpBuilder* builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_6(mht_6_v, 395, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "ReplaceBlockArgs");

  auto tuple_arg = block->addArgument(tuple_type, block->getParent()->getLoc());
  Detuple(tuple_arg, block->getArguments().drop_back(1), builder);
  for (int i = block->getNumArguments() - 2; i >= 0; --i)
    block->eraseArgument(i);
}

// Replaces implicitly captured value uses with block arguments.
llvm::SmallVector<Value, 4> ReplaceImplicitInputs(
    Block* block, int offset, ArrayRef<Value> implicit_inputs) {
  llvm::SmallVector<Value, 4> implicit_input_elements;
  implicit_input_elements.reserve(implicit_inputs.size());

  Region* region = block->getParent();

  for (auto& implicit_input : llvm::enumerate(implicit_inputs)) {
    Value implicit_input_value = implicit_input.value();
    BlockArgument arg = block->getArgument(implicit_input.index() + offset);
    implicit_input_elements.emplace_back(arg);
    for (auto& use :
         llvm::make_early_inc_range(implicit_input_value.getUses())) {
      if (!region->isAncestor(use.getOwner()->getParentRegion())) continue;
      use.set(arg);
    }
  }

  return implicit_input_elements;
}

// Replaces implicitly captured value uses with tuple block argument.
// get_tuple_element's are created to extract specific values. Values from
// get_tuple_element's are returned in the order of `implicit_inputs`.
llvm::SmallVector<Value, 4> ReplaceImplicitInputsWithTupleElements(
    Block* block, int offset, ArrayRef<Value> implicit_inputs,
    OpBuilder* builder) {
  llvm::SmallVector<Value, 4> implicit_input_elements;
  implicit_input_elements.reserve(implicit_inputs.size());

  Region* region = block->getParent();
  assert(block->getNumArguments() == 1);

  BlockArgument tuple_arg = block->getArgument(0);
  for (auto& implicit_input : llvm::enumerate(implicit_inputs)) {
    Value implicit_input_value = implicit_input.value();
    auto get_tuple_element = builder->create<mhlo::GetTupleElementOp>(
        implicit_input_value.getLoc(), tuple_arg,
        implicit_input.index() + offset);
    implicit_input_elements.emplace_back(get_tuple_element.getResult());
    for (auto& use :
         llvm::make_early_inc_range(implicit_input_value.getUses())) {
      if (!region->isAncestor(use.getOwner()->getParentRegion())) continue;
      use.set(get_tuple_element.getResult());
    }
  }

  return implicit_input_elements;
}

// Finds and replaces implicitly captured value uses with tuple block argument.
// A tuple of implicitly captured values is also created and returned, for use
// as an operand to the associated mhlo control flow op.
Value TupleImplicitInputs(Region& region, Location loc, OpBuilder* builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_7(mht_7_v, 459, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "TupleImplicitInputs");

  llvm::SetVector<Value> implicit_inputs;
  getUsedValuesDefinedAbove(region, region, implicit_inputs);
  llvm::ArrayRef<Value> implicit_inputs_ref = implicit_inputs.getArrayRef();
  Value tuple_input = builder->create<mhlo::TupleOp>(loc, implicit_inputs_ref);
  Block& block = region.front();
  // `tf.CaseRegion`/`tf.IfRegion` are expected to have no block arguments and
  // instead all inputs used by their branch regions are implicitly captured
  // from above.
  assert(block.getNumArguments() == 0);
  block.addArgument(tuple_input.getType(), loc);
  builder->setInsertionPointToStart(&block);
  ReplaceImplicitInputsWithTupleElements(&block, /*offset=*/0,
                                         implicit_inputs_ref, builder);
  return tuple_input;
}

// Replaces block terminator (tf.Yield) with `mhlo.return`. Additional results
// can be returned if `extra_results` is not empty. If `tuple_return` is
// set, a tuple of the return values will be set as the terminator operand.
void ReplaceTerminator(Block* block, ArrayRef<Value> extra_results,
                       OpBuilder* builder, bool tuple_return = true) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_8(mht_8_v, 483, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "ReplaceTerminator");

  Operation* terminator = block->getTerminator();
  assert(isa<TF::YieldOp>(terminator));
  Location loc = terminator->getLoc();

  builder->setInsertionPoint(terminator);
  auto results = llvm::to_vector<4>(terminator->getOperands());
  results.append(extra_results.begin(), extra_results.end());
  if (tuple_return) {
    auto tuple_results = builder->create<mhlo::TupleOp>(loc, results);
    builder->create<mhlo::ReturnOp>(loc, tuple_results.getResult());
  } else {
    builder->create<mhlo::ReturnOp>(loc, results);
  }

  terminator->erase();
}

void LowerIfRegion(TF::IfRegionOp op) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_9(mht_9_v, 504, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "LowerIfRegion");

  Location loc = op.getLoc();
  OpBuilder builder(op);

  builder.setInsertionPoint(op);
  ReplaceTerminator(&op.then_branch().front(), /*extra_results=*/{}, &builder,
                    /*tuple_return=*/false);

  builder.setInsertionPoint(op);
  ReplaceTerminator(&op.else_branch().front(), /*extra_results=*/{}, &builder,
                    /*tuple_return=*/false);

  // Create the new `mhlo.if` op and take ownership of regions from
  // `tf.IfRegion` op.
  builder.setInsertionPoint(op);
  auto if_op = builder.create<mhlo::IfOp>(loc, op.getResultTypes(), op.cond());
  if_op.true_branch().takeBody(op.then_branch());
  if_op.false_branch().takeBody(op.else_branch());

  // Replace all uses of `op` results with that of `mhlo.IfOp`.
  op->replaceAllUsesWith(if_op);

  op.erase();
}

void LowerCaseRegion(TF::CaseRegionOp op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_10(mht_10_v, 532, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "LowerCaseRegion");

  Location loc = op.getLoc();
  OpBuilder builder(op);

  for (Region& region : op.branches()) {
    builder.setInsertionPoint(op);
    ReplaceTerminator(&region.front(), /*extra_results=*/{}, &builder,
                      /*tuple_return=*/false);
  }

  // Create the new `mhlo.case` op and take ownership of regions from
  // `tf.CaseRegion` op.
  builder.setInsertionPoint(op);
  auto case_op = builder.create<mhlo::CaseOp>(
      loc, op.getResultTypes(), op.branch_index(), op.branches().size());
  for (auto region : llvm::zip(case_op.branches(), op.branches()))
    std::get<0>(region).takeBody(std::get<1>(region));

  // Replace all uses of `op` results with that of `mhlo.CaseOp`.
  op.replaceAllUsesWith(case_op);
  op.erase();
}

void LowerWhileRegion(TF::WhileRegionOp op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_11(mht_11_v, 558, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "LowerWhileRegion");

  Location loc = op.getLoc();
  OpBuilder builder(op);

  SmallVector<Value, 3> inputs(op.input());
  const int inputs_size = inputs.size();
  llvm::SetVector<Value> implicit_inputs;
  getUsedValuesDefinedAbove(op.getOperation()->getRegions(), implicit_inputs);
  inputs.append(implicit_inputs.begin(), implicit_inputs.end());

  builder.setInsertionPoint(op);

  // Create the new `mhlo.while` op with 'inputs'. Implicit inputs are also
  // returned.
  auto while_result_types = llvm::to_vector<4>(op.getResultTypes());
  while_result_types.reserve(while_result_types.size() +
                             implicit_inputs.size());
  for (const auto& implicit_input : implicit_inputs)
    while_result_types.emplace_back(implicit_input.getType());
  auto while_op =
      builder.create<mhlo::WhileOp>(loc, while_result_types, inputs);

  // Rewrite cond and associated block arguments and terminator. Ownership of
  // cond region is transfered over from `tf.WhileRegion` to `mhlo.while`.
  Region& cond = while_op.cond();
  cond.takeBody(op.cond());
  Block& cond_block = cond.front();
  builder.setInsertionPointToStart(&cond_block);

  // Add args corresponding to 'implicit_inputs'.
  for (const auto& implicit_input : implicit_inputs)
    cond_block.addArgument(implicit_input.getType(), loc);
  ReplaceImplicitInputs(&cond_block, inputs_size,
                        implicit_inputs.getArrayRef());
  // Cond always returns a single result of bool type.
  ReplaceTerminator(&cond_block, /*extra_results=*/{}, &builder,
                    /*tuple_return=*/false);

  // Rewrite body and associated block arguments and terminator. Ownership of
  // body region is transfered over from `tf.WhileRegion` to `mhlo.while`.
  Region& body = while_op.body();
  body.takeBody(op.body());
  Block& body_block = body.front();
  builder.setInsertionPointToStart(&body_block);
  // Add args corresponding to 'implicit_inputs'.
  for (const auto& implicit_input : implicit_inputs)
    body_block.addArgument(implicit_input.getType(), loc);
  auto implicit_input_elements = ReplaceImplicitInputs(
      &body_block, inputs_size, implicit_inputs.getArrayRef());
  ReplaceTerminator(&body_block, implicit_input_elements, &builder, false);

  // Replace all uses of `op` results with that of `mhlo.while`.
  builder.setInsertionPoint(op);
  if (while_op.getNumResults() > 1) {
    for (const auto& result_it : llvm::enumerate(op.getResults()))
      result_it.value().replaceAllUsesWith(
          while_op.getResult(result_it.index()));
  } else {
    op->replaceAllUsesWith(while_op);
  }
  op.erase();
}
}  // namespace

void LegalizeTFControlFlow::runOnOperation() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_control_flowDTcc mht_12(mht_12_v, 625, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_control_flow.cc", "LegalizeTFControlFlow::runOnOperation");

  getOperation().walk([&](Operation* op) {
    if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
      LowerWhile(while_op);
      return;
    }
    if (auto while_region_op = dyn_cast<TF::WhileRegionOp>(op)) {
      LowerWhileRegion(while_region_op);
      return;
    }
    if (auto if_op = dyn_cast<TF::IfOp>(op)) {
      LowerIf(if_op);
      return;
    }
    if (auto if_region_op = dyn_cast<TF::IfRegionOp>(op)) {
      LowerIfRegion(if_region_op);
      return;
    }
    if (auto case_op = dyn_cast<TF::CaseOp>(op)) {
      LowerCase(case_op);
      return;
    }
    if (auto case_region_op = dyn_cast<TF::CaseRegionOp>(op)) {
      LowerCaseRegion(case_region_op);
      return;
    }
  });
}
}  // namespace mhlo
}  // namespace mlir
