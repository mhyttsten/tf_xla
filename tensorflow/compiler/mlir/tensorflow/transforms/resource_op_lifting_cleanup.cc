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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.h"

#include "llvm/ADT/BitVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace {

bool IsResource(Value value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "IsResource");

  return getElementTypeOrSelf(value.getType()).isa<TF::ResourceType>();
}

// Checks if a cast op is casting a resource -> resource.
bool IsCastOfResource(Operation &op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "IsCastOfResource");

  auto cast = dyn_cast<TF::CastOp>(op);
  if (!cast) return false;
  return IsResource(cast.x());
}

// Removes passthrough ops in the block. The device computation does not need
// such nodes to carry information.
void RemovePassthroughOp(Block &block) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_2(mht_2_v, 219, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "RemovePassthroughOp");

  for (auto &op : llvm::make_early_inc_range(block)) {
    if (isa<TF::IdentityOp, TF::IdentityNOp>(op) || IsCastOfResource(op)) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    }
  }
}

// Eliminate local variables that are only assigned to but never read, and thus
// are dead.
void RemoveDeadLocalVariables(Block &block) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_3(mht_3_v, 233, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "RemoveDeadLocalVariables");

  llvm::SmallVector<TF::MlirLocalVarOp, 8> local_vars;
  for (Operation &op : block) {
    if (auto local_var = llvm::dyn_cast<TF::MlirLocalVarOp>(&op)) {
      local_vars.push_back(local_var);
    }
  }
  for (auto local_var : local_vars) {
    auto users = local_var.resource().getUsers();
    if (llvm::all_of(users, [](const Operation *user) {
          return isa<TF::AssignVariableOp>(user);
        })) {
      for (auto user : llvm::make_early_inc_range(users)) user->erase();
      local_var.erase();
    }
  }
}

LogicalResult CleanupAndCanonicalize(Operation *parent_op);

// Eliminates unusued results from an operation `op` by cloning it with reduced
// result types and doing appropriate use replacements. `results_to_eliminate`
// is a bitvector of result positions to eliminate. If its null, then all unused
// results of the operation will be eliminated.
void EliminateUnusedResults(
    Operation *op, const llvm::BitVector *results_to_eliminate = nullptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_4(mht_4_v, 261, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "EliminateUnusedResults");

  auto can_eliminate = [&](OpResult &result) -> bool {
    if (!result.use_empty()) return false;
    if (results_to_eliminate)
      return results_to_eliminate->test(result.getResultNumber());
    else
      return true;
  };
  SmallVector<Type, 4> new_result_types;
  for (OpResult result : op->getResults()) {
    if (can_eliminate(result)) continue;
    new_result_types.push_back(result.getType());
  }

  // Rebuild the new operation with lesser number of results.
  OpBuilder builder(op);
  Operation *new_op = Operation::create(
      op->getLoc(), op->getName(), new_result_types, op->getOperands(),
      op->getAttrs(), op->getSuccessors(), op->getNumRegions());
  builder.insert(new_op);

  // Move region bodies to the new operation.
  for (auto it : llvm::zip(op->getRegions(), new_op->getRegions())) {
    Region &old_region = std::get<0>(it);
    Region &new_region = std::get<1>(it);
    new_region.takeBody(old_region);
  }

  // Replace used results and erase the old op.
  int next_result_idx = 0;
  for (OpResult result : op->getResults()) {
    if (can_eliminate(result)) continue;
    result.replaceAllUsesWith(new_op->getResult(next_result_idx++));
  }
  op->erase();
}

// Clones a function if it cannot be patched in place. Clone if there are
// multiple uses or unknown uses (for external functions). The cloned function
// will be marked as private.
FuncOp CloneFunctionIfNeeded(FuncOp func) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_5(mht_5_v, 304, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "CloneFunctionIfNeeded");

  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto func_uses = SymbolTable::getSymbolUses(func, &module.getBodyRegion());
  if (func_uses.hasValue() && llvm::hasSingleElement(func_uses.getValue()))
    return func;
  FuncOp cloned = func.clone();
  cloned.setPrivate();
  cloned.setName(
      StringAttr::get(func.getContext(), func.getName().str() + "_lifted"));
  SymbolTable(module).insert(cloned);
  return cloned;
}

// Eliminates unused results for If/Case operations. Also patches up the
// branch functions to (a) drop the ununsed return values, and (b) as a result
// if some argument becomes unused in all branches, drop that argument and the
// corresponding if/case input operand.
void EliminateUnusedResultsForIfCase(Operation *op, ArrayRef<FuncOp> branches) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_6(mht_6_v, 324, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "EliminateUnusedResultsForIfCase");

  // Clone branch functions if needed since we will be mutating them.
  SmallVector<FuncOp, 2> cloned_branches;
  cloned_branches.reserve(branches.size());
  for (FuncOp func : branches) {
    FuncOp cloned = CloneFunctionIfNeeded(func);
    cloned_branches.push_back(cloned);
    if (cloned == func) continue;
    // Patch up the op attribute to point to the new function.
    for (NamedAttribute attr : op->getAttrs()) {
      auto symref = attr.getValue().dyn_cast<FlatSymbolRefAttr>();
      if (!symref) continue;
      if (symref.getValue() != func.getName()) continue;
      op->setAttr(attr.getName(),
                  FlatSymbolRefAttr::get(op->getContext(), cloned.getName()));
      break;
    }
  }

  // Traverse results backward so that indices to be deleted stay unchanged.
  for (OpResult result : llvm::reverse(op->getResults())) {
    if (!result.use_empty()) continue;
    int result_idx = result.getResultNumber();
    for (FuncOp func : cloned_branches)
      func.front().getTerminator()->eraseOperand(result_idx);
  }

  // Check which function arguments are unused in all branches. We can drop
  // those as well.
  int num_args = cloned_branches[0].getNumArguments();
  llvm::BitVector used_args(num_args);
  for (FuncOp func : branches) {
    for (BlockArgument arg : func.getArguments()) {
      if (!arg.use_empty()) used_args.set(arg.getArgNumber());
    }
  }

  // There are some unused args that we can drop. Also drop the corresponding
  // input operand.
  if (used_args.count() != num_args) {
    // Traverse arguments backward so that indices to be deleted stay unchanged.
    for (int idx = num_args - 1; idx >= 0; --idx) {
      if (used_args.test(idx)) continue;
      for (FuncOp func : cloned_branches) func.eraseArgument(idx);
      // For if/case, arg #i of attached function corresponds to operand #i+1
      op->eraseOperand(idx + 1);
    }
  }

  // Patch up function types (with less number of return values and potentially
  // less number of arguments)
  for (FuncOp func : cloned_branches) {
    func.setType(
        FunctionType::get(func.getContext(), func.front().getArgumentTypes(),
                          func.front().getTerminator()->getOperandTypes()));
  }

  EliminateUnusedResults(op);
}

// Eliminated unused results from a functional while.
void EliminateUnusedResultsForWhile(TF::WhileOp op) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_7(mht_7_v, 388, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "EliminateUnusedResultsForWhile");

  FuncOp cond = op.cond_function();
  FuncOp body = op.body_function();

  llvm::BitVector can_eliminate(op.getNumResults());
  for (OpResult result : llvm::reverse(op.getResults())) {
    if (!result.use_empty()) continue;
    int result_idx = result.getResultNumber();
    BlockArgument cond_arg = cond.getArgument(result_idx);
    BlockArgument body_arg = cond.getArgument(result_idx);
    Operation *body_ret = body.front().getTerminator();
    // We can eliminate a result if its unused and the corresponding argument
    // is unused in cond and the only use in body is use it as a return value.
    if (cond_arg.use_empty() && body_arg.hasOneUse() &&
        body_arg.use_begin()->getOperandNumber() == result_idx &&
        body_arg.use_begin()->getOwner() == body_ret) {
      can_eliminate.set(result_idx);
    }
  }

  if (can_eliminate.empty()) return;

  FuncOp cloned_cond = CloneFunctionIfNeeded(cond);
  FuncOp cloned_body = CloneFunctionIfNeeded(body);
  op.condAttr(FlatSymbolRefAttr::get(op.getContext(), cloned_cond.getName()));
  op.bodyAttr(FlatSymbolRefAttr::get(op.getContext(), cloned_body.getName()));

  // Drop cond/body args and return value. WhileOp result will be dropped later
  // in EliminateUnusedResults. Traverse in reverse order so that indices to be
  // deleted stay unchanged.
  for (int idx = op.getNumResults() - 1; idx >= 0; --idx) {
    if (!can_eliminate.test(idx)) continue;
    cloned_cond.eraseArgument(idx);
    cloned_body.front().getTerminator()->eraseOperand(idx);
    cloned_body.eraseArgument(idx);
  }

  // Patch up branch function types.
  for (FuncOp func : {cloned_cond, cloned_body}) {
    func.setType(
        FunctionType::get(func.getContext(), func.front().getArgumentTypes(),
                          func.front().getTerminator()->getOperandTypes()));
  }
  EliminateUnusedResults(op, &can_eliminate);
}

// For resource results, replace all uses with the resource input to which the
// result is tied to. After this, resource outputs of this op are expected to be
// unused.
LogicalResult ForwardCommonArgToOutput(Operation *op, ArrayRef<FuncOp> branches,
                                       ValueRange branch_args,
                                       bool &has_resource_result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_8(mht_8_v, 442, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "ForwardCommonArgToOutput");

  // For while, the branch inputs and outputs need to match.
  bool io_match = isa<TF::WhileOp>(op);

  has_resource_result = false;
  // Check if the same input argument number is passed through all functions.
  for (OpResult result : op->getResults()) {
    if (!IsResource(result)) continue;

    has_resource_result = true;
    int result_idx = result.getResultNumber();
    Optional<int> common_arg_index;
    for (FuncOp func : branches) {
      auto ret = func.front().getTerminator();
      auto block_arg = ret->getOperand(result_idx).dyn_cast<BlockArgument>();
      if (!block_arg) {
        return op->emitOpError("result #")
               << result_idx << " not tied to function argument for branch @"
               << func.getName();
      }
      if (!common_arg_index.hasValue()) {
        common_arg_index = block_arg.getArgNumber();
      } else if (common_arg_index.getValue() != block_arg.getArgNumber()) {
        return op->emitError("result #")
               << result_idx
               << " is not tied to the same argument across all branches";
      }
    }

    if (io_match && result_idx != common_arg_index.getValue()) {
      return op->emitOpError("Result #")
             << result_idx << " is tied to argument #"
             << common_arg_index.getValue();
    }

    // Forward the corresponding input to the output
    result.replaceAllUsesWith(branch_args[common_arg_index.getValue()]);
  }
  return success();
}

// Canonicalizes a function if. Forwards input argument to resource results and
// then deletes the resource results.
LogicalResult CanonicalizeFunctionalIfCase(Operation *op,
                                           ArrayRef<FuncOp> branches,
                                           ValueRange branch_args) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_9(mht_9_v, 490, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "CanonicalizeFunctionalIfCase");

  for (FuncOp func : branches) {
    if (failed(CleanupAndCanonicalize(func))) return failure();
  }

  bool has_resource_result = false;
  if (failed(ForwardCommonArgToOutput(op, branches, branch_args,
                                      has_resource_result)))
    return failure();

  // If no resource type results were found, no further cleanup needed.
  if (!has_resource_result) return success();

  // Drop unused results.
  EliminateUnusedResultsForIfCase(op, branches);
  return success();
}

// Canonicalizes a functional while. Forwards common argument to results and
// drop resource results if posible.
LogicalResult CanonicalizeFunctionalWhile(TF::WhileOp op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_10(mht_10_v, 513, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "CanonicalizeFunctionalWhile");

  for (FuncOp func : {op.cond_function(), op.body_function()}) {
    if (failed(CleanupAndCanonicalize(func))) return failure();
  }

  // For while, just use the body function to forward operand to result.
  bool has_resource_result = false;
  if (failed(ForwardCommonArgToOutput(op, {op.body_function()},
                                      op.getOperands(), has_resource_result)))
    return failure();
  // If no resource type results were found, no further cleanup needed.
  if (!has_resource_result) return success();

  // Drop unused results.
  EliminateUnusedResultsForWhile(op);
  return success();
}

// Canonicalizes region based if/case and cluster operations. If the same
// captured resource typed value is used for all region results, then that value
// is forwared to the result and the result is dropped.
LogicalResult CanonicalizeRegionIfCaseCluster(Operation *op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_11(mht_11_v, 537, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "CanonicalizeRegionIfCaseCluster");

  // Check if the same value is used for all region results for this output.
  bool has_resource_result = false;
  for (OpResult result : op->getResults()) {
    if (!IsResource(result)) continue;
    has_resource_result = true;
    int result_idx = result.getResultNumber();

    Value ret0 =
        op->getRegion(0).front().getTerminator()->getOperand(result_idx);
    for (Region &region : op->getRegions().drop_front()) {
      Value ret = region.front().getTerminator()->getOperand(result_idx);
      if (ret != ret0) {
        return op->emitError("Result #")
               << result_idx
               << " not tied to the same capture across all regions";
      }
    }
    result.replaceAllUsesWith(ret0);
  }

  if (!has_resource_result) return success();

  // Eliminate unused region results. Traverse in reverse order so that
  // indices to be deleted stay unchanged.
  for (OpResult result : llvm::reverse(op->getResults())) {
    if (!result.use_empty()) continue;
    int result_idx = result.getResultNumber();
    for (Region &region : op->getRegions())
      region.front().getTerminator()->eraseOperand(result_idx);
  }
  EliminateUnusedResults(op);
  return success();
}

// Canonicalizes a region based while. If the same value is passed through
// the body, the result is replaced with the operand and all argument/results
// and retuns values corresponding to that result are dropped.
LogicalResult CanonicalizeWhileRegion(TF::WhileRegionOp op) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_12(mht_12_v, 578, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "CanonicalizeWhileRegion");

  Region &body = op.body();
  Region &cond = op.cond();
  llvm::BitVector can_eliminate(op.getNumResults());

  // Traverse in reverse order so that indices to be deleted stay unchanged.
  for (OpResult result : llvm::reverse(op.getResults())) {
    if (!IsResource(result)) continue;
    int result_idx = result.getResultNumber();
    Operation *yield_op = body.front().getTerminator();
    Value yield_operand = yield_op->getOperand(result_idx);
    Value while_operand = op.getOperand(result_idx);
    Value body_arg = body.getArgument(result_idx);
    Value cond_arg = cond.getArgument(result_idx);
    if (yield_operand != body_arg && yield_operand != while_operand) {
      return op.emitOpError("Result #") << result_idx << " is not tied to arg #"
                                        << result_idx << " of the body";
    }
    body_arg.replaceAllUsesWith(while_operand);
    cond_arg.replaceAllUsesWith(while_operand);
    result.replaceAllUsesWith(while_operand);
    body.front().getTerminator()->eraseOperand(result_idx);
    body.eraseArgument(result_idx);
    cond.eraseArgument(result_idx);
    op.getOperation()->eraseOperand(result_idx);
    can_eliminate.set(result_idx);
  }
  EliminateUnusedResults(op, &can_eliminate);
  return success();
}

// Removes identities and canonicalizes all operations within `parent_op`.
LogicalResult CleanupAndCanonicalize(Operation *parent_op) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_13(mht_13_v, 613, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "CleanupAndCanonicalize");

  auto walk_result = parent_op->walk([](Operation *op) {
    // Cleanup code in attached regions.
    for (Region &region : op->getRegions()) {
      if (!llvm::hasSingleElement(region)) return WalkResult::interrupt();
      RemovePassthroughOp(region.front());
      RemoveDeadLocalVariables(region.front());
    }

    LogicalResult result = success();

    // While condition cannot write to resource variables.
    auto check_while_cond = [&](TF::AssignVariableOp assign) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_14(mht_14_v, 628, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "lambda");

      op->emitOpError("found resource write in loop condition.");
      return WalkResult::interrupt();
    };

    if (auto if_op = dyn_cast<TF::IfOp>(op)) {
      result = CanonicalizeFunctionalIfCase(
          op, {if_op.then_function(), if_op.else_function()}, if_op.input());
    } else if (auto case_op = dyn_cast<TF::CaseOp>(op)) {
      SmallVector<FuncOp, 4> branches;
      case_op.get_branch_functions(branches);
      result = CanonicalizeFunctionalIfCase(case_op, branches, case_op.input());
    } else if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
      if (while_op.cond_function().walk(check_while_cond).wasInterrupted())
        return WalkResult::interrupt();
      result = CanonicalizeFunctionalWhile(while_op);
    } else if (isa<TF::IfRegionOp, TF::CaseRegionOp, tf_device::ClusterOp>(
                   op)) {
      result = CanonicalizeRegionIfCaseCluster(op);
    } else if (auto while_region = dyn_cast<TF::WhileRegionOp>(op)) {
      if (while_region.cond().walk(check_while_cond).wasInterrupted())
        return WalkResult::interrupt();
      // For while region, the body input and output arg should match.
      result = CanonicalizeWhileRegion(while_region);
    } else if (auto call = dyn_cast<CallOpInterface>(op)) {
      FuncOp func = dyn_cast<FuncOp>(call.resolveCallable());
      if (!func) return WalkResult::interrupt();
      result = CleanupAndCanonicalize(func);
    }
    return failed(result) ? WalkResult::interrupt() : WalkResult::advance();
  });

  return failure(walk_result.wasInterrupted());
}

}  // anonymous namespace

namespace TF {

LogicalResult CleanupAndCanonicalizeForResourceOpLifting(FuncOp func) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_15(mht_15_v, 670, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "CleanupAndCanonicalizeForResourceOpLifting");

  return CleanupAndCanonicalize(func);
}

LogicalResult CleanupAndCanonicalizeForResourceOpLifting(ModuleOp module) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSresource_op_lifting_cleanupDTcc mht_16(mht_16_v, 677, "", "./tensorflow/compiler/mlir/tensorflow/transforms/resource_op_lifting_cleanup.cc", "CleanupAndCanonicalizeForResourceOpLifting");

  auto walk_result = module.walk([](tf_device::ClusterOp cluster) {
    if (failed(CleanupAndCanonicalize(cluster))) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(walk_result.wasInterrupted());
}

}  // namespace TF
}  // namespace mlir
