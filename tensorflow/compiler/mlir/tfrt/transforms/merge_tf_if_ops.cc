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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

// A special DenseMapInfo that hashes only operands of a operation, and treats
// two operations equivalent if their operands are the same.
struct OpWithSameArgsInfo : llvm::DenseMapInfo<mlir::Operation *> {
  static unsigned getHashValue(const mlir::Operation *const_op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/mlir/tfrt/transforms/merge_tf_if_ops.cc", "getHashValue");

    auto *op = const_cast<mlir::Operation *>(const_op);
    return llvm::hash_combine(
        llvm::hash_combine_range(op->operand_begin(), op->operand_end()));
  }

  static bool isEqual(const mlir::Operation *const_lhs,
                      const mlir::Operation *const_rhs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/mlir/tfrt/transforms/merge_tf_if_ops.cc", "isEqual");

    auto *lhs = const_cast<mlir::Operation *>(const_lhs);
    auto *rhs = const_cast<mlir::Operation *>(const_rhs);
    if (lhs == rhs) return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    return std::equal(lhs->operand_begin(), lhs->operand_end(),
                      rhs->operand_begin(), rhs->operand_end());
  }
};

// This pass merges non-side-effecting tf.If ops if their operands are the same.
// For example,
//  %r0 = tf.If(%cond, %x) {else = @else_0, then = @then_0}
//  %r1, %r2 = tf.If(%cond, %x) {else = @else_1, then = @then_1}
//
// will be converted to:
//  func private @merge_else(%arg) {
//    %r0 = tf.PartitionedCall(%arg) {f = @else_0}
//    %r1, %r2 = tf.PartitionedCall(%arg) {f = @else_1}
//    return %r0, %r1, %r2
//  }
//  func private @merge_then(%arg) {
//    %r0 = tf.PartitionedCall(%arg) {f = @then_0}
//    %r1, %r2 = tf.PartitionedCall(%arg) {f = @then_1}
//    return %r0, %r1, %r2
//  }
//
//  %r0, %r1, %r2 = tf.If(%cond, %arg) {else = @merge_else, then = @merge_then}
//
// Then the inliner pass is run on the module, so the bodies of else_0 and
// else_1 are inlined into the body of merge_else, and the bodies of then_0 and
// then_1 are inlined into the body of merge_then.
//
// Note that the results will be concatenated.
class MergeTfIfOpsPass
    : public mlir::PassWrapper<MergeTfIfOpsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/mlir/tfrt/transforms/merge_tf_if_ops.cc", "getArgument");
 return "tfrt-merge-tf-if-ops"; }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc mht_3(mht_3_v, 255, "", "./tensorflow/compiler/mlir/tfrt/transforms/merge_tf_if_ops.cc", "getDescription");

    return "Merge stateless tf.If ops with the same arguments.";
  }

  void runOnOperation() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc mht_4(mht_4_v, 262, "", "./tensorflow/compiler/mlir/tfrt/transforms/merge_tf_if_ops.cc", "runOnOperation");

    constexpr int kMaxIter = 10;
    auto module = getOperation();

    bool changed = true;
    for (int i = 0; i < kMaxIter && changed; ++i) {
      changed = false;
      for (auto func_op :
           llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
        changed |= ProcessFunction(func_op, i);
      }

      if (changed) {
        // Run inliner pass to expose more merge opportunities among the
        // then-branch functions and the else-branch functions that are now
        // respectively merged, for the next iteration.
        mlir::OpPassManager pm(module.getOperationName());
        pm.addPass(mlir::createInlinerPass());
        if (mlir::failed(runPipeline(pm, module))) {
          module.emitWarning(
              absl::StrCat("could not run inliner pass within the "
                           "tfrt-merge-tf-if-ops pass iteration ",
                           i));
          break;
        }
      }
    }
  }

  bool ProcessFunction(mlir::func::FuncOp op, int iteration) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc mht_5(mht_5_v, 294, "", "./tensorflow/compiler/mlir/tfrt/transforms/merge_tf_if_ops.cc", "ProcessFunction");

    // Use a hash map to group tf.If ops with the same operands.
    llvm::SmallDenseMap<mlir::Operation *, llvm::SmallVector<mlir::TF::IfOp, 2>,
                        2, OpWithSameArgsInfo>
        if_ops_to_merge;

    for (mlir::Operation &op : op.front()) {
      auto if_op = llvm::dyn_cast<mlir::TF::IfOp>(&op);

      // Skip non tf.If ops and tf.If ops that are side-effecting.
      if (!if_op || !if_op.is_stateless()) continue;

      if_ops_to_merge[if_op].push_back(if_op);
    }

    int id = 0;

    // Set the insertion point to the current function, as we will insert new
    // functions here.
    mlir::OpBuilder builder(op);

    // Track the tf.If ops that should be removed as they are merged.
    llvm::SmallVector<mlir::TF::IfOp, 4> if_ops_to_remove;

    bool changed = false;
    for (auto &iter : if_ops_to_merge) {
      if (iter.second.size() <= 1) continue;

      // Merge tf.If ops that have the same operands. The merged branches will
      // be given unique names.
      MergeIfOpsWithSameArgs(builder, iter.first->getLoc(),
                             /*branch_prefix=*/
                             absl::StrCat(op.getSymName().str(), "_merged_if_",
                                          iteration, "_", id++),
                             iter.second);

      if_ops_to_remove.append(iter.second.begin(), iter.second.end());
      changed = true;
    }

    // Now that we are no longer using `if_ops_to_merge` or any other data
    // structures that uses the operations that will be removed, we can now
    // erase these if ops.
    for (auto op : if_ops_to_remove) op->erase();

    return changed;
  }

  void MergeIfOpsWithSameArgs(mlir::OpBuilder &builder, mlir::Location loc,
                              absl::string_view branch_prefix,
                              llvm::MutableArrayRef<mlir::TF::IfOp> if_ops) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("branch_prefix: \"" + std::string(branch_prefix.data(), branch_prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc mht_6(mht_6_v, 348, "", "./tensorflow/compiler/mlir/tfrt/transforms/merge_tf_if_ops.cc", "MergeIfOpsWithSameArgs");

    assert(if_ops.size() > 1);

    // The results of the merged tf.If op are the concatenation of results of
    // the original tf.If ops.
    llvm::SmallVector<mlir::Type, 4> new_result_types;
    for (auto if_op : if_ops) {
      new_result_types.append(if_op->result_type_begin(),
                              if_op->result_type_end());
    }

    auto branch_function_type = builder.getFunctionType(
        if_ops.front().input().getTypes(), new_result_types);

    // Create new branches for the merged tf.If op.
    auto then_branch_name = CreateBranchFunction(
        builder, loc, branch_prefix,
        /*branch_suffix=*/"_then", branch_function_type, if_ops,
        [](mlir::TF::IfOp op) { return op.then_branchAttr(); });

    auto else_branch_name = CreateBranchFunction(
        builder, loc, branch_prefix,
        /*branch_suffix=*/"_else", branch_function_type, if_ops,
        [](mlir::TF::IfOp op) { return op.else_branchAttr(); });

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(if_ops.front());

    // Create the merged tf.If op using the new branches.
    auto new_if_op = builder.create<mlir::TF::IfOp>(
        loc, new_result_types, if_ops.front().cond(), if_ops.front().input(),
        then_branch_name, else_branch_name, /*is_stateless=*/true);

    // Replace the uses of results of the original tf.If ops with the results of
    // the merged tf.If op.
    auto new_result_iter = new_if_op.output().begin();
    for (auto if_op : if_ops) {
      for (auto result : if_op.output()) {
        assert(new_result_iter != new_if_op.output().end());
        result.replaceAllUsesWith(*new_result_iter);
        ++new_result_iter;
      }
    }
  }

  llvm::StringRef CreateBranchFunction(
      mlir::OpBuilder &builder, mlir::Location loc,
      absl::string_view branch_prefix, absl::string_view branch_suffix,
      mlir::FunctionType branch_function_type,
      llvm::ArrayRef<mlir::TF::IfOp> if_ops,
      llvm::function_ref<mlir::FlatSymbolRefAttr(mlir::TF::IfOp)> get_branch) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("branch_prefix: \"" + std::string(branch_prefix.data(), branch_prefix.size()) + "\"");
   mht_7_v.push_back("branch_suffix: \"" + std::string(branch_suffix.data(), branch_suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSmerge_tf_if_opsDTcc mht_7(mht_7_v, 403, "", "./tensorflow/compiler/mlir/tfrt/transforms/merge_tf_if_ops.cc", "CreateBranchFunction");

    std::string branch_name = absl::StrCat(branch_prefix, branch_suffix);
    auto branch = builder.create<mlir::func::FuncOp>(loc, branch_name,
                                                     branch_function_type);
    branch.setVisibility(mlir::func::FuncOp::Visibility::Private);

    mlir::OpBuilder::InsertionGuard guard(builder);

    // In the body of newly created branch function, we insert
    // tf.PartitionedCall ops to call the original branches.
    auto *block = branch.addEntryBlock();
    builder.setInsertionPointToStart(block);
    auto empty_string_attr = builder.getStringAttr("");

    llvm::SmallVector<mlir::Value, 4> results;
    results.reserve(branch_function_type.getNumResults());

    for (auto if_op : if_ops) {
      // Create the the call op to the original branch. The arguments are simply
      // the arguments from the wrapper function.
      auto call_op = builder.create<mlir::TF::PartitionedCallOp>(
          if_op.getLoc(), if_op.getResultTypes(), block->getArguments(),
          get_branch(if_op), empty_string_attr, empty_string_attr,
          empty_string_attr);

      // The results are the concatenation of the original branches.
      results.append(call_op.output().begin(), call_op.output().end());
    }

    builder.create<mlir::func::ReturnOp>(loc, results);

    return branch.getSymName();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateMergeTfIfOpsPass() {
  return std::make_unique<MergeTfIfOpsPass>();
}

static mlir::PassRegistration<MergeTfIfOpsPass> register_pass(
    CreateMergeTfIfOpsPass);

}  // namespace tfrt_compiler
}  // namespace tensorflow
