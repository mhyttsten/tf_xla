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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremove_tf_if_const_argsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremove_tf_if_const_argsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremove_tf_if_const_argsDTcc() {
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
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

// This pass removes tf.If ops' operands that are produced by tf.Const ops.
// These constants can be moved into branches' function body for further
// optimziation.
class RemoveTfIfConstArgs
    : public mlir::PassWrapper<RemoveTfIfConstArgs,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremove_tf_if_const_argsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/mlir/tfrt/transforms/remove_tf_if_const_args.cc", "getArgument");

    return "tfrt-remove-tf-if-const-args";
  }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremove_tf_if_const_argsDTcc mht_1(mht_1_v, 206, "", "./tensorflow/compiler/mlir/tfrt/transforms/remove_tf_if_const_args.cc", "getDescription");

    return "Remove const args from tf.If ops";
  }

  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremove_tf_if_const_argsDTcc mht_2(mht_2_v, 213, "", "./tensorflow/compiler/mlir/tfrt/transforms/remove_tf_if_const_args.cc", "runOnOperation");

    auto module = getOperation();
    for (auto func_op :
         llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      ProcessFunction(func_op);
    }
  }

  void ProcessFunction(mlir::func::FuncOp op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremove_tf_if_const_argsDTcc mht_3(mht_3_v, 224, "", "./tensorflow/compiler/mlir/tfrt/transforms/remove_tf_if_const_args.cc", "ProcessFunction");

    // Set the insertion point to the current function, as we will insert new
    // functions here.
    mlir::OpBuilder builder(op);
    for (mlir::Operation &op : op.front()) {
      auto if_op = llvm::dyn_cast<mlir::TF::IfOp>(&op);
      if (!if_op) continue;

      // Record the operands that are produced by tf.Const ops.
      llvm::SmallVector<mlir::TF::ConstOp, 2> const_args;
      // Record these operands's corresponding operand indices.
      llvm::SmallVector<unsigned, 2> const_arg_indices;
      // Record the remaining operands that won't be removed.
      llvm::SmallVector<mlir::Value, 2> remaining_args;
      for (auto iter : llvm::enumerate(if_op.input())) {
        mlir::Value operand = iter.value();
        if (auto const_op = operand.getDefiningOp<mlir::TF::ConstOp>()) {
          const_args.push_back(const_op);
          const_arg_indices.push_back(iter.index());
        } else {
          remaining_args.push_back(operand);
        }
      }

      if (const_args.empty()) continue;

      RemoveConstArgsFromTfIfOp(builder, if_op, const_args, const_arg_indices,
                                remaining_args);
    }
  }

  void RemoveConstArgsFromTfIfOp(mlir::OpBuilder &builder, mlir::TF::IfOp if_op,
                                 llvm::ArrayRef<mlir::TF::ConstOp> const_args,
                                 llvm::ArrayRef<unsigned> const_arg_indices,
                                 llvm::ArrayRef<mlir::Value> remaining_args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremove_tf_if_const_argsDTcc mht_4(mht_4_v, 261, "", "./tensorflow/compiler/mlir/tfrt/transforms/remove_tf_if_const_args.cc", "RemoveConstArgsFromTfIfOp");

    auto branch_suffix = absl::StrCat("_removed_const_args_", id_++);

    // Create wrapper functions with the new arguments (as const args are
    // removed) for both then function and else function.
    auto new_then_function_name =
        CreateBranchFunction(builder, if_op.then_function(), branch_suffix,
                             const_args, const_arg_indices);
    auto new_else_function_name =
        CreateBranchFunction(builder, if_op.else_function(), branch_suffix,
                             const_args, const_arg_indices);

    // Change the if_op's argumetns to the new arguments, branches to new
    // branches. Note that the outputs are not changed.
    if_op.inputMutable().assign(remaining_args);
    if_op.then_branchAttr(
        mlir::SymbolRefAttr::get(builder.getContext(), new_then_function_name));
    if_op.else_branchAttr(
        mlir::SymbolRefAttr::get(builder.getContext(), new_else_function_name));
  }

  llvm::StringRef CreateBranchFunction(
      mlir::OpBuilder &builder, mlir::func::FuncOp branch,
      absl::string_view branch_suffix,
      llvm::ArrayRef<mlir::TF::ConstOp> const_args,
      llvm::ArrayRef<unsigned> const_arg_indices) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("branch_suffix: \"" + std::string(branch_suffix.data(), branch_suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSremove_tf_if_const_argsDTcc mht_5(mht_5_v, 290, "", "./tensorflow/compiler/mlir/tfrt/transforms/remove_tf_if_const_args.cc", "CreateBranchFunction");

    // Get the new function type as const args are removed.
    llvm::BitVector const_arg_indices_bv(branch.getNumArguments());
    for (auto i : const_arg_indices) const_arg_indices_bv.set(i);
    auto new_branch_type = branch.getFunctionType().getWithoutArgsAndResults(
        const_arg_indices_bv, {});
    std::string new_branch_name =
        absl::StrCat(branch.getSymName().str(), branch_suffix);
    // Create the wrapper function with the new arguments that calls the
    // original branch.
    auto new_branch = builder.create<mlir::func::FuncOp>(
        branch.getLoc(), new_branch_name, new_branch_type);
    new_branch.setVisibility(mlir::func::FuncOp::Visibility::Private);

    // In its function body, we will add the corresponding const ops and call
    // the original branch.

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *block = new_branch.addEntryBlock();
    builder.setInsertionPointToStart(block);

    // Prepare the function arguments of the original branch.
    llvm::SmallVector<mlir::Value, 4> call_args(branch.getNumArguments());

    // For those removed const args, we copy the tf.Const op, and use that as
    // the corresponding argument when calling the original branch.
    for (const auto &iter : llvm::zip(const_args, const_arg_indices)) {
      auto const_op =
          llvm::cast<mlir::TF::ConstOp>(builder.clone(*std::get<0>(iter)));
      unsigned index = std::get<1>(iter);
      call_args[index] = const_op;
    }

    // For the rest, they are now coming from the wrapper function's arguments
    // in the original order.
    for (int i = 0, j = 0; i < call_args.size(); ++i) {
      if (!call_args[i]) {
        assert(j < block->getNumArguments());
        call_args[i] = block->getArgument(j++);
      }
    }

    // Now create the call op to the original branch.
    auto call_op = builder.create<mlir::TF::StatefulPartitionedCallOp>(
        new_branch.getLoc(), new_branch_type.getResults(), call_args,
        branch.getSymName(), "", "", "");
    // Note that the outputs are not changed.
    builder.create<mlir::func::ReturnOp>(new_branch.getLoc(), call_op.output());

    return new_branch.getSymName();
  }

  int id_ = 0;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRemoveTfIfConstArgsPass() {
  return std::make_unique<RemoveTfIfConstArgs>();
}

static mlir::PassRegistration<RemoveTfIfConstArgs> register_pass(
    CreateRemoveTfIfConstArgsPass);

}  // namespace tfrt_compiler
}  // namespace tensorflow
