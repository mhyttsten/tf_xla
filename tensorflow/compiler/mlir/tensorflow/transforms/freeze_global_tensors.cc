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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc() {
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

#include <algorithm>
#include <vector>

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/DataFlowAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/savedmodel_passes_detail.h"

namespace mlir {
namespace tf_saved_model {
namespace {

// The value of our lattice represents the GlobalTensorOp matching the value.
struct ResourceLatticeValue {
  explicit ResourceLatticeValue(GlobalTensorOp op = nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc", "ResourceLatticeValue");

    if (op) ops.insert(op);
  }

  static ResourceLatticeValue getPessimisticValueState(MLIRContext *context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc", "getPessimisticValueState");

    return ResourceLatticeValue();
  }
  static ResourceLatticeValue getPessimisticValueState(Value value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc mht_2(mht_2_v, 224, "", "./tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc", "getPessimisticValueState");

    if (auto barg = value.dyn_cast<BlockArgument>()) {
      if (FuncOp func = dyn_cast<FuncOp>(barg.getOwner()->getParentOp())) {
        SymbolTable symbol_table(func->getParentOfType<ModuleOp>());
        auto global_tensor = LookupBoundInputOfType<GlobalTensorOp>(
            func, barg.getArgNumber(), symbol_table);
        return ResourceLatticeValue(global_tensor);
      }
    }
    return ResourceLatticeValue();
  }

  bool operator==(const ResourceLatticeValue &rhs) const {
    return ops == rhs.ops;
  }

  static ResourceLatticeValue join(const ResourceLatticeValue &lhs,
                                   const ResourceLatticeValue &rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc mht_3(mht_3_v, 244, "", "./tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc", "join");

    // Take union of both sets of possible GlobalTensorOp values that can be
    // referenced here.
    ResourceLatticeValue ret;
    ret.ops.insert(lhs.ops.begin(), lhs.ops.end());
    ret.ops.insert(rhs.ops.begin(), rhs.ops.end());
    return ret;
  }

  // The location which originated the int value.
  DenseSet<GlobalTensorOp> ops;
};

class ResourceAnalysis : public ForwardDataFlowAnalysis<ResourceLatticeValue> {
 public:
  using LatticeElementT = LatticeElement<ResourceLatticeValue>;
  using ForwardDataFlowAnalysis<ResourceLatticeValue>::ForwardDataFlowAnalysis;
  ~ResourceAnalysis() override = default;

  ChangeResult visitOperation(Operation *op,
                              ArrayRef<LatticeElementT *> operands) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc mht_4(mht_4_v, 267, "", "./tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc", "visitOperation");

    return markAllPessimisticFixpoint(op->getResults());
  }
};

struct FreezeGlobalTensorsPass
    : public FreezeGlobalTensorsPassBase<FreezeGlobalTensorsPass> {
  explicit FreezeGlobalTensorsPass(bool allow_mutable_tensors) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc mht_5(mht_5_v, 277, "", "./tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc", "FreezeGlobalTensorsPass");

    this->allow_mutable_tensors = allow_mutable_tensors;
  }
  void runOnOperation() override;
};

void FreezeGlobalTensorsPass::runOnOperation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfreeze_global_tensorsDTcc mht_6(mht_6_v, 286, "", "./tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc", "FreezeGlobalTensorsPass::runOnOperation");

  auto module = getOperation();
  if (!tf_saved_model::HasTfSavedModelSemantics(module)) return;

  ResourceAnalysis analysis(&getContext());
  analysis.run(module);

  DenseSet<GlobalTensorOp> remaining_global_tensor_ops;
  {
    auto ops = module.getOps<GlobalTensorOp>();
    remaining_global_tensor_ops.insert(ops.begin(), ops.end());
  }

  for (auto global_tensor : remaining_global_tensor_ops) {
    // This pass assumes that all global tensors as immutable (e.g. by a
    // previous optimize global tensors pass). If not, this pass has to fail
    // since it cannot perform one of its goals.
    if (global_tensor.is_mutable()) {
      if (allow_mutable_tensors) continue;
      global_tensor.emitError()
          << "is not immutable, try removing mutable variables in your model "
             "since mutable variables are currently not supported through "
             "this converter";
      return signalPassFailure();
    }
  }

  // Collect all those freezable. This is an extra scan but allows for the
  // partial behavior from `allow_mutable_tensor`.
  DenseMap<BlockArgument, bool> freezeable;
  for (auto func : module.getOps<FuncOp>()) {
    for (BlockArgument val : func.getArguments()) {
      if (!getElementTypeOrSelf(val.getType()).isa<TF::ResourceType>())
        continue;

      // Check that there is only a single global tensor associated with arg.
      LatticeElement<ResourceLatticeValue> *latticeElement =
          analysis.lookupLatticeElement(val);
      if (!latticeElement || latticeElement->getValue().ops.size() != 1)
        continue;

      // Don't freeze mutable tensors.
      if (latticeElement->getValue().ops.begin()->is_mutable()) {
        freezeable[val] = false;
        continue;
      }

      freezeable[val] = true;

      // Verify users are supported kind.
      for (Operation *user : val.getUsers()) {
        if (!(isa<TF::ReadVariableOp>(user) || isa<CallOpInterface>(user))) {
          freezeable[val] = false;
          // Error out early if possible.
          if (!allow_mutable_tensors) {
            user->emitError()
                << "could not rewrite use of immutable bound input";
            return signalPassFailure();
          }
        }
      }
    }
  }

  DenseSet<GlobalTensorOp> frozen_global_tensors;
  for (auto func : module.getOps<FuncOp>()) {
    llvm::BitVector args_to_erase(func.getNumArguments());
    DenseMap<Operation *, llvm::BitVector> remove_operands;
    OpBuilder builder(func.getBody());

    for (BlockArgument val : func.getArguments()) {
      if (!freezeable[val]) continue;

      LatticeElement<ResourceLatticeValue> *latticeElement =
          analysis.lookupLatticeElement(val);
      GlobalTensorOp global_tensor = *latticeElement->getValue().ops.begin();

      SmallVector<TF::ReadVariableOp, 4> read_variable_ops_to_erase;
      frozen_global_tensors.insert(global_tensor);

      for (Operation *user : val.getUsers()) {
        if (auto read_op = llvm::dyn_cast<TF::ReadVariableOp>(user)) {
          // Collect all read variable ops so that all its uses can be replaced
          // with the tf.constant corresponding to the global tensor op.
          read_variable_ops_to_erase.push_back(read_op);
        } else {
          llvm::BitVector &bvector = remove_operands[user];
          bvector.resize(user->getNumOperands());
          for (OpOperand &use : user->getOpOperands())
            bvector.set(use.getOperandNumber());
        }
      }

      // Replace the arg with a tf.Const op in the function body.
      builder.setInsertionPointToStart(&func.getBody().front());
      auto const_op = builder.create<TF::ConstOp>(global_tensor.getLoc(),
                                                  global_tensor.value());
      args_to_erase.set(val.getArgNumber());
      for (auto read_op : read_variable_ops_to_erase) {
        read_op.getResult().replaceAllUsesWith(const_op.getResult());
        read_op.erase();
      }
    }
    // As the other uses are call operations, we simply remove the arguments
    // as the function arguments will be removed below once that function is
    // processed.
    for (auto it : remove_operands) {
      it.first->eraseOperands(it.second);
    }

    func.eraseArguments(args_to_erase);
  }

  // Erase all global tensors that were frozen.
  for (auto global_tensor : frozen_global_tensors) {
    remaining_global_tensor_ops.erase(global_tensor);
    global_tensor->erase();
  }

  // Verify that there are no remaining global tensors.
  if (!allow_mutable_tensors && !remaining_global_tensor_ops.empty()) {
    module.emitError() << "could not freeze all global tensors in the module";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateFreezeGlobalTensorsPass(
    bool allow_mutable_tensors) {
  return std::make_unique<FreezeGlobalTensorsPass>(allow_mutable_tensors);
}

}  // namespace tf_saved_model
}  // namespace mlir
