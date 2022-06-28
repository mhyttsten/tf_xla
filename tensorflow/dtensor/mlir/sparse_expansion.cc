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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSsparse_expansionDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSsparse_expansionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSsparse_expansionDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/sparse_expander.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

constexpr char kMainFunctionName[] = "main";

// Expand every op that consumes SparseTensor operands in topological order.
mlir::LogicalResult ConductSparseExpansion(mlir::ModuleOp module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSsparse_expansionDTcc mht_0(mht_0_v, 210, "", "./tensorflow/dtensor/mlir/sparse_expansion.cc", "ConductSparseExpansion");

  auto main_func = module.lookupSymbol<mlir::func::FuncOp>(kMainFunctionName);
  if (!main_func)
    return module.emitOpError(
        "could not find `main` function in module for SPMD expansion.");

  TopologicalIterator iterator(main_func);
  while (iterator.hasNext()) {
    mlir::Operation* op = iterator.next();

    mlir::Operation* expanded_op = nullptr;
    auto status = RunSparseExpansion(op, &expanded_op);
    if (!status.ok() || expanded_op == nullptr) {
      // Sometimes op may been erased and expanded_op set.
      // In this case we should emit the error on the expanded op.
      mlir::Operation* emit_op = op;
      if (expanded_op != nullptr) emit_op = expanded_op;
      return emit_op->emitError(WithContext(status, __FILE__, __LINE__,
                                            "While computing Sparse expansion")
                                    .error_message());
    }
  }
  return mlir::success();
}

// After Sparse Expansion pass, there may be unused SparseToDenseOps due to
// expanded ops possibly taking the operands of the SparseToDenseOps instead
// of the output of the SparseToDenseOps. So remove unused SparseToDenseOps
// and its corresponding dependent ops like DTensorLayout and Const ops.
void RemoveUnusedSparseToDenseOps(mlir::ModuleOp module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSsparse_expansionDTcc mht_1(mht_1_v, 242, "", "./tensorflow/dtensor/mlir/sparse_expansion.cc", "RemoveUnusedSparseToDenseOps");

  llvm::SmallVector<mlir::TF::SparseToDenseOp, 4> sparse_ops_to_erase;
  llvm::SmallVector<mlir::TF::DTensorLayout, 4> layout_ops_to_erase;

  module.walk([&](mlir::TF::SparseToDenseOp op) {
    // Delete this op if it either has no consuming ops or the only consuming
    // op is a DTensorLayout op that also has no consuming ops.
    if (op->use_empty()) {
      sparse_ops_to_erase.emplace_back(op);
    } else if (op->hasOneUse()) {
      if (auto layout_op = mlir::dyn_cast<mlir::TF::DTensorLayout>(
              op->getOpResult(0).getUses().begin().getUser())) {
        if (layout_op.use_empty()) {
          layout_ops_to_erase.emplace_back(layout_op);
          sparse_ops_to_erase.emplace_back(op);
        }
      }
    }
  });

  // First delete Layout ops and then delete SparseToDense ops.
  for (auto op : layout_ops_to_erase) op.erase();
  for (auto op : sparse_ops_to_erase) {
    // Also delete the corresponding Const ops that are no longer used
    // attached to the SparseToDense ops.
    auto const_op = op.getOperand(3).getDefiningOp();
    op.erase();
    if (const_op->use_empty()) const_op->erase();
  }
}

struct DTensorSparseExpansion
    : public DTensorSparseExpansionBase<DTensorSparseExpansion> {
  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSsparse_expansionDTcc mht_2(mht_2_v, 278, "", "./tensorflow/dtensor/mlir/sparse_expansion.cc", "runOnOperation");

    auto module = getOperation();
    if (failed(ConductSparseExpansion(module))) return signalPassFailure();

    // After Sparse Expansion, we may no longer use any SparseToDenseOp outputs,
    // so remove them if they are not used.
    RemoveUnusedSparseToDenseOps(module);
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSparseExpansion() {
  return std::make_unique<DTensorSparseExpansion>();
}

}  // namespace dtensor
}  // namespace tensorflow
