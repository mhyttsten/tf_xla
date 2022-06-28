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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSsplit_merged_operandsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSsplit_merged_operandsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSsplit_merged_operandsDTcc() {
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

#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/stateful_ops_utils.h"

// Background info:
// Currently the model taken to MLIRConverter is frozen (all the variables have
// been converted to constants, all the assign ops are gone, etc.). However,
// TFLite has these variable tensors semantics. So the variable mapping from TF
// to TFLite is actually broken here, we sort of hard-code the variable tensors
// based on the actual ops using them, such as unidirectional_sequence_lstm.
//
// MLIRConverter also benefits from lots of typical compiler optimization like
// merging same input values if they're identical. These optimizations are
// desirable but not for those TFLite ops which have variable tensors as inputs.
// Yes, they have identical input values, but those identical values are
// "stateful", their values can change during invocations.
//
// A typical example is unidirectional_sequence_lstm have two variable tensor
// inputs: activation state & cell state. They may have same initial values
// (typical zero-initialized), but their values will be changed. So we cannot
// just merge those values.
//
// This pass is more like short-term workaround since we don't have a good
// variable representation right now.
//
// This pass will duplicate input values for those variable tensor inputs.

namespace mlir {
namespace TFL {
namespace {

struct SplitMergedOperandsPass
    : public PassWrapper<SplitMergedOperandsPass, OperationPass<FuncOp>> {
  void runOnOperation() override;

  StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSsplit_merged_operandsDTcc mht_0(mht_0_v, 241, "", "./tensorflow/compiler/mlir/lite/transforms/split_merged_operands.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-split-merged-operands";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSsplit_merged_operandsDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/mlir/lite/transforms/split_merged_operands.cc", "getDescription");

    // This is a brief description of the pass.
    return "Split merged stateful operands for tfl operations.";
  }
};

LogicalResult DuplicateValueIfNeeded(Operation* op,
                                     llvm::DenseSet<Value>* values,
                                     OpBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSsplit_merged_operandsDTcc mht_2(mht_2_v, 260, "", "./tensorflow/compiler/mlir/lite/transforms/split_merged_operands.cc", "DuplicateValueIfNeeded");

  std::vector<int> stateful_operands_index;
  if (!IsStatefulOp(op, &stateful_operands_index)) return success();

  for (int index : stateful_operands_index) {
    Value operand = op->getOperand(index);
    auto inserted_value = values->insert(operand).second;
    if (inserted_value) continue;
    // We can only clone the constant op at this point.
    // Since all ops have been legalized to tflite ops, so we only care about
    // ConstOp or QConstOp or mlir constant op/
    Operation* input_op = operand.getDefiningOp();
    if (input_op == nullptr) return failure();

    Attribute attr;
    if (!matchPattern(input_op, m_Constant(&attr))) {
      op->emitError()
          << "We cannot duplicate the value since it's not constant.\n";
      return failure();
    }
    builder->setInsertionPoint(op);
    Operation* duplicated_input_op = builder->clone(*input_op);

    // Rewire the inputs.
    op->setOperand(index, duplicated_input_op->getResult(0));
  }
  return success();
}

void SplitMergedOperandsPass::runOnOperation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSsplit_merged_operandsDTcc mht_3(mht_3_v, 292, "", "./tensorflow/compiler/mlir/lite/transforms/split_merged_operands.cc", "SplitMergedOperandsPass::runOnOperation");

  llvm::DenseSet<Value> stateful_values;
  auto func = getOperation();
  OpBuilder builder(func);
  for (auto& bb : func.getBody()) {
    for (auto& op : bb) {
      if (failed(DuplicateValueIfNeeded(&op, &stateful_values, &builder))) {
        func.emitError() << "Failed to duplicate values for the stateful op\n";
        return signalPassFailure();
      }
    }
  }
}

}  // namespace

/// Creates an instance of the TensorFlow Lite dialect SplitMergedOperands
/// pass.
std::unique_ptr<OperationPass<FuncOp>> CreateSplitMergedOperandsPass() {
  return std::make_unique<SplitMergedOperandsPass>();
}

static PassRegistration<SplitMergedOperandsPass> pass;

}  // namespace TFL
}  // namespace mlir
