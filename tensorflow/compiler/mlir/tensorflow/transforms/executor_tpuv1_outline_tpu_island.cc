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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSexecutor_tpuv1_outline_tpu_islandDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSexecutor_tpuv1_outline_tpu_islandDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSexecutor_tpuv1_outline_tpu_islandDTcc() {
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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace tf_executor {

namespace {
constexpr llvm::StringRef kNestedModule = "_tpu_v1_compat_outlined";
constexpr llvm::StringRef kOutlinedFuncPrefix = "_tpu_v1_compat_outlined_func";

// Extract the islands containing a TPU cluster computation into an outlined
// function in a nested module. This will allow to run the usual bridge on this
// nested module which exhibit a more friendly "V2-like" structure.
// This is only intended for V1 compatibility mode where the bridge runs without
// feed/fetches on session create/extend.
struct TPUBridgeExecutorIslandOutlining
    : public TF::TPUBridgeExecutorIslandOutliningPassBase<
          TPUBridgeExecutorIslandOutlining> {
  void runOnOperation() override;
};

// Move FuncOp referenced by `symbol_ref` from one symbol table to another.
void MoveFuncOp(FlatSymbolRefAttr &symbol_ref, SymbolTable &from,
                SymbolTable &to) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSexecutor_tpuv1_outline_tpu_islandDTcc mht_0(mht_0_v, 224, "", "./tensorflow/compiler/mlir/tensorflow/transforms/executor_tpuv1_outline_tpu_island.cc", "MoveFuncOp");

  if (to.lookup<FuncOp>(symbol_ref.getValue())) return;
  FuncOp callee = from.lookup<FuncOp>(symbol_ref.getValue());
  callee.getOperation()->getBlock()->getOperations().remove(
      callee.getOperation());
  to.insert(callee);
}

void TPUBridgeExecutorIslandOutlining::runOnOperation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSexecutor_tpuv1_outline_tpu_islandDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/mlir/tensorflow/transforms/executor_tpuv1_outline_tpu_island.cc", "TPUBridgeExecutorIslandOutlining::runOnOperation");

  MLIRContext *ctx = &getContext();

  SymbolTable symbol_table(getOperation());
  if (Operation *nested_module = symbol_table.lookup(kNestedModule)) {
    nested_module->emitOpError("unexpected already present outlined module.");
    return signalPassFailure();
  }
  ModuleOp outlined_module = ModuleOp::create(getOperation().getLoc());
  outlined_module->setAttrs(getOperation()->getAttrDictionary());
  outlined_module->setAttr(SymbolTable::getSymbolAttrName(),
                           StringAttr::get(ctx, kNestedModule));
  symbol_table.insert(outlined_module);
  SymbolTable outlined_symbol_table(outlined_module);

  // Find every island that contains a TPUReplicateMetadata node and extract it
  // in a new module to run the V1 bridge there.
  SmallVector<IslandOp, 8> islands_to_outline;
  getOperation().walk([&](TF::TPUReplicateMetadataOp replicate_op) {
    auto island_op = cast<IslandOp>(replicate_op->getParentOp());
    if (!island_op || island_op.WrapsSingleOp()) return;
    islands_to_outline.push_back(island_op);
  });
  int prefix_id = 0;
  for (IslandOp island_op : islands_to_outline) {
    // Build the function signature.

    // First the captured values in the island are function arguments
    llvm::SetVector<Value> operands;
    getUsedValuesDefinedAbove(island_op.body(), operands);

    SmallVector<Type, 16> func_operand_types;
    func_operand_types.reserve(operands.size());
    for (Value operand : operands)
      func_operand_types.push_back(operand.getType());

    // Function results are the yield operands
    SmallVector<Type, 16> func_result_types;
    for (Value operand : island_op.GetYield().getOperands())
      func_result_types.push_back(operand.getType());
    FunctionType func_type =
        FunctionType::get(ctx, func_operand_types, func_result_types);

    // Create the outlined function
    SmallString<32> name = kOutlinedFuncPrefix;
    name += llvm::Twine(prefix_id++).str();
    auto outlined_func =
        OpBuilder(ctx).create<FuncOp>(island_op.getLoc(), name, func_type);
    outlined_symbol_table.insert(outlined_func);
    outlined_func.setNested();

    // We will "steal" the body of the island and replace it with a call to the
    // new function later.
    {
      YieldOp yield_op = island_op.GetYield();
      outlined_func.getBody().takeBody(island_op.body());

      // Replace the yield with a return
      OpBuilder replacer(yield_op);
      island_op.body().push_back(new Block);
      replacer.create<mlir::func::ReturnOp>(yield_op.getLoc(),
                                            yield_op.getOperands());
      yield_op.erase();
    }

    // Remap the captured operands in the (former) island block with newly
    // created entry block arguments in the function body.
    {
      Block &entry_block = outlined_func.getBody().front();
      auto loc = outlined_func.getLoc();
      for (Value operand : operands) {
        BlockArgument newArg = entry_block.addArgument(operand.getType(), loc);
        replaceAllUsesInRegionWith(operand, newArg, outlined_func.getBody());
      }
    }

    // The function is in place in the nested module, create a call and yield in
    // the original island.
    OpBuilder builder = OpBuilder::atBlockEnd(&island_op.GetBody());
    auto call_op = builder.create<mlir::TF::PartitionedCallOp>(
        island_op.getLoc(), func_result_types, operands.getArrayRef(),
        SymbolRefAttr::get(
            builder.getContext(), kNestedModule,
            SymbolRefAttr::get(builder.getContext(), outlined_func.getName())),
        /*config=*/builder.getStringAttr(""),
        /*config_proto=*/builder.getStringAttr(""),
        /*executor_type=*/builder.getStringAttr(""));
    SmallVector<Value, 16> yield_operands(call_op.getResults());
    builder.create<YieldOp>(island_op.getLoc(), yield_operands);
  }

  // Outlined all the transitively called functions by moving them in the
  // outlined module.
  for (FuncOp func : outlined_module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      for (NamedAttribute attr : op->getAttrs()) {
        if (auto symbol_ref = attr.getValue().dyn_cast<FlatSymbolRefAttr>()) {
          MoveFuncOp(symbol_ref, symbol_table, outlined_symbol_table);
          continue;
        }
        if (auto array_attr = attr.getValue().dyn_cast<ArrayAttr>()) {
          for (const Attribute &attribute : array_attr) {
            auto symbol_ref = attribute.dyn_cast<FlatSymbolRefAttr>();
            if (!symbol_ref) continue;
            MoveFuncOp(symbol_ref, symbol_table, outlined_symbol_table);
          }
        }
      }
    });
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandOutliningPass() {
  return std::make_unique<TPUBridgeExecutorIslandOutlining>();
}

}  // namespace tf_executor
}  // namespace mlir
