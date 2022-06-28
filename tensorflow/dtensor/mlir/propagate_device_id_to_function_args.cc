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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_device_id_to_function_argsDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_device_id_to_function_argsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_device_id_to_function_argsDTcc() {
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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Holds information on functions to rewrite. `function` is the function
// definition or function that needs to be updated and `callsite_ops` holds a
// list of ops that calls the `function`.
struct FunctionToChangeInfo {
  mlir::func::FuncOp function;
  llvm::SmallVector<mlir::Operation*, 4> callsite_ops;
};

// Finds all functions in graph that is not a public functions and retrieves
// their callsite operations.
llvm::SmallVector<FunctionToChangeInfo, 4> FindFunctionsToRewrite(
    mlir::ModuleOp module) {
  llvm::SmallVector<FunctionToChangeInfo, 4> functions_to_change;
  module.walk([&](mlir::Operation* op) {
    if (!llvm::isa<mlir::TF::StatefulPartitionedCallOp,
                   mlir::TF::PartitionedCallOp>(op))
      return;

    // Extract function symbol from PartitionedCall or StatefulPartitionedCall
    // op.
    llvm::StringRef symbol;
    if (auto call_op =
            llvm::dyn_cast<mlir::TF::StatefulPartitionedCallOp>(op)) {
      symbol = call_op.f();
    } else {
      auto symbol_ref = llvm::dyn_cast<mlir::TF::PartitionedCallOp>(op).f();
      if (!symbol_ref.isa<mlir::FlatSymbolRefAttr>()) return;
      symbol = symbol_ref.getRootReference().getValue();
    }

    // If function definition could be found, then extract all function usages.
    auto function = MaybeFindFunction(op);
    if (!function || function->isPublic()) return;

    auto function_uses = mlir::SymbolTable::getSymbolUses(
        mlir::StringAttr::get(module.getContext(), symbol),
        &module.getBodyRegion());
    if (!function_uses) return;

    llvm::SmallVector<mlir::Operation*, 4> function_use_ops;
    for (auto function_use : *function_uses)
      function_use_ops.emplace_back(function_use.getUser());

    functions_to_change.emplace_back(
        FunctionToChangeInfo{function.value(), function_use_ops});
  });

  return functions_to_change;
}

// Rewrites function such that 0th argument of type `type` is added to
// `function`.
void PrependArgumentToFunction(mlir::func::FuncOp function, mlir::Type type,
                               mlir::OpBuilder* builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_device_id_to_function_argsDTcc mht_0(mht_0_v, 264, "", "./tensorflow/dtensor/mlir/propagate_device_id_to_function_args.cc", "PrependArgumentToFunction");

  auto& function_body = function.front();
  function_body.insertArgument(static_cast<unsigned>(0), type,
                               function.getLoc());
  auto new_argument_types =
      llvm::to_vector<4>(function_body.getArgumentTypes());
  function.setType(
      mlir::FunctionType::get(builder->getContext(), new_argument_types,
                              function.getFunctionType().getResults()));
}

// Rewrites function callsites ops. As function signatures are already updated,
// simply add 0th argument of the parent function to 0th operand of the callsite
// operation.
mlir::LogicalResult PrependDeviceIdToCallsites(mlir::OpBuilder* builder,
                                               mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_device_id_to_function_argsDTcc mht_1(mht_1_v, 282, "", "./tensorflow/dtensor/mlir/propagate_device_id_to_function_args.cc", "PrependDeviceIdToCallsites");

  auto device_id_or_status = DeviceId(op);
  if (!device_id_or_status.ok())
    return op->emitOpError(
        "Failed during PropagateDeviceIdToFunctionArgs pass. All functions "
        "must have device id as 0th argument.");

  auto new_operands = llvm::to_vector<4>(op->getOperands());
  new_operands.insert(new_operands.begin(), device_id_or_status.ValueOrDie());

  builder->setInsertionPoint(op);
  mlir::Operation* new_call = nullptr;
  if (auto stateful_partitioned_call =
          llvm::dyn_cast<mlir::TF::StatefulPartitionedCallOp>(op)) {
    new_call = builder->create<mlir::TF::StatefulPartitionedCallOp>(
        op->getLoc(), op->getResultTypes(), new_operands,
        stateful_partitioned_call.f(), stateful_partitioned_call.config(),
        stateful_partitioned_call.config_proto(),
        stateful_partitioned_call.executor_type());
  } else {
    auto partitioned_call = llvm::cast<mlir::TF::PartitionedCallOp>(op);
    new_call = builder->create<mlir::TF::PartitionedCallOp>(
        op->getLoc(), op->getResultTypes(), new_operands, partitioned_call.f(),
        partitioned_call.config(), partitioned_call.config_proto(),
        partitioned_call.executor_type());
  }

  for (auto results : llvm::zip(op->getResults(), new_call->getResults()))
    std::get<0>(results).replaceAllUsesWith(std::get<1>(results));

  op->erase();

  return mlir::success();
}

// Pass that rewrites the functions in graph so that 0th argument of the main
// function (i.e. device_id) is present on all functions in the graph.
struct DTensorPropagateDeviceIdToFunctionArgs
    : public DTensorPropagateDeviceIdToFunctionArgsBase<
          DTensorPropagateDeviceIdToFunctionArgs> {
  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_device_id_to_function_argsDTcc mht_2(mht_2_v, 325, "", "./tensorflow/dtensor/mlir/propagate_device_id_to_function_args.cc", "runOnOperation");

    mlir::MLIRContext& context = getContext();
    auto module = getOperation();
    mlir::OpBuilder builder(&context);

    // Extracts device id argument from main function.
    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>("main");
    auto device_id_or_status = DeviceId(&main_func.getBody().front().front());
    if (!device_id_or_status.ok()) {
      main_func.emitOpError(
          "Error in PropagateDeviceIdToFunctionArgs pass. Main function must "
          "have device id as 0th function argument.");
      return signalPassFailure();
    }
    auto device_id_from_main_function = device_id_or_status.ValueOrDie();
    // First iterate through all functions to rewrite and update the signatures
    // first.
    const auto functions_to_update = FindFunctionsToRewrite(module);
    for (const auto& function_to_update : functions_to_update)
      PrependArgumentToFunction(function_to_update.function,
                                device_id_from_main_function.getType(),
                                &builder);

    // Once all function signatures are updated, rewrite the callsite ops.
    for (const auto& function_to_update : functions_to_update) {
      for (auto call_site_op : function_to_update.callsite_ops) {
        if (mlir::failed(PrependDeviceIdToCallsites(&builder, call_site_op)))
          return signalPassFailure();
      }
    }
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorPropagateDeviceIdToFunctionArgs() {
  return std::make_unique<DTensorPropagateDeviceIdToFunctionArgs>();
}

}  // namespace dtensor
}  // namespace tensorflow
