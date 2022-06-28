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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSexpand_hlo_tuplesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSexpand_hlo_tuplesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSexpand_hlo_tuplesDTcc() {
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

#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace mhlo {
namespace {

// This pass assumes the function to be expanded has no callees, to be specific,
// the function is more like the main function.
class ExpandHloTuplesPass
    : public ExpandHloTuplesPassBase<ExpandHloTuplesPass> {
 public:
  ExpandHloTuplesPass() = default;
  ExpandHloTuplesPass(const ExpandHloTuplesPass&) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSexpand_hlo_tuplesDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/expand_hlo_tuples.cc", "ExpandHloTuplesPass");
}
  explicit ExpandHloTuplesPass(const std::string& entry_function_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("entry_function_name: \"" + entry_function_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSexpand_hlo_tuplesDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/expand_hlo_tuples.cc", "ExpandHloTuplesPass");

    entry_function_name_ = entry_function_name;
  }

  // Expands the mhlo.tuple used in return op. Also updates function
  // signature accordingly.
  void ExpandTupledTensorInReturnOp(FuncOp func) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSexpand_hlo_tuplesDTcc mht_2(mht_2_v, 228, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/expand_hlo_tuples.cc", "ExpandTupledTensorInReturnOp");

    FunctionType old_func_type = func.getFunctionType();
    // Update input signatures.
    // We will flatten the tuples for the function inputs as well.
    // So if an input is tuple, will be flattened and packed as following:
    // func_1(%arg0: tuple<input1, input2>) =>
    //
    // func_1(%arg0: <input1>, %arg1: <input2>) {
    //  %0 = mhlo.tuple(%arg0, %arg1)
    // }
    SmallVector<Type, 4> expanded_input_types;
    SmallVector<BlockArgument, 20> func_arguments(func.getArguments().begin(),
                                                  func.getArguments().end());
    for (auto argument : func_arguments) {
      auto type = argument.getType();
      auto tuple_type = type.dyn_cast_or_null<TupleType>();
      if (!tuple_type) {
        expanded_input_types.push_back(type);
      } else {
        // We need to
        // 1) expand the tuple
        // 2) insert a new tuple
        // 3) rewire the new tuple
        int original_argument_index = argument.getArgNumber();
        int argument_index = original_argument_index;
        SmallVector<Value, 4> flattened_operands;
        // insert the flattened tuples after the original tuple.
        Location loc = func.getBody().getLoc();
        for (auto flattened_type : tuple_type.getTypes()) {
          expanded_input_types.push_back(flattened_type);
          func.insertArgument(++argument_index, flattened_type, {}, loc);
          flattened_operands.push_back(func.getArgument(argument_index));
        }

        // Construct a new tuple and rewire it.
        OpBuilder builder(func.getBody());
        builder.setInsertionPointToStart(&func.getBody().front());
        auto new_tuple =
            builder.create<mhlo::TupleOp>(loc, tuple_type, flattened_operands);
        func.getArgument(original_argument_index).replaceAllUsesWith(new_tuple);

        // Now the original argument has been rewired, we should be able to
        // safely erase it.
        func.eraseArgument(original_argument_index);
      }
    }

    // Update output signatures.
    auto return_op = cast<mlir::func::ReturnOp>(func.getBody().back().back());

    // Expand all tuples in old return operands.
    SmallVector<Value, 4> expanded_return_operands;
    SmallVector<Type, 4> expanded_result_types;
    for (auto value : return_op.getOperands()) {
      auto tuple = dyn_cast_or_null<mhlo::TupleOp>(value.getDefiningOp());
      if (!tuple) {
        expanded_return_operands.push_back(value);
        expanded_result_types.push_back(value.getType());
        continue;
      }

      for (auto tuple_operand : tuple.getOperands()) {
        expanded_return_operands.push_back(tuple_operand);
        expanded_result_types.push_back(tuple_operand.getType());
      }
    }

    if (expanded_return_operands.empty()) return;

    OpBuilder builder(return_op);
    builder.create<mlir::func::ReturnOp>(return_op.getLoc(),
                                         expanded_return_operands);
    return_op.erase();
    auto new_func_type =
        FunctionType::get(old_func_type.getContext(), expanded_input_types,
                          expanded_result_types);
    func.setType(new_func_type);
  }

  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSexpand_hlo_tuplesDTcc mht_3(mht_3_v, 310, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/expand_hlo_tuples.cc", "runOnOperation");

    auto module = getOperation();
    // Find `main` function.
    auto entry_function = module.lookupSymbol<FuncOp>(entry_function_name_);
    if (!entry_function) {
      return;
    }

    ExpandTupledTensorInReturnOp(entry_function);
  }
};

}  // end namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateExpandHloTuplesPass(
    const std::string& entry_function_name) {
  return std::make_unique<ExpandHloTuplesPass>(entry_function_name);
}

}  // namespace mhlo
}  // namespace mlir
