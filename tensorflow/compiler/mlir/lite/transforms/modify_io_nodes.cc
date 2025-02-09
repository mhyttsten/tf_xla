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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc() {
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

// NOLINTNEXTLINE
static llvm::cl::list<std::string> io_node_types(
    "tfl-test-io-types", llvm::cl::value_desc("list"),
    llvm::cl::desc("comma separated type strings. Allowed values: "
                   "'int8', 'uint8', 'float32']"),
    llvm::cl::CommaSeparated);

namespace mlir {
namespace TFL {
namespace {

// This transformation pass modifies the input and output types of the function
// to what are specified. The task was not just adding cast operations, but,
// instead, using tfl.quantize and tfl.dequantize ops to scale the tensors.
struct ModifyIONodesPass
    : public PassWrapper<ModifyIONodesPass, OperationPass<FuncOp>> {
 public:
  explicit ModifyIONodesPass() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/mlir/lite/transforms/modify_io_nodes.cc", "ModifyIONodesPass");
}
  explicit ModifyIONodesPass(mlir::Type input_type, mlir::Type output_type)
      : input_type(input_type), output_type(output_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/lite/transforms/modify_io_nodes.cc", "ModifyIONodesPass");
}

  void runOnOperation() override;

  StringRef getArgument() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc mht_2(mht_2_v, 231, "", "./tensorflow/compiler/mlir/lite/transforms/modify_io_nodes.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-modify-io-nodes";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc mht_3(mht_3_v, 239, "", "./tensorflow/compiler/mlir/lite/transforms/modify_io_nodes.cc", "getDescription");

    // This is a brief description of the pass.
    return "Modify the type of the model io nodes.";
  }

 private:
  // Assign the io types from the command line flag. This is only required for
  // tests.
  LogicalResult SetupInputOutputTypesIfNull(OpBuilder builder);

  // Modifies the element types of entry block arguments to be user specified
  // and returns  the new argument types.
  LogicalResult ModifyInputNodes(FuncOp func,
                                 llvm::SmallVectorImpl<Type>& new_input_types,
                                 OpBuilder builder);

  // Modifies the element types of entry block returns to be user specified
  // and returns the new return types.
  LogicalResult ModifyOutputNodes(FuncOp func,
                                  llvm::SmallVectorImpl<Type>& new_output_types,
                                  OpBuilder builder);

  mlir::Type input_type;
  mlir::Type output_type;
};

LogicalResult ModifyIONodesPass::SetupInputOutputTypesIfNull(
    OpBuilder builder) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc mht_4(mht_4_v, 269, "", "./tensorflow/compiler/mlir/lite/transforms/modify_io_nodes.cc", "ModifyIONodesPass::SetupInputOutputTypesIfNull");

  if (input_type && output_type) return success();

  auto convert_str_to_type = [&builder](absl::string_view str) -> Type {
    if (str == "int8") {
      return builder.getIntegerType(8);
    } else if (str == "uint8") {
      return builder.getIntegerType(8, /*isSigned=*/false);
    } else if (str == "float32") {
      return builder.getF32Type();
    } else {
      return {};
    }
  };
  if (io_node_types.size() < 2) return failure();
  if (!input_type) input_type = convert_str_to_type(io_node_types[0]);
  if (!output_type) output_type = convert_str_to_type(io_node_types[1]);
  return success();
}

LogicalResult ModifyIONodesPass::ModifyInputNodes(
    FuncOp func, llvm::SmallVectorImpl<Type>& new_input_types,
    OpBuilder builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc mht_5(mht_5_v, 294, "", "./tensorflow/compiler/mlir/lite/transforms/modify_io_nodes.cc", "ModifyIONodesPass::ModifyInputNodes");

  if (input_type.isa<FloatType>()) {
    return success();
  }

  Block& block = func.front();
  builder.setInsertionPointToStart(&block);

  for (int i = 0; i != block.getNumArguments(); ++i) {
    Value arg = block.getArgument(0);
    Type arg_type = arg.getType();
    Value new_arg = arg;
    Location loc = func.getLoc();
    if (arg.hasOneUse() && llvm::isa<QuantizeOp>(*arg.user_begin())) {
      auto quantize_op = llvm::cast<QuantizeOp>(*arg.user_begin());
      auto quantize_output = quantize_op.output();
      auto current_type = quant::QuantizedType::getQuantizedElementType(
                              quantize_output.getType())
                              .getStorageType();
      if (current_type == input_type) {  // int8 == int8
        arg_type = quantize_output.getType();
        new_arg = block.addArgument(arg_type, loc);
        quantize_output.replaceAllUsesWith(new_arg);
      } else if (input_type.isUnsignedInteger(
                     current_type.getIntOrFloatBitWidth())) {  // int8 != uint8
        arg_type = quant::ConvertSignedQuantizedToUnsigned(
            quantize_output.getType(), loc);
        new_arg = block.addArgument(arg_type, loc);
        quantize_op.setOperand(new_arg);
      } else {
        input_type.print(llvm::errs() << "Requested input type ");
        quantize_op.emitError(" Couldn't be modified to the requested type.");
        return failure();
      }
      new_input_types[i] = arg_type;
      arg.dropAllUses();
      if (quantize_op.use_empty()) {
        quantize_op.erase();
      }
    } else {
      // `arg` has multiple uses or the user isn't a quantiz op (so we couldn't
      // rewrite it to a different type. Make a copy of the `arg` and replace
      // its use.
      new_arg = block.addArgument(arg_type, loc);
      arg.replaceAllUsesWith(new_arg);
    }
    block.eraseArgument(0);
  }
  return success();
}

LogicalResult ModifyIONodesPass::ModifyOutputNodes(
    FuncOp func, llvm::SmallVectorImpl<Type>& new_output_types,
    OpBuilder builder) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc mht_6(mht_6_v, 350, "", "./tensorflow/compiler/mlir/lite/transforms/modify_io_nodes.cc", "ModifyIONodesPass::ModifyOutputNodes");

  Block& block = func.front();
  auto* terminator = block.getTerminator();
  builder.setInsertionPoint(terminator);

  if (output_type.isa<FloatType>()) {
    return success();
  }

  int num_return_operands = terminator->getNumOperands();
  new_output_types.reserve(num_return_operands);
  for (int i = 0; i != num_return_operands; ++i) {
    auto returned_value = terminator->getOperand(i);
    Type returned_type = returned_value.getType();
    Operation* returned_op = returned_value.getDefiningOp();
    if (returned_op && llvm::isa<DequantizeOp>(returned_op)) {
      auto dequantize_op = llvm::cast<DequantizeOp>(returned_op);
      auto dequantize_input = dequantize_op.input();
      Type current_type = quant::QuantizedType::getQuantizedElementType(
                              dequantize_input.getType())
                              .getStorageType();
      if (current_type == output_type) {  // int8 == int8
        returned_type = dequantize_input.getType();
        returned_value = dequantize_input;
      } else if (output_type.isUnsignedInteger(
                     current_type.getIntOrFloatBitWidth())) {  // int8 != uint8
        returned_type = quant::ConvertSignedQuantizedToUnsigned(
            dequantize_input.getType(), dequantize_op.getLoc());
        // replace the dequantize op by a quantize op
        TypeAttr type_attr = TypeAttr::get(returned_type);
        auto quantize_op = builder.create<QuantizeOp>(
            dequantize_op.getLoc(), returned_type, dequantize_input, type_attr);
        returned_value = quantize_op.output();
      } else {
        output_type.print(llvm::errs() << "Requested output type ");
        dequantize_op.emitError(" Couldn't be modified to the requested type.");
        return failure();
      }
      new_output_types[i] = returned_type;
      terminator->setOperand(i, returned_value);
      if (dequantize_op.use_empty()) {
        dequantize_op.erase();
      }
    }
  }
  return success();
}

void ModifyIONodesPass::runOnOperation() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSmodify_io_nodesDTcc mht_7(mht_7_v, 401, "", "./tensorflow/compiler/mlir/lite/transforms/modify_io_nodes.cc", "ModifyIONodesPass::runOnOperation");

  auto func = getOperation();
  auto attrs = func->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");

  // Handle the entry functions only.
  if (func.getName() != "main" && (!attrs || attrs.empty())) {
    return;
  }

  OpBuilder builder(func);
  FunctionType func_type = func.getFunctionType();
  llvm::SmallVector<Type, 4> new_input_types(func_type.getInputs().begin(),
                                             func_type.getInputs().end());
  llvm::SmallVector<Type, 4> new_output_types(func_type.getResults().begin(),
                                              func_type.getResults().end());

  if (failed(SetupInputOutputTypesIfNull(builder))) {
    return;
  }

  if (failed(ModifyInputNodes(func, new_input_types, builder))) {
    return;
  }

  if (failed(ModifyOutputNodes(func, new_output_types, builder))) {
    return;
  }

  auto new_func_type =
      builder.getFunctionType(new_input_types, new_output_types);
  func.setType(new_func_type);
}
}  // namespace

// Creates an instance of the TensorFlow Lite modify io nodes pass.
std::unique_ptr<OperationPass<FuncOp>> CreateModifyIONodesPass(
    Type input_type, Type output_type) {
  return std::make_unique<ModifyIONodesPass>(input_type, output_type);
}

static PassRegistration<ModifyIONodesPass> pass;

}  // namespace TFL
}  // namespace mlir
