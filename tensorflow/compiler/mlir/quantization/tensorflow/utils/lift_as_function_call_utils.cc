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
class MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSutilsPSlift_as_function_call_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSutilsPSlift_as_function_call_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSutilsPSlift_as_function_call_utilsDTcc() {
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
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.h"

#include <queue>
#include <stack>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

constexpr char kAttrMapAttribute[] = "attr_map";
// This attribute will be set for functions created by this pass.
constexpr char kFusedFunctionAttr[] = "tf_quant.fused_function";
// The keyword to detect if this is a `NullAttribute`.
constexpr char kNullAttributeValue[] = "N/A";

// Checks if the op is inside a lifted function.
bool IsInLiftedFunc(Operation *op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSutilsPSlift_as_function_call_utilsDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.cc", "IsInLiftedFunc");

  return op->getParentOfType<FuncOp>()->hasAttr(kFusedFunctionAttr);
}

// Inserts the function to the symbol table of the module thread-safely.
StringAttr InsertToSymbolTable(Operation *module, Operation *function,
                               const std::string &func_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSutilsPSlift_as_function_call_utilsDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.cc", "InsertToSymbolTable");

  static tensorflow::mutex *mtx = new tensorflow::mutex();
  tensorflow::mutex_lock lock(*mtx);

  SymbolTable symbol_table(module);
  std::string unique_name = func_name;
  int32_t uniquing_counter = 0;
  while (symbol_table.lookup(unique_name) != nullptr) {
    ++uniquing_counter;
    unique_name = func_name + "_" + std::to_string(uniquing_counter);
  }
  function->setAttr("sym_name",
                    StringAttr::get(module->getContext(), unique_name));
  return symbol_table.insert(function);
}

ValueRange createFusedFnCall(OpBuilder builder, Location location,
                             StringRef func_name, TypeRange output_types,
                             ValueRange args) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSutilsPSlift_as_function_call_utilsDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.cc", "createFusedFnCall");

  TF::PartitionedCallOp call_op = builder.create<TF::PartitionedCallOp>(
      location, output_types, args,
      FlatSymbolRefAttr::get(builder.getStringAttr(func_name)),
      /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
  call_op->setAttr(
      kQuantTraitAttrName,
      builder.getStringAttr(llvm::StringRef(
          std::string(QuantTraitValues[QuantizationTrait::FullyQuantizable]))));

  return call_op.output();
}

// Finds ops in the paths from arguments to results. The ops is listed in an
// order that the former ops shouldn't have any dependencies on the later ones.
llvm::SmallVector<Operation *> FindOpsFromArgumentsToResults(
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results) {
  std::queue<Value> value_queue;
  for (Value result : results) {
    value_queue.push(result);
  }
  absl::flat_hash_set<mlir::detail::ValueImpl *> argument_set;
  for (Value argument : arguments) {
    argument_set.insert(argument.getImpl());
  }

  // Searching for ops from results to arguments. Duplicate ops in the op stack
  // are intentional in order to make sure the op on the top of the stack
  // doesn't depends on any ops below it.
  std::stack<Operation *> op_stack;
  while (!value_queue.empty()) {
    Value current_value = value_queue.front();
    value_queue.pop();

    Operation *defining_node = current_value.getDefiningOp();
    if (defining_node == nullptr) continue;
    op_stack.push(defining_node);
    for (const auto &arg : defining_node->getOperands()) {
      if (!argument_set.contains(arg.getImpl())) {
        value_queue.push(arg);
      }
    }
  }

  // Remove duplicate ops from the op stack.
  llvm::SmallVector<Operation *> sorted_ops;
  absl::flat_hash_set<Operation *> unique_ops;
  while (!op_stack.empty()) {
    Operation *current_op = op_stack.top();
    op_stack.pop();
    if (unique_ops.contains(current_op)) continue;
    sorted_ops.push_back(current_op);
    unique_ops.insert(current_op);
  }
  return sorted_ops;
}

// Finds the name of each attribute in `attributes` and set the attr_map
// attribute which maps an attribute identifier to its attribute name. The
// identifier is the order of that attribute in `attributes`. This map
// is then used to set attributes in the quantized functions in the
// QuantizeCompositeFunctionsPass.
// This function returns success if all attributes could be found.
LogicalResult SetAttributeMap(MLIRContext *context,
                              const llvm::SmallVector<Attribute> &attributes,
                              const llvm::SmallVector<Operation *> &ops) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSutilsPSlift_as_function_call_utilsDTcc mht_3(mht_3_v, 306, "", "./tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.cc", "SetAttributeMap");

  // A map to find which operation an attribute belongs to.
  llvm::SmallDenseMap<Attribute, Operation *> attr_to_op_map;
  // A map from the attribute to its name.
  llvm::SmallDenseMap<Attribute, llvm::StringRef> attr_to_name_map;
  for (Operation *op : ops) {
    for (const auto &named_attr : op->getAttrs()) {
      attr_to_op_map.insert({named_attr.getValue(), op});
      attr_to_name_map.insert(
          {named_attr.getValue(), named_attr.getName().getValue()});
    }
  }

  for (int idx : llvm::seq<int>(0, attributes.size())) {
    const Attribute &attribute = attributes[idx];
    // Skip following steps if this attribute is a `NullAttribute`.
    auto string_attr = attribute.dyn_cast_or_null<StringAttr>();
    if (string_attr != nullptr &&
        string_attr.getValue().equals(kNullAttributeValue)) {
      continue;
    }

    if (attr_to_op_map.count(attribute) == 0) {
      return failure();
    }

    llvm::StringRef attribute_name = attr_to_name_map[attribute];
    std::string identifier = std::to_string(idx);

    Operation *owner_op = attr_to_op_map[attribute];
    std::string new_attr_map_str;
    if (owner_op->hasAttr(kAttrMapAttribute)) {
      new_attr_map_str =
          owner_op->getAttrOfType<StringAttr>(kAttrMapAttribute).str();
      absl::StrAppend(&new_attr_map_str, ",");
    }
    absl::StrAppend(&new_attr_map_str, identifier, ":", attribute_name.str());
    owner_op->setAttr(kAttrMapAttribute,
                      StringAttr::get(context, new_attr_map_str));
  }
  return success();
}

// Creates a function to wrap the section between arguments and results.
llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, StringRef func_name,
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results,
    const llvm::SmallVector<Attribute> &attributes) {
  MLIRContext *context = builder.getContext();
  if (results.empty()) {
    mlir::emitError(UnknownLoc::get(context), "No result values specified");
    return {};
  }
  Operation *result_op = results[0].getDefiningOp();
  auto module = result_op->getParentOfType<ModuleOp>();

  // Create a private function and copy all ops between arguments and results.
  auto current_func = result_op->getParentOfType<FuncOp>();
  auto guard = OpBuilder::InsertionGuard(builder);
  builder.setInsertionPointAfter(current_func);
  TypeRange arg_types(
      llvm::ArrayRef<Value>(arguments.begin(), arguments.end()));
  TypeRange result_types(llvm::ArrayRef<Value>(results.begin(), results.end()));
  auto func_type = FunctionType::get(context, arg_types, result_types);

  llvm::SmallVector<Location> arg_locs;
  for (const auto &arg : arguments) {
    arg_locs.push_back(arg.getLoc());
  }
  auto wrap_func = builder.create<FuncOp>(location, func_name, func_type);
  wrap_func.setVisibility(SymbolTable::Visibility::Private);
  wrap_func->setAttr(kFusedFunctionAttr, builder.getUnitAttr());
  builder.createBlock(&wrap_func.getBody(), wrap_func.begin(), arg_types,
                      arg_locs);

  BlockAndValueMapping mapping;
  for (int32_t i : llvm::seq<int32_t>(0, arguments.size())) {
    mapping.map(arguments[i], wrap_func.getArgument(i));
  }

  auto cloning_ops = FindOpsFromArgumentsToResults(arguments, results);
  if (failed(SetAttributeMap(context, attributes, cloning_ops))) {
    current_func.emitError() << "Some attributes couldn't be found.";
  }
  for (Operation *op : cloning_ops) {
    builder.clone(*op, mapping);
  }

  llvm::SmallVector<Value> return_values;
  for (Value result : results) {
    return_values.push_back(mapping.lookupOrNull(result));
  }
  builder.create<mlir::func::ReturnOp>(location, return_values);

  // Create a function call to the newly created function.
  StringAttr new_func_name =
      InsertToSymbolTable(module, wrap_func, func_name.str());
  builder.setInsertionPointAfter(result_op);
  ValueRange new_results = createFusedFnCall(
      builder, location, new_func_name.getValue(), result_types, arguments);
  return llvm::SmallVector<Value, 4>(new_results.begin(), new_results.end());
}

llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, StringRef func_name,
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results) {
  llvm::SmallVector<Attribute> attributes;
  return LiftAsFunctionCall(builder, location, func_name, arguments, results,
                            attributes);
}

}  // namespace quant
}  // namespace mlir
