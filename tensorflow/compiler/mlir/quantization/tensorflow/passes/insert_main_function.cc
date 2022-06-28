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
class MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc() {
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
#include <string>

#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/macros.h"

namespace mlir {
namespace quant {
namespace {

constexpr char kEntryFunctionAttr[] = "tf.entry_function";
constexpr char kExportedNameAttr[] = "tf_saved_model.exported_names";
constexpr char kIndexPathAttr[] = "tf_saved_model.index_path";

// The ConvertMlirToGraphdef requires the provided input module to have a main
// function, which might not exist in case of multi-signature graphs. In that
// case, this pass will create a new main function, which calls signature
// functions.
class InsertMainFunctionPass
    : public PassWrapper<InsertMainFunctionPass, OperationPass<ModuleOp>> {
 public:
  explicit InsertMainFunctionPass() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/insert_main_function.cc", "InsertMainFunctionPass");
}

  StringRef getArgument() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/insert_main_function.cc", "getArgument");
 return "quant-add-main-function"; }

  StringRef getDescription() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc mht_2(mht_2_v, 223, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/insert_main_function.cc", "getDescription");

    return "Insert the main function to the module if it is missing.";
  }

  void runOnOperation() override;
};

// Checks if the module has a main function.
bool HasMainFunction(ModuleOp& module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc mht_3(mht_3_v, 234, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/insert_main_function.cc", "HasMainFunction");

  StringAttr main_func_id = StringAttr::get(module.getContext(), "main");
  for (auto function : module.getOps<FuncOp>()) {
    if (function.getName() == main_func_id) return true;
  }
  return false;
}

// Checks if a FuncOp is exported.
bool IsExported(FuncOp& op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc mht_4(mht_4_v, 246, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/insert_main_function.cc", "IsExported");

  auto exported_names = op->getAttrOfType<ArrayAttr>(kExportedNameAttr);
  return exported_names && !exported_names.empty();
}

// Check if a function is an entry function.
bool IsEntryFunction(FuncOp& op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc mht_5(mht_5_v, 255, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/insert_main_function.cc", "IsEntryFunction");
 return op->hasAttr(kEntryFunctionAttr); }

// Sets a function to be private so it can be referred internally.
void SetFunctionPrivate(FuncOp& func) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc mht_6(mht_6_v, 261, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/insert_main_function.cc", "SetFunctionPrivate");

  func.setVisibility(SymbolTable::Visibility::Private);

  // The `tf_saved_model` attributes can only be appied to public functions.
  for (auto& attr : func->getAttrs()) {
    StringRef attr_name = attr.getName().getValue();
    if (attr_name.startswith("tf_saved_model.")) {
      func->removeAttr(attr_name);
    }
  }

  for (int i = 0; i < func.getNumArguments(); ++i) {
    for (auto& attr : func.getArgAttrs(i)) {
      const StringAttr& attr_name = attr.getName();
      if (attr_name.getValue().startswith("tf_saved_model.")) {
        func.removeArgAttr(i, attr_name);
      }
    }
  }
  for (int i = 0; i < func.getNumResults(); ++i) {
    for (auto& attr : func.getResultAttrs(i)) {
      const StringAttr& attr_name = attr.getName();
      if (attr_name.getValue().startswith("tf_saved_model.")) {
        func.removeResultAttr(i, attr_name);
      }
    }
  }
}

// Creates a main function which calls other exported functions.
bool CreateMainFunction(ModuleOp& module) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc mht_7(mht_7_v, 294, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/insert_main_function.cc", "CreateMainFunction");

  MLIRContext* context = module.getContext();
  OpBuilder builder(context);

  // Collects argument and result types.
  llvm::SmallVector<Location> arg_locs;
  llvm::SmallVector<Type> arg_types, result_types;
  std::vector<std::string> input_names, output_names;
  for (auto function : module.getOps<FuncOp>()) {
    if (function.isPrivate() || !IsExported(function)) continue;
    arg_types.append(function.getArgumentTypes().begin(),
                     function.getArgumentTypes().end());
    auto& return_op = function.getBody().getBlocks().front().back();
    result_types.append(return_op.getOperandTypes().begin(),
                        return_op.getOperandTypes().end());
    for (const auto& arg : function.getArguments()) {
      arg_locs.push_back(arg.getLoc());
    }

    // Collects input and output node names. These names are prefixed with the
    // signature key in SavedModel. They also contain the index suffix. Ex:
    // "<signature key>_<name>:0", where 0 is the index.
    if (auto tf_attrs =
            function->getAttrOfType<DictionaryAttr>(kEntryFunctionAttr)) {
      if (auto inputs_attr = tf_attrs.get("inputs")) {
        std::string inputs_attr_str =
            inputs_attr.cast<StringAttr>().getValue().str();
        std::vector<std::string> inputs_attr_vec =
            absl::StrSplit(inputs_attr_str, ',', absl::SkipEmpty());
        input_names.insert(input_names.end(), inputs_attr_vec.begin(),
                           inputs_attr_vec.end());
      }
      if (auto outputs_attr = tf_attrs.get("outputs")) {
        std::string outputs_attr_str =
            outputs_attr.cast<StringAttr>().getValue().str();
        std::vector<std::string> outputs_attr_vec =
            absl::StrSplit(outputs_attr_str, ',', absl::SkipEmpty());
        output_names.insert(output_names.end(), outputs_attr_vec.begin(),
                            outputs_attr_vec.end());
      }
    }
  }

  // Creates a new main function.
  auto func_type = FunctionType::get(context, arg_types, result_types);
  auto main_func = builder.create<FuncOp>(module.getLoc(), "main", func_type);
  builder.createBlock(&main_func.getBody(), main_func.begin(), arg_types,
                      arg_locs);
  SmallVector<NamedAttribute> func_attrs;
  func_attrs.push_back(
      {StringAttr::get(context, "inputs"),
       StringAttr::get(context, absl::StrJoin(input_names, ","))});
  func_attrs.push_back(
      {StringAttr::get(context, "outputs"),
       StringAttr::get(context, absl::StrJoin(output_names, ","))});
  auto dictAttr = DictionaryAttr::get(context, func_attrs);
  main_func->setAttr(StringAttr::get(context, kEntryFunctionAttr), dictAttr);
  main_func->setAttr(kExportedNameAttr, builder.getStrArrayAttr({"main"}));

  if (input_names.size() != main_func.getNumArguments() ||
      output_names.size() != main_func.getNumResults()) {
    module.emitError() << "number of inputs and outputs in the "
                          "tf.entry_function attribute mismatched.";
    return false;
  }

  int numArgs = main_func.getNumArguments();
  for (int i = 0; i < numArgs; ++i) {
    main_func.setArgAttr(
        i, kIndexPathAttr,
        mlir::ArrayAttr::get(context,
                             {mlir::StringAttr::get(context, input_names[i])}));
  }

  int numResults = main_func.getNumResults();
  for (int i = 0; i < numResults; ++i) {
    main_func.setResultAttr(
        i, kIndexPathAttr,
        mlir::ArrayAttr::get(
            context, {mlir::StringAttr::get(context, output_names[i])}));
  }

  // Creates PartitionedCall ops to call exported functions.
  auto guard = OpBuilder::InsertionGuard(builder);
  int arg_idx = 0;
  int result_idx = 0;
  llvm::SmallVector<Value> returning_values;
  for (auto function : module.getOps<FuncOp>()) {
    if (function.isPrivate() || !IsExported(function) ||
        !IsEntryFunction(function)) {
      continue;
    }

    llvm::ArrayRef<BlockArgument> new_args = llvm::makeArrayRef(
        main_func.getArguments().begin() + arg_idx, function.getNumArguments());
    arg_idx += function.getNumArguments();
    llvm::ArrayRef<Type> new_types = llvm::makeArrayRef(
        result_types.begin() + result_idx, function.getNumResults());
    result_idx += function.getNumResults();

    auto call_op = builder.create<TF::PartitionedCallOp>(
        module.getLoc(), new_types, new_args,
        SymbolRefAttr::get(context, function.getSymName()),
        /*config=*/builder.getStringAttr(""),
        /*config_proto=*/builder.getStringAttr(""),
        /*executor_type=*/builder.getStringAttr(""));
    returning_values.append(call_op.getResults().begin(),
                            call_op.getResults().end());
    SetFunctionPrivate(function);
  }
  builder.create<mlir::func::ReturnOp>(main_func.getBody().getLoc(),
                                       returning_values);

  // Adds the new function to symbol table.
  SymbolTable symbol_table(module);
  symbol_table.insert(main_func);
  return true;
}

void InsertMainFunctionPass::runOnOperation() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSinsert_main_functionDTcc mht_8(mht_8_v, 416, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/insert_main_function.cc", "InsertMainFunctionPass::runOnOperation");

  ModuleOp module = getOperation();
  if (!HasMainFunction(module)) {
    if (!CreateMainFunction(module)) {
      signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateInsertMainFunctionPass() {
  return std::make_unique<InsertMainFunctionPass>();
}

static PassRegistration<InsertMainFunctionPass> pass([] {
  return CreateInsertMainFunctionPass();
});

}  // namespace quant
}  // namespace mlir
