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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_saved_model {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool IsStrArrayAttr(Attribute attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "IsStrArrayAttr");

  auto array = attr.dyn_cast<ArrayAttr>();
  if (!array) return false;

  return llvm::all_of(array,
                      [](Attribute attr) { return attr.isa<StringAttr>(); });
}

//===----------------------------------------------------------------------===//
// TensorFlowSavedModelDialect Op's
//===----------------------------------------------------------------------===//

LogicalResult VerifyTensorTypesCompatible(Type t1, Type t2) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "VerifyTensorTypesCompatible");

  if (!t1.isa<TensorType>() || !t2.isa<TensorType>()) {
    return failure();
  }
  return verifyCompatibleShape(t1.cast<TensorType>(), t2.cast<TensorType>());
}

LogicalResult GlobalTensorOp::verify() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_2(mht_2_v, 239, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "GlobalTensorOp::verify");

  GlobalTensorOp global_tensor = *this;
  if (failed(VerifyTensorTypesCompatible(
          global_tensor.type(), global_tensor.value().Attribute::getType()))) {
    return global_tensor.emitError() << "'type' and 'value' attributes should "
                                        "have compatible tensor types";
  }
  if (!global_tensor.is_mutable()) {
    if (!global_tensor.type().cast<TensorType>().hasStaticShape()) {
      return global_tensor.emitError()
             << "'type' attribute for immutable 'tf_saved_model.global_tensor' "
                "should have a static shape";
    }
  }
  return success();
}

LogicalResult SessionInitializerOp::verify() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_3(mht_3_v, 259, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "SessionInitializerOp::verify");

  SessionInitializerOp session_initializer = *this;
  mlir::SymbolTable symbol_table(
      session_initializer->getParentOfType<ModuleOp>());

  for (auto sym_ref : session_initializer.initializers()) {
    auto init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
        sym_ref.cast<FlatSymbolRefAttr>().getValue());

    if (!init_func_op)
      return session_initializer.emitOpError()
             << "the initializer function does not exist";

    if (!init_func_op.getFunctionType().getResults().empty())
      return session_initializer.emitOpError()
             << "the initializer function should have no output";

    auto exported_names = GetExportedNames(init_func_op);

    if (exported_names.empty())
      return session_initializer.emitOpError()
             << "the initializer function should be exported";

    if (exported_names.size() != 1)
      return session_initializer.emitOpError()
             << "the initializer function should have only one exported names";
  }

  return success();
}

}  // namespace tf_saved_model
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc.inc"

namespace mlir {
namespace tf_saved_model {

//===----------------------------------------------------------------------===//
// TensorFlowSavedModelDialect Dialect
//===----------------------------------------------------------------------===//

TensorFlowSavedModelDialect::TensorFlowSavedModelDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf_saved_model", context,
              TypeID::get<TensorFlowSavedModelDialect>()) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_4(mht_4_v, 308, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "TensorFlowSavedModelDialect::TensorFlowSavedModelDialect");

  // The TensorFlow Dialect is needed in the verifier and other routines
  // associated to this dialect. It makes little sense anyway to use the
  // SavedModel dialect without the TensorFlow Dialect.
  context->loadDialect<TF::TensorFlowDialect>();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc.inc"
      >();
}

static LogicalResult VerifyIndexPath(Operation *op, NamedAttribute named_attr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_5(mht_5_v, 323, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "VerifyIndexPath");

  auto attr = named_attr.getValue().dyn_cast<ArrayAttr>();
  if (!attr) {
    return op->emitError()
           << "'tf_saved_model.index_path' attribute should be an ArrayAttr";
  }
  for (auto element : attr) {
    if (element.isa<StringAttr>()) {
      continue;
    }
    if (auto integer = element.dyn_cast<IntegerAttr>()) {
      if (integer.getValue().getBitWidth() == 64) {
        continue;
      }
    }
    return op->emitError() << "'tf_saved_model.index_path' elements should "
                              "be strings or 64-bit integers";
  }
  return mlir::success();
}

Type GetBoundInputArgTypeFor(mlir::Operation *op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_6(mht_6_v, 347, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "GetBoundInputArgTypeFor");

  if (auto global_tensor = llvm::dyn_cast<GlobalTensorOp>(op)) {
    auto type = global_tensor.type().cast<TensorType>();
    return RankedTensorType::get(
        {}, TF::ResourceType::get({type}, type.getContext()));
  }

  if (auto asset = llvm::dyn_cast<AssetOp>(op)) {
    return RankedTensorType::get({}, TF::StringType::get(asset.getContext()));
  }

  op->emitError() << "unknown symbol operation";
  return {};
}

static LogicalResult VerifyBoundInputArgType(Operation *op_for_diagnostics,
                                             Type arg_type,
                                             mlir::Operation *symbol_op) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_7(mht_7_v, 367, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "VerifyBoundInputArgType");

  auto expected_type = GetBoundInputArgTypeFor(symbol_op);
  if (!expected_type) return failure();

  if (arg_type != expected_type) {
    return op_for_diagnostics->emitError()
           << "bound input with type " << arg_type << " expected to have type "
           << expected_type;
  }
  return success();
}

LogicalResult TensorFlowSavedModelDialect::verifyRegionArgAttribute(
    Operation *op, unsigned region_index, unsigned arg_index,
    NamedAttribute named_attr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_8(mht_8_v, 384, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "TensorFlowSavedModelDialect::verifyRegionArgAttribute");

  if (named_attr.getName() == "tf_saved_model.bound_input") {
    if (!named_attr.getValue().isa<FlatSymbolRefAttr>()) {
      return op->emitError() << "'tf_saved_model.bound_input' attribute should "
                                "be a FlatSymbolRefAttr";
    }
    auto symbol_name =
        named_attr.getValue().cast<FlatSymbolRefAttr>().getValue();
    auto module = op->getParentOfType<ModuleOp>();
    mlir::Operation *symbol_op = module.lookupSymbol(symbol_name);
    if (!symbol_op) {
      return op->emitError() << "'tf_saved_model.bound_input' attribute must "
                                "reference a valid symbol, got invalid symbol '"
                             << symbol_name << "'";
    }
    auto arg_type = cast<FuncOp>(op).getArgument(arg_index).getType();
    return VerifyBoundInputArgType(op, arg_type, symbol_op);
  }
  if (named_attr.getName() == "tf_saved_model.index_path") {
    return VerifyIndexPath(op, named_attr);
  }

  return op->emitError() << "unknown tf_saved_model dialect arg attribute '"
                         << named_attr.getName().getValue() << "'";
}

LogicalResult TensorFlowSavedModelDialect::verifyRegionResultAttribute(
    Operation *op, unsigned region_index, unsigned result_index,
    NamedAttribute named_attr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_9(mht_9_v, 415, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "TensorFlowSavedModelDialect::verifyRegionResultAttribute");

  if (named_attr.getName() == "tf_saved_model.index_path") {
    return VerifyIndexPath(op, named_attr);
  }

  return op->emitError() << "unknown tf_saved_model dialect result attribute '"
                         << named_attr.getName().getValue() << "'";
}

static bool HasAnyTfSavedModelArgAttr(FuncOp func) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_10(mht_10_v, 427, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "HasAnyTfSavedModelArgAttr");

  for (int i = 0, e = func.getNumArguments(); i < e; i++) {
    if (func.getArgAttr(i, "tf_saved_model.index_path") ||
        func.getArgAttr(i, "tf_saved_model.bound_input")) {
      return true;
    }
  }
  for (int i = 0, e = func.getNumResults(); i < e; i++) {
    if (func.getResultAttr(i, "tf_saved_model.index_path") ||
        func.getResultAttr(i, "tf_saved_model.bound_input")) {
      return true;
    }
  }
  return false;
}

static LogicalResult VerifySavedModelModule(
    ModuleOp module, TensorFlowSavedModelDialect *dialect) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_11(mht_11_v, 447, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "VerifySavedModelModule");

  auto exported_names_ident =
      StringAttr::get(dialect->getContext(), "tf_saved_model.exported_names");
  // Check that there are no duplicated exported_names.
  DenseMap<StringRef, Operation *> exported_name_to_op;
  for (auto &op : module) {
    auto attr = op.getAttr(exported_names_ident);
    if (!attr) continue;
    // If this verifier is called before we verify the
    // 'tf_saved_model.exported_names' attribute, then it might be invalid.
    // Forward to the dialect's verification to establish that precondition.
    if (failed(dialect->verifyOperationAttribute(
            &op, {exported_names_ident, attr}))) {
      return failure();
    }
    for (auto str : attr.cast<ArrayAttr>()) {
      auto exported_name = str.cast<StringAttr>().getValue();
      auto p = exported_name_to_op.insert({exported_name, &op});
      if (!p.second) {
        return op.emitError()
            .append("duplicate exported name '", exported_name, "'")
            .attachNote(p.first->getSecond()->getLoc())
            .append("previously seen here");
      }
    }
  }
  for (auto func : module.getOps<FuncOp>()) {
    const bool is_exported = IsExported(func);

    if (is_exported && !func.isPublic()) {
      return func.emitError()
             << "exported function @" << func.getName() << " should be public";
    }

    if (!is_exported && func.isPublic()) {
      return func.emitError() << "non-exported function @" << func.getName()
                              << " should be private";
    }
    if (!is_exported && HasAnyTfSavedModelArgAttr(func)) {
      return func.emitError() << "can only apply 'tf_saved_model' argument "
                                 "attributes to exported functions";
    }
  }

  auto session_initializers = module.getOps<SessionInitializerOp>();
  if (!session_initializers.empty() &&
      !llvm::hasSingleElement(session_initializers)) {
    return (*++session_initializers.begin()).emitError()
           << "there must be no more than one session_initializer op";
  }

  auto is_init = [&session_initializers](mlir::func::FuncOp func) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_12(mht_12_v, 501, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "lambda");

    if (session_initializers.empty()) return false;
    auto init_syms = (*session_initializers.begin()).initializers();
    return std::any_of(
        init_syms.begin(), init_syms.end(), [&](Attribute sym_ref) {
          return sym_ref.cast<FlatSymbolRefAttr>().getValue() == func.getName();
        });
  };

  SymbolTable symbol_table(module);
  auto symbol_uses = SymbolTable::getSymbolUses(&module.getBodyRegion());
  if (!symbol_uses.hasValue()) {
    return module.emitError() << "modules with 'tf_saved_model.semantics' must "
                                 "have analyzable symbol uses";
  }
  for (auto symbol_use : *symbol_uses) {
    auto func = symbol_table.lookupNearestSymbolFrom<FuncOp>(
        symbol_use.getUser(), symbol_use.getSymbolRef());
    if (func && IsExported(func)) {
      // If it is an init function, then it can be used by the unique
      // session_initializer op.
      if (is_init(func) &&
          llvm::isa<SessionInitializerOp>(symbol_use.getUser()))
        continue;

      return symbol_use.getUser()
          ->emitError("exported function cannot be internally referenced")
          .attachNote(func.getLoc())
          .append("references this exported function");
    }
  }
  return success();
}

LogicalResult VerifyExportedFunc(FuncOp func) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_13(mht_13_v, 538, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "VerifyExportedFunc");

  bool reached_bound_inputs = false;
  auto module = func->getParentOfType<ModuleOp>();
  for (int i = 0, e = func.getNumArguments(); i < e; i++) {
    if (func.getArgAttr(i, "tf_saved_model.bound_input")) {
      reached_bound_inputs = true;
      continue;
    }
    if (func.getArgAttr(i, "tf_saved_model.index_path")) {
      if (reached_bound_inputs) {
        return func.emitError()
               << "all 'tf_saved_model.index_path' arg attributes should "
                  "precede all 'tf_saved_model.bound_input' arg attributes";
      }
      continue;
    }
    if (func.getArgAttr(i, "tf.resource_name")) {
      if (module->getAttr("tf_saved_model.under_construction")) continue;
      return func.emitError() << "'tf.resource_name' attribute is not allowed "
                                 "unless it is being under construction";
    }
    return func.emitError()
           << "all arguments should have 'tf_saved_model.index_path', "
              "'tf_saved_model.bound_input' or 'tf.resource_name' attributes";
  }
  llvm::SmallDenseSet<StringRef, 8> unique_bound_inputs;
  for (int i = 0, e = func.getNumArguments(); i < e; i++) {
    if (auto attr = func.getArgAttrOfType<FlatSymbolRefAttr>(
            i, "tf_saved_model.bound_input")) {
      if (!unique_bound_inputs.insert(attr.getValue()).second) {
        if (module->getAttr("tf_saved_model.under_construction")) continue;
        return func.emitError()
               << "duplicate 'tf_saved_model.bound_input' binding";
      }
    }
  }

  for (int i = 0, e = func.getNumResults(); i < e; i++) {
    if (!func.getResultAttr(i, "tf_saved_model.index_path")) {
      return func.emitError() << "all results should have "
                                 "'tf_saved_model.index_path' attributes";
    }
  }

  return success();
}

LogicalResult TensorFlowSavedModelDialect::verifyOperationAttribute(
    Operation *op, NamedAttribute named_attr) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_14(mht_14_v, 589, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "TensorFlowSavedModelDialect::verifyOperationAttribute");

  if (named_attr.getName() == "tf_saved_model.exported_names") {
    if (!isa<FuncOp, GlobalTensorOp>(op)) {
      return op->emitError() << "'tf_saved_model.exported_names' must be on a "
                                "'func' or 'tf_saved_model.global_tensor' op";
    }
    if (!IsStrArrayAttr(named_attr.getValue())) {
      return op->emitError()
             << "'tf_saved_model.exported_names' must be an array of strings";
    }
    if (!op->getParentOp()->getAttr("tf_saved_model.semantics")) {
      return op->emitError()
             << "'tf_saved_model.exported_names' must be on an op "
                "whose immediate parent has attribute "
                "'tf_saved_model.semantics'";
    }
    if (auto func = dyn_cast<FuncOp>(op)) {
      if (failed(VerifyExportedFunc(func))) {
        return failure();
      }
    }
    return success();
  }
  if (named_attr.getName() == "tf_saved_model.semantics") {
    auto module = dyn_cast<ModuleOp>(op);
    if (!module) {
      return op->emitError() << "'tf_saved_model.semantics' must "
                                "be on a module op";
    }
    return VerifySavedModelModule(module, this);
  }
  if (named_attr.getName() == "tf_saved_model.under_construction") {
    return success();
  }

  return op->emitError() << "unknown tf_saved_model dialect attribute '"
                         << named_attr.getName().getValue() << "'";
}

SmallVector<StringRef, 2> GetExportedNames(Operation *op) {
  SmallVector<StringRef, 2> ret;
  auto exported_names =
      op->getAttrOfType<ArrayAttr>("tf_saved_model.exported_names");
  if (exported_names) {
    for (auto name : exported_names) {
      ret.push_back(name.cast<StringAttr>().getValue());
    }
  }
  return ret;
}

bool IsExported(Operation *op) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_15(mht_15_v, 643, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "IsExported");

  auto exported_names =
      op->getAttrOfType<ArrayAttr>("tf_saved_model.exported_names");
  return exported_names && !exported_names.empty();
}

bool HasTfSavedModelSemantics(ModuleOp module) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_16(mht_16_v, 652, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "HasTfSavedModelSemantics");

  return module->getAttr("tf_saved_model.semantics") != nullptr;
}

Operation *LookupBoundInput(FuncOp func, int arg_index,
                            const SymbolTable &symbol_table) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_17(mht_17_v, 660, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "LookupBoundInput");

  auto attr = func.getArgAttrOfType<FlatSymbolRefAttr>(
      arg_index, "tf_saved_model.bound_input");
  if (!attr) return nullptr;
  return symbol_table.lookup(attr.getValue());
}

SessionInitializerOp GetSessionInitializerOp(mlir::ModuleOp op) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_18(mht_18_v, 670, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "GetSessionInitializerOp");

  auto initializers = op.getOps<SessionInitializerOp>();
  if (initializers.empty()) return {};
  return *initializers.begin();
}

class OptimizeSessionInitializerPattern
    : public OpRewritePattern<SessionInitializerOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SessionInitializerOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_19(mht_19_v, 685, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "matchAndRewrite");

    SymbolTable symbol_table(op->getParentOfType<ModuleOp>());

    SmallVector<FuncOp, 2> to_remove;
    SmallVector<mlir::Attribute, 2> to_keep;
    for (auto sym_ref : op.initializers()) {
      auto init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
          sym_ref.cast<FlatSymbolRefAttr>().getValue());

      // The init function can only be referenced from the SessionInitializerOp.
      // And there is at most one SessionInitializerOp in the module. So if both
      // ops have no other uses or have one NoOp only, they can be simply
      // erased.
      auto &operations = init_func_op.front().getOperations();
      if ((operations.size() == 1 &&
           operations.front().hasTrait<OpTrait::IsTerminator>()) ||
          (operations.size() == 2 &&
           dyn_cast<mlir::TF::NoOp>(operations.front()) &&
           operations.back().hasTrait<OpTrait::IsTerminator>())) {
        to_remove.push_back(init_func_op);
      } else {
        to_keep.push_back(sym_ref);
      }
    }

    for (auto func_op : to_remove) rewriter.eraseOp(func_op);

    if (to_keep.empty())
      rewriter.eraseOp(op);
    else
      op->setAttr("initializers", rewriter.getArrayAttr(to_keep));

    return success();
  }
};

void SessionInitializerOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_saved_modelDTcc mht_20(mht_20_v, 725, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc", "SessionInitializerOp::getCanonicalizationPatterns");

  results.add<OptimizeSessionInitializerPattern>(context);
}

SmallVector<StringRef, 2> GetSessionInitializerExportedName(ModuleOp op) {
  auto session_initializer_op = GetSessionInitializerOp(op);
  if (!session_initializer_op) return {};

  SymbolTable symbol_table(op);

  SmallVector<StringRef, 2> results;
  for (auto sym_ref : session_initializer_op.initializers()) {
    auto init_func_op = symbol_table.lookup<mlir::func::FuncOp>(
        sym_ref.cast<FlatSymbolRefAttr>().getValue());
    auto exported_names = GetExportedNames(init_func_op);
    assert(exported_names.size() == 1);
    results.push_back(exported_names[0]);
  }

  return results;
}

}  // namespace tf_saved_model
}  // namespace mlir
