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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc() {
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
#include "tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_attr.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.h"
#include "tensorflow/compiler/mlir/tfr/passes/passes.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace tfr {

const char* const kTFRLibEnv = "TF_MLIR_TFR_LIB_DIR";

StatusOr<std::unique_ptr<TFRDecomposeContext>> TFRDecomposeContext::Get(
    mlir::MLIRContext* mlir_ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc mht_0(mht_0_v, 230, "", "./tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.cc", "TFRDecomposeContext::Get");

  Env* env = Env::Default();
  std::string tfr_lib_dir;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar(
      kTFRLibEnv, "tensorflow/compiler/mlir/tfr/resources", &tfr_lib_dir));
  string composite_mlir_dir = io::JoinPath(env->GetRunfilesDir(), tfr_lib_dir);
  std::vector<string> files;
  TF_RETURN_IF_ERROR(env->GetChildren(composite_mlir_dir, &files));
  if (files.empty()) {
    return errors::Internal(absl::StrCat(
        "Failed to find the decomposition lib from path ", composite_mlir_dir));
  }
  std::string tfr_raw_text;
  for (const auto& file : files) {
    string fullpath = io::JoinPath(composite_mlir_dir, file);
    if (env->MatchPath(fullpath, io::JoinPath(composite_mlir_dir, "*.mlir"))) {
      std::string text;
      TF_RETURN_IF_ERROR(ReadFileToString(env, fullpath, &text));
      tfr_raw_text.append(text);
    }
  }

  auto ctx = TFRDecomposeContext::GetFromText(tfr_raw_text, mlir_ctx);
  if (!ctx) {
    return errors::Internal(absl::StrCat(
        "Failed to load the imported decomposition lib: ", tfr_raw_text));
  }
  return ctx;
}

std::unique_ptr<TFRDecomposeContext> TFRDecomposeContext::GetFromText(
    StringPiece tfr_raw_text, mlir::MLIRContext* mlir_ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc mht_1(mht_1_v, 264, "", "./tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.cc", "TFRDecomposeContext::GetFromText");

  mlir_ctx->allowUnregisteredDialects(/*allow=*/true);
  // Load dialects involved in the conversion
  mlir::DialectRegistry registry;
  // clang-format off
  registry.insert<mlir::arith::ArithmeticDialect,
                  mlir::func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::shape::ShapeDialect,
                  mlir::TF::TensorFlowDialect,
                  mlir::tf_device::TensorFlowDeviceDialect,
                  mlir::tf_executor::TensorFlowExecutorDialect,
                  mlir::TFR::TFRDialect>();
  // clang-format on
  mlir_ctx->appendDialectRegistry(registry);
  mlir_ctx->loadAllAvailableDialects();

  // Load the TFR functions in a mlir::ModuleOp
  auto memory_buffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(tfr_raw_text.data(), tfr_raw_text.size()));
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(memory_buffer), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, mlir_ctx);
  // The MLIRContext owns the module
  auto module_op = module.release();

  // Create the context
  return absl::make_unique<TFRDecomposeContext>(module_op);
}

StatusOr<FunctionDef> TFRDecomposeContext::ExpandNode(const NodeDef& node_def,
                                                      StringPiece func_name) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc mht_2(mht_2_v, 299, "", "./tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.cc", "TFRDecomposeContext::ExpandNode");

  const OpDef* op_def;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def));
  DataTypeVector input_dtys, output_dtys;
  TF_RETURN_IF_ERROR(InputTypesForNode(node_def, *op_def, &input_dtys));
  TF_RETURN_IF_ERROR(OutputTypesForNode(node_def, *op_def, &output_dtys));

  mlir::MLIRContext* context = tfr_module_.getContext();
  llvm::SmallVector<mlir::Type, 4> input_tys, output_tys;
  mlir::Builder builder(context);
  for (auto ty : input_dtys) {
    mlir::Type elt_ty;
    TF_RETURN_IF_ERROR(ConvertDataType(ty, builder, &elt_ty));
    mlir::TensorType mlir_ty = mlir::UnrankedTensorType::get(elt_ty);
    input_tys.push_back(mlir_ty);
  }
  for (auto ty : output_dtys) {
    mlir::Type elt_ty;
    TF_RETURN_IF_ERROR(ConvertDataType(ty, builder, &elt_ty));
    mlir::TensorType mlir_ty = mlir::UnrankedTensorType::get(elt_ty);
    output_tys.push_back(mlir_ty);
  }
  llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
  for (const auto& attr : node_def.attr()) {
    TF_ASSIGN_OR_RETURN(auto mlir_attr,
                        ConvertAttributeValue(attr.second, &builder));
    attrs.push_back({mlir::StringAttr::get(context, attr.first), mlir_attr});
  }

  mlir::Location loc = mlir::UnknownLoc::get(context);
  mlir::ModuleOp module = mlir::ModuleOp::create(loc);
  mlir::FunctionType func_type =
      mlir::FunctionType::get(context, input_tys, output_tys);
  llvm::StringRef func_name_str(func_name.data(), func_name.size());
  auto func = mlir::func::FuncOp::create(loc, func_name_str, func_type, {});
  module.push_back(func);
  func.addEntryBlock();
  mlir::OpBuilder op_builder(func.getBody());

  // Create the TF op
  const std::string tf_op_full_name = absl::StrCat("tf.", node_def.op());
  mlir::OperationState op_state(loc, tf_op_full_name);
  op_state.addOperands(func.getArguments());
  op_state.addTypes(output_tys);
  op_state.addAttributes(attrs);
  mlir::Operation* tf_op = op_builder.create(op_state);
  op_builder.create<mlir::func::ReturnOp>(loc, tf_op->getResults());

  // Run the decompose passes on the module
  TF_RETURN_IF_ERROR(DecomposeGraph(module));

  // Export the result as a FunctionDef.
  FunctionDef func_def;
  TF_RETURN_IF_ERROR(
      ConvertMlirFunctionToFunctionLibraryDef(func, export_confs_, &func_def));
  module.erase();
  return func_def;
}

Status TFRDecomposeContext::DecomposeGraph(mlir::ModuleOp user_module) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc mht_3(mht_3_v, 361, "", "./tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.cc", "TFRDecomposeContext::DecomposeGraph");

  // Call the decompose passes by using the external symbol table.
  if (failed(pm_.run(user_module))) {
    return errors::Internal("Failed to run the decompose passes.");
  }
  return Status::OK();
}

// Constructor of the decompose context.
TFRDecomposeContext::TFRDecomposeContext(mlir::ModuleOp tfr_module)
    : tfr_module_(tfr_module), pm_(tfr_module_.getContext()) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc mht_4(mht_4_v, 374, "", "./tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.cc", "TFRDecomposeContext::TFRDecomposeContext");

  mlir::OpPassManager& func_pm = pm_.nest<mlir::func::FuncOp>();

  // Prepare the imported graph.
  func_pm.addPass(mlir::CreateExecutorDialectToFunctionalConversionPass());

  // Run TFR lowering, inlining and raising to tf.
  func_pm.addPass(mlir::TFR::CreateDecomposeTFOpsPass(tfr_module_));
  func_pm.addPass(mlir::TFR::CreateRaiseToTFOpsPass(
      tfr_module_, /*materialize_derived_attrs=*/true));

  // Prepare to be exported.
  func_pm.addPass(mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm_.addPass(mlir::CreateBreakUpIslandsPass());
}

void TFRDecomposeContext::Destroy() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc mht_5(mht_5_v, 393, "", "./tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.cc", "TFRDecomposeContext::Destroy");
 tfr_module_.erase(); }

StatusOr<FunctionDef> ExpandNode(const NodeDef& node_def,
                                 StringPiece func_name) {
  mlir::MLIRContext mlir_ctx;
  TF_ASSIGN_OR_RETURN(auto ctx, TFRDecomposeContext::Get(&mlir_ctx));
  return ctx->ExpandNode(node_def, func_name);
}

Status DecomposeGraph(mlir::ModuleOp user_module) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSintegrationPStfr_decompose_ctxDTcc mht_6(mht_6_v, 405, "", "./tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.cc", "DecomposeGraph");

  mlir::MLIRContext* mlir_ctx = user_module.getContext();
  TF_ASSIGN_OR_RETURN(auto ctx, TFRDecomposeContext::Get(mlir_ctx));
  return ctx->DecomposeGraph(user_module);
}

}  // namespace tfr
}  // namespace tensorflow
