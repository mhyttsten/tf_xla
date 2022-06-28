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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSxla_mlir_translate_registrationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSxla_mlir_translate_registrationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSxla_mlir_translate_registrationDTcc() {
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

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/mlir/xla/xla_mlir_translate.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
namespace {
// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_use_tuple_arg(
    "emit-use-tuple-args",
    llvm::cl::desc(
        "Emit HLO modules using tuples as args for the entry computation"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_return_tuple(
    "emit-return-tuple",
    llvm::cl::desc("Emit HLO modules with entry computations returning tuple"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> optimize_xla_hlo(
    "optimize-xla-hlo",
    llvm::cl::desc("Enable optimizations when translating XLA HLO -> LHLO"),
    llvm::cl::init(true));

// NOLINTNEXTLINE
llvm::cl::opt<bool> legalize_node_names(
    "legalize-node-names",
    llvm::cl::desc("Legalize nodes names when translating MHLO->XLA HLO"),
    llvm::cl::init(true));

// NOLINTNEXTLINE
llvm::cl::opt<bool> with_layouts(
    "with-layouts",
    llvm::cl::desc("Propagate layouts when translating MHLO->XLA HLO"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_layouts(
    "print-layouts", llvm::cl::desc("Print layouts in the generated HLO text"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> via_builder(
    "via-builder", llvm::cl::desc("Translate MHLO->XLA HLO via XLA Builder"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> import_all_computations(
    "hlo-import-all-computations",
    llvm::cl::desc("Enable importing unreachable computations."));
}  // namespace

namespace xla {

static mlir::LogicalResult MlirHloToHloTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSxla_mlir_translate_registrationDTcc mht_0(mht_0_v, 246, "", "./tensorflow/compiler/mlir/xla/xla_mlir_translate_registration.cc", "MlirHloToHloTranslateFunction");

  if (!module) return mlir::failure();

  HloProto hloProto;
  Status status = mlir::ConvertMlirHloToHlo(
      module, &hloProto, emit_use_tuple_arg, emit_return_tuple);
  if (!status.ok()) {
    LOG(ERROR) << "Module conversion failed: " << status;
    return mlir::failure();
  }

  output << hloProto.DebugString();
  return mlir::success();
}

static StatusOr<std::unique_ptr<HloModule>> HloModuleFromProto(
    const HloProto& hlo_proto) {
  const HloModuleProto& module_proto = hlo_proto.hlo_module();
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          module_proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(module_proto, module_config);
}

// Wraps BuildHloFromMlirHlo to output an HloProto that's the same as
// ConvertMlirHloToHlo.
Status ConvertMlirHloToHloViaBuilder(mlir::ModuleOp module,
                                     ::xla::HloProto* hlo_proto,
                                     mlir::MlirToHloConversionOptions options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSxla_mlir_translate_registrationDTcc mht_1(mht_1_v, 277, "", "./tensorflow/compiler/mlir/xla/xla_mlir_translate_registration.cc", "ConvertMlirHloToHloViaBuilder");

  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  mlir::Block& block = main.getRegion().front();
  xla::XlaBuilder builder("main");

  // Create xla_params.
  std::vector<xla::XlaOp> xla_params;
  for (mlir::BlockArgument& arg : block.getArguments()) {
    auto num = arg.getArgNumber();
    xla::Shape shape = xla::TypeToShape(arg.getType());
    XlaOp argop =
        xla::Parameter(&builder, num, shape, absl::StrCat("Arg_", num));
    xla_params.push_back(argop);
  }

  std::vector<xla::XlaOp> returns(1);
  TF_RETURN_IF_ERROR(
      mlir::BuildHloFromMlirHlo(block, builder, xla_params, returns, options));

  xla::XlaOp return_value;
  if (returns.size() == 1)
    return_value = returns[0];
  else if (returns.size() > 1)
    return_value = xla::Tuple(&builder, returns);

  TF_ASSIGN_OR_RETURN(
      xla::XlaComputation computation,
      return_value.valid() ? builder.Build(return_value) : builder.Build());
  auto hlo_module = computation.proto();
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);

  return Status::OK();
}

static mlir::LogicalResult MlirHloToHloTextTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSxla_mlir_translate_registrationDTcc mht_2(mht_2_v, 315, "", "./tensorflow/compiler/mlir/xla/xla_mlir_translate_registration.cc", "MlirHloToHloTextTranslateFunction");

  if (!module) return mlir::failure();

  HloProto hloProto;
  mlir::MlirToHloConversionOptions options;
  options.propagate_layouts = with_layouts;
  options.legalize_node_names = legalize_node_names;
  Status status =
      via_builder
          ? ConvertMlirHloToHloViaBuilder(module, &hloProto, options)
          : mlir::ConvertMlirHloToHlo(
                module, &hloProto, emit_use_tuple_arg, emit_return_tuple,
                /*shape_representation_fn=*/nullptr, options);
  if (!status.ok()) {
    LOG(ERROR) << "Module conversion failed: " << status;
    return mlir::failure();
  }

  auto statusOrHloModule = HloModuleFromProto(hloProto);

  if (!statusOrHloModule.ok()) {
    LOG(ERROR) << "Conversion to HLO module failed: "
               << statusOrHloModule.status();
    return mlir::failure();
  }

  HloModule* hlo_module = statusOrHloModule.ValueOrDie().get();

  output << hlo_module->ToString(
      HloPrintOptions().set_include_layout_in_shapes(print_layouts));

  // Output alias information as comments in the HLO text.
  hlo_module->input_output_alias_config().ForEachAlias(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) {
        output << "// OutputIndex " << output_index.ToString()
               << " aliases with input " << alias.parameter_number << " at "
               << alias.parameter_index.ToString() << "\n";
      });

  return mlir::success();
}

}  // namespace xla

//----------------------------------------------------------------------------//
// Hooks for tf-mlir-translate
//----------------------------------------------------------------------------/

static mlir::OwningOpRef<mlir::ModuleOp> HloToMlirHloTranslate(
    llvm::StringRef input, mlir::MLIRContext* context) {
  return xla::HloToMlirHloTranslateFunction(input, context,
                                            import_all_computations);
}

static mlir::OwningOpRef<mlir::ModuleOp> HloTextToMlirHloTranslate(
    llvm::StringRef input, mlir::MLIRContext* context) {
  return xla::HloTextToMlirHloTranslateFunction(input, context,
                                                import_all_computations);
}

static void RegisterInputDialects(mlir::DialectRegistry& registry) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSxla_mlir_translate_registrationDTcc mht_3(mht_3_v, 379, "", "./tensorflow/compiler/mlir/xla/xla_mlir_translate_registration.cc", "RegisterInputDialects");

  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::mhlo::MhloDialect, mlir::tensor::TensorDialect>();
}

static mlir::TranslateFromMLIRRegistration MlirHloToHloTranslate(
    "mlir-hlo-to-hlo", xla::MlirHloToHloTranslateFunction,
    RegisterInputDialects);

static mlir::TranslateFromMLIRRegistration MlirHloToHloTextTranslate(
    "mlir-hlo-to-hlo-text", xla::MlirHloToHloTextTranslateFunction,
    RegisterInputDialects);

static mlir::TranslateToMLIRRegistration HloToHloMlirTranslate(
    "hlo-to-mlir-hlo", HloToMlirHloTranslate);

static mlir::TranslateToMLIRRegistration HloTextToHloMlirTranslate(
    "hlo-text-to-mlir-hlo", HloTextToMlirHloTranslate);

// MHLO doesn't support explicit layouts, while XLA service does.
// TODO(timshen): remove it once MHLO supports explicit layouts.
static mlir::TranslateToMLIRRegistration HloTextToLhloMlirTranslate(
    "hlo-text-to-lhlo", [](llvm::StringRef input, mlir::MLIRContext* context) {
      return mlir::HloTextToLhloTranslateFunction(input, context,
                                                  optimize_xla_hlo);
    });
