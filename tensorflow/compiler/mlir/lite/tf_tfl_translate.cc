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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_translateDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_translateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_translateDTcc() {
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

#include <iostream>

#include "absl/strings/str_split.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export_flags.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_translate_cl.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/xla/xla_mlir_translate.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::func::FuncOp;
using stream_executor::port::StatusOr;

// Debugging flag to print function mapping in the flatbuffer.
// NOLINTNEXTLINE
static llvm::cl::opt<bool> print_function_result_mapping(
    "print-function-result-mapping",
    llvm::cl::desc(
        "Print the mapping of function result to flatbuffer output buffer"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> weight_quantization(
    "weight_quantization",
    llvm::cl::desc("The type of the quantized weight buffer. Must be NONE, "
                   "INT8, FLOAT16."),
    llvm::cl::init("NONE"));

enum TranslationStatus { kTrSuccess, kTrFailure };

static int PrintFunctionResultMapping(const std::string &result,
                                      ModuleOp module) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("result: \"" + result + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_translateDTcc mht_0(mht_0_v, 252, "", "./tensorflow/compiler/mlir/lite/tf_tfl_translate.cc", "PrintFunctionResultMapping");

  // Build model from the resultant string to extract the return values from
  // their source of truth.
  auto model =
      tflite::FlatBufferModel::BuildFromBuffer(result.data(), result.size());
  if (!model) return kTrFailure;

  // Get an unknown location for where we don't have a terminator to get the
  // location of the return value from.
  auto unknown_loc = mlir::UnknownLoc::get(module.getContext());

  auto print_buffer = [&](const tflite::SubGraph &subgraph, int id, int buffer,
                          std::function<mlir::Location(int)> loc) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_translateDTcc mht_1(mht_1_v, 267, "", "./tensorflow/compiler/mlir/lite/tf_tfl_translate.cc", "lambda");

    const auto &output_tensor = (*subgraph.tensors())[buffer];
    std::cout << "\tname: '"
              << (output_tensor->name() ? output_tensor->name()->str()
                                        : "<<unnamed>>")
              << "' buffer: " << buffer;
    if (loc) std::cout << llvm::formatv(" {0}", loc(id)).str();
    std::cout << '\n';
  };

  // For every subgraph print out the name (if available), each result's output
  // buffer number and location of the return value (if available).
  for (auto *subgraph : *(*model)->subgraphs()) {
    std::string subgraph_name =
        subgraph->name() ? subgraph->name()->str() : "<<unnamed subgraph>>";

    std::cout << '\'' << subgraph_name << "' inputs:\n";
    int i = 0;
    for (auto input : *subgraph->inputs())
      print_buffer(*subgraph, i++, input, nullptr);

    std::cout << '\'' << subgraph_name << "' outputs:\n";
    mlir::Operation *terminator = nullptr;
    if (subgraph->name()) {
      if (auto fn = module.lookupSymbol<FuncOp>(subgraph->name()->str()))
        terminator = fn.back().getTerminator();
    }
    i = 0;
    for (auto output : *subgraph->outputs()) {
      print_buffer(*subgraph, i, output, [&](int i) {
        return terminator ? terminator->getOperand(i).getLoc() : unknown_loc;
      });
    }
  }
  return kTrSuccess;
}

int main(int argc, char **argv) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStf_tfl_translateDTcc mht_2(mht_2_v, 307, "", "./tensorflow/compiler/mlir/lite/tf_tfl_translate.cc", "main");

  // TODO(jpienaar): Revise the command line option parsing here.
  tensorflow::InitMlir y(&argc, &argv);

  // TODO(antiagainst): We are pulling in multiple transformations as follows.
  // Each transformation has its own set of command-line options; options of one
  // transformation can essentially be aliases to another. For example, the
  // -tfl-annotate-inputs has -tfl-input-arrays, -tfl-input-data-types, and
  // -tfl-input-shapes, which are the same as -graphdef-to-mlir transformation's
  // -tf_input_arrays, -tf_input_data_types, and -tf_input_shapes, respectively.
  // We need to disable duplicated ones to provide a cleaner command-line option
  // interface. That also means we need to relay the value set in one option to
  // all its aliases.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "TF GraphDef to TFLite FlatBuffer converter\n");

  MLIRContext context;
  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(source_mgr, &context);

  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module;
  std::unordered_set<std::string> tags;

  tensorflow::GraphImportConfig specs;
  specs.upgrade_legacy = upgrade_legacy;
  specs.prune_unused_nodes = true;

  if (!select_user_tf_ops.empty() && !emit_select_tf_ops) {
    llvm::errs() << "You must specify `emit-select-tf-ops=true` when passing "
                    "`select-user-tf-ops` flag.";
    return kTrFailure;
  }

  std::unique_ptr<tensorflow::SavedModelBundle> bundle;

  // TODO(b/147435528): We need to test the e2e behavior once the graph freezing
  // inside mlir is done.
  if ((import_saved_model_object_graph || import_saved_model_signature_defs) &&
      import_hlo) {
    llvm::errs() << "Import saved model and import hlo cannot be both set.";
    return kTrFailure;
  }

  if (import_saved_model_object_graph || import_saved_model_signature_defs) {
    // Saved model import path.
    int saved_model_version;
    if (import_saved_model_object_graph) {
      saved_model_version = 2;
    } else {
      saved_model_version = 1;
    }
    if (input_mlir)
      module = tensorflow::errors::InvalidArgument(
          "Importing saved model should not have input_mlir set");

    tags = absl::StrSplit(saved_model_tags, ',');
    std::vector<std::string> exported_names_vector =
        absl::StrSplit(saved_model_exported_names, ',', absl::SkipEmpty());
    absl::Span<std::string> exported_names(exported_names_vector);

    if (exported_names.size() != 1) {
      llvm::errs() << "There should be only one exported name";
      return kTrFailure;
    }
    std::vector<std::string> extra_opdefs(custom_opdefs.begin(),
                                          custom_opdefs.end());
    module = tensorflow::ImportSavedModel(
        input_file_name, saved_model_version, tags, extra_opdefs,
        exported_names, specs, /*enable_variable_lifting=*/true, &context,
        &bundle);
  } else if (import_hlo) {
    // HLO import path.
    std::string error;
    std::unique_ptr<llvm::MemoryBuffer> buffer =
        mlir::openInputFile(input_file_name, &error);
    if (buffer == nullptr) {
      llvm::errs() << "Cannot open input file: " << input_file_name << " "
                   << error;
      return kTrFailure;
    }

    auto content = buffer->getBuffer();
    if (hlo_import_type == HloImportType::hlotxt) {
      module = xla::HloTextToMlirHloTranslateFunction(content, &context, false);
    } else if (hlo_import_type == HloImportType::proto) {
      module = xla::HloToMlirHloTranslateFunction(content, &context, false);
    } else {
      module = mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::parseSourceString<mlir::ModuleOp>(content, &context));
    }
  } else {
    // Graphdef import path.
    module = tensorflow::LoadFromGraphdefOrMlirSource(
        input_file_name, input_mlir, use_splatted_constant, custom_opdefs,
        specs, debug_info_file, input_arrays, input_dtypes, input_shapes,
        output_arrays, control_output_arrays, &source_mgr, &context);
  }

  // If errors occur, the library call in the above already logged the error
  // message. So we can just return here.
  if (!module.ok()) return kTrFailure;

  // Set the quantization specifications from the command line flags.
  mlir::quant::QuantizationSpecs quant_specs;
  if (mlir::quant::ParseInputNodeQuantSpecs(
          input_arrays, min_values, max_values, inference_type, &quant_specs)) {
    llvm::errs() << "Failed to get input quant spec.";
    return kTrFailure;
  }
  if (weight_quantization != "NONE") {
    quant_specs.weight_quantization = true;
    if (weight_quantization == "INT8") {
      quant_specs.inference_type = tensorflow::DT_QINT8;
    } else if (weight_quantization == "FLOAT16") {
      quant_specs.inference_type = tensorflow::DT_HALF;
    } else {
      llvm::errs() << "Unknown weight quantization " << weight_quantization;
      return kTrFailure;
    }
  }
  if (!emit_quant_adaptor_ops) {
    quant_specs.inference_input_type = quant_specs.inference_type;
  }

  if (!quant_stats_file_name.empty()) {
    std::string error_message;
    auto file = mlir::openInputFile(quant_stats_file_name, &error_message);
    if (!file) {
      llvm::errs() << "fail to open quant stats file: "
                   << quant_stats_file_name;
      return kTrFailure;
    }
    quant_specs.serialized_quant_stats = file->getBuffer().str();
  }

  mlir::TFL::PassConfig pass_config(quant_specs);
  pass_config.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
  pass_config.lower_tensor_list_ops = lower_tensor_list_ops;
  pass_config.unfold_batch_matmul = unfold_batchmatmul;
  pass_config.unfold_large_splat_constant = unfold_large_splat_constant;
  pass_config.guarantee_all_funcs_one_use = guarantee_all_funcs_one_use;
  pass_config.enable_dynamic_update_slice = enable_dynamic_update_slice;
  pass_config.runtime_verification = true;
  pass_config.outline_tf_while = true;
  pass_config.preserve_assert_op = preserve_assert_op;

  if (enable_hlo_to_tf_conversion) {
    pass_config.enable_hlo_to_tf_conversion = true;
  }

  toco::TocoFlags toco_flags;
  toco_flags.set_force_select_tf_ops(!emit_builtin_tflite_ops);
  toco_flags.set_enable_select_tf_ops(emit_select_tf_ops);
  toco_flags.set_allow_custom_ops(emit_custom_ops);
  toco_flags.set_allow_all_select_tf_ops(allow_all_select_tf_ops);
  toco_flags.set_enable_dynamic_update_slice(enable_dynamic_update_slice);
  // Read list of user select ops.
  llvm::SmallVector<llvm::StringRef, 2> user_ops;
  (llvm::StringRef(select_user_tf_ops))
      .split(user_ops, ',', /*MaxSplit=*/-1,
             /*KeepEmpty=*/false);
  llvm::for_each(user_ops, [&toco_flags](llvm::StringRef op_name) {
    *(toco_flags.add_select_user_tf_ops()) = op_name.str();
  });

  std::string result;
  // TODO(b/153507667): Pass the session object when importing logic is removed.
  auto status = tensorflow::ConvertTFExecutorToTFLOrFlatbuffer(
      module.ValueOrDie().get(), output_mlir, toco_flags, pass_config, tags,
      /*saved_model_dir=*/"", /*session=*/llvm::None, &result);
  if (!status.ok()) return kTrFailure;

  std::string error_msg;
  auto output = mlir::openOutputFile(output_file_name, &error_msg);
  if (output == nullptr) {
    llvm::errs() << error_msg << '\n';
    return kTrFailure;
  }
  output->os() << result;
  output->keep();

  // Print out debugging info related to function mapping.
  if (print_function_result_mapping)
    return PrintFunctionResultMapping(result, module.ValueOrDie().get());
  return kTrSuccess;
}
