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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_tfjs_translateDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_tfjs_translateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_tfjs_translateDTcc() {
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

#include <iostream>
#include <string>

#include "absl/strings/str_split.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/tfjs/tf_tfjs_passes.h"
#include "tensorflow/compiler/mlir/tfjs/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfjs/translate/tf_to_tfjs_json.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

using llvm::cl::opt;
using mlir::MLIRContext;
using stream_executor::port::StatusOr;

// NOLINTNEXTLINE
opt<std::string> input_file_name(llvm::cl::Positional,
                                 llvm::cl::desc("<input file>"),
                                 llvm::cl::init("-"));

// NOLINTNEXTLINE
opt<bool> import_saved_model_object_graph(
    "savedmodel-objectgraph-to-mlir",
    llvm::cl::desc("Import a saved model to its MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
opt<bool> import_saved_model_signature_defs(
    "savedmodel-signaturedefs-to-mlir",
    llvm::cl::desc("Import a saved model V1 to its MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
opt<std::string> saved_model_tags(
    "tf-savedmodel-tags",
    llvm::cl::desc("Tags used to indicate which MetaGraphDef to import, "
                   "separated by ','"),
    llvm::cl::init("serve"));

// NOLINTNEXTLINE
opt<std::string> saved_model_exported_names(
    "tf-savedmodel-exported-names",
    llvm::cl::desc("Names to export from SavedModel, separated by ','. Empty "
                   "(the default) means export all."),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> output_file_name("o", llvm::cl::desc("<output file>"),
                                  llvm::cl::value_desc("filename"),
                                  llvm::cl::init("-"));
// NOLINTNEXTLINE
opt<bool> input_mlir(
    "input-mlir",
    llvm::cl::desc("Take input TensorFlow model in textual MLIR instead of "
                   "GraphDef format"),
    llvm::cl::init(false), llvm::cl::Hidden);
// NOLINTNEXTLINE
opt<bool> output_mlir(
    "output-mlir",
    llvm::cl::desc("Output MLIR rather than JSON for the generated TFJS model"),
    llvm::cl::init(false));

// The following approach allows injecting opdefs in addition
// to those that are already part of the global TF registry  to be linked in
// prior to importing the graph. The primary goal is for support of custom ops.
// This is not intended to be a general solution for custom ops for the future
// but mainly for supporting older models like mobilenet_ssd. More appropriate
// mechanisms, such as op hints or using functions to represent composable ops
// like https://github.com/tensorflow/community/pull/113 should be encouraged
// going forward.
// NOLINTNEXTLINE
llvm::cl::list<std::string> custom_opdefs(
    "tf-custom-opdefs", llvm::cl::desc("List of custom opdefs when importing "
                                       "graphdef"));

// Debugging flag to print function mapping in the JSON.
// NOLINTNEXTLINE
static opt<bool> print_function_result_mapping(
    "print-function-result-mapping",
    llvm::cl::desc(
        "Print the mapping of function result to json output buffer"),
    llvm::cl::init(false));

enum TranslationStatus { kTrSuccess, kTrFailure };

static int PrintFunctionResultMapping(const std::string& result) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("result: \"" + result + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_tfjs_translateDTcc mht_0(mht_0_v, 286, "", "./tensorflow/compiler/mlir/tfjs/translate/tf_tfjs_translate.cc", "PrintFunctionResultMapping");

  std::cout << result << std::endl;
  return kTrSuccess;
}

int main(int argc, char** argv) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_tfjs_translateDTcc mht_1(mht_1_v, 294, "", "./tensorflow/compiler/mlir/tfjs/translate/tf_tfjs_translate.cc", "main");

  tensorflow::InitMlir y(&argc, &argv);

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TF GraphDef to TFJS JSON converter\n");

  MLIRContext context;
  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(source_mgr, &context);

  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module;

  if (import_saved_model_object_graph || import_saved_model_signature_defs) {
    if (input_mlir)
      module = tensorflow::errors::InvalidArgument(
          "Importing saved model should not have input_mlir set");
    module = tensorflow::ImportSavedModel(
        import_saved_model_object_graph, import_saved_model_signature_defs,
        custom_opdefs, input_file_name, saved_model_tags,
        saved_model_exported_names, &context);
  } else {
    module = tensorflow::LoadFromGraphdefOrMlirSource(
        input_file_name, input_mlir, custom_opdefs, debug_info_file,
        input_arrays, input_dtypes, input_shapes, output_arrays,
        /*prune_unused_nodes=*/true, &source_mgr, &context);
  }

  // If errors occur, the library call in the above already logged the error
  // message. So we can just return here.
  if (!module.ok()) return kTrFailure;

  mlir::PassManager pm(&context);

  tensorflow::AddTFToTFJSConversionPasses(&pm);

  std::string result;
  auto status = tensorflow::ConvertTFOpsToTfjsJSON(module.ValueOrDie().get(),
                                                   output_mlir, &result, &pm);
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
  if (print_function_result_mapping) return PrintFunctionResultMapping(result);
  return kTrSuccess;
}
