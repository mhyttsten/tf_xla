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
class MHTracer_DTPStensorflowPScompilerPSmlirPStf_mlir_translate_mainDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStf_mlir_translate_mainDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStf_mlir_translate_mainDTcc() {
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

#include <unordered_set>

#include "absl/strings/str_split.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/ToolUtilities.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/core/platform/init_main.h"

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> output_filename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each chunk "
                   "independently"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> import_saved_model_object_graph(
    "savedmodel-objectgraph-to-mlir",
    llvm::cl::desc(
        "Import a saved model's object graph to its MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> import_saved_model_signature_defs(
    "savedmodel-signaturedefs-to-mlir",
    llvm::cl::desc(
        "Import a saved model's SignatureDefs to their MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> import_saved_model_signature_defs_lite(
    "savedmodel-signaturedefs-to-mlir-lite",
    llvm::cl::desc("Import a saved model's SignatureDefs to to their MLIR "
                   "representation without any graph transformation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> saved_model_tags(
    "tf-savedmodel-tags",
    llvm::cl::desc("Tags used to indicate which MetaGraphDef to import, "
                   "separated by ','"),
    llvm::cl::init("serve"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> saved_model_exported_names(
    "tf-savedmodel-exported-names",
    llvm::cl::desc("Names to export from SavedModel, separated by ','. Empty "
                   "(the default) means export all."),
    llvm::cl::init(""));

int main(int argc, char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStf_mlir_translate_mainDTcc mht_0(mht_0_v, 256, "", "./tensorflow/compiler/mlir/tf_mlir_translate_main.cc", "main");

  tensorflow::InitMlir y(&argc, &argv);

  // Add flags for all the registered translations.
  llvm::cl::opt<const mlir::TranslateFunction*, false, mlir::TranslationParser>
      requested_translation("", llvm::cl::desc("Translation to perform"));
  mlir::registerAsmPrinterCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "TF MLIR translation driver\n");

  if (!import_saved_model_object_graph && !import_saved_model_signature_defs &&
      !import_saved_model_signature_defs_lite && !requested_translation) {
    llvm::errs() << "error: need to specify one translation to perform\n";
    return 1;
  } else if (import_saved_model_object_graph +
                 import_saved_model_signature_defs +
                 import_saved_model_signature_defs_lite +
                 (requested_translation != nullptr) >
             1) {
    llvm::errs()
        << "error: cannot specify more than one translation to perform\n";
    return 1;
  }

  std::string error_message;
  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!output) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  std::unordered_set<std::string> tags = absl::StrSplit(saved_model_tags, ',');
  std::vector<std::string> exported_names_vector =
      absl::StrSplit(saved_model_exported_names, ',', absl::SkipEmpty());
  absl::Span<std::string> exported_names(exported_names_vector);

  if (import_saved_model_object_graph) {
    mlir::MLIRContext context;
    auto module_or = tensorflow::SavedModelObjectGraphToMlirImport(
        input_filename, tags, exported_names, &context);
    if (!module_or.status().ok()) return 1;

    module_or.ConsumeValueOrDie()->print(output->os());
  } else if (import_saved_model_signature_defs) {
    mlir::MLIRContext context;
    tensorflow::MLIRImportOptions import_options;
    import_options.upgrade_legacy = upgrade_legacy;
    auto module_or = tensorflow::SavedModelSignatureDefsToMlirImport(
        input_filename, tags, exported_names, &context, import_options);
    if (!module_or.status().ok()) return 1;

    module_or.ConsumeValueOrDie()->print(output->os());
  } else if (import_saved_model_signature_defs_lite) {
    mlir::MLIRContext context;
    tensorflow::MLIRImportOptions import_options;
    import_options.upgrade_legacy = upgrade_legacy;
    auto module_or = tensorflow::SavedModelSignatureDefsToMlirImportLite(
        input_filename, tags, exported_names, &context, import_options);
    if (!module_or.status().ok()) return 1;

    module_or.ConsumeValueOrDie()->print(output->os());
  } else {
    auto input = mlir::openInputFile(input_filename, &error_message);

    if (!input) {
      llvm::errs() << error_message << "\n";
      return 1;
    }

    // Processes the memory buffer with a new MLIRContext.
    auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                             llvm::raw_ostream& os) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStf_mlir_translate_mainDTcc mht_1(mht_1_v, 329, "", "./tensorflow/compiler/mlir/tf_mlir_translate_main.cc", "lambda");

      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
      mlir::MLIRContext context;
      mlir::SourceMgrDiagnosticHandler diagnostic_handler(sourceMgr, &context);
      return (*requested_translation)(sourceMgr, os, &context);
    };

    if (splitInputFile) {
      if (failed(mlir::splitAndProcessBuffer(std::move(input), processBuffer,
                                             output->os())))
        return 1;
    } else {
      if (failed(processBuffer(std::move(input), output->os()))) return 1;
    }
  }

  output->keep();
  return 0;
}
