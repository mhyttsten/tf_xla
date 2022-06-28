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
class MHTracer_DTPStensorflowPStoolsPStfg_graph_transformsPStfg_graph_transforms_mainDTcc {
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
   MHTracer_DTPStensorflowPStoolsPStfg_graph_transformsPStfg_graph_transforms_mainDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPStfg_graph_transformsPStfg_graph_transforms_mainDTcc() {
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

#include <string>
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/ir/importexport/import.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/transforms/pass_registration.h"
#include "tensorflow/tools/tfg_graph_transforms/utils.h"

namespace {

llvm::cl::OptionCategory tfg_graph_transform_category(
    "TFG graph transform options");

// NOLINTNEXTLINE
llvm::cl::opt<std::string> input_file(
    llvm::cl::Positional, llvm::cl::desc("<Input model>"),
    llvm::cl::value_desc("Full path to the input model"),
    llvm::cl::cat(tfg_graph_transform_category), llvm::cl::Required);

// NOLINTNEXTLINE
llvm::cl::opt<std::string> output_file(
    "o", llvm::cl::desc("Output model"),
    llvm::cl::value_desc("Full path to the output model"),
    llvm::cl::cat(tfg_graph_transform_category), llvm::cl::Required);

enum class DataFormat { SavedModel = 0, GraphDef = 1 };

// NOLINTNEXTLINE
llvm::cl::opt<DataFormat> data_format(
    "data_format",
    llvm::cl::desc(
        "Data format for both input and output, e.g., SavedModel or GraphDef"),
    values(clEnumValN(DataFormat::SavedModel, "savedmodel",
                      "SavedModel format"),
           clEnumValN(DataFormat::GraphDef, "graphdef", "GraphDef format")),
    llvm::cl::init(DataFormat::SavedModel),
    llvm::cl::cat(tfg_graph_transform_category));

// Validate CL options and returns false in case of an error.
bool CheckCLParams() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPStfg_graph_transformsPStfg_graph_transforms_mainDTcc mht_0(mht_0_v, 238, "", "./tensorflow/tools/tfg_graph_transforms/tfg_graph_transforms_main.cc", "CheckCLParams");

  if (input_file == output_file) {
    LOG(WARNING)
        << "Input and output files are set to the same location. "
           "The resulted model protobuf will overwrite the original one.\n";
  }
  if (!tensorflow::Env::Default()->FileExists(input_file).ok()) {
    LOG(ERROR) << "Provided file or directory does not exist: '" << input_file
               << "'\n";
    return false;
  }

  if (tensorflow::Env::Default()->IsDirectory(input_file).ok()) {
    LOG(ERROR)
        << "Expected full path to the model protobuf file, given directory: '"
        << input_file << "'\n";
    return false;
  }

  return true;
}

void RegisterDialects(mlir::DialectRegistry& registry) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPStfg_graph_transformsPStfg_graph_transforms_mainDTcc mht_1(mht_1_v, 263, "", "./tensorflow/tools/tfg_graph_transforms/tfg_graph_transforms_main.cc", "RegisterDialects");

  // This potentially could be limited, for now keep all TF.
  mlir::RegisterAllTensorFlowDialects(registry);
}

tensorflow::Status RunOptimizationPasses(
    const mlir::PassPipelineCLParser& passPipeline, mlir::ModuleOp module,
    mlir::MLIRContext* context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPStfg_graph_transformsPStfg_graph_transforms_mainDTcc mht_2(mht_2_v, 273, "", "./tensorflow/tools/tfg_graph_transforms/tfg_graph_transforms_main.cc", "RunOptimizationPasses");

  mlir::PassManager pm(context);
  mlir::applyPassManagerCLOptions(pm);

  auto error_handler = [&](const llvm::Twine& msg) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPStfg_graph_transformsPStfg_graph_transforms_mainDTcc mht_3(mht_3_v, 280, "", "./tensorflow/tools/tfg_graph_transforms/tfg_graph_transforms_main.cc", "lambda");

    emitError(mlir::UnknownLoc::get(pm.getContext())) << msg;
    return mlir::failure();
  };
  if (failed(passPipeline.addToPipeline(pm, error_handler))) {
    return tensorflow::errors::InvalidArgument(
        "Pipeline initialization failed");
  }

  mlir::StatusScopedDiagnosticHandler diagnostics_handler(context);
  if (failed(pm.run(module))) {
    return diagnostics_handler.Combine(
        tensorflow::errors::InvalidArgument("MLIR Pass Manager failure: "));
  }

  return diagnostics_handler.ConsumeStatus();
}

// Import model to the TFG MLIR module.
tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportModel(
    DataFormat data_format, const std::string& input_file,
    mlir::MLIRContext* mlir_context) {
  tensorflow::GraphDebugInfo debug_info;

  switch (data_format) {
    case DataFormat::SavedModel: {
      tensorflow::SavedModel saved_model;
      TF_RETURN_IF_ERROR(
          mlir::tfg::graph_transforms::ReadModelProto<tensorflow::SavedModel>(
              input_file, saved_model));
      return mlir::tfg::ImportSavedModelToMlir(mlir_context, debug_info,
                                               saved_model);
    }
    case DataFormat::GraphDef: {
      tensorflow::GraphDef graph_def;
      TF_RETURN_IF_ERROR(
          mlir::tfg::graph_transforms::ReadModelProto<tensorflow::GraphDef>(
              input_file, graph_def));
      return mlir::tfg::ImportGraphDefToMlir(mlir_context, debug_info,
                                             graph_def);
    }
  }
}

tensorflow::Status ExportTFGModule(mlir::ModuleOp module_op,
                                   DataFormat data_format,
                                   const std::string& input_file,
                                   const std::string& output_file) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("input_file: \"" + input_file + "\"");
   mht_4_v.push_back("output_file: \"" + output_file + "\"");
   MHTracer_DTPStensorflowPStoolsPStfg_graph_transformsPStfg_graph_transforms_mainDTcc mht_4(mht_4_v, 332, "", "./tensorflow/tools/tfg_graph_transforms/tfg_graph_transforms_main.cc", "ExportTFGModule");

  switch (data_format) {
    case DataFormat::SavedModel: {
      tensorflow::SavedModel original_saved_model;
      TF_RETURN_IF_ERROR(
          mlir::tfg::graph_transforms::ReadModelProto<tensorflow::SavedModel>(
              input_file, original_saved_model));
      tensorflow::SavedModel final_saved_model;

      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          tensorflow::ExportMlirToSavedModel(module_op, original_saved_model,
                                             &final_saved_model),
          "while converting TFG to SavedModel");

      VLOG(1) << "Serializing resulting SavedModel to " << output_file;
      return mlir::tfg::graph_transforms::SerializeProto<
          tensorflow::SavedModel>(final_saved_model, output_file);
    }
    case DataFormat::GraphDef: {
      tensorflow::GraphDef new_graphdef;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          tensorflow::ExportMlirToGraphdef(module_op, &new_graphdef),
          "while converting TFG to GraphDef");

      VLOG(1) << "Serializing resulting GraphDef to " << output_file;
      return mlir::tfg::graph_transforms::SerializeProto<tensorflow::GraphDef>(
          new_graphdef, output_file);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPStfg_graph_transformsPStfg_graph_transforms_mainDTcc mht_5(mht_5_v, 368, "", "./tensorflow/tools/tfg_graph_transforms/tfg_graph_transforms_main.cc", "main");

  tensorflow::InitMlir y(&argc, &argv);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::tfg::registerTFGraphPasses();
  mlir::registerSymbolPrivatizePass();
  mlir::registerSymbolDCEPass();

  mlir::PassPipelineCLParser pass_pipeline("", "TFG passes to run");
  llvm::cl::ParseCommandLineOptions(argc, argv, "TFG optimization tool\n");

  if (!CheckCLParams()) {
    LOG(QFATAL) << "Command line parameters are invalid";
  }

  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  mlir::MLIRContext context(registry);

  // Import model to the TFG MLIR module.
  auto module_ref_status = ImportModel(data_format, input_file, &context);

  if (!module_ref_status.ok()) {
    LOG(QFATAL) << "Model import failed: "
                << module_ref_status.status().ToString();
  }
  auto module_ref = std::move(module_ref_status.ValueOrDie());

  // Parse the optimization pipeline configuration and run requested graph
  // optimizations.
  tensorflow::Status pass_pipeline_status =
      RunOptimizationPasses(pass_pipeline, *module_ref, &context);
  if (!pass_pipeline_status.ok()) {
    LOG(QFATAL) << pass_pipeline_status.ToString() << "\n";
  }

  // Export MLIR TFG module to the resulting model proto.
  tensorflow::Status export_status =
      ExportTFGModule(*module_ref, data_format, input_file, output_file);

  if (!export_status.ok()) {
    LOG(QFATAL) << "Export of TFG module failed: " << export_status.ToString()
                << "\n";
  }

  return EXIT_SUCCESS;
}
