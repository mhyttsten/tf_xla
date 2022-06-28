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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_to_tfjs_jsonDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_to_tfjs_jsonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_to_tfjs_jsonDTcc() {
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

#include "tensorflow/compiler/mlir/tfjs/translate/tf_to_tfjs_json.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfjs/translate/json_translate.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OwningOpRef;
using stream_executor::port::StatusOr;

namespace {
tensorflow::Status RegisterCustomOps(
    const std::vector<std::string>& extra_tf_opdefs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_to_tfjs_jsonDTcc mht_0(mht_0_v, 223, "", "./tensorflow/compiler/mlir/tfjs/translate/tf_to_tfjs_json.cc", "RegisterCustomOps");

  for (const auto& tf_opdefs_string : extra_tf_opdefs) {
    tensorflow::OpDef opdef;
    if (!tensorflow::protobuf::TextFormat::ParseFromString(tf_opdefs_string,
                                                           &opdef)) {
      LOG(ERROR) << "OpDef parsing failed for: " << tf_opdefs_string;
      return errors::InvalidArgument("fail to parse extra OpDef");
    }
    // Register extra opdefs.
    tensorflow::OpRegistry::Global()->Register(
        [opdef](tensorflow::OpRegistrationData* op_reg_data) -> Status {
          *op_reg_data = tensorflow::OpRegistrationData(opdef);
          return Status::OK();
        });
  }
  return Status::OK();
}
}  // namespace

StatusOr<OwningOpRef<ModuleOp>> LoadFromGraphdefOrMlirSource(
    const std::string& input_filename, bool input_mlir,
    const std::vector<std::string>& extra_tf_opdefs,
    absl::string_view debug_info_file, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, bool prune_unused_nodes,
    llvm::SourceMgr* source_mgr, MLIRContext* context) {
  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  if (!file) {
    llvm::errs() << error_message << "\n";
    return errors::InvalidArgument("fail to open input file");
  }

  if (input_mlir) {
    source_mgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    return OwningOpRef<ModuleOp>(
        mlir::parseSourceFile<mlir::ModuleOp>(*source_mgr, context));
  }

  TF_RETURN_IF_ERROR(RegisterCustomOps(extra_tf_opdefs));

  return tensorflow::GraphdefToMlirTranslateFunction(
      file->getBuffer(), debug_info_file, input_arrays, input_dtypes,
      input_shapes, output_arrays, /*control_output_arrays=*/"",
      prune_unused_nodes, /*convert_legacy_fed_inputs=*/true,
      /*graph_as_function=*/false, /*upgrade_legacy=*/true,
      /*enable_shape_inference=*/true,
      /*unconditionally_use_set_output_shapes=*/false, context);
}

Status ConvertTFOpsToTfjsJSON(mlir::ModuleOp module, bool export_to_mlir,
                              std::string* result,
                              mlir::PassManager* pass_manager) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfjsPStranslatePStf_to_tfjs_jsonDTcc mht_1(mht_1_v, 279, "", "./tensorflow/compiler/mlir/tfjs/translate/tf_to_tfjs_json.cc", "ConvertTFOpsToTfjsJSON");

  mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext(),
                                                    /*propagate=*/true);
  if (failed(pass_manager->run(module))) {
    return statusHandler.ConsumeStatus();
  }

  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return Status::OK();
  }

  return tfjs::MlirToJSONTranslateFunction(module, result)
             ? Status::OK()
             : statusHandler.ConsumeStatus();
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    bool import_saved_model, bool import_saved_model_v1,
    const std::vector<std::string>& extra_tf_opdefs,
    const std::string& input_filename, const std::string& saved_model_tags,
    const std::string& saved_model_exported_names, mlir::MLIRContext* context) {
  std::unordered_set<std::string> tags = absl::StrSplit(saved_model_tags, ',');
  std::vector<std::string> exported_names_in_vector =
      absl::StrSplit(saved_model_exported_names, ',', absl::SkipEmpty());
  absl::Span<std::string> exported_names(exported_names_in_vector);
  if (import_saved_model) {
    auto module_or = tensorflow::SavedModelObjectGraphToMlirImport(
        input_filename, tags, absl::Span<std::string>(exported_names), context);
    if (!module_or.status().ok()) return module_or.status();
    TF_RETURN_IF_ERROR(RegisterCustomOps(extra_tf_opdefs));
    return module_or.ConsumeValueOrDie();
  } else if (import_saved_model_v1) {
    tensorflow::MLIRImportOptions import_options;
    auto module_or = tensorflow::SavedModelSignatureDefsToMlirImport(
        input_filename, tags, exported_names, context, import_options);

    if (!module_or.status().ok()) return module_or.status();
    TF_RETURN_IF_ERROR(RegisterCustomOps(extra_tf_opdefs));
    return module_or.ConsumeValueOrDie();
  } else {
    return tensorflow::errors::InvalidArgument(
        "Should be either saved model v1 or v2");
  }
}

}  // namespace tensorflow
