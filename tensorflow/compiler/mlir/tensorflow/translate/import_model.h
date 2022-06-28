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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_MODEL_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSimport_modelDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSimport_modelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSimport_modelDTh() {
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


#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

ABSL_CONST_INIT extern const char kImportModelDefaultGraphFuncName[];

// Given a GraphDef, returns a MLIR module containing the graph, expressed with
// tf_executor dialect.
stream_executor::port::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConvertGraphdefToMlir(const GraphDef& graphdef,
                      const GraphDebugInfo& debug_info,
                      const GraphImportConfig& specs,
                      mlir::MLIRContext* context,
                      bool add_default_attributes = true);

// Given a Graph, returns a MLIR module containing the graph, expressed with
// tf_executor dialect.
stream_executor::port::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConvertGraphToMlir(const Graph& graph, const GraphDebugInfo& debug_info,
                   const FunctionLibraryDefinition& flib_def,
                   const GraphImportConfig& specs, mlir::MLIRContext* context);

// [Experimental]
// Given a Function, returns a MLIR module containing the graph, expressed with
// tf_executor dialect.
stream_executor::port::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConvertFunctionToMlir(const FunctionBody* fbody,
                      const FunctionLibraryDefinition& flib_def,
                      mlir::MLIRContext* context);

// Given a SavedModel, returns a MLIR module containing the functions, expressed
// with tf_executor dialect.
stream_executor::port::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConvertSavedModelToMlir(SavedModelV2Bundle* saved_model,
                        mlir::MLIRContext* context,
                        absl::Span<std::string> exported_names,
                        bool add_default_attributes = true,
                        bool unconditionally_use_set_output_shapes = false);

// Given a V1 SavedModel, returns a MLIR module containing the functions,
// expressed with tf_executor dialect.
stream_executor::port::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConvertSavedModelV1ToMlir(const SavedModelBundle& saved_model,
                          absl::Span<std::string> exported_names,
                          mlir::MLIRContext* context, MLIRImportOptions options,
                          bool lift_variables = true);

// Given a V1 SavedModel, returns a MLIR module containing the functions,
// expressed with tf_executor dialect. It does not require a session to be
// created and it does not perform any graph transformation. If `exported_names`
// is absl::nullopt, all signatures will be imported. Otherwise, only names
// in `exported_names` are imported.
//
// Note that the word `Lite` means it is a lighter version compared to
// ConvertSavedModelV1ToMlir(), and is not related to TFLite.
//
// TODO(b/179683149): Rename this class to avoid confusion with TFLite.
stream_executor::port::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConvertSavedModelV1ToMlirLite(
    const MetaGraphDef& meta_graph_def, const GraphDebugInfo& debug_info,
    absl::optional<absl::Span<const std::string>> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions options);

// SavedModelMLIRImportInput is an adapter class for users to inject custom
// graph transformation logic on Tensorflow graphs before importing to MLIR. It
// serves as the source that provides the subgraphs requested by the savedmodel
// MLIR importer, and at the same time it allows the implementation of this
// class to transform the graph before feeding it to the importer.
class SavedModelMLIRImportInput {
 public:
  SavedModelMLIRImportInput(const MetaGraphDef* meta_graph_def,
                            const GraphDebugInfo& debug_info)
      : meta_graph_def_(meta_graph_def), debug_info_(debug_info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSimport_modelDTh mht_0(mht_0_v, 275, "", "./tensorflow/compiler/mlir/tensorflow/translate/import_model.h", "SavedModelMLIRImportInput");

    DCHECK(meta_graph_def);
  }

  virtual ~SavedModelMLIRImportInput();

  // The original MetaGraphDef of the savedmodel.
  const MetaGraphDef& meta_graph_def() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSimport_modelDTh mht_1(mht_1_v, 285, "", "./tensorflow/compiler/mlir/tensorflow/translate/import_model.h", "meta_graph_def");
 return *meta_graph_def_; }

  const GraphDebugInfo& debug_info() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSimport_modelDTh mht_2(mht_2_v, 290, "", "./tensorflow/compiler/mlir/tensorflow/translate/import_model.h", "debug_info");
 return debug_info_; }

  // GetSubGraph() is expected to return a tensorflow::Graph that contains the
  // node set specified in `specs`. The implementation is free to transform the
  // graph in the original savedmodel as needed, as long as it produces the same
  // results and effects. If the transformation requires some configs in `spec`
  // (e.g., control_outputs) to be changed, they should be updated accordingly
  // and remain valid for the graph.
  // `name` is a unique identifier for this subgraph, so the implementation can
  // use it for eg. debugging or caching compilation results.
  virtual stream_executor::port::StatusOr<const Graph*> GetSubGraph(
      absl::string_view name, GraphImportConfig& specs) = 0;

 private:
  const MetaGraphDef* meta_graph_def_ = nullptr;
  GraphDebugInfo debug_info_;
};

// Given the SavedModelMLIRImportInput for a saved model, returns a MLIR module
// containing the functions, expressed with tf_executor dialect. It does not
// require a session to be created. If `exported_names` is absl::nullopt, all
// signatures will be imported. Otherwise, only names in `exported_names` are
// imported.

//
// Note that the word `Lite` means it is a lighter version compared to
// ConvertSavedModelV1ToMlir(), and is not related to TFLite.
//
// TODO(b/179683149): Rename this class to avoid confusion with TFLite.
stream_executor::port::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConvertSavedModelV1ToMlirLite(
    SavedModelMLIRImportInput& input,
    absl::optional<absl::Span<const std::string>> exported_names,
    mlir::MLIRContext* context,
    bool unconditionally_use_set_output_shapes = false);

// Serialize a MLIR module to a string.
std::string MlirModuleToString(mlir::ModuleOp module,
                               mlir::OpPrintingFlags flags);
std::string MlirModuleToString(mlir::ModuleOp m, bool show_debug_info = false);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_MODEL_H_
