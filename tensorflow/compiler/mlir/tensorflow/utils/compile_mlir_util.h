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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_COMPILE_MLIR_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_COMPILE_MLIR_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPScompile_mlir_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPScompile_mlir_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPScompile_mlir_utilDTh() {
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


#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// Populates the supplied passmanager with the passes required to run the
// TF MLIR to XLA HLO MLIR conversion/legalization. Custom legalization passes
// can be populated in `custom_legalization_passes`.
void CreateConvertMlirToXlaHloPipeline(
    mlir::OpPassManager& pm, llvm::StringRef device_type, bool prefer_tf2xla,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes);

// Lowers MLIR module to XLA HLO inside an XlaComputation. The input module
// should only contain operations in tf dialect. If the input module contains
// operation in the tf_executor dialect, for example, returns an error.
// Exception to this are tf_executor dialect ops that are optimized away through
// canonicalization.
//
// Operations in tf dialect are lowered to XLA HLO through the following steps:
//   . Legalizes control flow operations.
//   . Decomposes compound resource operations so that the only remaining
//     operations on resource variables are resource reads/writes..
//   . Replaces resource reads/writes with function inputs/outputs and
//     eliminates the use of resource variables.
//   . Legalizes the operations to XLA HLO operations.
//   . Canonicalizes the XLA HLO operations.
//
// device_type: XLA JIT device to use for compilation such as "XLA_CPU_JIT",
//   "XLA_GPU_JIT" or "XLA_TPU_JIT".
// use_tuple_args: when this is true, always create a tuple argument for the
//   entry computation.
// prefer_tf2xla: when this is true, prefer tf2xla fallback kernels over MLIR
//   native kernels for legalization to HLO.
// return_tuple: when this is true, always create a tuple result for the
//   entry computation.
// shape_representation_fn: when this is set, this shape representation function
//   will be used to determine argument and result shapes. Otherwise the
//   original shape will be used as is.
// custom_legalization_passes: passes to run before the default TF legalization
//   passes for backend-specific ops.
//
// TODO(hinsu): Migrate options to a separate struct.
Status ConvertMLIRToXlaComputation(
    mlir::ModuleOp module_op, llvm::StringRef device_type,
    xla::XlaComputation* xla_computation, bool use_tuple_args,
    bool prefer_tf2xla, bool return_tuple,
    const XlaHelpers::ShapeRepresentationFn shape_representation_fn = nullptr,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {});

// Helper struct representing argument tensor or resource handle shapes.
struct TensorOrResourceShape {
  TensorShape shape;
  bool is_resource = false;
};

// Refine MLIR types based on new shape information.
Status RefineShapes(llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
                    mlir::ModuleOp module);

// Lower TF to MHLO and insert HLO into the XlaBuilder. xla_params are HLO-level
// inputs to module_op that have already been added to the XlaBuilder. returns
// are the returned XlaOps.
Status BuildHloFromTf(mlir::ModuleOp module_op, xla::XlaBuilder& builder,
                      llvm::ArrayRef<xla::XlaOp> xla_params,
                      std::vector<xla::XlaOp>& returns,
                      llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
                      llvm::StringRef device_type,
                      llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
                          custom_legalization_passes);

// Apply shape, description, and resource information to inputs and outputs
// in the XlaCompilationResult. This should be called after
// compilation_result->computation was set.
Status PopulateResultIOInfo(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    bool use_tuple_args, bool use_resource_updates_for_aliases,
    XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    XlaCompilationResult* compilation_result);

// Compiles a MLIR module into XLA HLO, generates all accompanying metadata and
// stores them in CompilationResult.
//
// If analyse_graph is set to true, graph is legalized only if the graph
// analysis for the graph is successful. Otherwise, an error is returned.
Status CompileMlirToXlaHlo(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args, bool analyse_graph,
    bool use_return_tuple, bool use_resource_updates_for_aliases,
    XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes);

// Compiles a serialized MLIR module into XLA HLO, generates all accompanying
// metadata and stores them in CompilationResult.
Status CompileSerializedMlirToXlaHlo(
    llvm::StringRef mlir_module_string, llvm::ArrayRef<TensorShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args, bool analyse_graph,
    const XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {});

// Compiles a TensorFlow Graph (already converted to MLIR, imported with
// tf_executor dialect still present) into XLA HLO, generates all accompanying
// metadata and stores them in CompilationResult. This will rewrite arguments
// and run the TensorFlow standard pipeline prior to invoking
// `CompileMlirToXlaHlo`.
Status CompileGraphToXlaHlo(
    mlir::ModuleOp module_op, llvm::ArrayRef<XlaArgument> args,
    llvm::StringRef device_type, bool use_tuple_args, bool analyse_graph,
    bool use_return_tuple,
    const XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes);

// Compiles a TensorFlow Graph into XLA HLO, generates all accompanying metadata
// and stores them in CompilationResult.
Status CompileGraphToXlaHlo(
    const Graph& graph, llvm::ArrayRef<XlaArgument> args,
    llvm::ArrayRef<std::string> control_rets, llvm::StringRef device_type,
    bool use_tuple_args, bool analyse_graph,
    const FunctionLibraryDefinition& flib_def, const GraphDebugInfo& debug_info,
    const XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {});

// Compiles a Graph from TF to HLO and adds the resulting HLO to the
// XlaBuilder. This function adds HLO to a larger HLO computation, so
// HLO-level inputs are supplied, and HLO-level outputs are produced.
// xla_params is the HLO-level inputs and returns is the HLO-level outputs.
Status BuildHloFromGraph(
    const Graph& graph, xla::XlaBuilder& builder,
    mlir::MLIRContext& mlir_context, llvm::ArrayRef<xla::XlaOp> xla_params,
    std::vector<xla::XlaOp>& returns, llvm::ArrayRef<XlaArgument> args,
    llvm::ArrayRef<std::string> control_rets, llvm::StringRef device_type,
    const FunctionLibraryDefinition& flib_def, const GraphDebugInfo& debug_info,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {});

static inline Status CompileToHloGraphAnalysisFailedError() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPScompile_mlir_utilDTh mht_0(mht_0_v, 345, "", "./tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h", "CompileToHloGraphAnalysisFailedError");

  return errors::Internal("disabled after graph analysis");
}

// Register a convenient pipeline for invoking TF/XLA lowering from the command
// line.
void RegisterConvertMlirToXlaHloPipelineWithDefaults();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_COMPILE_MLIR_UTIL_H_
