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
class MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_graph_to_mlir_passDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_graph_to_mlir_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_graph_to_mlir_passDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/dtensor_graph_to_mlir_pass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {

DTensorMlirPassRunner::DTensorMlirPassRunner()
    : pass_manager_(&context_), logging_enabled_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_graph_to_mlir_passDTcc mht_0(mht_0_v, 220, "", "./tensorflow/dtensor/cc/dtensor_graph_to_mlir_pass.cc", "DTensorMlirPassRunner::DTensorMlirPassRunner");

  logging_enabled_ = dtensor::MaybeEnableLogging(&pass_manager_);
  if (logging_enabled_) pass_manager_.getContext()->enableMultithreading();

  // TODO(hinsu, hongjunchoi): Figure out a better place to explicitly enable
  // the MLIR bridge.
  // Explicitly enable MLIR bridge as DTensor introduces some ops like
  // XlaAllReduce are only supported in MLIR.
  GetMlirCommonFlags()->tf_mlir_enable_mlir_bridge =
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;

  // Creates a pipeline that include each DTensor related passes.
  mlir::TF::StandardPipelineOptions pipeline_options;
  dtensor::CreateDTensorMLIRPass(pipeline_options, &pass_manager_);
}

Status DTensorMlirPassRunner::RunOnGraph(
    const DeviceSet& device_set, bool is_func,
    FunctionLibraryDefinition* flib_def, std::unique_ptr<Graph>* graph,
    absl::flat_hash_set<Node*>& control_ret_nodes, Fprint128 cache_key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_graph_to_mlir_passDTcc mht_1(mht_1_v, 242, "", "./tensorflow/dtensor/cc/dtensor_graph_to_mlir_pass.cc", "DTensorMlirPassRunner::RunOnGraph");

  Graph* input_graph = graph->get();
  GraphDebugInfo debug_info;
  GraphImportConfig import_config;
  import_config.graph_as_function = true;
  // DTensor relies on importing with shape_inference to work properly ATM.
  // Make it explicit so that we're not affected by potential flipping of the
  // flag.
  import_config.enable_shape_inference = true;
  // Graph pruning will prune away an op (may be side effecting) if the op is
  // not reachable from a fetch/result or target/control ret. With how the entry
  // function/Graph is created, it is possible if the op has no data results. To
  // make sure this op does not get pruned away, the op is defined as a
  // target/control ret.
  import_config.control_outputs = {"eager_operation"};

  // Import GraphDef to TF MLIR.
  stream_executor::port::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
      module_ref = ConvertGraphToMlir(*input_graph, debug_info, *flib_def,
                                      import_config, &context_);
  if (!module_ref.ok())
    return errors::InvalidArgument(
        absl::StrCat(
            "Can not convert the graph to MLIR, errors from MLIR converter : ",
            module_ref.status().error_message())
            .c_str());

  mlir::ModuleOp module = module_ref.ValueOrDie().get();

  AddDevicesToOp(module, &device_set);

  // Tag the module for logging or not depending on flag.
  if (!is_func && !dtensor::LogOpByOp())
    module->setAttr(dtensor::kDoNotLog, mlir::UnitAttr::get(&context_));

  // Set the cache key for the module as an attribute. This attribute will be
  // used to rename all private functions in the module (by appending the
  // cache key) so they have unique names.
  module->setAttr(
      dtensor::kCacheKey,
      mlir::StringAttr::get(&context_, absl::StrCat("_", cache_key.low64, "_",
                                                    cache_key.high64)));

  // Executes and collects results from the passes.
  mlir::StatusScopedDiagnosticHandler diag_handler(&context_);

  if (logging_enabled_ && !module->hasAttr(dtensor::kDoNotLog))
    pass_manager_.getContext()->disableMultithreading();
  mlir::LogicalResult result = pass_manager_.run(module);
  (void)result;
  TF_RETURN_IF_ERROR(diag_handler.ConsumeStatus());

  if (logging_enabled_) pass_manager_.getContext()->enableMultithreading();

  // Convert MLIR to graphdef for execution.
  GraphExportConfig export_config;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertMlirToGraph(module, export_config, graph, flib_def,
                         &control_ret_nodes),
      "Error converting MLIR module back to graph");
  Graph* output_graph = graph->get();
  VLOG(4) << DumpGraphToFile("dtensor_mlir_pass_after", *output_graph,
                             flib_def);
  return Status::OK();
}

}  // namespace tensorflow
