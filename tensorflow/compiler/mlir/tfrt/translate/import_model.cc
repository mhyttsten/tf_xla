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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStranslatePSimport_modelDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStranslatePSimport_modelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStranslatePSimport_modelDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"

#include "absl/strings/match.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime

namespace tensorflow {

Status ConvertFunctionToBef(
    mlir::StringRef function_name, const tensorflow::FunctionBody* fbody,
    const FunctionLibraryDefinition& flib_def,
    tfrt::ArrayRef<tfrt::string_view> devices,
    const tensorflow::TfrtFunctionCompileOptions& options,
    tfrt::BefBuffer* bef_buffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStranslatePSimport_modelDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/mlir/tfrt/translate/import_model.cc", "ConvertFunctionToBef");

  mlir::MLIRContext context;
  // FunctionDef -> TF Dialect
  auto expected_module =
      tensorflow::ConvertFunctionToMlir(fbody, flib_def, &context);

  if (!expected_module.ok())
    return tensorflow::errors::Internal(
        "Failed to convert function to mlir for function ", function_name.str(),
        ". Error: ", expected_module.status().error_message());

  auto module = expected_module.ConsumeValueOrDie();

  // Attach devices to the MLIR module.
  if (!devices.empty()) {
    mlir::Builder builder(module->getContext());
    module->getOperation()->setAttr("tf.devices",
                                    builder.getStrArrayAttr(devices));
  }

  // TF Dialect -> BEF
  return tensorflow::CompileTFMLIRToBEF(options, module.get(), bef_buffer);
}

Status ConvertTfMlirToBef(const TfrtCompileOptions& options,
                          mlir::ModuleOp module, tfrt::BefBuffer* bef_buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStranslatePSimport_modelDTcc mht_1(mht_1_v, 235, "", "./tensorflow/compiler/mlir/tfrt/translate/import_model.cc", "ConvertTfMlirToBef");

  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());

  if (options.tpu_target == TfrtTpuInfraTarget::kTpurt) {
    VLOG(1) << "Running MLIR TPU bridge for tpurt";
    if (VLOG_IS_ON(1)) {
      tensorflow::DumpMlirOpToFile("tpu_bct_conversion_before", module);
    }

    TfrtTpuCompileOptions tpu_compile_options;
    tpu_compile_options.move_resource_gather_to_host =
        options.tpu_move_resource_gather_to_host;
    tpu_compile_options.gather_table_width_threshold_bytes =
        options.tpu_gather_table_width_threshold_bytes;

    auto backward_compat_result =
        tensorflow::RunTPUBackwardCompatConversion(module, tpu_compile_options);
    if (mlir::failed(backward_compat_result)) {
      return diag_handler.Combine(
          tensorflow::errors::Internal("Failed to handle legacy TPU Ops"));
    }

    if (VLOG_IS_ON(1)) {
      tensorflow::DumpMlirOpToFile("tpu_bct_conversion_after", module);
    }

    TF_RETURN_IF_ERROR(
        mlir::TFTPU::TPUBridge(module, /*enable_logging=*/VLOG_IS_ON(1)));
  } else if (options.tpu_target == TfrtTpuInfraTarget::kTfFallback) {
    auto tpu_partitioned_call_fallback_compat_result =
        tensorflow::RunTPUPartitionedCallFallbackCompatConversion(module);
    if (mlir::failed(tpu_partitioned_call_fallback_compat_result)) {
      return diag_handler.Combine(tensorflow::errors::Internal(
          "Failed to process TPUPartitionedCallOp for fallback execution"));
    }
  }

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tf_dialect", module);
  }

  // Lower MLIR TF Dialect to MLIR TFRT CoreRT dialect.
  mlir::PassManager pm(module.getContext());

  tensorflow::TfrtPipelineOptions pass_options;
  if (!options.default_device.empty()) {
    pass_options.default_device = options.default_device;
  }
  if (!options.force_data_format.empty()) {
    pass_options.force_data_format = options.force_data_format;
  }

  // TODO(b/187991150): Consider only decomposing read-only resource variable
  // ops.
  pass_options.decompose_resource_ops = true;
  pass_options.enable_optimizer = options.enable_optimizer;
  pass_options.enable_native_ops = options.enable_native_ops;
  pass_options.target_tpurt =
      (options.tpu_target == TfrtTpuInfraTarget::kTpurt);
  pass_options.tpu_fuse_ops = options.tpu_fuse_ops;
  pass_options.use_tpu_host_allocator_for_inputs =
      options.use_tpu_host_allocator_for_inputs;
  pass_options.hoist_invariant_ops = options.hoist_invariant_ops;
  pass_options.func_use_fallback_tensor = true;
  pass_options.enable_while_parallel_iterations =
      options.enable_while_parallel_iterations;
  pass_options.auto_fusion_oplist = options.auto_fusion_oplist;
  pass_options.auto_fusion_min_cluster_size =
      options.auto_fusion_min_cluster_size;
  pass_options.cost_threshold = options.cost_threshold;
  pass_options.upper_cost_threshold = options.upper_cost_threshold;
  pass_options.merge_inter_dependent_streams =
      options.merge_inter_dependent_streams;
  tensorflow::CreateTfExecutorToTfrtPipeline(pm, pass_options);

  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(tensorflow::errors::Internal(
        "failed to lower TF Dialect to CoreRT dialect."));

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tfrt_dialect", module);
  }

  *bef_buffer =
      tfrt::ConvertMLIRToBEF(module, /*disable_optional_sections=*/true);
  if (bef_buffer->empty())
    return diag_handler.Combine(
        tensorflow::errors::Internal("failed to convert MLIR to BEF."));

  bef_buffer->shrink_to_fit();

  return Status::OK();
}

}  // namespace tensorflow
