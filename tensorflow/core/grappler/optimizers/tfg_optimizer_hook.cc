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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc() {
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

#include "tensorflow/core/grappler/optimizers/tfg_optimizer_hook.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/ir/importexport/import.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/transforms/pass_registration.h"

using tensorflow::Status;
using tensorflow::errors::InvalidArgument;

namespace mlir {
namespace tfg {

// The default pipeline is empty.
void DefaultGrapplerPipeline(PassManager& mgr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/grappler/optimizers/tfg_optimizer_hook.cc", "DefaultGrapplerPipeline");
}

// The implementation of the TFG optimizer. It holds the MLIR context and the
// pass manager.
class TFGGrapplerOptimizer::Impl {
 public:
  // Builds the pass pipeline. The context is initialized without threading.
  // Creating and destroying the threadpool each time Grappler is invoked is
  // prohibitively expensive.
  // TODO(jeffniu): Some passes may run in parallel on functions. Find a way to
  // hold and re-use a threadpool.
  explicit Impl(TFGPassPipelineBuilder builder)
      : ctx_(MLIRContext::Threading::DISABLED), mgr_(&ctx_) {
    builder(mgr_);
  }

  // Runs the pass manager.
  LogicalResult RunPipeline(ModuleOp module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/grappler/optimizers/tfg_optimizer_hook.cc", "RunPipeline");
 return mgr_.run(module); }

  // Get the context.
  MLIRContext* GetContext() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/grappler/optimizers/tfg_optimizer_hook.cc", "GetContext");
 return &ctx_; }

  // Convert the pass pipeline to a textual string.
  std::string GetPipelineString() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc mht_3(mht_3_v, 250, "", "./tensorflow/core/grappler/optimizers/tfg_optimizer_hook.cc", "GetPipelineString");

    std::string pipeline;
    llvm::raw_string_ostream os(pipeline);
    mgr_.printAsTextualPipeline(os);
    return os.str();
  }

 private:
  // The MLIR context.
  MLIRContext ctx_;
  // The pass manager containing the loaded TFG pass pipeline.
  PassManager mgr_;
};

TFGGrapplerOptimizer::TFGGrapplerOptimizer(TFGPassPipelineBuilder builder)
    : impl_(std::make_unique<Impl>(std::move(builder))) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc mht_4(mht_4_v, 268, "", "./tensorflow/core/grappler/optimizers/tfg_optimizer_hook.cc", "TFGGrapplerOptimizer::TFGGrapplerOptimizer");
}

TFGGrapplerOptimizer::~TFGGrapplerOptimizer() = default;

std::string TFGGrapplerOptimizer::name() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/grappler/optimizers/tfg_optimizer_hook.cc", "TFGGrapplerOptimizer::name");

  return absl::StrCat("tfg_optimizer{", impl_->GetPipelineString(), "}");
}

Status TFGGrapplerOptimizer::Optimize(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item,
    tensorflow::GraphDef* optimized_graph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPStfg_optimizer_hookDTcc mht_6(mht_6_v, 285, "", "./tensorflow/core/grappler/optimizers/tfg_optimizer_hook.cc", "TFGGrapplerOptimizer::Optimize");

  VLOG(5) << "TFG Before Graph: \n" << item.graph.DebugString();

  // Import the GraphDef to TFG.
  tensorflow::GraphDebugInfo debug_info;
  tensorflow::metrics::ScopedCounter<2> metrics(
      tensorflow::metrics::GetGraphOptimizationCounter(),
      {"TfgOptimizer", "convert_graphdef_to_tfg"});
  auto error_or_module =
      ImportGraphDefToMlir(impl_->GetContext(), debug_info, item.graph);
  if (!error_or_module.ok()) {
    auto status = error_or_module.status();
    tensorflow::errors::AppendToMessage(
        &status, "when importing GraphDef to MLIR module in GrapplerHook");
    VLOG(4) << "GraphDef import error: " << status.ToString();
    return status;
  }
  metrics.ReportAndStop();

  // Run the pipeline on the graph.
  ModuleOp module = (*error_or_module).get();
  StatusScopedDiagnosticHandler error_handler(impl_->GetContext());
  if (failed(impl_->RunPipeline(module)))
    return error_handler.Combine(
        InvalidArgument("MLIR Graph Optimizer failed: "));

  // Export the TFG module to GraphDef.
  tensorflow::GraphDef graphdef;
  *graphdef.mutable_library() = item.graph.library();
  metrics.Reset({"TfgOptimizer", "convert_tfg_to_graphdef"});
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      tensorflow::ExportMlirToGraphdef(module, &graphdef),
      "when exporting MLIR module to GraphDef in GrapplerHook");
  metrics.ReportAndStop();
  *optimized_graph = std::move(graphdef);

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "TFG After Graph: \n"
            << optimized_graph->DebugString() << "\nMLIR module: \n";
    module.dump();
  }

  return Status::OK();
}

}  // end namespace tfg
}  // end namespace mlir
