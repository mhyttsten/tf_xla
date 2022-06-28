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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#define DEBUG_TYPE "run-tf-graph-optimization"

namespace tensorflow {
namespace {
// Creates a pass to convert MLIR to Graph, run user-specified Graph
// Optimization Passes and convert back to MLIR.
// Constraints: This pass expects that all operations in the MLIR module either
// belong to 'tf' or '_tf' dialect. The output is in '_tf' dialect.
class GraphOptPass
    : public mlir::PassWrapper<GraphOptPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "getDependentDialects");

    mlir::RegisterAllTensorFlowDialects(registry);
  }

 public:
  explicit GraphOptPass(std::vector<tensorflow::GraphOptimizationPass*> passes)
      : passes_(std::move(passes)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "GraphOptPass");
}

 protected:
  void runOnOperation() override;

  // The passes to run on the module.
  std::vector<GraphOptimizationPass*> passes_;
};
}  // anonymous namespace

void GraphOptPass::runOnOperation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "GraphOptPass::runOnOperation");

  mlir::ModuleOp module_in = getOperation();
  mlir::MLIRContext& ctx = getContext();

  // Convert MLIR to Graph
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphExportConfig confs;
  auto graph = absl::make_unique<Graph>(flib_def);
  Status status = ConvertMlirToGraph(module_in, confs, &graph, &flib_def);
  if (!status.ok()) {
    mlir::emitError(mlir::UnknownLoc::get(&ctx)) << status.error_message();
    return signalPassFailure();
  }

  // Run each of the passes that were selected.
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = false;

  GraphOptimizationPassOptions options;
  SessionOptions sess_options;
  options.graph = &graph;
  options.flib_def = &flib_def;
  options.session_options = &sess_options;

  for (auto pass : passes_) {
    assert(pass != nullptr);
    Status status = pass->Run(options);
    if (!status.ok()) {
      mlir::emitError(mlir::UnknownLoc::get(&ctx))
          << pass->name() << ": " << status.error_message();
      return signalPassFailure();
    }
  }

  // Convert Graph to MLIR
  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  auto module_or_status =
      ConvertGraphToMlir(**options.graph, debug_info, flib_def, specs, &ctx);
  if (!module_or_status.ok()) {
    mlir::emitError(mlir::UnknownLoc::get(&ctx))
        << module_or_status.status().error_message();
    return signalPassFailure();
  }
  auto module_out = std::move(module_or_status).ValueOrDie();

  // We cannot replace the module in a ModulePass. So we simply copy the
  // operation list from module_out to module_in.
  auto& module_in_ops = module_in.getBody()->getOperations();
  module_in_ops.clear();
  module_in_ops.splice(module_in_ops.end(),
                       module_out->getBody()->getOperations());
}

// Returns a vector of passes from their names. If a pass is not found, then the
// corresponding return entry is null.
static std::vector<GraphOptimizationPass*> FindRegisteredPassesByName(
    const std::vector<std::string>& pass_names) {
  std::vector<GraphOptimizationPass*> pass_ids(pass_names.size(), nullptr);

  for (const auto& group : OptimizationPassRegistry::Global()->groups()) {
    for (const auto& phase : group.second) {
      for (const auto& pass : phase.second) {
        // Iterate over the pass_names_ and insert the pass pointer at all the
        // corresponding indices in the pass_ids vector.
        auto iter = pass_names.begin();
        while ((iter = std::find(iter, pass_names.end(), pass->name())) !=
               pass_names.end()) {
          pass_ids[std::distance(pass_names.begin(), iter)] = pass.get();
          iter++;
        }
      }
    }
  }
  return pass_ids;
}

// TODO(prakalps): Move these flags and pass registration to a header file so
// that it is clear that this is a generic pass library and command line is used
// for testing only.

// NOLINTNEXTLINE
static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

// NOLINTNEXTLINE
static llvm::cl::list<std::string> cl_pass_list(
    "graph-passes", llvm::cl::value_desc("list"),
    llvm::cl::desc("comma separated list of GraphOptimizationPass to run."),
    llvm::cl::CommaSeparated, llvm::cl::cat(clOptionsCategory));

class GraphOptByNamePass : public GraphOptPass {
 public:
  explicit GraphOptByNamePass() : GraphOptByNamePass(cl_pass_list) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_3(mht_3_v, 334, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "GraphOptByNamePass");
}
  explicit GraphOptByNamePass(const std::vector<std::string>& pass_names)
      : GraphOptPass(FindRegisteredPassesByName(pass_names)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_4(mht_4_v, 339, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "GraphOptByNamePass");
}

  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_5(mht_5_v, 344, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "getArgument");

    return "run-tf-graph-optimization";
  }

  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_6(mht_6_v, 351, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "getDescription");

    return "runs passes registered as tensorflow::GraphOptimizationPass";
  }

 private:
  void runOnOperation() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_7(mht_7_v, 359, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "runOnOperation");

    // Verify all passes requested were registered/found.
    for (auto pass_it : llvm::enumerate(passes_)) {
      if (pass_it.value() == nullptr) {
        mlir::emitError(mlir::UnknownLoc::get(&getContext()))
            << "could not find pass " << cl_pass_list[pass_it.index()];
        return signalPassFailure();
      }
    }
    return GraphOptPass::runOnOperation();
  }
};

}  // namespace tensorflow

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
tensorflow::CreateTensorFlowGraphOptimizationPass(
    std::vector<tensorflow::GraphOptimizationPass*> tf_passes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_8(mht_8_v, 379, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "tensorflow::CreateTensorFlowGraphOptimizationPass");

  return std::make_unique<GraphOptPass>(std::move(tf_passes));
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
tensorflow::CreateTensorFlowGraphOptimizationPass(
    const std::vector<std::string>& pass_names) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_9(mht_9_v, 388, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "tensorflow::CreateTensorFlowGraphOptimizationPass");

  return std::make_unique<GraphOptByNamePass>(pass_names);
}

void tensorflow::RegisterGraphOptimizationPasses() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStf_graph_optimization_passDTcc mht_10(mht_10_v, 395, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.cc", "tensorflow::RegisterGraphOptimizationPasses");

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<GraphOptByNamePass>();
  });
}
