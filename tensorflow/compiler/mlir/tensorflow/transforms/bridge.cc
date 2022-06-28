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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/metrics.h"

namespace mlir {
namespace {

// Add logger to bridge passmanager.
// Enable timing statistics per pass for the bridge passmanager.
void EnableDetailedLogging(PassManager *pm) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "EnableDetailedLogging");

  // Print the whole module after each pass, which requires disabling
  // multi-threading as well.
  pm->getContext()->disableMultithreading();
  pm->enableIRPrinting(std::make_unique<tensorflow::BridgeLoggerConfig>(
      /*print_module_scope=*/true));
  pm->enableTiming();
}
}  // namespace

namespace TFTPU {

namespace {
// Run the TF XLA Bridge based on the input pipeline, which can be either TPU
// bridge pipeline or non TPU bridge pipeline.
tensorflow::Status RunTFXLABridge(
    ModuleOp module, bool enable_logging,
    llvm::function_ref<void(OpPassManager &pm)> pipeline_builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "RunTFXLABridge");

  PassManager bridge(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(bridge);
  if (enable_logging || VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tf_xla_bridge_before", module);
    if (VLOG_IS_ON(2)) EnableDetailedLogging(&bridge);
  }

  // Populate a passmanager with the list of passes that implement the bridge.
  pipeline_builder(bridge);

  // Add set of passes to lower back to graph (from tf_executor).
  TF::AddGraphExportLoweringPasses(bridge);

  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  LogicalResult result = bridge.run(module);
  (void)result;
  if (enable_logging || VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("tf_xla_bridge_after", module);
  return diag_handler.ConsumeStatus();
}

void CreateTPUBridgePipelineImpl(OpPassManager &pm) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "CreateTPUBridgePipelineImpl");

  // The following ops must be preserved regardless of reachability. Ideally,
  // all graphs should have control dependencies to enforce this but this is
  // currently not the case (see b/177478741).
  const llvm::SmallVector<std::string, 4> ops_to_preserve = {
      "tf.TPUReplicateMetadata", "tf.TPUCompilationResult",
      "tf.TPUReplicatedOutput"};
  pm.addNestedPass<FuncOp>(
      tf_executor::CreateTFExecutorGraphPruningPass(ops_to_preserve));
  // It is assumed at this stage there are no V1 control flow ops as Graph
  // functionalization is ran before import. Ops can be lifted out of
  // tf_executor dialect islands/graphs.
  pm.addNestedPass<FuncOp>(CreateExecutorDialectToFunctionalConversionPass());
  // Guarantee all functions have one use, which enables more exact shape
  // inference.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  // Run shape inference so that tf_executor/tf_device ops created later will
  // likely to inherit more concrete types.
  pm.addPass(TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(CreateTPUReorderReplicateAndPartitionedInputsPass());
  pm.addPass(CreateTPUClusterFormationPass());
  // Run TPU cluster cleanup attributes so ops with no outside compiled
  // attribute have no host device attribute.
  pm.addPass(CreateTPUClusterCleanupAttributesPass());
  pm.addPass(CreateOutsideCompiledToHostLaunchPass());
  pm.addNestedPass<FuncOp>(TFDevice::CreateDeviceAttributeToLaunchPass());
  // Running canonicalizer before decomposing resource ops in cluster helps the
  // latter pass to converge faster as it does not have to spend time folding
  // away dead ops.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  // Place DecomposeResourceOpsPass before TFExecutorConstantSinking pass
  // because DecomposeResourceOpsPass uses pattern rewriter which hoists
  // changed constants out of tf_device.Launch.
  pm.addPass(TFDevice::CreateDecomposeResourceOpsInClusterPass());
  // Encode this in its own scope so that func_pm is not mistakenly used
  // later on.
  {
    OpPassManager &func_pm = pm.nest<FuncOp>();
    func_pm.addPass(CreateTPUHostComputationExpansionPass());
    func_pm.addPass(CreateTPUUpdateEmbeddingEnqueueOpInputsPass());
  }
  // TODO(b/173622615): This should incrementally be moved down as
  // more passes support this representation and then can be removed once
  // all passes support it.
  pm.addPass(TFDevice::CreateHostLaunchToOutsideCompiledPass());

  // TODO(b/173622615): Once OutsideCompilation is represented by launch op and
  // the remaining passes including Inliner support it, remove this
  // LaunchToDeviceAttributePass. This LaunchToDeviceAttribute pass needs to
  // come before TPUClusterCleanupAttributes pass or else the device attribute
  // will be removed from launch causing an error.
  pm.addNestedPass<FuncOp>(TFDevice::CreateLaunchToDeviceAttributePass());

  // TODO(b/173622615): This can be removed once more passes support outside
  // compilation represented by op and conversion back to attribute is removed.
  pm.addPass(CreateOutsideCompiledToHostLaunchPass());
  // Note that the region-based control-flow produced here still contains
  // function call ops which get inlined by the subsequent inliner pass.
  pm.addPass(TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<FuncOp>(
      TF::CreateDropWhileShapeInvariantInDeviceClusterPass());
  // Run another shape inference pass because resource decomposition might have
  // created new partial types. Also, after dropping `shape_invariant` attribute
  // from While/WhileRegion ops within cluster would lead to more precise
  // shapes.
  pm.addPass(TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addPass(CreateTPUClusterCleanupAttributesPass());
  pm.addPass(TFDevice::CreateResourceOpLiftingPass());
  // Re-run the canonicalizer pass as some cleanup during resource op lifting
  // pass opens up some opportunities for canonicalization of cluster ops.
  // Specifically, we want to eliminate pass through results from the cluster
  // op.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  // TODO(b/173622615): This should incrementally be moved down as
  // more passes support this representation and then can be removed once
  // all passes support it.
  pm.addPass(TFDevice::CreateHostLaunchToOutsideCompiledPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_merge_control_flow_pass) {
    pm.addPass(TFDevice::CreateMergeControlFlowPass());
  }

  pm.addPass(TFDevice::CreateMarkOpsForOutsideCompilationPass());
  pm.addPass(CreateTPUExtractHeadTailOutsideCompilationPass());
  pm.addPass(CreateTPUExtractOutsideCompilationPass());

  pm.addNestedPass<FuncOp>(TFDevice::CreateClusterConstantSinkingPass());
  pm.addPass(TF::CreateResourceDeviceInferencePass());
  pm.addPass(TFDevice::CreateClusterOutliningPass());
  pm.addPass(CreateTPUResourceReadForWritePass());
  pm.addPass(TFDevice::CreateMarkInputOutputAliasesPass());
  pm.addPass(CreateTPUShardingIdentificationPass());
  pm.addNestedPass<FuncOp>(CreateTPUResourceReadsWritesPartitioningPass());
  pm.addPass(TFDevice::CreateAnnotateParameterReplicationPass());
  pm.addPass(CreateTPURewritePass());
  pm.addPass(createSymbolDCEPass());
  pm.addNestedPass<FuncOp>(TFDevice::CreateReplicateInvariantOpHoistingPass());
  pm.addPass(CreateTPUMergeVariablesWithExecutePass());
  pm.addNestedPass<FuncOp>(
      TF::CreateHoistReplicateInvariantResourceWritesPass());
  pm.addNestedPass<FuncOp>(CreateTPUColocateCompositeResourceOps());
  pm.addPass(CreateTPUVariableRuntimeReformattingPass());
  pm.addPass(TF::CreateTFRegionControlFlowToFunctional());
}
}  // namespace

void CreateTPUBridgePipeline(OpPassManager &pm) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_3(mht_3_v, 366, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "CreateTPUBridgePipeline");

  pm.addNestedPass<FuncOp>(
      CreateCanonicalizeCompileAndReplicateAttributesPass());
  CreateTPUBridgePipelineImpl(pm);
}

void CreateTPUBridgePipelineV1(OpPassManager &pm) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_4(mht_4_v, 375, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "CreateTPUBridgePipelineV1");

  // Convert to unified compilation and replication attributes.
  pm.addNestedPass<FuncOp>(
      CreateCanonicalizeCompileAndReplicateAttributesPass());
  // Guarantee all functions have one use, which enables more exact shape
  // inference.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  pm.addPass(TF::CreateTFShapeInferencePass());
  // For V1 compatibility, we process a module where the graph does not have
  // feeds and fetched. We extract first the TPU computation in a submodule,
  // where it'll be in a function with args and returned values, much more like
  // a TF v2 module. We can then run the usual pipeline on this nested module.
  // Afterward we inline back in the parent module and delete the nested one.
  pm.addPass(tf_executor::CreateTFExecutorTPUV1IslandCoarseningPass());
  pm.addPass(tf_executor::CreateTFExecutorTPUV1IslandOutliningPass());
  OpPassManager &nested_module = pm.nest<ModuleOp>();
  CreateTPUBridgePipelineImpl(nested_module);
  pm.addPass(tf_executor::CreateTFExecutorTPUV1IslandInliningPass());
  // There are cases where we don't consume all compilation and replication
  // attributes like we do for the V2 pipeline, so we need to convert them from
  // unified to legacy attributes before they get exposed to outside of the
  // bridge.
  pm.addNestedPass<FuncOp>(
      CreateConvertToLegacyCompileAndReplicateAttributesPass());
}

tensorflow::Status TPUBridge(ModuleOp module, bool enable_logging,
                             bool fallback_enabled) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_5(mht_5_v, 405, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "TPUBridge");

  Status status =
      RunTFXLABridge(module, enable_logging, CreateTPUBridgePipeline);
  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      "tpu", "v2", fallback_enabled,
      status == Status::OK() ? "success" : "failure");
  return status;
}
tensorflow::Status TPUBridgeV1Compat(ModuleOp module, bool enable_logging,
                                     bool fallback_enabled) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_6(mht_6_v, 417, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "TPUBridgeV1Compat");

  Status status =
      RunTFXLABridge(module, enable_logging, CreateTPUBridgePipelineV1);
  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      "tpu", "v1", fallback_enabled,
      status == Status::OK() ? "success" : "failure");
  return status;
}

}  // namespace TFTPU

namespace TF {

void AddGraphExportLoweringPasses(OpPassManager &pm) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_7(mht_7_v, 433, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "AddGraphExportLoweringPasses");

  auto add_pass = [&](std::unique_ptr<Pass> pass) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_8(mht_8_v, 437, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "lambda");

    pm.addNestedPass<FuncOp>(std::move(pass));
    pm.addPass(CreateBreakUpIslandsPass());
  };

  add_pass(CreateFunctionalToExecutorDialectConversionPass());
  add_pass(TFDevice::CreateReplicateToIslandPass());
  add_pass(TFDevice::CreateReplicaIDToDeviceOrdinalPass());
  add_pass(TFDevice::CreateParallelExecuteToIslandsPass());
  add_pass(TFDevice::CreateLaunchToDeviceAttributePass());
  pm.addNestedPass<FuncOp>(TFTPU::CreateTPUDevicePropagationPass());
  pm.addPass(createSymbolDCEPass());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_convert_control_to_data_outputs_pass) {
    pm.addPass(tf_executor::CreateTFExecutorConvertControlToDataOutputsPass());
  }
  pm.addPass(CreateVerifySuitableForExportPass());
}

tensorflow::Status RunBridgeWithStandardPipeline(ModuleOp module,
                                                 bool enable_logging,
                                                 bool enable_inliner) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_9(mht_9_v, 461, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "RunBridgeWithStandardPipeline");

  PassManager bridge(module.getContext());
  if (enable_logging || VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("standard_pipeline_before", module);
    if (VLOG_IS_ON(2)) EnableDetailedLogging(&bridge);
  }

  StandardPipelineOptions pipeline_options;
  pipeline_options.enable_inliner.setValue(enable_inliner);
  CreateTFStandardPipeline(bridge, pipeline_options);

  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  LogicalResult result = bridge.run(module);
  (void)result;
  if (enable_logging || VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("standard_pipeline_after", module);
  return diag_handler.ConsumeStatus();
}

namespace {
void CreateTFXLABridgePipeline(OpPassManager &pm) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_10(mht_10_v, 487, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "CreateTFXLABridgePipeline");

  // The following ops must be preserved regardless of reachability. Ideally,
  // all graphs should have control dependencies to enforce this.
  VLOG(2) << "Create TF XLA Bridge pipeline";
  const llvm::SmallVector<std::string, 4> ops_to_preserve = {};
  pm.addNestedPass<FuncOp>(
      tf_executor::CreateTFExecutorGraphPruningPass(ops_to_preserve));
  // It is assumed at this stage there are no V1 control flow ops as Graph
  // functionalization is ran before import. Ops can be lifted out of
  // tf_executor dialect islands/graphs.
  pm.addNestedPass<FuncOp>(CreateExecutorDialectToFunctionalConversionPass());
  // Guarantee all functions have one use, which enables more exact shape
  // inference.
  pm.addPass(TF::CreateTFShapeInferencePass());
  // Running canonicalizer before decomposing resource ops in cluster helps the
  // latter pass to converge faster as it does not have to spend time folding
  // away dead ops.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  // Encapsulate StatefulPartitionedCallOp within a cluster so that the
  // composite resource ops can be decomposed.
  pm.addPass(TFDevice::CreateXlaClusterFormationPass());
  // Place DecomposeResourceOpsPass.
  pm.addPass(TFDevice::CreateDecomposeResourceOpsInClusterPass());
  // Run another shape inference pass because resource decomposition might have
  // created new partial types. Also, after dropping `shape_invariant` attribute
  // from While/WhileRegion ops within cluster would lead to more precise
  // shapes.
  pm.addPass(TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addPass(TFDevice::CreateResourceOpLiftingPass());
  // Inline the StatefulPartitionedCallOp op based in the parent region.
  pm.addPass(TFDevice::CreateXlaInlineDeviceOpsPass());
  // Re-run the canonicalizer pass as some cleanup during resource op lifting
  // pass opens up some opportunities for canonicalization of cluster ops.
  // Specifically, we want to eliminate pass through results from the cluster
  // op.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(TF::CreateTFRegionControlFlowToFunctional());
}

}  // namespace

tensorflow::Status RunTFXLABridge(ModuleOp module, bool enable_logging) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSbridgeDTcc mht_11(mht_11_v, 535, "", "./tensorflow/compiler/mlir/tensorflow/transforms/bridge.cc", "RunTFXLABridge");

  return mlir::TFTPU::RunTFXLABridge(module, enable_logging,
                                     CreateTFXLABridgePipeline);
}

}  // namespace TF
}  // namespace mlir
