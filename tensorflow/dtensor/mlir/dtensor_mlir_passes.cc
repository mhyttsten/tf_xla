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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mlir_passesDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mlir_passesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mlir_passesDTcc() {
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

#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/mlir/create_dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/utils/dtensor_mlir_passes_internal.h"

namespace tensorflow {
namespace dtensor {
namespace {
class ConditionalPrinter : public BridgeLoggerConfig {
 private:
  bool do_not_print_;

 public:
  explicit ConditionalPrinter(bool print_module_scope = false,
                              bool print_after_only_on_change = true)
      : BridgeLoggerConfig(print_module_scope, print_after_only_on_change) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mlir_passesDTcc mht_0(mht_0_v, 213, "", "./tensorflow/dtensor/mlir/dtensor_mlir_passes.cc", "ConditionalPrinter");

    do_not_print_ = !(LogOnAllTasks() || (ClientId() == 0));
  }

  void printBeforeIfEnabled(mlir::Pass *pass, mlir::Operation *operation,
                            PrintCallbackFn print_callback) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mlir_passesDTcc mht_1(mht_1_v, 221, "", "./tensorflow/dtensor/mlir/dtensor_mlir_passes.cc", "printBeforeIfEnabled");
}

  void printAfterIfEnabled(mlir::Pass *pass, mlir::Operation *operation,
                           PrintCallbackFn print_callback) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mlir_passesDTcc mht_2(mht_2_v, 227, "", "./tensorflow/dtensor/mlir/dtensor_mlir_passes.cc", "printAfterIfEnabled");

    mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(operation);
    if (!module) module = operation->getParentOfType<mlir::ModuleOp>();
    if (module && !module->hasAttr(dtensor::kDoNotLog) && !do_not_print_)
      BridgeLoggerConfig::printAfterIfEnabled(pass, operation, print_callback);
  }
};
}  // namespace

// Adds logger to DTensor transformation passmanager.
bool MaybeEnableLogging(mlir::PassManager *pm) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mlir_passesDTcc mht_3(mht_3_v, 240, "", "./tensorflow/dtensor/mlir/dtensor_mlir_passes.cc", "MaybeEnableLogging");

  if (VLOG_IS_ON(1)) {
    // Print the whole module after each pass, which requires disabling
    // multi-threading as well.
    pm->getContext()->disableMultithreading();
    pm->enableIRPrinting(std::make_unique<ConditionalPrinter>(
        /*print_module_scope=*/true));
    return true;
  }
  return false;
}

void CreateDTensorMLIRPass(const mlir::TF::StandardPipelineOptions &options,
                           mlir::OpPassManager *pm) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mlir_passesDTcc mht_4(mht_4_v, 256, "", "./tensorflow/dtensor/mlir/dtensor_mlir_passes.cc", "CreateDTensorMLIRPass");

  // Remove ops that cannot be reached from the sink node.
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass());
  // Remove graph-def executor dialect and represent IR as a flattened list of
  // TF ops in functions.
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::CreateExecutorDialectToFunctionalConversionPass());

  // This does not guarantee that shape are inferred for all ops. For ops with
  // dynamic shapes, shape information may still be missing.
  pm->addPass(mlir::TF::CreateTFShapeInferencePass());

  // If V2 layout propagation algorithm, layouts are expressed as DTensorLayout
  // op and Canonicalize and Inliner passes will not lose layout information.
  pm->addNestedPass<mlir::func::FuncOp>(CreateDTensorPropagateDefaultLayout());
  pm->addPass(mlir::createSCCPPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());
  pm->addPass(mlir::createInlinerPass());

  // Ensure that all functions have `device_id` as 0th argument.
  pm->addPass(CreateDTensorPropagateDeviceIdToFunctionArgs());

  // Ensure that all functions with SparseTensor input is converted to its
  // three component tensors and SparseToDenseOps are emitted for every usage
  // of a SparseTensor.
  pm->addPass(CreateDTensorSparseTensorToDenseTensor());

  AddDTensorEmbeddingPass(pm);

  // After shape inference, there may be unused constants ops added when
  // propagating caller-callee constants. As DTensor mesh/layout propgation
  // passes assumes that there are no unreachable ops, removes trivial unused
  // ops. Note that `Canonicalizer` pass in TF includes similar optimization.
  // However, canonicalizer pass also rewrites some ops and may remove `_layout`
  // or `_mesh` attributes in the re-written TF ops.
  // TODO(hongjunchoi): Remove this pass once shape inference pass no longer
  // creates unnecessary constants ops.
  pm->addNestedPass<mlir::func::FuncOp>(CreateDTensorDCE());

  // Propagate mesh cluster config and cluster ops by mesh cluster so that
  // SPMD expansion can be isolated to a single device mesh.
  pm->addNestedPass<mlir::func::FuncOp>(CreateDTensorOpToDeviceClusterPass());
  pm->addPass(CreateDTensorMeshPropagationPass());

  {
    mlir::OpPassManager &func_pm = pm->nest<mlir::func::FuncOp>();
    func_pm.addPass(CreateDTensorDeviceMeshClusterCoarsening());
    // Set empty layout to cluster wrapping `tf.VarHandleOp`. VarHandle op
    // always runs in the default device where client program executes.
    func_pm.addPass(CreateDTensorDesignateResourceHandleMesh());
  }

  // Validates that all cross mesh data transfers are expressed via
  // DTensorLayout operation and lowers it to send/recvs.
  pm->addPass(CreateDTensorHandleCrossClusterDependencies());

  // Mark all ops and functions with global shape attribute to preserve global
  // shape information as it is needed during Layout Propagation and SPMD
  // expansion.
  pm->addPass(CreateDTensorAnnotateGlobalShape());

  // Propagate layout to all ops in graph.
  pm->addPass(CreateDTensorMergeClustersPass());

  AddDTensorEmbeddingPassV2(pm);

  pm->addPass(CreateDTensorLayoutPropagationPassV2());

  // Expand graph to SPMD form given layouts are annotated to all ops.
  // Remove all DTensorLayout ops after the expansion is done.
  pm->addPass(CreateDTensorSPMDExpansion());

  // Expand all ops that consume SparseTensors to possibly new ops.
  // Remove any unused SparseToDense, Layout, and Const Ops after
  // the expansion is done.
  //
  // Note that this pass assumes that SparseTensor operands is represented
  // as an operand from the output of a SparseToDenseOp. Thus, this pass
  // must happen after SparseTensorToDenseTensor pass and after
  // the SPMD Expansion pass.
  pm->addPass(CreateDTensorSparseExpansion());

  // Do a round of CSE: this helps reduce the number of consts in the graph now
  // that SPMD expansion is done. We had replicated all Consts (so that each
  // const only had one usage) as part of layout propagation.
  pm->addPass(mlir::createCSEPass());

  // Lower the AllGather collectives. This has to happen before the all reduce
  // optimizations and AllGather may emit an AllReduce.
  pm->addPass(CreateDTensorAllGatherLoweringPass());

  // Fuses AllReduce and AllScatter into ReduceScatter.
  if (!DoNotFuseReduceScatter()) {
    pm->addNestedPass<mlir::func::FuncOp>(
        CreateDTensorAllReduceScatterOptimization());
  }

  // Changes order of DTensorAllReduce + Add to Add + DTensorAllReduce to
  // minimize number of all reduce operations.
  pm->addNestedPass<mlir::func::FuncOp>(
      CreateDTensorAllReduceSumOptimization());

  AddDTensorAllReduceCombineOptimization(pm);

  // DTensorReduceScatter lowering should come before DTensorAllReduce
  // and DTensorAllScatter lowerings since for some devices DTensorReduceScatter
  // will be decomposed into an DTensorAllReduce+DTensorScatter.
  pm->addPass(CreateDTensorReduceScatterLoweringPass());

  // For large enough reduction groups in reduction ops, upcast the input
  // tensors to higher precision type (e.g. bfloat16 -> float32).
  if (EnableMixedPrecisionReduce()) {
    pm->addNestedPass<mlir::func::FuncOp>(
        CreateDTensorMixedPrecisionReducePass());
  }

  // Lower device-agnostic logical AllReduce ops into device-specific physical
  // AllReduce ops.
  //
  // First, find DTensor collective ops such as DTensorAllReduce, which are
  // generated by SPMD expansion. Lower them into device-specific forms. For
  // most devices, there is a one-to-one mapping: DTensorAllReduce becomes
  // CollectiveReduce on CPUs/GPUs and XlaAllReduce on TPU pods.
  // Optionally, for special topologies, DTensorAllReduce
  // could become a chain of collectives running on different devices:
  // XlaAllReduce on each donut followed by CollectiveReduce on the hosts. Those
  // collective ops running on hosts will have their _mesh attribute set to
  // empty by this pass. The other ops continue to have no _mesh attributes,
  // which means they run on the cluster mesh.
  pm->addPass(CreateDTensorAllReduceLoweringPass());

  pm->addPass(CreateDTensorAllScatterLoweringPass());

  // Group together multiple device clusters assigned to the same mesh. Repeat
  // this for every mesh to support multi-mesh. Collective lowering may have
  // created multiple CPU mesh clusters for executing collective operations on
  // CPUs.
  // As so, we merge newly created CPU clusters after collective lowering
  // especially for special topologies.
  pm->addPass(CreateDTensorMergeClustersPass());
  pm->addPass(CreateDTensorLowerSendRecv());

  // Convert tf_device.cluster into a function call op.
  pm->addPass(mlir::TFDevice::CreateClusterOutliningPass());
  pm->addPass(CreateDTensorClusterFunctionConversion());

  // During layout propagation, we clone all constants with multiple consumers
  // for easier analaysis.
  // This may create multiple same constants ops. Apply constant folding on
  // duplicated constant operations to reduce graph size.
  pm->addNestedPass<mlir::func::FuncOp>(CreateDTensorConstantFolding());
  // DTensor SPMD lowering passes may have created auxiliary operations that are
  // no longer used. Add additional DCE pass to remove unused non-side effecting
  // ops.
  pm->addNestedPass<mlir::func::FuncOp>(CreateDTensorDCE());

  // DTensor SPMD Expansion may have caused multiple control flows and
  // duplicate ops to calculate device ordinal. Re-run SCCP and merge
  // controlflows if possible.
  pm->addNestedPass<mlir::func::FuncOp>(mlir::createSCCPPass());
  pm->addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm->addPass(mlir::TFDevice::CreateMergeControlFlowPass());

  // TF2XLA Integration
  {
    // Make sure clusters that run on TPU's are correct metadata ops and
    // attributes attached to be compatible with later TPU specific optimization
    // passes.
    pm->addPass(CreateDTensorTPUIntegration());

    pm->addNestedPass<mlir::func::FuncOp>(
        mlir::TFDevice::CreateDecomposeResourceOpsPass());
    // Sink constant ops into cluster region as DecomposeResourceOpsPass() could
    // lift constant out due to folding.
    pm->addNestedPass<mlir::func::FuncOp>(
        mlir::TFDevice::CreateClusterConstantSinkingPass());

    // Run another shape inference pass (and following DCE pass) because
    // resource decomposition might have created new partial types.
    pm->addPass(mlir::TF::CreateTFShapeInferencePass());
    pm->addNestedPass<mlir::func::FuncOp>(CreateDTensorDCE());
    pm->addPass(mlir::TFDevice::CreateResourceOpLiftingPass());
    pm->addPass(mlir::TFDevice::CreateClusterOutliningPass());

    // Rename functions with unique names, to avoid collisions in the function
    // library.
    pm->addPass(CreateFunctionRenamingPass());

    // As DTensor SPMD expansion handles sharded inputs for model
    // parallelism, we set input/output sharding to maximal sharding
    // for inputs/outputs of the TPU computation.
    pm->addNestedPass<mlir::func::FuncOp>(CreateDTensorSetDefaultSharding());

    // Creates a pass that marks TPU cluster input-output pairs reading and
    // writing to same resource variable as aliases.
    pm->addPass(mlir::TFDevice::CreateMarkInputOutputAliasesPass());

    // Convert compilation and replication attributes to unified attributes
    // expected by TPURewritePass.
    pm->addNestedPass<mlir::FuncOp>(
        mlir::TFTPU::CreateCanonicalizeCompileAndReplicateAttributesPass());
    // Create TPU Compile and TPU Execute ops for each TPU devices.
    pm->addPass(mlir::TFTPU::CreateTPURewritePass());
    // Convert unified compilation and replication attributes back to legacy
    // attributes for subsequent passes.
    pm->addNestedPass<mlir::FuncOp>(
        mlir::TFTPU::CreateConvertToLegacyCompileAndReplicateAttributesPass());

    // Add placeholder device attributes to resource arguments of TPU
    // computation. This ensures the following
    // CreateTPUMergeVariablesWithExecutePass correctly merges resource
    // operations with TPUExecute op.
    pm->addPass(CreateDTensorTpuAddResourceDeviceAttribute());
    // Translate TPUExecute op to TPUExecuteAndUpdateVariable op to enable
    // buffer aliasing.
    pm->addPass(mlir::TFTPU::CreateTPUMergeVariablesWithExecutePass());

    pm->addPass(CreateDTensorUpdateTPUMetadata());
    // If send/recv exists between TPU and CPU, then TPU Compilation program key
    // is used as input for recv op in host computation as well as TPUExecute op
    // in device computation. As so, move TPUCompile logic to host computation
    // and transfer program key using send/recv operations.
    pm->addPass(CreateDTensorMoveCompilationToHost());
    pm->addPass(mlir::createSymbolDCEPass());
  }

  pm->addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());

  // Convert graph into graph executor dialect so that transformed graph can be
  // exported back to Graphdef.
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm->addPass(mlir::CreateBreakUpIslandsPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateLaunchToDeviceAttributePass());
  // Add additional BreakUpIslandPass as LaunchToDeviceAttribute pass may have
  // created additional islands.
  pm->addPass(mlir::CreateBreakUpIslandsPass());
}

}  // namespace dtensor
}  // namespace tensorflow
