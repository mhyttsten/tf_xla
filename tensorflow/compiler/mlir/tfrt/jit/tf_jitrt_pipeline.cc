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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrt_pipelineDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrt_pipelineDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrt_pipelineDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.h"

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

// -------------------------------------------------------------------------- //
// Custom passes that are missing upstream.
// -------------------------------------------------------------------------- //

namespace tensorflow {
namespace {

using mlir::OpPassManager;
using mlir::func::FuncOp;

// Adds a Tensorflow producer version to the module to enable shape inference.
struct AddTensorflowProducerVersion
    : public mlir::PassWrapper<AddTensorflowProducerVersion,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrt_pipelineDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.cc", "runOnOperation");

    mlir::ModuleOp module = getOperation();

    // Tensorflow producer version does not really impact anything during the
    // shape inference. Set it to `0` (any random number will do the work) to
    // bypass attribute checks.
    mlir::Builder builder(module);
    auto version =
        builder.getNamedAttr("producer", builder.getI32IntegerAttr(0));
    module->setAttr("tf.versions", builder.getDictionaryAttr({version}));
  }
};

// Adds Linalg passes to perform fusion, tiling, peeling and vectorization.
void AddLinalgTransformations(OpPassManager& pm,
                              const TfJitRtPipelineOptions& options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrt_pipelineDTcc mht_1(mht_1_v, 239, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.cc", "AddLinalgTransformations");

  pm.addNestedPass<FuncOp>(CreateFusionPass());

  if (!options.vectorize) return;

  pm.addNestedPass<FuncOp>(CreateDetensorizeLinalgPass());

  // Unfortunately, at the moment there is no way to provide default values for
  // ListOption. That's why we have to provide them here. When
  // https://github.com/llvm/llvm-project/issues/52667 feature request is
  // accepted and implemented, this line will have to be removed.
  mlir::SmallVector<int64_t, 2> reduction_2d_tile_sizes = {4, 4};
  if (options.reduction_2d_tile_sizes.hasValue()) {
    reduction_2d_tile_sizes.assign(options.reduction_2d_tile_sizes.begin(),
                                   options.reduction_2d_tile_sizes.end());
  }
  pm.addNestedPass<FuncOp>(CreateTileReductionPass(
      options.vector_size, options.reduction_1d_tile_size,
      reduction_2d_tile_sizes));

  if (options.vectorize && options.codegen_transpose)
    pm.addNestedPass<FuncOp>(CreateTileTransposePass());
  pm.addNestedPass<FuncOp>(CreateTileCWisePass(options.vector_size));
  if (options.peel) {
    pm.addNestedPass<FuncOp>(CreatePeelTiledLoopsPass());
  }
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if (options.fuse_fill) {
    pm.addNestedPass<FuncOp>(CreateFuseFillIntoTiledReductionPass());
  }
  pm.addNestedPass<FuncOp>(CreateTileFillPass(options.vector_size));
  pm.addNestedPass<FuncOp>(CreateVectorizeTiledOpsPass());
}

void AddBufferizationPasses(OpPassManager& pm) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrt_pipelineDTcc mht_2(mht_2_v, 277, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.cc", "AddBufferizationPasses");

  // Now bufferize all the compute operations (hlo + linalg) and func signature.
  pm.addPass(
      mlir::kernel_gen::transforms::CreateComputeOpAndFuncBufferizePass());
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateTiledLoopBufferizePass());
  // Always run CSE and canonicalizer (which does dead code removal) before
  // bufferizing anything.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(
      mlir::kernel_gen::transforms::CreateFinalBufferizePass(/*alignment=*/64));
}

}  // namespace

// -------------------------------------------------------------------------- //
// Assemble a TF JitRt pipeline to lower from Tensorflow dialects to Linalg on
// buffers via progressive lowering to MHLO and Linalg.
// -------------------------------------------------------------------------- //
void CreateTfJitRtPipeline(OpPassManager& pm,
                           const TfJitRtPipelineOptions& options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrt_pipelineDTcc mht_3(mht_3_v, 301, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.cc", "CreateTfJitRtPipeline");

  // Break Tensorflow fused operations into primitive operations before
  // lowering to HLO.
  pm.addNestedPass<FuncOp>(CreateFissionPass());

  // Run shape inference to propagate potentially specialized input shapes.
  pm.addPass(std::make_unique<AddTensorflowProducerVersion>());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Transform TF operation to HLO.
  pm.addPass(mlir::mhlo::createLegalizeTFControlFlowPass());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeTFPass());

  if (options.legalize_i1_tensors) {
    // Convert 'i1' tensors into 'i8' tensors.
    pm.addPass(CreateJitRtLegalizeI1TypesPass());
  }

  // Remove redundant shape operations left after legalizing to HLO.
  pm.addPass(mlir::createCSEPass());

  // Resolve all shape constraints (e.g. broadcast constraints that can be
  // proved statically and changed to const witness) early to allow more
  // efficient broadcast operations moving.
  pm.addNestedPass<FuncOp>(
      CreateSymbolicShapeOptimizationPass(/*constraints_only=*/true));

  // Analyze shapes and try to simplify the IR as early as possible.
  pm.addNestedPass<FuncOp>(mlir::createSymbolicShapeOptimizationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Move up broadcasting operations to allow for more fusion opportunities.
  // Add the broadcast propagation pass first, because it can help to avoid
  // exponential complexity from the EarlyBroadcastInDimOp pattern which is used
  // in the merge assuming ops pass further down.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createMergeAssumingOpsPass());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // After all shape constraints removed and broadcasts moved to the top, try
  // to resolve broadcasts that can be converted to linalg generic operations.
  pm.addNestedPass<FuncOp>(CreateSymbolicShapeOptimizationPass());

  // Group reduction and parallel dimensions of reduction operations and realize
  // them through equivalent 1D or 2D reductions, if possible.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createGroupReductionDimensionsPass());

  // Also, try to simplify reshape operations.
  pm.addNestedPass<FuncOp>(mlir::createSymbolicShapeOptimizationPass());

  // Transform HLO operations to Linalg and Standard.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeControlFlowPass());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeHloToLinalgPass());
  pm.addPass(mlir::mhlo::createLegalizeToArithmeticPass());
  pm.addNestedPass<FuncOp>(
      mlir::mhlo::createLegalizeHloShapeOpsToStandardPass());

  // Now that all compute operations are converted to standard (as a side effect
  // of bufferizing to memref dialect) we can remove the remaining references
  // to unsigned types.
  pm.addPass(mlir::kernel_gen::transforms::CreateConvertToSignlessPass());

  // Lower shape dialect to standard to enable linalg canonicalizations (e.g.
  // use linalg inputs instead of outputs for memref.dim operations).
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateShapeSimplification());
  pm.addNestedPass<FuncOp>(mlir::createShapeToShapeLowering());
  pm.addPass(mlir::createConvertShapeToStandardPass());
  pm.addNestedPass<FuncOp>(mlir::createConvertShapeConstraintsPass());

  // Fuse Linalg on tensors operations.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  // Lower index cast on tensors to tensor.generate.
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateLowerIndexCastPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Add linalg passes to perform fusion, tiling, peeling and vectorization.
  AddLinalgTransformations(pm, options);

  // Inline everything, bufferization doesn't model ownership across calls.
  pm.addPass(mlir::createInlinerPass());

  // Always run canonicalizer (which does dead code removal) before bufferizing
  // anything.
  pm.addPass(mlir::createCanonicalizerPass());

  AddBufferizationPasses(pm);

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Deallocate all temporary buffers.
  pm.addNestedPass<FuncOp>(mlir::bufferization::createBufferDeallocationPass());

  // Do trivial buffer forwarding across linalg.generic operations.
  pm.addNestedPass<FuncOp>(CreateLinalgTrivialBufferForwardingPass());

  // Remove trivial copy operations.
  pm.addNestedPass<FuncOp>(CreateLinalgTrivialCopyRemovalPass());

  if (options.vectorize)
    pm.addNestedPass<FuncOp>(mlir::gml_st::createGmlStToScfPass());

  pm.addPass(mlir::createBufferizationToMemRefPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (options.vectorize && options.codegen_transpose)
    pm.addNestedPass<FuncOp>(CreateLowerVectorTransposePass());

  mlir::VectorTransferToSCFOptions vec_to_scf_options;
  vec_to_scf_options.unroll = true;
  pm.addNestedPass<FuncOp>(
      mlir::createConvertVectorToSCFPass(vec_to_scf_options));
  pm.addNestedPass<FuncOp>(createRewriteVectorMultiReductionPass());

  pm.addNestedPass<FuncOp>(CreateMathApproximationPass({"all"}));
}

void CreateDefaultTfJitRtPipeline(OpPassManager& pm) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrt_pipelineDTcc mht_4(mht_4_v, 429, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.cc", "CreateDefaultTfJitRtPipeline");

  TfJitRtPipelineOptions options;
  options.vectorize = tensorflow::GetJitRtFlags().vectorize;
  CreateTfJitRtPipeline(pm, options);
}

void CreateJitRtSpecializationPipeline(mlir::OpPassManager& pm) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStf_jitrt_pipelineDTcc mht_5(mht_5_v, 438, "", "./tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.cc", "CreateJitRtSpecializationPipeline");

  pm.addPass(std::make_unique<AddTensorflowProducerVersion>());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
}

static mlir::PassPipelineRegistration<TfJitRtPipelineOptions> tf_jitrt_pipeline(
    "tf-jitrt-pipeline",
    "Convert Tensorflow dialect to TFRT's JitRt compatible dialects",
    CreateTfJitRtPipeline);

}  // namespace tensorflow
