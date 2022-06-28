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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSlmhlo_to_gpuDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSlmhlo_to_gpuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSlmhlo_to_gpuDTcc() {
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

// This file implements logic for lowering LHLO GPU dialect to TFRT CUDA
// dialect.

#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu.h"

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime

namespace tensorflow {

void populateCclConversionPattern(RewritePatternSet&, TypeConverter&);
void populateCholeskyConversionPattern(RewritePatternSet&, TypeConverter&);
void populateConvolutionConversionPattern(RewritePatternSet&, TypeConverter&);
void populateCustomCallConversionPattern(RewritePatternSet&, TypeConverter&);
void populateFftConversionPattern(RewritePatternSet&, TypeConverter&);
void populateGemmConversionPattern(RewritePatternSet&, TypeConverter&);
void populateInfeedAndOutfeedConversionPattern(RewritePatternSet&,
                                               TypeConverter&);
void populateReplicaAndPartitionConversionPattern(RewritePatternSet&,
                                                  TypeConverter&);
void populateTriangularSolveConversionPattern(RewritePatternSet&,
                                              TypeConverter&);

namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gpu_passes.h.inc"

struct ConvertLmhloToGpuPass
    : public ConvertLmhloToGpuPassBase<ConvertLmhloToGpuPass> {
 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSlmhlo_to_gpuDTcc mht_0(mht_0_v, 235, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu.cc", "getDependentDialects");

    registry.insert<mlir::gpu::GPUDialect, tfrt::compiler::TFRTDialect,
                    tfrt::gpu::GpuDialect,
                    tfrt::gpu::conversion::GpuConversionDialect,
                    xla::gpu::XlirDialect>();
  }
};

}  // namespace

void ConvertLmhloToGpuPass::runOnOperation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSlmhlo_to_gpuDTcc mht_1(mht_1_v, 248, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu.cc", "ConvertLmhloToGpuPass::runOnOperation");

  auto* context = &getContext();
  TypeConverter converter = tfrt::gpu::createMemrefToTfrtGpuConverter();

  RewritePatternSet patterns(context);
  populateCclConversionPattern(patterns, converter);
  populateCholeskyConversionPattern(patterns, converter);
  populateConvolutionConversionPattern(patterns, converter);
  populateCustomCallConversionPattern(patterns, converter);
  populateGemmConversionPattern(patterns, converter);
  populateInfeedAndOutfeedConversionPattern(patterns, converter);
  populateReplicaAndPartitionConversionPattern(patterns, converter);
  populateTriangularSolveConversionPattern(patterns, converter);
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, converter);
  populateReturnOpTypeConversionPattern(patterns, converter);
  populateFftConversionPattern(patterns, converter);

  // Set of ops that need to be wrapped in tfrt_gpu_conversion.async.execute
  // before lowering directly to tfrt_gpu ops (and therefore require some chain
  // and stream, which the wrapper op provides as block arguments). On the other
  // hand, ops which lower to the gpu dialect do not need to be wrapped.
  ConversionTarget wrap_target(*context);
  wrap_target.addLegalDialect<lmhlo_gpu::LmhloGpuDialect>();
  wrap_target.addLegalOp<
      lmhlo::AllGatherOp, lmhlo::AllReduceOp, lmhlo::ReduceScatterOp,
      lmhlo::AllToAllOp, lmhlo::CollectivePermuteOp, lmhlo::CustomCallOp,
      lmhlo::TriangularSolveOp, lmhlo::ReplicaIdOp, lmhlo::PartitionIdOp,
      lmhlo::InfeedOp, lmhlo::OutfeedOp, lmhlo::FftOp>();
  tfrt::gpu::populateGpuAsyncConversionPatterns(patterns, converter,
                                                wrap_target);

  ConversionTarget target(*context);
  target.addIllegalOp<memref::ReinterpretCastOp, memref::ViewOp,
                      memref::AllocaOp, memref::AllocOp, memref::DeallocOp>();
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType()) &&
           converter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<tfrt::gpu::conversion::AsyncExecuteOp>(
      [&](tfrt::gpu::conversion::AsyncExecuteOp op) {
        return converter.isLegal(&op.body());
      });
  target.markUnknownOpDynamicallyLegal([&](Operation* op) {
    if (op->hasTrait<OpTrait::ReturnLike>()) return converter.isLegal(op);
    return !wrap_target.isLegal(op);  // Wrapped ops are immediately lowered.
  });

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>> createConvertLmhloToGpuPass() {
  return std::make_unique<ConvertLmhloToGpuPass>();
}

void registerConvertLmhloToGpuPass() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSlmhlo_to_gpuDTcc mht_2(mht_2_v, 307, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu.cc", "registerConvertLmhloToGpuPass");

  ::mlir::registerPass([] { return createConvertLmhloToGpuPass(); });
}

}  // namespace tensorflow
