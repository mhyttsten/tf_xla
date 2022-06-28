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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for translating mixed IR to buffer form.
// Currently it supports MHLO and some operations from the Standard dialect.

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/bufferizable_op_interface_impl.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

/// A helper type converter class that automatically populates the relevant
/// materializations and type conversions for bufferization.

static Value materializeToTensor(OpBuilder& builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  return builder.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
}

// TODO(pifon): Remove as soon as https://reviews.llvm.org/D93126 is landed.
class CustomBufferizeTypeConverter
    : public bufferization::BufferizeTypeConverter {
 public:
  CustomBufferizeTypeConverter() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_0(mht_0_v, 263, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "CustomBufferizeTypeConverter");

    // Keep all types unchanged.
    addConversion([](Type type) { return type; });
    // Convert RankedTensorType to MemRefType.
    addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    // Convert UnrankedTensorType to UnrankedMemRefType.
    addConversion([](UnrankedTensorType type) -> Type {
      return UnrankedMemRefType::get(type.getElementType(), 0);
    });
    addArgumentMaterialization(materializeToTensor);
    addSourceMaterialization(materializeToTensor);
    addTargetMaterialization([](OpBuilder& builder, BaseMemRefType type,
                                ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1);
      // Target materialization is invoked if the new operand type does not
      // match the expected type. A special case is when the new operand type is
      // a memref with a specified layout, i.e. non-empty affine map.
      // TODO(pifon) : Change how target materialization is invoked in dialect
      // conversion.
      if (auto memref_type = inputs[0].getType().dyn_cast<MemRefType>()) {
        assert(!memref_type.getLayout().isIdentity());
        return inputs[0];
      }
      assert(inputs[0].getType().isa<TensorType>());
      return builder.create<bufferization::ToMemrefOp>(loc, type, inputs[0]);
    });
  }
};

struct ComputeOpAndFuncBufferizePass
    : public ComputeOpAndFuncBufferizePassBase<ComputeOpAndFuncBufferizePass> {
  // TODO(b/173201243): Move to tablegen.
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_1(mht_1_v, 300, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "getDependentDialects");

    registry.insert<linalg::LinalgDialect, bufferization::BufferizationDialect,
                    lmhlo::LmhloDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, mhlo::MhloDialect,
                    shape::ShapeDialect, vector::VectorDialect>();
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mhlo::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_2(mht_2_v, 314, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "runOnOperation");

    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.allowDialectInFilter<linalg::LinalgDialect, mhlo::MhloDialect,
                                 shape::ShapeDialect, tensor::TensorDialect,
                                 vector::VectorDialect>();
    // Ops inside TiledLoopOps have special handling.
    options.denyOperationInFilter([](Operation* op) {
      return mlir::isa<gml_st::LoopOp>(op->getParentOp());
    });
    // Configure bufferization options for mhlo ops.
    options.addDialectStateInitializer(
        mhlo::MhloDialect::getDialectNamespace(), []() {
          auto dialect_state = std::make_unique<mhlo::MhloBufferizationState>();
          dialect_state->enforce_identity_map_fn = [](Operation* op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_3(mht_3_v, 335, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "lambda");

            // Force identity maps for several ops which don't support memrefs
            // with affine_maps.
            return llvm::any_of(op->getUsers(), [](Operation* user) {
              return isa<gml_st::LoopOp, func::ReturnOp, mhlo::DynamicReshapeOp,
                         tensor::CastOp, tensor::CollapseShapeOp,
                         tensor::ExpandShapeOp>(user);
            });
          };
          return dialect_state;
        });

    if (failed(bufferization::bufferizeOp(getOperation(), options))) {
      signalPassFailure();
      return;
    }

    // Bufferize the remaining IR with dialect conversion. This will disappear
    // eventually once all bufferization is done via BufferizableOpInterface.
    if (failed(runDialectConversionBasedBufferization())) signalPassFailure();
  }

 private:
  LogicalResult runDialectConversionBasedBufferization() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_4(mht_4_v, 361, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "runDialectConversionBasedBufferization");

    RewritePatternSet patterns(&getContext());
    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<
        arith::ArithmeticDialect, complex::ComplexDialect, lmhlo::LmhloDialect,
        AffineDialect, vector::VectorDialect, memref::MemRefDialect,
        func::FuncDialect, tensor::TensorDialect, math::MathDialect>();
    target.addLegalOp<UnrealizedConversionCastOp, gml_st::LoopOp>();
    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addDynamicallyLegalOp<tensor::ExtractSliceOp, tensor::InsertSliceOp>(
        [&](Operation* op) {
          return mlir::isa<gml_st::LoopOp>(op->getParentOp());
        });

    CustomBufferizeTypeConverter converter;
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    // Configure legality and structural patterns.
    bufferization::populateBufferizeMaterializationLegality(target);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    // TODO(herhut): Move this legality configuration to bufferize itself?
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto inputs = op.getFunctionType().getInputs();
      auto results = op.getFunctionType().getResults();
      return converter.isLegal(inputs) && converter.isLegal(results) &&
             converter.isLegal(&op.getBody());
    });
    auto isLegalOp = [&](Operation* op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_5(mht_5_v, 398, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "lambda");
 return converter.isLegal(op); };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(isLegalOp);

    auto isLegalOrInsideTiledLoop = [&](Operation* op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_6(mht_6_v, 404, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "lambda");

      return converter.isLegal(op) ||
             mlir::isa<gml_st::LoopOp>(op->getParentOp());
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
        isLegalOrInsideTiledLoop);
    target
        .addDynamicallyLegalOp<vector::TransferWriteOp, vector::TransferReadOp>(
            isLegalOrInsideTiledLoop);

    return applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

struct TiledLoopBufferizePass
    : public TiledLoopBufferizePassBase<TiledLoopBufferizePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_7(mht_7_v, 423, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "getDependentDialects");

    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_8(mht_8_v, 430, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "runOnOperation");

    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    RewritePatternSet patterns(&getContext());
    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.allowDialectInFilter<shape::ShapeDialect>();
    if (failed(bufferization::bufferizeOp(getOperation(), options))) {
      signalPassFailure();
      return;
    }

    // Bufferize the remaining IR with dialect conversion. This will disappear
    // eventually once all bufferization is done via BufferizableOpInterface.
    if (failed(runDialectConversionBasedBufferization())) signalPassFailure();
  }

 private:
  LogicalResult runDialectConversionBasedBufferization() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_9(mht_9_v, 453, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "runDialectConversionBasedBufferization");

    RewritePatternSet patterns(&getContext());
    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<
        arith::ArithmeticDialect, bufferization::BufferizationDialect,
        complex::ComplexDialect, lmhlo::LmhloDialect, AffineDialect,
        vector::VectorDialect, memref::MemRefDialect, func::FuncDialect,
        tensor::TensorDialect, math::MathDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<tensor::ExtractSliceOp, tensor::InsertSliceOp>();

    CustomBufferizeTypeConverter converter;
    mhlo::RemoveSignTypeConverter remove_sign_converter;

    // Configure bufferize pattern.
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    bufferization::populateBufferizeMaterializationLegality(target);
    populateTiledLoopBufferizePattern(&getContext(), &converter, &patterns);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    // Configure legality.
    auto isLegalOp = [&](Operation* op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_10(mht_10_v, 481, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "lambda");
 return converter.isLegal(op); };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOp);
    target.addDynamicallyLegalOp<func::CallOp, gml_st::LoopOp, gml_st::YieldOp,
                                 LLVM::InlineAsmOp, vector::TransferWriteOp,
                                 vector::TransferReadOp>(isLegalOp);

    return applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

struct FinalBufferizePass : public FinalBufferizePassBase<FinalBufferizePass> {
  // TODO(b/173201243): Move to tablegen.
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_11(mht_11_v, 496, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "getDependentDialects");

    registry.insert<AffineDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, shape::ShapeDialect, tensor::TensorDialect,
                    tf_framework::TFFrameworkDialect, lmhlo::LmhloDialect,
                    arith::ArithmeticDialect, vector::VectorDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }
  // Default alignment_ specified in passes.td
  FinalBufferizePass() = default;

  explicit FinalBufferizePass(uint64_t alignment) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_12(mht_12_v, 513, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "FinalBufferizePass");
 alignment_ = alignment; }

  void runOnOperation() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_13(mht_13_v, 518, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "runOnOperation");

    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    options.bufferAlignment = alignment_;
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.allowDialectInFilter<
        arith::ArithmeticDialect, linalg::LinalgDialect, func::FuncDialect,
        shape::ShapeDialect, tensor::TensorDialect, vector::VectorDialect>();
    if (failed(bufferization::bufferizeOp(getOperation(), options))) {
      signalPassFailure();
      return;
    }

    // Bufferize the remaining IR with dialect conversion. This will disappear
    // eventually once all bufferization is done via BufferizableOpInterface.
    if (failed(runDialectConversionBasedBufferization())) signalPassFailure();
  }

 private:
  LogicalResult runDialectConversionBasedBufferization() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_14(mht_14_v, 543, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "runDialectConversionBasedBufferization");

    auto& context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<
        arith::ArithmeticDialect, bufferization::BufferizationDialect,
        cf::ControlFlowDialect, complex::ComplexDialect, memref::MemRefDialect,
        func::FuncDialect, scf::SCFDialect, tensor::TensorDialect,
        tf_framework::TFFrameworkDialect, AffineDialect, shape::ShapeDialect,
        lmhlo::LmhloDialect, linalg::LinalgDialect, math::MathDialect,
        vector::VectorDialect>();
    target.addLegalOp<FuncOp, ModuleOp>();

    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<tensor::GenerateOp, tensor::ExtractOp,
                        tensor::FromElementsOp, tensor::CastOp, tensor::DimOp,
                        tensor::RankOp, chlo::MinimumBroadcastShapesOp,
                        bufferization::ToTensorOp, bufferization::ToMemrefOp,
                        tensor::ExpandShapeOp, tensor::CollapseShapeOp>();
    bufferization::BufferizeTypeConverter converter;
    auto typesAreLegal = [&converter](Operation* op) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_passDTcc mht_15(mht_15_v, 565, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_pass.cc", "lambda");

      return converter.isLegal(op->getOperandTypes()) &&
             converter.isLegal(op->getResultTypes());
    };
    target.addDynamicallyLegalOp<func::ConstantOp, arith::ConstantOp,
                                 arith::IndexCastOp, arith::SelectOp,
                                 gml_st::LoopOp, gml_st::YieldOp,
                                 tf_framework::JITExecuteOp>(typesAreLegal);

    RewritePatternSet patterns(&getContext());
    populateEliminateBufferizeMaterializationsPatterns(converter, patterns);
    populateExtraBufferizePatterns(&getContext(), &converter, &patterns);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    return applyFullConversion(getOperation(), target, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateComputeOpAndFuncBufferizePass() {
  return std::make_unique<ComputeOpAndFuncBufferizePass>();
}

std::unique_ptr<OperationPass<FuncOp>> CreateTiledLoopBufferizePass() {
  return std::make_unique<TiledLoopBufferizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateFinalBufferizePass() {
  return std::make_unique<FinalBufferizePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> CreateFinalBufferizePass(
    uint64_t alignment) {
  return std::make_unique<FinalBufferizePass>(alignment);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
