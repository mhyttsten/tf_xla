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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_gpuDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_gpuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_gpuDTcc() {
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

// This file implements logic for lowering LHLO dialect to GPU dialect.

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace lmhlo {
namespace {

// A simple translation of LHLO reduce operations to a corresponding gpu
// launch operation. The transformation does no tiling and also only supports
// 1d results.
class LhloReduceToGPULaunchConverter : public OpConversionPattern<ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReduceOp reduce_op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_gpuDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_gpu.cc", "matchAndRewrite");

    auto loc = reduce_op.getLoc();
    // Only support 1d reductions for now.
    int64_t size = 0;
    for (auto result : reduce_op.out()) {
      auto shaped_type = result.getType().dyn_cast<ShapedType>();
      if (!shaped_type || shaped_type.getRank() != 1) {
        return failure();
      }
      auto dim_size = shaped_type.getDimSize(0);
      if (size && size != dim_size) {
        return failure();
      }
      size = dim_size;
    }

    auto reducing_dimension = *reduce_op.dimensions().value_begin<APInt>();

    // Require all inputs to have the same shape.
    int64_t reduce_dim_size = 0;
    for (auto input : reduce_op.inputs()) {
      auto shaped_type = input.getType().dyn_cast<ShapedType>();
      if (!shaped_type || !shaped_type.hasStaticShape()) {
        return failure();
      }
      reduce_dim_size =
          shaped_type.getDimSize(reducing_dimension.getSExtValue());
    }

    // Create a launch that is parallel in the result dimension.
    auto block_size_x = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), size));
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    auto launch_op = rewriter.create<mlir::gpu::LaunchOp>(
        loc, one, one, one, block_size_x, one, one);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&launch_op.body().front());
      auto index = launch_op.getThreadIds().x;

      // Load the initial value and store it to the output.
      for (auto pair : llvm::zip(reduce_op.init_values(), reduce_op.out())) {
        auto init_value =
            rewriter.create<mlir::memref::LoadOp>(loc, std::get<0>(pair));
        rewriter.create<mlir::memref::StoreOp>(
            loc, init_value, std::get<1>(pair), ArrayRef<Value>{index});
      }

      // Insert a loop into the body to compute the reduction. The loop ranges
      // from [0.dim).
      auto zero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
      // TODO(b/137624192) Use dimOp to make it shape independent.
      auto upper = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), reduce_dim_size));
      auto step = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
      auto loop = rewriter.create<mlir::scf::ForOp>(loc, zero, upper, step);

      rewriter.setInsertionPointToStart(loop.getBody());
      // Compute memrefs for the value to reduce. This makes it easier to just
      // inline the body.
      auto output = *reduce_op.out().begin();
      auto resType = MemRefType::get(
          llvm::None, getElementTypeOrSelf(output.getType()),
          makeStridedLinearLayoutMap(llvm::None,
                                     MemRefType::getDynamicStrideOrOffset(),
                                     rewriter.getContext()));
      OpFoldResult offset = launch_op.getThreadIds().x;
      auto oneAttr = rewriter.getI64IntegerAttr(1);
      OpFoldResult size = oneAttr;
      OpFoldResult stride = oneAttr;
      auto accumulator = rewriter.create<memref::SubViewOp>(
          loc, resType, output, offset, size, stride);
      llvm::SmallVector<Value, 4> indexings;
      Value input_buffer = reduce_op.inputs().front();
      auto input_type_rank =
          input_buffer.getType().cast<MemRefType>().getRank();

      Value input = *reduce_op.operand_begin();
      SmallVector<OpFoldResult> offsets = llvm::to_vector<4>(llvm::map_range(
          llvm::seq<int>(0, input_type_rank), [&](int dim) -> OpFoldResult {
            return dim == reducing_dimension ? loop.getInductionVar()
                                             : launch_op.getThreadIds().x;
          }));
      SmallVector<OpFoldResult> sizes(input_type_rank, oneAttr);
      SmallVector<OpFoldResult> strides(input_type_rank, oneAttr);
      auto rhs = rewriter.create<memref::SubViewOp>(
          loc, accumulator.getType(), input, offsets, sizes, strides);

      // Now copy over the actual body of the reduction, leaving out the
      // terminator.
      BlockAndValueMapping mapping;
      mapping.map(reduce_op.body().getArgument(0), accumulator);
      mapping.map(reduce_op.body().getArgument(1), rhs);
      mapping.map(reduce_op.body().getArgument(2), accumulator);
      for (auto& nested : reduce_op.body().front().without_terminator()) {
        auto* clone = rewriter.clone(nested, mapping);
        for (auto pair : llvm::zip(nested.getResults(), clone->getResults())) {
          mapping.map(std::get<0>(pair), std::get<1>(pair));
        }
      }

      // Finally, insert the terminator for the launchOp.
      rewriter.setInsertionPointToEnd(&launch_op.body().front());
      rewriter.create<mlir::gpu::TerminatorOp>(loc);
    }

    rewriter.eraseOp(reduce_op);
    return success();
  };
};

struct LhloLegalizeToGpuPass
    : public LhloLegalizeToGpuPassBase<LhloLegalizeToGpuPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_gpuDTcc mht_1(mht_1_v, 349, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_gpu.cc", "getDependentDialects");

    registry.insert<AffineDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSlhloPStransformsPSlhlo_legalize_to_gpuDTcc mht_2(mht_2_v, 357, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/lhlo/transforms/lhlo_legalize_to_gpu.cc", "runOnOperation");

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, func::FuncDialect,
                           gpu::GPUDialect, scf::SCFDialect, LmhloDialect>();
    target.addIllegalOp<ReduceOp>();
    auto func = getOperation();
    patterns.add<LhloReduceToGPULaunchConverter>(func.getContext());
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLegalizeToGpuPass() {
  return std::make_unique<LhloLegalizeToGpuPass>();
}

}  // namespace lmhlo
}  // namespace mlir
