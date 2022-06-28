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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc() {
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

// This file implements conversion of `gml_st.loop` to buffer form.

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

using bufferization::ToMemrefOp;
using bufferization::ToTensorOp;
using gml_st::LoopOp;
using linalg::FillOp;
using linalg::InitTensorOp;
using memref::SubViewOp;
using tensor::ExtractSliceOp;
using tensor::InsertSliceOp;
using vector::TransferReadOp;
using vector::TransferWriteOp;

/// Convert `tensor.extract_slice` to `memref.subview` in-place.
struct BufferizeExtractSliceOp : public OpConversionPattern<ExtractSliceOp> {
  using OpConversionPattern<ExtractSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_0(mht_0_v, 227, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "matchAndRewrite");

    if (!op->getParentOfType<LoopOp>()) return failure();

    rewriter.replaceOpWithNewOp<SubViewOp>(
        op, adaptor.source(), op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    return success();
  }
};

/// Convert `linalg.init_tensor` of `memref.alloc`.
struct BufferizeInitTensorOp : public OpConversionPattern<InitTensorOp> {
  using OpConversionPattern<InitTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InitTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "matchAndRewrite");

    if (!op->getParentOfType<LoopOp>()) return failure();

    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, getTypeConverter()->convertType(op.getType()).cast<MemRefType>(),
        adaptor.sizes());
    return success();
  }
};

bool IsBlockArgOfTiledLoop(Value value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_2(mht_2_v, 259, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "IsBlockArgOfTiledLoop");

  if (auto block_arg = value.dyn_cast<BlockArgument>())
    return isa<LoopOp>(block_arg.getOwner()->getParentOp());
  return false;
}

// Attempts to find an existing `memref.subview` of `destMemRef` in the tiled
// loop. The assumption is that in `gml_st.loop` the tile of the output
// tensor that we read and the tile that we write to are the same.
Value FindExistingSubview(Value destMemRef) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_3(mht_3_v, 271, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "FindExistingSubview");

  if (auto to_memref = destMemRef.getDefiningOp<ToMemrefOp>()) {
    if (auto to_tensor = to_memref.tensor().getDefiningOp<ToTensorOp>()) {
      if (!IsBlockArgOfTiledLoop(to_tensor.memref())) return Value{};
      // Scan through users of the block argument to find `subview` op.
      for (Operation *tensor_user : to_memref.tensor().getUsers()) {
        if (auto another_cast = mlir::dyn_cast<ToMemrefOp>(tensor_user)) {
          for (Operation *memref_user : another_cast.memref().getUsers()) {
            if (auto subview = mlir::dyn_cast<SubViewOp>(memref_user)) {
              if (subview.source() == destMemRef) return subview;
            }
          }
        }
      }
    }
  }
  return Value{};
}

/// Convert `tensor.insert_slice` to `memref.subview` in-place.
struct BufferizeInsertSliceOp : public OpConversionPattern<InsertSliceOp> {
 public:
  using OpConversionPattern<InsertSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InsertSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_4(mht_4_v, 300, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "matchAndRewrite");

    Value sourceMemRef = adaptor.source();
    assert(sourceMemRef.getType().isa<MemRefType>());

    Value destMemRef = adaptor.dest();
    assert(destMemRef.getType().isa<MemRefType>());

    if (!op->getParentOfType<LoopOp>()) return failure();

    Value subview = FindExistingSubview(destMemRef);
    if (!subview) {
      subview = rewriter.create<SubViewOp>(
          op.getLoc(), destMemRef, op.getMixedOffsets(), op.getMixedSizes(),
          op.getMixedStrides());
    }
    rewriter.create<memref::CopyOp>(op.getLoc(), sourceMemRef, subview);
    rewriter.replaceOp(op, destMemRef);
    return success();
  }
};

/// Create linalg op on buffers given the original tensor-based operation and
/// the buffers for the outputs.
static linalg::LinalgOp createLinalgOpOnBuffers(
    ConversionPatternRewriter &rewriter, linalg::LinalgOp linalgOp,
    ValueRange inputs, ValueRange outputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_5(mht_5_v, 328, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "createLinalgOpOnBuffers");

  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  auto *newOp = linalgOp.cloneWithoutRegions(rewriter, linalgOp.getLoc(),
                                             /*resultTypes=*/ArrayRef<Type>{},
                                             newOperands);
  for (auto regions : llvm::zip(linalgOp->getRegions(), newOp->getRegions())) {
    auto &oldRegion = std::get<0>(regions);
    auto &newRegion = std::get<1>(regions);
    rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.begin());
  }
  return newOp;
}

// Bufferize LinalgOps in-place.
struct BufferizeLinalgOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
  using OpInterfaceConversionPattern<
      linalg::LinalgOp>::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::LinalgOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_6(mht_6_v, 353, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "matchAndRewrite");

    if (!op->getParentOfType<LoopOp>()) return failure();

    // GenericOpAdaptor below expects an `operand_segment_sizes` attribute.
    if (!op->hasAttr("operand_segment_sizes")) return failure();

    // TODO(b/199046880): Replace this with LinalgOp::Adaptor or equivalent.
    linalg::GenericOpAdaptor adaptor(operands, op->getAttrDictionary());

    createLinalgOpOnBuffers(rewriter, op, adaptor.inputs(), adaptor.outputs());
    rewriter.replaceOp(op, adaptor.outputs());
    return success();
  }
};

// Convert `gml_st.yield` terminator of `gml_st.loop` to `gml_st.yield` with no
// arguments.
struct BufferizeLinalgYieldOp : public OpConversionPattern<gml_st::YieldOp> {
  using OpConversionPattern<gml_st::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      gml_st::YieldOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_7(mht_7_v, 378, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "matchAndRewrite");

    if (!mlir::dyn_cast<LoopOp>(op->getParentOp()) ||
        adaptor.getOperands().empty())
      return failure();

    rewriter.replaceOpWithNewOp<gml_st::YieldOp>(op);
    return success();
  }
};

// FuncOp-like bufferization pattern for `gml_st.loop` that inserts
// `memref.tensor_load` ops for every memref block argument.
struct BufferizeLoopOp : public OpConversionPattern<LoopOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LoopOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_8(mht_8_v, 398, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "matchAndRewrite");

    if (op.getNumResults() == 0) return failure();

    SmallVector<NamedAttribute> attr_list;
    for (auto &item : adaptor.getAttributes()) {
      attr_list.push_back(item);
    }
    auto newOp = rewriter.create<LoopOp>(op.getLoc(), mlir::TypeRange{},
                                         adaptor.getOperands(), attr_list);
    // Take the region from the old op and put it in the new op.
    rewriter.inlineRegionBefore(op.getLoopBody(), newOp.getLoopBody(),
                                newOp.getLoopBody().end());

    // Convert the type of the entry block of the LoopOp's body.
    if (failed(rewriter.convertRegionTypes(&newOp.getLoopBody(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }

    // Change the clone to use the updated operands. We could have cloned with
    // a BlockAndValueMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());

    rewriter.replaceOp(op, newOp.outputs());
    return success();
  }
};

// TODO(b/199045477): The pattern for vector.transfer_read/write have to be
// moved out of Linalg bufferization to a VectorOps bufferization pass.
struct BufferizeVectorTransferReadOp
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp readOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_9(mht_9_v, 437, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "matchAndRewrite");

    if (readOp.getShapedType().isa<MemRefType>()) return failure();
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        readOp, readOp.getType(), adaptor.getSource(), adaptor.getIndices(),
        adaptor.getPermutationMapAttr(), adaptor.getPadding(),
        adaptor.getMask(),
        adaptor.getInBounds() ? adaptor.getInBoundsAttr() : ArrayAttr());
    return success();
  }
};

struct BufferizeVectorTransferWriteOp
    : public OpConversionPattern<vector::TransferWriteOp> {
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp writeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_10(mht_10_v, 457, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "matchAndRewrite");

    if (writeOp.getShapedType().isa<MemRefType>()) return failure();
    rewriter.create<vector::TransferWriteOp>(
        writeOp.getLoc(), adaptor.getVector(), adaptor.getSource(),
        adaptor.getIndices(), adaptor.getPermutationMapAttr(),
        adaptor.getInBounds() ? adaptor.getInBoundsAttr() : ArrayAttr());
    rewriter.replaceOp(writeOp, adaptor.getSource());
    return success();
  }
};

}  // namespace

void populateTiledLoopBufferizePattern(
    MLIRContext *context, bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSbufferize_tiled_loopDTcc mht_11(mht_11_v, 475, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/bufferize_tiled_loop.cc", "populateTiledLoopBufferizePattern");

  // clang-format off
  patterns->add<
    BufferizeExtractSliceOp,
    BufferizeInitTensorOp,
    BufferizeInsertSliceOp,
    BufferizeLinalgOp,
    BufferizeLinalgYieldOp,
    BufferizeLoopOp,
    BufferizeVectorTransferReadOp,
    BufferizeVectorTransferWriteOp
  >(*converter, context);
  // clang-format on
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
