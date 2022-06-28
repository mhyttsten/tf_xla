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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc() {
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

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

// Prepends argument type list of the function with an OpKernelContextType arg.
class FuncOpConverter : public OpConversionPattern<FuncOp> {
 public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FuncOp func, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_tf_framework.cc", "matchAndRewrite");

    // Convert function arguments using the provided TypeConverter.
    auto func_type = func.getFunctionType();
    TypeConverter::SignatureConversion conversion(func_type.getNumInputs());

    conversion.addInputs(OpKernelContextType::get(rewriter.getContext()));
    for (auto arg_type : llvm::enumerate(func_type.getInputs())) {
      conversion.addInputs(arg_type.index(), arg_type.value());
    }

    rewriter.applySignatureConversion(&func.getBody(), conversion);

    // Update the signature of the function.
    rewriter.updateRootInPlace(func, [&] {
      func.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                            func_type.getResults()));
    });
    return success();
  }
};

llvm::Optional<Value> FindOpKernelContext(Operation *op) {
  auto func = op->getParentOfType<FuncOp>();
  if (func.getNumArguments() == 0) {
    return llvm::None;
  }
  Value ctx = func.getArgument(0);
  if (!ctx.getType().isa<OpKernelContextType>()) {
    return llvm::None;
  }
  return ctx;
}

// Converts std.alloc to tf_framework.alloc_raw using OpKernelContextType arg of
// the parent function.
struct AllocOpConverter : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::AllocOp alloc, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc mht_1(mht_1_v, 250, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_tf_framework.cc", "matchAndRewrite");

    llvm::Optional<Value> ctx = FindOpKernelContext(alloc);
    if (!ctx) return failure();

    // Symbolic operands that bind to the symbols of the memref's layout map are
    // not supported by TFAllocOp.
    if (!alloc.symbolOperands().empty()) {
      return failure();
    }
    auto reuse_input_candidates = alloc->getAttrOfType<ArrayAttr>(
        TFAllocOp::kReuseInputCandidatesAttrName);
    auto reuse_output_index =
        alloc->getAttrOfType<IntegerAttr>(TFAllocOp::kReuseOutputAttrName);
    Value buffer = rewriter.replaceOpWithNewOp<TFAllocOp>(
        alloc, alloc.getType(), *ctx, adaptor.getOperands(),
        reuse_input_candidates, reuse_output_index);
    Location loc = buffer.getLoc();
    Value cond = rewriter.create<IsValidMemRefOp>(
        loc, rewriter.getIntegerType(1), buffer);
    rewriter.create<TFAssertOp>(loc, *ctx, cond, ErrorCode::RESOURCE_EXHAUSTED,
                                "failed to allocate memory");
    return success();
  }
};

// Converts std.dealloc to tf_framework.dealloc_raw using OpKernelContextType
// arg of the parent function.
struct DeallocOpConverter : public OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::DeallocOp dealloc, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc mht_2(mht_2_v, 285, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_tf_framework.cc", "matchAndRewrite");

    llvm::Optional<Value> ctx = FindOpKernelContext(dealloc);
    if (!ctx) return failure();

    // Operand with no layout is expected.
    auto operand_memref_type = dealloc.memref().getType().cast<MemRefType>();
    if (!operand_memref_type.getLayout().isIdentity()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<TFDeallocOp>(dealloc, *ctx, adaptor.memref());
    return success();
  }
};

// Converts std.assert to tf_framework.assert with using OpKernelContextType
// arg of the parent function.
struct AssertOpConverter : public OpConversionPattern<cf::AssertOp> {
 public:
  using OpConversionPattern<cf::AssertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cf::AssertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc mht_3(mht_3_v, 310, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_tf_framework.cc", "matchAndRewrite");

    llvm::Optional<Value> ctx = FindOpKernelContext(op);
    if (!ctx) return failure();
    rewriter.replaceOpWithNewOp<TFAssertOp>(op, *ctx, adaptor.getArg(),
                                            ErrorCode::INVALID_ARGUMENT,
                                            adaptor.getMsg());
    return success();
  }
};

// Amends `tf_framework.jit_execute` with the newly introduced OpKernelContext.
struct JITExecuteOpConverter : public OpConversionPattern<JITExecuteOp> {
  using OpConversionPattern<JITExecuteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      JITExecuteOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc mht_4(mht_4_v, 329, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_tf_framework.cc", "matchAndRewrite");

    llvm::Optional<Value> ctx = FindOpKernelContext(op);
    if (!ctx) return failure();
    rewriter.replaceOpWithNewOp<JITExecuteOp>(op, op.getResultTypes(), *ctx,
                                              op.callable(), op.operands());
    return success();
  }
};

// Amends `tf_framework.jit_compile_from_str` with the newly introduced
// OpKernelContext.
struct JITCompileFromStrOpConverter
    : public OpConversionPattern<JITCompileFromStrOp> {
  using OpConversionPattern<JITCompileFromStrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      JITCompileFromStrOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc mht_5(mht_5_v, 349, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_tf_framework.cc", "matchAndRewrite");

    llvm::Optional<Value> ctx = FindOpKernelContext(op);
    if (!ctx) return failure();
    rewriter.replaceOpWithNewOp<JITCompileFromStrOp>(
        op, rewriter.getType<JITCallableType>(), *ctx, op->getAttrs());
    return success();
  }
};

}  // namespace

void PopulateEmbedTFFrameworkAssertPattern(RewritePatternSet *patterns) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc mht_6(mht_6_v, 363, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_tf_framework.cc", "PopulateEmbedTFFrameworkAssertPattern");

  patterns->add<AssertOpConverter>(patterns->getContext());
}

void PopulateEmbedTFFrameworkPatterns(RewritePatternSet *patterns) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSembed_tf_frameworkDTcc mht_7(mht_7_v, 370, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/embed_tf_framework.cc", "PopulateEmbedTFFrameworkPatterns");

  // clang-format off
  patterns->add<
      AllocOpConverter,
      AssertOpConverter,
      DeallocOpConverter,
      FuncOpConverter,
      JITCompileFromStrOpConverter,
      JITExecuteOpConverter>(patterns->getContext());
  // clang-format on
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
