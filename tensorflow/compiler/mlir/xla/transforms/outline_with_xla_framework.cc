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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSoutline_with_xla_frameworkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSoutline_with_xla_frameworkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSoutline_with_xla_frameworkDTcc() {
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

#include <memory>
#include <stdexcept>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Constants.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/LLVMCommon/Pattern.h"  // from @llvm-project
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/xla_framework.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes_detail.h"

namespace mlir {
namespace mhlo {
namespace {

// Given a FuncOp with only memref args/outputs, create a new function that
// wraps/unwraps xla_framework.buffer types and then calls the original
// function.
//
// For example:
//   func @func_to_outline(%arg0: memref<?xf32>) -> memref<?xf32>
//
// Will generate:
//   func @func_to_outline_xla_framework(%arg0: !xla_framework.buffer)
//     -> !xla_framework.buffer attributes {xla_entry = true} {
//    %0 = xla_framework.buffer_to_mem %arg0 : memref<?xf32>
//    %1 = call @func_to_outline(%0) : (memref<?xf32>) -> memref<?xf32>
//    %2 = xla_framework.mem_to_buffer %1 : memref<?xf32>
//    return %2 : !xla_framework.buffer
//   }
struct OutlineXLAFunc : public RewritePattern {
  explicit OutlineXLAFunc(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(FuncOp::getOperationName(), benefit, context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSoutline_with_xla_frameworkDTcc mht_0(mht_0_v, 240, "", "./tensorflow/compiler/mlir/xla/transforms/outline_with_xla_framework.cc", "OutlineXLAFunc");
}

  static void filterFuncAttributes(ArrayRef<NamedAttribute> attrs,
                                   bool argAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSoutline_with_xla_frameworkDTcc mht_1(mht_1_v, 247, "", "./tensorflow/compiler/mlir/xla/transforms/outline_with_xla_framework.cc", "filterFuncAttributes");

    for (const auto &attr : attrs) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == FunctionOpInterface::getTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (argAttrs && attr.getName() == FuncOp::getArgDictAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSoutline_with_xla_frameworkDTcc mht_2(mht_2_v, 262, "", "./tensorflow/compiler/mlir/xla/transforms/outline_with_xla_framework.cc", "matchAndRewrite");

    auto func = dyn_cast<FuncOp>(op);
    auto ctx = rewriter.getContext();
    auto loc = func.getLoc();
    SmallVector<Location> locs(func.getFunctionType().getNumInputs(), loc);

    // Functions should only be outlined once and should only use memrefs
    if (!func) return failure();
    if (llvm::any_of(op->getOperandTypes(),
                     [](Type t) { return !t.isa<MemRefType>(); }) ||
        op->getNumResults() != 0)
      return failure();
    if (func->hasAttr("outlined")) return failure();
    func->setAttr("outlined", BoolAttr::get(ctx, true));

    // Prepare new func attribute information
    func.setSymNameAttr(mlir::StringAttr::get(ctx, func.getName()));
    SmallVector<Type> operands(func.getFunctionType().getNumInputs(),
                               ::mlir::xla_framework::BufferType::get(ctx));
    SmallVector<Type> result_array(func.getFunctionType().getNumResults(),
                                   ::mlir::xla_framework::BufferType::get(ctx));
    auto func_type = FunctionType::get(ctx, operands, result_array);
    SmallVector<NamedAttribute> attrs;
    filterFuncAttributes(func->getAttrs(), true, attrs);
    SmallVector<DictionaryAttr> arg_attrs;
    func.getAllArgAttrs(arg_attrs);

    // The wrapper function will have the same name but with _xla_framework
    // appended and will be annotated with the attribute "xla_entry".
    auto outline_func =
        rewriter.create<FuncOp>(loc, func.getSymName().str() + "_xla_framework",
                                func_type, attrs, arg_attrs);
    outline_func->setAttr("outlined", BoolAttr::get(ctx, true));
    outline_func->setAttr("xla_entry", BoolAttr::get(ctx, true));
    auto *b = rewriter.createBlock(&outline_func.getBody(), {},
                                   func_type.getInputs(), locs);

    // Unwrap arguments
    SmallVector<Value> args;
    for (const auto &t : llvm::enumerate(func.getFunctionType().getInputs())) {
      args.push_back(rewriter.create<xla_framework::XLABufferToMemOp>(
          loc, t.value(), b->getArgument(t.index())));
    }

    auto call = rewriter.create<func::CallOp>(
        loc, func.getSymName(), func.getFunctionType().getResults(), args);
    // Wrap results
    SmallVector<Value> results;
    for (auto t : call.getResults()) {
      results.push_back(rewriter.create<xla_framework::MemToXLABufferOp>(
          loc, ::mlir::xla_framework::BufferType::get(ctx), t));
    }

    rewriter.create<func::ReturnOp>(loc, results);
    return success();
  }
};

class OutlineWithXLAFrameworkPass
    : public OutlineWithXLAFrameworkBase<OutlineWithXLAFrameworkPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSoutline_with_xla_frameworkDTcc mht_3(mht_3_v, 325, "", "./tensorflow/compiler/mlir/xla/transforms/outline_with_xla_framework.cc", "getDependentDialects");

    registry.insert<xla_framework::XLAFrameworkDialect, mlir::BuiltinDialect>();
  }

 public:
  explicit OutlineWithXLAFrameworkPass() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSoutline_with_xla_frameworkDTcc mht_4(mht_4_v, 333, "", "./tensorflow/compiler/mlir/xla/transforms/outline_with_xla_framework.cc", "OutlineWithXLAFrameworkPass");
}

  void runOnOperation() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSoutline_with_xla_frameworkDTcc mht_5(mht_5_v, 338, "", "./tensorflow/compiler/mlir/xla/transforms/outline_with_xla_framework.cc", "runOnOperation");

    ModuleOp m = getOperation();

    // Populate type conversions.
    MLIRContext *ctx = m.getContext();

    // Populate patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<OutlineXLAFunc>(ctx);
    //  Set target.

    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
      signalPassFailure();
    }
    m->walk([](FuncOp f) {
      if (f->hasAttr("outlined")) f->removeAttr("outlined");
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp> > CreateOutlineWithXLAFrameworkPass() {
  return std::make_unique<OutlineWithXLAFrameworkPass>();
}

}  // namespace mhlo
}  // namespace mlir
