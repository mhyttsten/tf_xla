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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_arithmeticDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_arithmeticDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_arithmeticDTcc() {
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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

struct RngGetAndUpdateStatePattern
    : public OpConversionPattern<mhlo::XlaRngGetAndUpdateStateOp> {
  using OpConversionPattern<
      mhlo::XlaRngGetAndUpdateStateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::XlaRngGetAndUpdateStateOp op,
      XlaRngGetAndUpdateStateOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_arithmeticDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_arithmetic.cc", "matchAndRewrite");

    // Get various type related information
    auto loc = op->getLoc();

    const auto global_name = rewriter.getStringAttr("rng_state");
    constexpr auto initial_seed = 0x7012395ull;
    auto seed_type = rewriter.getIntegerType(128);
    auto memref_type = MemRefType::get({}, seed_type);

    auto result_type = op.getType();
    auto word_size = result_type.getElementType().getIntOrFloatBitWidth();
    auto smaller_int_type = rewriter.getIntegerType(word_size);
    auto num_elements = result_type.getNumElements();

    // Get or define the global variable
    auto* global_op =
        mlir::SymbolTable::lookupNearestSymbolFrom(op, global_name);
    if (!global_op) {
      auto* parent = mlir::SymbolTable::getNearestSymbolTable(op);
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&parent->getRegions().front().front());

      const auto priv = rewriter.getStringAttr("private");
      auto initial_value = mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({}, seed_type),
          rewriter.getIntegerAttr(seed_type, initial_seed));
      global_op =
          rewriter.create<memref::GlobalOp>(loc, global_name, priv, memref_type,
                                            initial_value, /*constant=*/false,
                                            /*alignment=*/IntegerAttr());
    }
    assert(isa<memref::GlobalOp>(global_op) &&
           "rng_state was defined somewhere else, not as a global op");

    // Get and update
    Value rng_state =
        rewriter.create<memref::GetGlobalOp>(loc, memref_type, global_name);
    Value old_val = rewriter.create<memref::LoadOp>(loc, rng_state);
    Value delta = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(seed_type,
                                     static_cast<int64_t>(adaptor.delta())));
    Value new_val = rewriter.create<arith::AddIOp>(loc, old_val, delta);
    (void)rewriter.create<memref::StoreOp>(loc, new_val, rng_state);

    // Create the proper return type by packing the old seed into a tensor
    SmallVector<Value> pieces;
    for (int i = (num_elements - 1) * word_size; i >= 0; i -= word_size) {
      Value shift_distance = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(seed_type, i));
      pieces.push_back(rewriter.create<arith::TruncIOp>(
          loc, smaller_int_type,
          rewriter.create<arith::ShRUIOp>(loc, old_val, shift_distance)));
    }

    // Obtain a tensor with the correct shape and bit widths but the incorrect
    // integer signedness, then cast the tensor to the correct signedness to
    // ensure that unrealized casts will successfully lower later.
    Value result_tensor = rewriter.create<tensor::FromElementsOp>(
        loc,
        mlir::RankedTensorType::get(result_type.getShape(), smaller_int_type),
        pieces);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, result_type,
                                                            result_tensor);
    return success();
  }
};

struct HloLegalizeToArithmeticPass
    : public HloLegalizeToArithmeticPassBase<HloLegalizeToArithmeticPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_arithmeticDTcc mht_1(mht_1_v, 285, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_arithmetic.cc", "getDependentDialects");

    registry.insert<arith::ArithmeticDialect, memref::MemRefDialect,
                    tensor::TensorDialect>();
  }

 public:
  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_arithmeticDTcc mht_2(mht_2_v, 294, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_arithmetic.cc", "runOnOperation");

    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    populateHLOToArithmeticConversionPatterns(&patterns);

    target.addIllegalOp<XlaRngGetAndUpdateStateOp>();
    target.addLegalDialect<arith::ArithmeticDialect, BuiltinDialect,
                           memref::MemRefDialect, tensor::TensorDialect>();

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

void populateHLOToArithmeticConversionPatterns(RewritePatternSet* patterns) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPShlo_legalize_to_arithmeticDTcc mht_3(mht_3_v, 316, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/hlo_legalize_to_arithmetic.cc", "populateHLOToArithmeticConversionPatterns");

  patterns->add<RngGetAndUpdateStatePattern>(patterns->getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToArithmeticPass() {
  return std::make_unique<HloLegalizeToArithmeticPass>();
}

}  // namespace mhlo
}  // namespace mlir
