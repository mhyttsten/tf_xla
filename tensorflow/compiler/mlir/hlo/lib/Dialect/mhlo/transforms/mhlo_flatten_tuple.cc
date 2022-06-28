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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc() {
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

// This file implements logic for flattening tuples in HLO ops.

#include <cassert>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

// Calculates the flatten types of a value.
void FlattenTupleType(Value value, llvm::SmallVectorImpl<Type> &types) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_flatten_tuple.cc", "FlattenTupleType");

  if (!value.getType().isa<TupleType>()) {
    types.push_back(value.getType());
    return;
  }

  // This function doesn't handle nested tuple.
  auto tupleType = value.getType().cast<TupleType>();
  types.append(tupleType.begin(), tupleType.end());
}

// Flattens value into flatten_values.
void FlattenTupleValue(OpBuilder &builder, Location loc, Value value,
                       llvm::SmallVectorImpl<Value> &flatten_values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_flatten_tuple.cc", "FlattenTupleValue");

  if (!value.getType().isa<TupleType>()) {
    flatten_values.push_back(value);
    return;
  }

  // This function doesn't handle nested tuple.
  int flattenIdx = 0;
  auto tupleType = value.getType().cast<TupleType>();
  for (auto childType : tupleType.getTypes()) {
    auto getTupleOp = builder.create<mhlo::GetTupleElementOp>(
        loc, childType, value, builder.getI32IntegerAttr(flattenIdx++));
    flatten_values.push_back(getTupleOp);
  }
}

// FlattenTupleValue and CreateTupleValue is a pair of functions to create and
// flatten tuples in the exact same order. CreateTupleValue returns the result
// of the root TupleOp or given value if the type is not TupleType.
Value CreateTupleValue(OpBuilder &builder, Location loc,
                       const llvm::ArrayRef<Value> &flatten_values,
                       Type tuple_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_flatten_tuple.cc", "CreateTupleValue");

  if (!tuple_type.isa<TupleType>()) {
    assert(flatten_values.size() == 1);
    return flatten_values[0];
  }

  assert(tuple_type.cast<TupleType>().getTypes().size() ==
         flatten_values.size());
  return builder.create<mhlo::TupleOp>(loc, flatten_values);
}

// Flattens the tuples in the region's arguments and returning values.
void FlattenTupleInRegion(Region &region, PatternRewriter &rewriter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc mht_3(mht_3_v, 268, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_flatten_tuple.cc", "FlattenTupleInRegion");

  Location loc = region.getLoc();
  OpBuilder regionOpBuilder(region);

  // Flatten tuples in arguments. The order of arguments must match the order
  // in FlattenTupleType, FlattenTupleValue and CreateTupleValue.
  const int originalNumArgs = region.getNumArguments();
  for (int argIdx : llvm::seq<int>(0, originalNumArgs)) {
    auto argument = region.getArgument(argIdx);

    // Adds new arguments to replace the tuple argument.
    llvm::SmallVector<Type, 4> newTypes;
    llvm::SmallVector<Value, 4> newArguments;
    FlattenTupleType(argument, newTypes);
    for (auto type : newTypes) {
      newArguments.push_back(region.addArgument(type, loc));
    }

    // Replaces uses of the replacing argument.
    auto tupleValue = CreateTupleValue(regionOpBuilder, loc, newArguments,
                                       argument.getType());
    argument.replaceAllUsesWith(tupleValue);
  }
  // Removes old tuple arguments.
  for (int argIdx = originalNumArgs - 1; argIdx >= 0; --argIdx) {
    region.eraseArgument(argIdx);
  }

  // Flatten tuples in results.
  for (auto &block : region.getBlocks()) {
    Operation *terminator = block.getTerminator();
    assert(isa<mhlo::ReturnOp>(terminator));
    auto returnOp = llvm::cast<mhlo::ReturnOp>(terminator);

    // Creates a new ReturnOp with flatten values.
    OpBuilder builder(returnOp);
    llvm::SmallVector<Value, 4> results;
    for (auto operand : returnOp.getOperands()) {
      FlattenTupleValue(builder, returnOp.getLoc(), operand, results);
    }
    builder.create<mhlo::ReturnOp>(loc, results);
    rewriter.eraseOp(returnOp);
  }
}

// Applies tuple flattening patterns to given target. This helper
// function is used to flatten ops recursively.
template <typename T>
void ApplyFlatteningTuplePatterns(T target, MLIRContext *context);

struct FlattenWhileOp : public RewritePattern {
  explicit FlattenWhileOp(MLIRContext *context)
      : RewritePattern(mhlo::WhileOp::getOperationName(), 1, context,
                       {mhlo::WhileOp::getOperationName(),
                        mhlo::TupleOp::getOperationName(),
                        mhlo::GetTupleElementOp::getOperationName()}),
        context(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc mht_4(mht_4_v, 327, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_flatten_tuple.cc", "FlattenWhileOp");
}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc mht_5(mht_5_v, 333, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_flatten_tuple.cc", "matchAndRewrite");

    auto whileOp = cast<mhlo::WhileOp>(op);
    // HLO WhileOp should have two regions: cond and body.
    if (whileOp->getNumRegions() != 2) return failure();

    // Operands of mhlo::WhileOp can be a variadic list of tensors and
    // tuples. Tuples need to be flattened in order to be used in
    // TF::WhileOp. Note that in WhileOp, operand and result types are
    // always the same.
    OpBuilder builder(whileOp);
    llvm::SmallVector<Value, 4> flattenedOperands;
    llvm::SmallVector<Type, 4> flattenedOperandTypes;
    for (auto operand : whileOp->getOperands()) {
      FlattenTupleType(operand, flattenedOperandTypes);
      FlattenTupleValue(builder, whileOp.getLoc(), operand, flattenedOperands);
    }

    // The applyPatternsAndFoldGreedily can't be called on child regions, so
    // creates temporary regions to apply flattening rules recursively.
    auto module = whileOp->getParentOfType<ModuleOp>();
    BlockAndValueMapping mapping;
    Region newCond(module);
    whileOp.cond().cloneInto(&newCond, mapping);
    Region newBody(module);
    whileOp.body().cloneInto(&newBody, mapping);

    // Flattens the tuples in child regions.
    FlattenTupleInRegion(newCond, rewriter);
    FlattenTupleInRegion(newBody, rewriter);

    // There might be WhileOp in child regions, flattens tuple in them too.
    ApplyFlatteningTuplePatterns<MutableArrayRef<Region>>(newCond, context);
    ApplyFlatteningTuplePatterns<MutableArrayRef<Region>>(newBody, context);

    // Creates a new mhlo::WhileOp with no tuples.
    auto newWhile = builder.create<mhlo::WhileOp>(
        whileOp.getLoc(), flattenedOperandTypes, flattenedOperands);
    newCond.cloneInto(&newWhile.cond(), mapping);
    newBody.cloneInto(&newWhile.body(), mapping);

    // Replaces uses of the old WhileOp.
    auto newResultIter = newWhile.result_begin();
    for (auto oldResult : whileOp.getResults()) {
      llvm::SmallVector<Type, 4> flattenedTypes;
      FlattenTupleType(oldResult, flattenedTypes);
      llvm::SmallVector<Value, 4> flattenedResults;
      while (flattenedResults.size() < flattenedTypes.size()) {
        assert(newResultIter != newWhile->result_end());
        flattenedResults.push_back(*newResultIter++);
      }
      auto tupleValue = CreateTupleValue(builder, whileOp.getLoc(),
                                         flattenedResults, oldResult.getType());
      oldResult.replaceAllUsesWith(tupleValue);
    }
    rewriter.eraseOp(whileOp);
    return success();
  }

 private:
  MLIRContext *context;
};

template <typename T>
void ApplyFlatteningTuplePatterns(T target, MLIRContext *context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc mht_6(mht_6_v, 399, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_flatten_tuple.cc", "ApplyFlatteningTuplePatterns");

  RewritePatternSet patterns(context);
  patterns.add<FlattenWhileOp>(context);
  (void)applyPatternsAndFoldGreedily(target, std::move(patterns));
}

class FlattenTuplePass : public FlattenTuplePassBase<FlattenTuplePass> {
 public:
  void runOnOperation() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSmhlo_flatten_tupleDTcc mht_7(mht_7_v, 410, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/mhlo_flatten_tuple.cc", "runOnOperation");

    MLIRContext *ctx = &getContext();
    ApplyFlatteningTuplePatterns(getOperation(), ctx);
  }
};
}  // end namespace

static PassRegistration<FlattenTuplePass> pass;

std::unique_ptr<OperationPass<FuncOp>> createFlattenTuplePass() {
  return std::make_unique<FlattenTuplePass>();
}

}  // end namespace mhlo
}  // end namespace mlir
