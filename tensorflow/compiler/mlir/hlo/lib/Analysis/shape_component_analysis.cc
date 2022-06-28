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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc() {
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

#include "mlir-hlo/Analysis/shape_component_analysis.h"

#include <algorithm>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

using SymbolicShapeConstraintsMap =
    ShapeComponentAnalysis::SymbolicShapeConstraintsMap;
using ShapeOrValueInfo = ShapeComponentAnalysis::ShapeOrValueInfo;
using Symbol = ShapeComponentAnalysis::Symbol;
using SymbolicExpr = ShapeComponentAnalysis::SymbolicExpr;
using SymbolicExprsMap = ShapeComponentAnalysis::SymbolicExprsMap;

namespace {
// Shape visitor. This implements a symbolic interpreter for MHLO with some
// shape and tensor dialect ops mixed in. We are interested in shapes (e.g., the
// dimensions of a tensor) and values (e.g, the elements of a shape tensor). The
// goal is to assign every component of a shape or value either a symbol, a
// constant, or a symbolic expression. We propagate these symbolic expressions
// through the various operations. Later optimization passes can use this
// information for optimizations, e.g., exploiting the equality of dimensions.
//
// The visitation happens in two phases:
//   1. Find the sources of a value's shape or value. This climbs up the
//      operations from a given value until an unknown op or a function argument
//      is found. These sources are assigned the initial symbols for each of
//      their components.
//   2. Propagate the initial symbols downwards. This builds symbolic
//      expressions so users of the analysis can pattern match things like
//      "two dimensions are multiplied".
//
// Conceptually, this is defined recursively. For each op, we compute the
// required shape or value information for the operands and then derive the
// resulting symbolic expression.
struct ShapeVisitor {
  ShapeVisitor(SymbolicExprsMap *symbolicExprsMap,
               SymbolicShapeConstraintsMap *symbolicShapeConstraintsMap)
      : symbolicExprsMap(symbolicExprsMap),
        symbolicShapeConstraintsMap(symbolicShapeConstraintsMap) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_0(mht_0_v, 233, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "ShapeVisitor");
}

  void visit(ShapeOrValueInfo requestedInfo) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_1(mht_1_v, 238, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "visit");

    backwards_worklist.push_back(requestedInfo);

    // First, we climb up the operations so we get the set of all ops taking
    // part in this shape or value computation. An alternative would be
    // analyzing everything eagerly. This backwards pass allows us to be lazy.
    while (!backwards_worklist.empty()) {
      // Skip if already processed.
      ShapeOrValueInfo transitivelyRequestedInfo =
          backwards_worklist.pop_back_val();
      if (symbolicExprsMap->count(transitivelyRequestedInfo)) continue;

      // Skip irrelevant cases early.
      Value value = transitivelyRequestedInfo.value();
      Type ty = value.getType();
      if (!ty.isIntOrIndexOrFloat() && !ty.isa<RankedTensorType>()) continue;

      // Handle shapes.
      if (transitivelyRequestedInfo.isShapeInfo()) {
        if (value.getDefiningOp<shape::AssumingOp>()) {
          backwardAssumingShape(value);
        } else if (auto bcast =
                       value.getDefiningOp<mhlo::DynamicBroadcastInDimOp>()) {
          backwardDynamicBroadcastInDimShape(bcast);
        } else if (auto reshape =
                       value.getDefiningOp<mhlo::DynamicReshapeOp>()) {
          backwardDynamicReshapeShape(reshape);
        } else if (value.getDefiningOp<mhlo::ReduceOp>()) {
          backwardReduceShape(value);
        } else if (auto transpose = value.getDefiningOp<mhlo::TransposeOp>()) {
          backwardTransposeShape(transpose);
        } else if (auto select = value.getDefiningOp<mhlo::SelectOp>()) {
          backwardSelectShape(select);
        } else if (auto arg = value.dyn_cast<BlockArgument>()) {
          backwardBlockArgumentShape(arg);
        } else if (value.getDefiningOp() &&
                   value.getDefiningOp()
                       ->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
          backwardSameOperandsAndResultShape(value);
        } else {
          backwardUnknownShape(value);
        }
        continue;
      }

      // Skip irrelevant cases early.
      auto ranked_ty = ty.dyn_cast<RankedTensorType>();
      bool is_possibly_interesting_scalar = ty.isIntOrIndex();
      bool is_possibly_interesting_tensor =
          ranked_ty && ranked_ty.getRank() <= 1 && ranked_ty.hasStaticShape();
      if (!is_possibly_interesting_scalar && !is_possibly_interesting_tensor) {
        continue;
      }

      // Handle values.
      assert(transitivelyRequestedInfo.isValueInfo() &&
             "Expect value info at this point.");
      if (auto shapeof = value.getDefiningOp<shape::ShapeOfOp>()) {
        backwardShapeOf(shapeof);
      } else if (auto bcast = value.getDefiningOp<shape::BroadcastOp>()) {
        backwardBroadcast(bcast);
      } else if (auto num_elements =
                     value.getDefiningOp<shape::NumElementsOp>()) {
        backwardNumElements(num_elements);
      } else if (auto dim = value.getDefiningOp<tensor::DimOp>()) {
        backwardDim(dim);
      } else if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        backwardIndexCast(cast);
      } else if (auto fromElements =
                     value.getDefiningOp<tensor::FromElementsOp>()) {
        backwardTensorFromElements(fromElements);
      } else if (auto extract = value.getDefiningOp<tensor::ExtractOp>()) {
        backwardTensorExtract(extract);
      } else if (auto add = value.getDefiningOp<mhlo::AddOp>()) {
        backwardBinOp(add);
      } else if (auto mul = value.getDefiningOp<mhlo::MulOp>()) {
        backwardBinOp(mul);
      } else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        backwardBinOp(add);
      } else if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
        backwardBinOp(mul);
      } else if (auto concat = value.getDefiningOp<mhlo::ConcatenateOp>()) {
        backwardConcatenate(concat);
      } else if (auto reshape = value.getDefiningOp<mhlo::ReshapeOp>()) {
        backwardReshape(reshape);
      } else if (auto slice = value.getDefiningOp<mhlo::SliceOp>()) {
        backwardSlice(slice);
      } else if (matchPattern(value, m_Constant())) {
        backwardConstant(value);
      } else {
        backwardUnknown(value);
      }
    }

    // Second, we walk down from the defs to the uses, building symbolic
    // expressions for shape and value components.
    while (!forwards_worklist.empty()) {
      auto transitivelyRequestedInfo = forwards_worklist.pop_back_val();

      // Skip if already processed.
      if (symbolicExprsMap->count(transitivelyRequestedInfo)) continue;

      // Handle shapes.
      Value value = transitivelyRequestedInfo.value();
      if (!transitivelyRequestedInfo.isValueInfo()) {
        if (value.getDefiningOp<shape::AssumingOp>()) {
          forwardAssumingShape(value);
        } else if (auto broadcast =
                       value.getDefiningOp<mhlo::DynamicBroadcastInDimOp>()) {
          forwardDynamicBroadcastInDimShape(broadcast);
        } else if (auto reshape =
                       value.getDefiningOp<mhlo::DynamicReshapeOp>()) {
          forwardDynamicReshapeShape(reshape);
        } else if (value.getDefiningOp<mhlo::ReduceOp>()) {
          forwardReduceShape(value);
        } else if (auto transpose = value.getDefiningOp<mhlo::TransposeOp>()) {
          forwardTransposeShape(transpose);
        } else if (auto select = value.getDefiningOp<mhlo::SelectOp>()) {
          forwardSelectShape(select);
        } else if (value.getDefiningOp() &&
                   value.getDefiningOp()
                       ->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
          forwardSameOperandsShape(value);
        } else {
          forwardUnknownShape(value);
        }
        continue;
      }

      // Handle values.
      assert(transitivelyRequestedInfo.isValueInfo() &&
             "Expect value info at this point.");
      if (auto shapeof = value.getDefiningOp<shape::ShapeOfOp>()) {
        forwardShapeOf(shapeof);
      } else if (auto bcast = value.getDefiningOp<shape::BroadcastOp>()) {
        forwardBroadcast(bcast);
      } else if (auto num_elements =
                     value.getDefiningOp<shape::NumElementsOp>()) {
        forwardNumElements(num_elements);
      } else if (auto dim = value.getDefiningOp<tensor::DimOp>()) {
        forwardDim(dim);
      } else if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        forwardIndexCast(cast);
      } else if (auto fromElements =
                     value.getDefiningOp<tensor::FromElementsOp>()) {
        forwardTensorFromElements(fromElements);
      } else if (auto extract = value.getDefiningOp<tensor::ExtractOp>()) {
        forwardTensorExtract(extract);
      } else if (auto add = value.getDefiningOp<mhlo::AddOp>()) {
        forwardBinOp(add, [](AffineExpr a, AffineExpr b) { return a + b; });
      } else if (auto mul = value.getDefiningOp<mhlo::MulOp>()) {
        forwardBinOp(mul, [](AffineExpr a, AffineExpr b) { return a * b; });
      } else if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        forwardBinOp(add, [](AffineExpr a, AffineExpr b) { return a + b; });
      } else if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
        forwardBinOp(mul, [](AffineExpr a, AffineExpr b) { return a * b; });
      } else if (auto concat = value.getDefiningOp<mhlo::ConcatenateOp>()) {
        forwardConcatenate(concat);
      } else if (auto reshape = value.getDefiningOp<mhlo::ReshapeOp>()) {
        forwardReshape(reshape);
      } else if (auto slice = value.getDefiningOp<mhlo::SliceOp>()) {
        forwardSlice(slice);
      } else if (matchPattern(value, m_Constant())) {
        forwardConstant(value);
      } else {
        forwardUnknown(value);
      }
    }
  }

 private:
  // ===
  // Functions that traverse the shapes of operations.
  // ===

  void backwardAssumingShape(Value op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_2(mht_2_v, 416, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardAssumingShape");

    auto assumingOp = op.getDefiningOp<shape::AssumingOp>();
    auto number = op.cast<OpResult>().getResultNumber();
    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(
        cast<shape::AssumingYieldOp>(
            assumingOp.getDoRegion().back().getTerminator())
            .getOperand(number)));
  }
  void forwardAssumingShape(Value op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_3(mht_3_v, 428, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardAssumingShape");

    auto assumingOp = op.getDefiningOp<shape::AssumingOp>();
    auto number = op.cast<OpResult>().getResultNumber();
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    dims = lookup(ShapeOrValueInfo::getShapeInfoOf(
        cast<shape::AssumingYieldOp>(
            assumingOp.getDoRegion().back().getTerminator())
            .getOperand(number)));
  }
  void backwardBroadcast(shape::BroadcastOp op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_4(mht_4_v, 440, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardBroadcast");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    for (Value s : op.getShapes())
      backwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(s));
  }
  void forwardBroadcast(shape::BroadcastOp op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_5(mht_5_v, 448, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardBroadcast");

    auto *ctx = op.getContext();

    // Get operands' info.
    SmallVector<ArrayRef<SymbolicExpr>> args_info =
        llvm::to_vector(llvm::map_range(op.getShapes(), [&](Value s) {
          return lookup(ShapeOrValueInfo::getValueInfoOf(s));
        }));

    // Determine broadcasted rank.
    size_t rank = 0;
    for (auto &info : args_info) rank = std::max(rank, info.size());

    // Evaluate broadcast per result dimension.
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    for (size_t i = 0; i < rank; ++i) {
      // Init with neural element.
      SymbolicExpr bcasted_expr;
      bcasted_expr.expr = getAffineConstantExpr(1, ctx);

      // Consider all the operands.
      for (auto &info : args_info) {
        // Find corresponding symbolic expression for the ith result dimension,
        // if the operand contributes.
        size_t arg_rank = info.size();
        if (i + arg_rank < rank) continue;
        size_t j = i + arg_rank - rank;
        SymbolicExpr expr = info[j];

        // One dimensions are neutral.
        if (expr.isConstant(1)) continue;

        // If a dimension is known not to be 1, we can use this expression.
        if (expr.isKnownNotOne()) {
          bcasted_expr = expr;
          break;
        }

        // If all other dimensions were neutral, try using this expression.
        if (bcasted_expr.isConstant(1)) {
          bcasted_expr = expr;
          continue;
        }

        // If we have contradicting expressions, give up and create a new
        // symbol.
        if (bcasted_expr != expr) {
          bcasted_expr.expr = getAffineSymbolExpr(0, ctx);
          bcasted_expr.symbols = {{ShapeOrValueInfo::getValueInfoOf(op), i}};
          break;
        }
      }

      dims.push_back(bcasted_expr);
    }
    assert(dims.size() == rank && "expect one expression per dimension");
  }
  void backwardDynamicBroadcastInDimShape(mhlo::DynamicBroadcastInDimOp op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_6(mht_6_v, 508, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardDynamicBroadcastInDimShape");

    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.output_dimensions()));
  }
  void forwardDynamicBroadcastInDimShape(mhlo::DynamicBroadcastInDimOp op) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_7(mht_7_v, 516, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardDynamicBroadcastInDimShape");

    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    dims = lookup(ShapeOrValueInfo::getValueInfoOf(op.output_dimensions()));
  }
  void backwardDynamicReshapeShape(mhlo::DynamicReshapeOp op) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_8(mht_8_v, 523, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardDynamicReshapeShape");

    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.output_shape()));
  }
  void forwardDynamicReshapeShape(mhlo::DynamicReshapeOp op) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_9(mht_9_v, 531, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardDynamicReshapeShape");

    auto ranked_ty = op.getResult().getType().cast<RankedTensorType>();
    auto shape_dims =
        lookup(ShapeOrValueInfo::getValueInfoOf(op.output_shape()));
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    dimsFromStaticShape(ranked_ty, shape_dims, &dims);
  }
  void backwardReduceShape(Value op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_10(mht_10_v, 541, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardReduceShape");

    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    auto reduceOp = op.getDefiningOp<mhlo::ReduceOp>();
    if (reduceOp.inputs().size() == 1)
      backwards_worklist.push_back(
          ShapeOrValueInfo::getShapeInfoOf(reduceOp.inputs().back()));
  }
  void forwardReduceShape(Value op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_11(mht_11_v, 551, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardReduceShape");

    auto reduceOp = op.getDefiningOp<mhlo::ReduceOp>();
    if (reduceOp.inputs().size() != 1) return forwardUnknownShape(op);
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    for (const auto &dim : llvm::enumerate(lookup(
             ShapeOrValueInfo::getShapeInfoOf(reduceOp.inputs().back())))) {
      if (!llvm::is_contained(reduceOp.dimensions(), dim.index()))
        dims.push_back(dim.value());
    }
  }
  void backwardTransposeShape(mhlo::TransposeOp op) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_12(mht_12_v, 564, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardTransposeShape");

    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getShapeInfoOf(op.operand()));
  }
  void forwardTransposeShape(mhlo::TransposeOp op) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_13(mht_13_v, 572, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardTransposeShape");

    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    auto in = lookup(ShapeOrValueInfo::getShapeInfoOf(op.operand()));
    auto elem = op.permutation().cast<DenseIntElementsAttr>();
    for (const auto &val : elem) dims.push_back(in[val.getZExtValue()]);
  }
  void backwardSelectShape(mhlo::SelectOp op) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_14(mht_14_v, 581, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardSelectShape");

    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getShapeInfoOf(op.on_true()));
  }
  void forwardSelectShape(mhlo::SelectOp op) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_15(mht_15_v, 589, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardSelectShape");

    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(op));
    // Forward the `on_true` operand, it has the same shape as the output.
    dims = lookup(ShapeOrValueInfo::getShapeInfoOf(op.on_true()));
  }
  void backwardSameOperandsAndResultShape(Value v) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_16(mht_16_v, 597, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardSameOperandsAndResultShape");

    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(v));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getShapeInfoOf(v.getDefiningOp()->getOperand(0)));
  }
  void forwardSameOperandsShape(Value v) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_17(mht_17_v, 605, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardSameOperandsShape");

    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(v));
    dims = lookup(
        ShapeOrValueInfo::getShapeInfoOf(v.getDefiningOp()->getOperand(0)));
  }
  void backwardBlockArgumentShape(BlockArgument argument) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_18(mht_18_v, 613, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardBlockArgumentShape");

    // JitRT uses jitrt.symbolic_shape to describe identical dimensions. Make
    // use of that when it exists.
    //
    // Example:
    //   func @compute(
    //     %arg0: tensor<?xf32> {jitrt.symbolic_shape = dense<-2> :
    //     tensor<1xi64>}, %arg1: tensor<?xf32> {jitrt.symbolic_shape =
    //     dense<-2> : tensor<1xi64>})
    //   } { ... }
    //
    // Symbolic shape is a negative value smaller than `-1`. The concrete value
    // is not known at compile time, and in this particular example it is only
    // known that both arguments have the same shape.
    //
    // TODO(ezhulenev): Add symbolic shape attribute verifier to the jitrt
    // dialect.
    if (auto func =
            dyn_cast_or_null<FuncOp>(argument.getOwner()->getParentOp())) {
      if (auto shape = func.getArgAttrOfType<DenseIntElementsAttr>(
              argument.getArgNumber(), "jitrt.symbolic_shape")) {
        auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(argument));
        auto id = getAffineSymbolExpr(0, argument.getContext());
        for (const auto &symbol : llvm::enumerate(shape.getValues<ssize_t>())) {
          dims.emplace_back();
          auto &dim = dims.back();
          if (symbol.value() >= 0) {
            dim.expr =
                getAffineConstantExpr(symbol.value(), argument.getContext());
          } else {
            auto it = symbolicShapeConstraintsMap->try_emplace(
                symbol.value(),
                Symbol{ShapeOrValueInfo::getShapeInfoOf(argument),
                       symbol.index()});
            dim.symbols.push_back(it.first->second);
            dim.expr = id;
          }
        }
        return;
      }
    }
    forwardUnknownShape(argument);
  }
  void backwardUnknownShape(Value v) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_19(mht_19_v, 659, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardUnknownShape");

    forwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(v));
  }
  void forwardUnknownShape(Value v) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_20(mht_20_v, 665, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardUnknownShape");

    auto ranked_ty = v.getType().dyn_cast<RankedTensorType>();
    if (!ranked_ty) return;
    auto id = getAffineSymbolExpr(0, v.getContext());
    auto &dims = insert(ShapeOrValueInfo::getShapeInfoOf(v));
    return dimsFromStaticShape(
        ranked_ty,
        [&](size_t i) {
          SymbolicExpr d;
          d.symbols.push_back({ShapeOrValueInfo::getShapeInfoOf(v), i});
          d.expr = id;
          return d;
        },
        &dims);
  }

  // ===
  // Functions that traverse values. These can be shape tensors (e.g., of type
  // tensor<3xindex>) or interesting scalars (e.g., of type index).
  // ===

  void backwardShapeOf(shape::ShapeOfOp op) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_21(mht_21_v, 689, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardShapeOf");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op.getArg()));
  }
  void forwardShapeOf(shape::ShapeOfOp op) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_22(mht_22_v, 696, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardShapeOf");

    auto ranked_ty = op.getArg().getType().cast<RankedTensorType>();
    auto arg = lookup(ShapeOrValueInfo::getShapeInfoOf(op.getArg()));
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    return dimsFromStaticShape(ranked_ty, arg, &dims);
  }
  void backwardNumElements(shape::NumElementsOp op) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_23(mht_23_v, 705, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardNumElements");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.getShape()));
  }
  void forwardNumElements(shape::NumElementsOp op) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_24(mht_24_v, 713, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardNumElements");

    auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.getShape()));

    // Accumulate product symbolically and concrete where possible.
    int64_t concrete_product = 1;
    SymbolicExpr dim;
    for (auto &it : in) {
      // For constant expressions, we can accumulate a concrete product.
      if (auto cexpr = it.expr.dyn_cast<AffineConstantExpr>()) {
        assert(cexpr.getValue() > 0 && "shape value must be positive");
        concrete_product *= cexpr.getValue();
        continue;
      }

      // Simply copy the first sybolic factor.
      if (!dim.expr) {
        dim = it;
        continue;
      }

      // Multiply remaining symbolic factors.
      dim.expr = dim.expr *
                 it.expr.shiftSymbols(dim.symbols.size(), it.symbols.size());
      dim.symbols.append(it.symbols);
    }

    // Combine concrete and symbolic product.
    if (concrete_product != 1 || !dim.expr) {
      auto cexpr = getAffineConstantExpr(concrete_product, op.getContext());
      if (dim.expr)
        dim.expr = cexpr * dim.expr;
      else
        dim.expr = cexpr;
    }

    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    dims.push_back(dim);
  }
  void backwardDim(tensor::DimOp op) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_25(mht_25_v, 754, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardDim");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getShapeInfoOf(op.source()));
  }
  void forwardDim(tensor::DimOp op) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_26(mht_26_v, 761, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardDim");

    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    if (auto index = op.index().getDefiningOp<arith::ConstantOp>()) {
      int64_t i = index.getValue().cast<IntegerAttr>().getInt();
      auto in = lookup(ShapeOrValueInfo::getShapeInfoOf(op.source()));
      dims.push_back({in[i].symbols, in[i].expr});
    } else {
      forwardUnknown(op);
    }
  }
  template <typename Op>
  void backwardBinOp(Op op) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_27(mht_27_v, 775, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardBinOp");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    // TODO(jpienaar): Switch to named accessors when MHLO uses prefixed form.
    backwards_worklist.append(
        {ShapeOrValueInfo::getValueInfoOf(op.getOperand(0)),
         ShapeOrValueInfo::getValueInfoOf(op.getOperand(1))});
  }
  template <typename Op, typename Combiner>
  void forwardBinOp(Op op, Combiner &&combiner) {
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    // TODO(jpienaar): Switch to named accessors when MHLO uses prefixed form.
    auto lhs = lookup(ShapeOrValueInfo::getValueInfoOf(op.getOperand(0)));
    auto rhs = lookup(ShapeOrValueInfo::getValueInfoOf(op.getOperand(1)));
    for (int64_t i = 0, e = dim0size(op.getType()); i != e; ++i) {
      dims.emplace_back();
      auto &dim = dims.back();
      dim.symbols.append(lhs[i].symbols);
      dim.symbols.append(rhs[i].symbols);
      dim.expr = combiner(lhs[i].expr,
                          rhs[i].expr.shiftSymbols(rhs[i].symbols.size(),
                                                   lhs[i].symbols.size()));
    }
  }
  void backwardIndexCast(arith::IndexCastOp op) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_28(mht_28_v, 801, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardIndexCast");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op.getIn()));
  }
  void forwardIndexCast(arith::IndexCastOp op) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_29(mht_29_v, 808, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardIndexCast");

    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.getIn()));
    for (int64_t i = 0, e = dim0size(op.getType()); i != e; ++i) {
      // This is intentionally not modelling the truncation/zero extension of
      // index_cast. While it's incorrect it doesn't really matter for shape
      // computations.
      dims.push_back({in[i].symbols, in[i].expr});
    }
  }
  void backwardTensorFromElements(tensor::FromElementsOp op) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_30(mht_30_v, 821, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardTensorFromElements");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    for (auto operand : op.getOperands())
      backwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(operand));
  }
  void forwardTensorFromElements(tensor::FromElementsOp op) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_31(mht_31_v, 829, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardTensorFromElements");

    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    for (auto operand : op.getOperands()) {
      auto in = lookup(ShapeOrValueInfo::getValueInfoOf(operand));
      assert(in.size() == 1);
      dims.push_back({in[0].symbols, in[0].expr});
    }
  }
  void backwardTensorExtract(tensor::ExtractOp op) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_32(mht_32_v, 840, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardTensorExtract");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op.tensor()));
  }
  void forwardTensorExtract(tensor::ExtractOp op) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_33(mht_33_v, 847, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardTensorExtract");

    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    assert(op.indices().size() == 1);
    if (auto index = op.indices().front().getDefiningOp<arith::ConstantOp>()) {
      int64_t i = index.getValue().cast<IntegerAttr>().getInt();
      // We asssume this is in bounds.
      auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.tensor()));
      dims.push_back({in[i].symbols, in[i].expr});
    } else {
      forwardUnknown(op);
    }
  }
  void backwardConstant(Value v) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_34(mht_34_v, 862, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardConstant");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(v));
  }
  void forwardConstant(Value v) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_35(mht_35_v, 868, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardConstant");

    IntegerAttr intAttr;
    DenseIntElementsAttr denseAttr;
    if (matchPattern(v, m_Constant(&denseAttr))) {
      auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(v));
      for (uint64_t i = 0, e = dim0size(v.getType()); i != e; ++i) {
        dims.emplace_back();
        auto &dim = dims.back();
        dim.expr = getAffineConstantExpr(
            denseAttr.getValues<APInt>()[i].getSExtValue(), v.getContext());
      }
    } else if (matchPattern(v, m_Constant(&intAttr))) {
      auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(v));
      dims.emplace_back();
      auto &dim = dims.back();
      dim.expr = getAffineConstantExpr(intAttr.getInt(), v.getContext());
    } else {
      forwardUnknown(v);
    }
  }
  void backwardConcatenate(mhlo::ConcatenateOp op) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_36(mht_36_v, 891, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardConcatenate");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    for (auto operand : op.getOperands())
      backwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(operand));
  }
  void forwardConcatenate(mhlo::ConcatenateOp op) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_37(mht_37_v, 899, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardConcatenate");

    for (auto operand : op.getOperands()) {
      auto in = lookup(ShapeOrValueInfo::getValueInfoOf(operand));
      if (in.size() != 1) return forwardUnknown(op);
    }
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    for (auto operand : op.getOperands()) {
      auto in = lookup(ShapeOrValueInfo::getValueInfoOf(operand));
      dims.push_back({in[0].symbols, in[0].expr});
    }
  }
  void backwardReshape(mhlo::ReshapeOp op) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_38(mht_38_v, 913, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardReshape");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.operand()));
  }
  void forwardReshape(mhlo::ReshapeOp op) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_39(mht_39_v, 921, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardReshape");

    auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.operand()));
    if (in.size() != 1) return forwardUnknown(op);
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    dims.push_back({in[0].symbols, in[0].expr});
  }
  void backwardSlice(mhlo::SliceOp op) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_40(mht_40_v, 930, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardSlice");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(op));
    backwards_worklist.push_back(
        ShapeOrValueInfo::getValueInfoOf(op.operand()));
  }
  void forwardSlice(mhlo::SliceOp op) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_41(mht_41_v, 938, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardSlice");

    // Only handle slices equivalent to an extract.
    if (!op.getType().hasStaticShape({1})) {
      return forwardUnknown(op);
    }
    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(op));
    auto in = lookup(ShapeOrValueInfo::getValueInfoOf(op.operand()));
    auto elem = op.start_indices().cast<DenseIntElementsAttr>();
    auto i = (*elem.begin()).getZExtValue();
    if (i >= in.size()) {  // Bounds check.
      return forwardUnknown(op);
    }
    dims.push_back({in[i].symbols, in[i].expr});
  }
  void backwardUnknown(Value v) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_42(mht_42_v, 955, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "backwardUnknown");

    forwards_worklist.push_back(ShapeOrValueInfo::getValueInfoOf(v));
  }
  void forwardUnknown(Value v) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_43(mht_43_v, 961, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "forwardUnknown");

    auto &dims = insert(ShapeOrValueInfo::getValueInfoOf(v));
    auto id = getAffineSymbolExpr(0, v.getContext());
    for (size_t i = 0, e = dim0size(v.getType()); i != e; ++i) {
      dims.emplace_back();
      auto &dim = dims.back();
      dim.symbols.push_back({ShapeOrValueInfo::getValueInfoOf(v), i});
      dim.expr = id;
    }
  }

  // ===
  // Helpers
  // ===

  static void dimsFromStaticShape(
      RankedTensorType ranked_ty,
      llvm::function_ref<SymbolicExpr(int64_t)> fallback,
      std::vector<SymbolicExpr> *merged_dims) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_44(mht_44_v, 982, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "dimsFromStaticShape");

    auto *ctx = ranked_ty.getContext();
    for (int64_t i = 0, e = ranked_ty.getRank(); i != e; ++i) {
      if (ranked_ty.isDynamicDim(i)) {
        merged_dims->push_back(fallback(i));
      } else {
        merged_dims->emplace_back();
        auto &d = merged_dims->back();
        d.expr = getAffineConstantExpr(ranked_ty.getDimSize(i), ctx);
      }
    }
  }

  static void dimsFromStaticShape(RankedTensorType ranked_ty,
                                  ArrayRef<SymbolicExpr> fallback,
                                  std::vector<SymbolicExpr> *merged_dims) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_45(mht_45_v, 1000, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "dimsFromStaticShape");

    return dimsFromStaticShape(
        ranked_ty, [&](int64_t i) { return fallback[i]; }, merged_dims);
  }

  // Return the size of the first dimension. Returns 1 for scalars.
  static int64_t dim0size(Type type) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_46(mht_46_v, 1009, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "dim0size");

    if (auto rankedType = type.dyn_cast<RankedTensorType>())
      return rankedType.getRank() == 0 ? 1 : rankedType.getDimSize(0);
    return 1;
  }

  // Retrieves the existing information from the cache.
  ArrayRef<SymbolicExpr> lookup(ShapeOrValueInfo requestedInfo) {
    auto i = symbolicExprsMap->find(requestedInfo);
    assert(i != symbolicExprsMap->end() && "op not processed yet?");
    return llvm::makeArrayRef(i->second);
  }

  // Inserts a new entry into the cache and returns a reference to its result
  // components.
  std::vector<SymbolicExpr> &insert(ShapeOrValueInfo requestedInfo) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_47(mht_47_v, 1027, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "insert");

    auto i = symbolicExprsMap->try_emplace(requestedInfo);
    assert(i.second && "op already processed?");
    return i.first->second;
  }

  SymbolicExprsMap *symbolicExprsMap;
  SymbolicShapeConstraintsMap *symbolicShapeConstraintsMap;

  // Worklists for the forward and backward passes.
  SmallVector<ShapeOrValueInfo> backwards_worklist;
  SmallVector<ShapeOrValueInfo> forwards_worklist;
};
}  // namespace

void ShapeComponentAnalysis::compute(ShapeOrValueInfo requestedInfo) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_48(mht_48_v, 1045, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "ShapeComponentAnalysis::compute");

  ShapeVisitor(&symbolicExprsMap, &symbolicShapeConstraintsMap)
      .visit(requestedInfo);
}

Optional<ArrayRef<SymbolicExpr>>
ShapeComponentAnalysis::ShapeComponentAnalysis::GetShapeInfo(Value value) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_49(mht_49_v, 1054, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "ShapeComponentAnalysis::ShapeComponentAnalysis::GetShapeInfo");

  auto request = ShapeOrValueInfo::getShapeInfoOf(value);
  compute(request);
  auto found = symbolicExprsMap.find(request);
  if (found == symbolicExprsMap.end()) return {};
  return llvm::makeArrayRef(found->second);
}

Optional<ArrayRef<SymbolicExpr>>
ShapeComponentAnalysis::ShapeComponentAnalysis::GetValueInfo(Value shape) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_50(mht_50_v, 1066, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "ShapeComponentAnalysis::ShapeComponentAnalysis::GetValueInfo");

  auto request = ShapeOrValueInfo::getValueInfoOf(shape);
  compute(request);
  auto found = symbolicExprsMap.find(request);
  if (found == symbolicExprsMap.end()) return {};
  return llvm::makeArrayRef(found->second);
}

void ShapeComponentAnalysis::reset() {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_51(mht_51_v, 1077, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "ShapeComponentAnalysis::reset");

  symbolicExprsMap.clear();
  symbolicShapeConstraintsMap.clear();
}

bool SymbolicExpr::isConstant(int64_t value) const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_52(mht_52_v, 1085, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "SymbolicExpr::isConstant");

  return expr.isa<AffineConstantExpr>() &&
         expr.cast<AffineConstantExpr>().getValue() == value;
}

bool SymbolicExpr::isKnownNotNegativeOne() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_53(mht_53_v, 1093, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "SymbolicExpr::isKnownNotNegativeOne");

  // If the symbol is coming from a shape it can't be a -1. Also allow results
  // of shape_of, compute_reshape_shape, and num_elements. This is correct, not
  // complete.
  auto isGoodSymbol = [](const Symbol &symbol) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_54(mht_54_v, 1100, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "lambda");

    if (symbol.source.isShapeInfo()) return true;
    Operation *op = symbol.source.value().getDefiningOp();
    if (op == nullptr) return false;
    return llvm::isa<shape::ShapeOfOp, mhlo::ComputeReshapeShapeOp,
                     shape::NumElementsOp>(op);
  };

  // For constants we know if it's -1 or not. Checking the sign is sufficient
  // here and allows for reuse below. This is correct, not complete.
  auto isGoodSymbolOrGoodConstantExpr = [&](AffineExpr expr) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_55(mht_55_v, 1113, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "lambda");

    if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>())
      return isGoodSymbol(symbols[symExpr.getPosition()]);
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>())
      return constExpr.getValue() >= 0;
    return false;
  };

  if (isGoodSymbolOrGoodConstantExpr(expr)) return true;

  // Multiplying non-negative symbols and non-negative constants will always
  // give a positive result. This is correct, not complete.
  // TODO(kramerb): Could the analysis provide a generic interface for this?
  if (auto bexpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    return bexpr.getKind() == AffineExprKind::Mul &&
           isGoodSymbolOrGoodConstantExpr(bexpr.getLHS()) &&
           isGoodSymbolOrGoodConstantExpr(bexpr.getRHS());
  }

  return false;
}

bool SymbolicExpr::isKnownNotOne() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_56(mht_56_v, 1138, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "SymbolicExpr::isKnownNotOne");

  if (auto const_expr = expr.dyn_cast<AffineConstantExpr>()) {
    return const_expr.getValue() != 1;
  }
  return false;
}

llvm::Optional<Symbol> SymbolicExpr::singleton() const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_57(mht_57_v, 1148, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "SymbolicExpr::singleton");

  if (expr.isa<AffineSymbolExpr>() &&
      expr.cast<AffineSymbolExpr>().getPosition() == 0) {
    assert(symbols.size() == 1);
    return symbols[0];
  }
  return llvm::None;
}

void SymbolicExpr::dump(llvm::raw_ostream &os) const {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSAnalysisPSshape_component_analysisDTcc mht_58(mht_58_v, 1160, "", "./tensorflow/compiler/mlir/hlo/lib/Analysis/shape_component_analysis.cc", "SymbolicExpr::dump");

  expr.print(os);
  if (!symbols.empty()) os << " with";
  os << "\n";
  if (symbols.empty()) return;
  for (const auto &sym : llvm::enumerate(symbols)) {
    os.indent(4);
    os << 's' << sym.index() << " = ";
    if (!sym.value().source.isValueInfo()) os << "shapeof(";
    sym.value().source.value().print(os);
    if (!sym.value().source.isValueInfo()) os << ")";
    os << '[' << sym.value().index << "]\n";
  }
}
