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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

#include <algorithm>
#include <iterator>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// TF Executor Dialect
//===----------------------------------------------------------------------===//

namespace {

struct TensorFlowExecutorInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_0(mht_0_v, 235, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "isLegalToInline");

    return true;
  }
  // Override the inlining hook to determine if 'src' can be inlined into
  // 'dest'.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &value_mapping) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_1(mht_1_v, 244, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "isLegalToInline");

    // Allow inlining into tf.island regions if the incoming region has a single
    // block.
    return llvm::isa<tf_executor::IslandOp>(dest->getParentOp()) &&
           llvm::hasSingleElement(*src);
  }
};

struct TensorFlowExecutorDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  // Registered hook to check if the given region, which is attached to an
  // operation that is *not* isolated from above (i.e. no internal regions
  // reference values defined in an enclosing region), should be used when
  // materializing constants.
  // In the executor dialect we materialize inside an island.
  bool shouldMaterializeInto(Region *region) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_2(mht_2_v, 263, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "shouldMaterializeInto");

    return isa<tf_executor::IslandOp>(region->getParentOp());
  }
};

}  // namespace

TensorFlowExecutorDialect::TensorFlowExecutorDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf_executor", context,
              TypeID::get<TensorFlowExecutorDialect>()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_3(mht_3_v, 275, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "TensorFlowExecutorDialect::TensorFlowExecutorDialect");

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc.inc"
      >();

  addInterfaces<TensorFlowExecutorInlinerInterface,
                TensorFlowExecutorDialectFoldInterface>();

  addTypes<ControlType, TokenType>();
}

Type TensorFlowExecutorDialect::parseType(DialectAsmParser &parser) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_4(mht_4_v, 290, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "TensorFlowExecutorDialect::parseType");

  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();

  if (data_type == "control") return ControlType::get(getContext());
  if (data_type == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc())
      << "unknown tf_executor type: " << data_type;
  return nullptr;
}

void TensorFlowExecutorDialect::printType(Type type,
                                          DialectAsmPrinter &os) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_5(mht_5_v, 305, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "TensorFlowExecutorDialect::printType");

  if (type.isa<ControlType>()) {
    os << "control";
    return;
  }
  if (type.isa<TokenType>()) {
    os << "token";
    return;
  }
  os << "<unknown tf_executor type>";
}

//===----------------------------------------------------------------------===//
// Implementation for all the operations defined in ODS (op definition spec).
//===----------------------------------------------------------------------===//

namespace {

// Verifies that every control operands are at the end of the list.
// Used by the constraint `ControlOperandsAfterAllData` in ODS.
LogicalResult VerifyControlOperandsAfterAllData(Operation *op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_6(mht_6_v, 328, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "VerifyControlOperandsAfterAllData");

  bool found_control = false;
  for (int operand_idx : llvm::seq<int>(0, op->getNumOperands())) {
    if (op->getOperand(operand_idx).getType().isa<ControlType>()) {
      found_control = true;
      continue;
    }
    if (found_control)
      return op->emitOpError() << "found non-control operand #" << operand_idx
                               << " after control operand";
  }
  return success();
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// tf_executor.graph
//===----------------------------------------------------------------------===//

FetchOp GraphOp::GetFetch() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_7(mht_7_v, 351, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "GraphOp::GetFetch");
 return llvm::cast<FetchOp>(GetBody().back()); }

LogicalResult GraphOp::verify() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_8(mht_8_v, 356, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "GraphOp::verify");

  GraphOp graph = *this;
  auto *executorDialect = graph->getDialect();

  if (graph.GetBody().empty())
    return graph.emitOpError() << "expects a non-empty body";

  // Only tf_executor dialect operations are allowed to be immediately nested
  // in a tf_executor.graph region.
  for (Operation &op : graph.GetBody()) {
    if (op.getDialect() != executorDialect)
      return op.emitOpError() << "unallowed inside a tf_executor.graph region";
    if (isa<GraphOp>(op))
      return op.emitOpError()
             << "unallowed directly inside another tf_executor.graph";
  }

  Operation &fetch = graph.GetBody().back();
  if (!isa<FetchOp>(fetch))
    return fetch.emitOpError()
           << "invalid tf_executor.graph terminator, fetch expected";

  // Ensure that the fetch terminator operands matches the graph result type.
  // All the non-control operands of the fetch operation must match the graph
  // returned value.
  if (fetch.getNumOperands() < graph.getNumResults())
    return fetch.emitOpError() << "does not have enough operands to cover the "
                                  "graph returned values";
  for (int i : llvm::seq<int>(0, fetch.getNumOperands())) {
    Value operand = fetch.getOperand(i);
    // Break out of the loop at the first control operand encountered.
    const int64_t num_results = graph.getNumResults();
    if (operand.getType().isa<ControlType>()) {
      if (i != num_results)
        return fetch.emitOpError()
               << "operand #" << i
               << " is a control type, can't be bound to a graph result";
      break;
    }
    if (i >= num_results)
      return fetch.emitOpError()
             << "operand #" << i << " does not have a graph results to bind";
    if (graph.getResult(i).getType() != operand.getType()) {
      return fetch.emitOpError()
             << "operand #" << i << " type mismatch graph results ("
             << graph.getResult(i).getType() << " != " << operand.getType()
             << ")";
    }
  }
  return success();
}

void GraphOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_9(mht_9_v, 411, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "GraphOp::print");

  p << ' ';
  p.printRegion(getOperation()->getRegion(0));
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_10(mht_10_v, 420, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "GraphOp::parse");

  llvm::SMLoc loc = parser.getCurrentLocation();

  // Parse the body region.
  Region &body = *result.addRegion();
  if (parser.parseRegion(body, llvm::None, llvm::None)) return failure();

  // Ensure that the region is well formed: it contains at least a block with
  // a FetchOp terminator.
  GraphOp::ensureTerminator(body, parser.getBuilder(), result.location);

  if (!llvm::hasSingleElement(body))
    return parser.emitError(loc) << "expects a single block region";

  // Get the results type from the terminator type inside the graph.
  Operation &fetch = body.back().back();
  if (!isa<FetchOp>(fetch))
    return parser.emitError(loc) << "expects a tf_executor.fetch terminator";

  // The return value of the graph operation are the non-control operands of
  // the fetch operation.
  result.types.reserve(fetch.getNumOperands());
  for (Type type : fetch.getOperandTypes()) {
    if (type.isa<ControlType>()) break;
    result.types.push_back(type);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// tf_executor.fetch
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_executor.island
//===----------------------------------------------------------------------===//

YieldOp IslandOp::GetYield() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_11(mht_11_v, 464, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "IslandOp::GetYield");
 return llvm::cast<YieldOp>(GetBody().back()); }

// Checks if a tf_executor.island wraps a single operation and the single
// operation results are perfectly forwarded to the islands yield.
bool IslandOp::WrapsSingleOp() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_12(mht_12_v, 471, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "IslandOp::WrapsSingleOp");

  auto body = GetBody().without_terminator();
  if (!hasSingleElement(body)) return false;

  Operation &wrapped_op = *body.begin();
  YieldOp yield = GetYield();
  return wrapped_op.getNumResults() == yield.getNumOperands() &&
         std::equal(wrapped_op.getResults().begin(),
                    wrapped_op.getResults().end(), yield.getOperands().begin());
}

mlir::LogicalResult IslandOp::verify() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_13(mht_13_v, 485, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "IslandOp::verify");

  IslandOp island = *this;
  if (!island.GetBody().args_empty())
    return island.emitOpError() << "expects body without any arguments";

  Operation &yield = island.GetBody().back();
  if (!isa<YieldOp>(yield))
    return yield.emitOpError()
           << "invalid tf_executor.island terminator, yield expected";

  // Ensure that the yield terminator operands matches the island results type.
  int result_count = island.getNumResults() - 1;  // -1 for the control token
  const int num_operands = yield.getNumOperands();
  if (num_operands != result_count)
    return yield.emitOpError()
           << "has " << yield.getNumOperands()
           << " operand, but island returns " << result_count;
  for (int operand_idx : llvm::seq<int>(0, yield.getNumOperands())) {
    if (island.getResult(operand_idx).getType() !=
        yield.getOperand(operand_idx).getType())
      return yield.emitOpError()
             << "operand #" << operand_idx << " type mismatch island results";
  }

  // Check that there aren't any control results other than the last one.
  Type control_type = ControlType::get(island.getContext());
  for (int operand_idx : llvm::seq<int>(0, island.getNumResults() - 1)) {
    if (island.getResult(operand_idx).getType() == control_type)
      return yield.emitOpError()
             << "unexpected control type for operand #" << operand_idx;
  }
  return success();
}

void IslandOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_14(mht_14_v, 522, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "IslandOp::print");

  if (getNumOperands()) {
    // These are always control operand, no explicit type needed.
    p << '(';
    p.printOperands(getOperands());
    p << ')';
  }

  // Check if we can print the short "wraps" form: that is if the island
  // contains a single operation and the result of this operation are perfectly
  // forwarded to the yield.
  if (getOperation()->getAttrs().empty() && WrapsSingleOp()) {
    Operation &wrapped_op = GetBody().front();
    YieldOp yield_op = GetYield();
    // The "wraps" syntax only encodes a single location.
    // In order to correctly round-trip, we can only use this syntax when all
    // the locations are identical.
    if (wrapped_op.getLoc() == getLoc() && yield_op.getLoc() == getLoc()) {
      p << " wraps ";
      p.printGenericOp(&wrapped_op);
      return;
    }
  }
  p << ' ';
  p.printRegion(getOperation()->getRegion(0));
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult IslandOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_15(mht_15_v, 553, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "IslandOp::parse");

  llvm::SMLoc loc = parser.getCurrentLocation();
  Type control_type = ControlType::get(parser.getBuilder().getContext());

  // Parse optional argument list (control dependencies only).
  SmallVector<OpAsmParser::UnresolvedOperand, 4> op_infos;
  if (parser.parseOperandList(op_infos, OpAsmParser::Delimiter::OptionalParen))
    return failure();
  if (!op_infos.empty()) {
    SmallVector<Type, 2> types(op_infos.size(), control_type);
    parser.resolveOperands(op_infos, types, loc, result.operands);
  }

  // Parse the body region.
  Region &body = *result.addRegion();

  if (succeeded(parser.parseOptionalKeyword("wraps"))) {
    // If we parse the short version of the island, we have an operation in the
    // generic form that follows the "wraps" keyword. Parse it inside the region
    // and forward all of its results as-is to the yield operation.
    body.push_back(new Block);
    Block &block = body.back();
    Operation *wrapped_op = parser.parseGenericOperation(&block, block.begin());
    if (!wrapped_op) return failure();
    OpBuilder builder(parser.getBuilder().getContext());
    builder.setInsertionPointToEnd(&block);
    builder.create<YieldOp>(wrapped_op->getLoc(), wrapped_op->getResults());
    result.location = wrapped_op->getLoc();
  } else if (parser.parseRegion(body, llvm::None, llvm::None)) {
    return failure();
  }

  IslandOp::ensureTerminator(body, parser.getBuilder(), result.location);

  // Get the results type for the island from the terminator operands.
  Operation &yield = body.back().back();
  result.types.reserve(yield.getNumOperands() + 1);
  result.types.append(yield.operand_type_begin(), yield.operand_type_end());
  result.types.push_back(control_type);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// tf_executor.yield
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_executor.Switch
//===----------------------------------------------------------------------===//

ParseResult SwitchOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_16(mht_16_v, 609, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "SwitchOp::parse");

  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  SmallVector<Type, 1> types;
  if (parser.parseOperandList(op_infos) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 1)
    return parser.emitError(parser.getNameLoc())
           << " expects only a single data type";

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // input and the outputs are all using this type and predicate is tensor<i1>
  // type).
  if (types.front().isa<FunctionType>()) {
    FunctionType type = types.front().cast<FunctionType>();
    if (type.getNumInputs() < 2)
      return parser.emitError(parser.getNameLoc())
             << " expects a single data type and a predicate";
    result.types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    if (op_infos.size() < 2)
      return parser.emitError(parser.getNameLoc())
             << " expects a single data type and a predicate";
    Type control_type = ControlType::get(parser.getBuilder().getContext());
    result.types.append(2, types[0]);
    result.types.push_back(control_type);
    Type i1_type = parser.getBuilder().getI1Type();
    RankedTensorType predicate_type = RankedTensorType::get({}, i1_type);
    types.push_back(predicate_type);
    types.append(op_infos.size() - 2, control_type);
  }

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

void SwitchOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_17(mht_17_v, 652, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "SwitchOp::print");

  p << ' ';
  p.printOperands(getOperands());
  Type data_operand_ty = data().getType();
  // If the types aren't perfectly matching, print the functional type syntax
  // else print the shorter single type.
  p << " : ";
  if (trueOutput().getType() != data_operand_ty ||
      falseOutput().getType() != data_operand_ty ||
      predicate().getType().isa<UnrankedTensorType>()) {
    p.printFunctionalType(getOperation());
  } else {
    p << getType(0);
  }
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

//===----------------------------------------------------------------------===//
// tf_executor.SwitchN
//===----------------------------------------------------------------------===//

LogicalResult SwitchNOp::verify() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_18(mht_18_v, 676, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "SwitchNOp::verify");

  SwitchNOp switchn = *this;
  IntegerAttr num_outs = switchn->getAttrOfType<IntegerAttr>("num_outs");
  if (!num_outs)
    return switchn.emitOpError() << "expects a `num_outs` integer attribute";

  // Expect num_outs results + 1 control output.
  if (switchn.getNumResults() != num_outs.getInt() + 1)
    return switchn.emitOpError()
           << "expect `num_outs` (" << num_outs.getInt() << ") results but got "
           << (switchn.getNumResults() - 1);

  // Check that operand can be broadcasted to each output type.
  auto operand0_type = switchn.getOperand(0).getType();
  TensorType operand0_tensor_type = operand0_type.dyn_cast<TensorType>();
  if (!operand0_tensor_type) {
    return switchn.emitOpError()
           << "expects data operand to have tensor type but got "
           << operand0_type;
  }
  for (Type output_type : switchn.getResultTypes()) {
    if (output_type.isa<ControlType>()) break;

    TensorType output_tensor_type = output_type.dyn_cast<TensorType>();
    if (!output_tensor_type) {
      return switchn.emitOpError()
             << "expects outputs to have tensor type but got " << output_type;
    }

    // If the output type is a ref type, then the operand type should also be of
    // the same ref type. However, if the output type is a non-ref type T, then
    // the operand can be tensor of type T or T_REF.
    bool is_output_ref =
        output_tensor_type.getElementType().isa<tf_type::TensorFlowRefType>();
    if (is_output_ref && !operand0_tensor_type.getElementType()
                              .isa<tf_type::TensorFlowRefType>()) {
      return switchn.emitOpError()
             << "expects same operand and output element type but got "
             << operand0_tensor_type << " vs " << output_tensor_type;
    }
    Type broadcasted_type = OpTrait::util::getBroadcastedType(
        tf_type::DropRefAndSubTypes(operand0_tensor_type),
        tf_type::DropRefAndSubTypes(output_tensor_type));
    if (!broadcasted_type) {
      return switchn.emitOpError()
             << "expects data operand to be broadcastable with all output types"
             << " but got " << operand0_tensor_type << " vs "
             << output_tensor_type;
    }
  }
  return success();
}

void SwitchNOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_19(mht_19_v, 732, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "SwitchNOp::print");

  p << ' ';
  auto operands = getOperands();
  // Print the 2 data operands.
  p.printOperands(operands.begin(), std::next(operands.begin(), 2));
  p << " of " << (getNumResults() - 1);
  // print control dependencies if any
  if (!llvm::empty(controlInputs())) {
    p << " (";
    p.printOperands(controlInputs());
    p << ")";
  }
  p << " : " << getType(0);
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"num_outs"});
}

ParseResult SwitchNOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_20(mht_20_v, 751, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "SwitchNOp::parse");

  // Parsing:
  //       %2:6 = tf_executor.SwitchN %0, %1 of 5 : tensor<??xf32>
  // Where the first operand is the data to replicate, the second is an i32
  // indicating which output to populate, followed by the keyword `of` and the
  // number of outputs (+1 for the control token).
  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  SmallVector<Type, 1> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  IntegerAttr num_outs;
  Type i64_type = parser.getBuilder().getIntegerType(64);
  if (parser.parseOperandList(op_infos, 2) || parser.parseKeyword("of") ||
      parser.parseAttribute(num_outs, i64_type, "num_outs",
                            result.attributes) ||
      parser.parseOperandList(op_infos,
                              OpAsmParser::Delimiter::OptionalParen) ||
      parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 1)
    return parser.emitError(parser.getNameLoc())
           << " expects only a single data type";

  if (num_outs.getInt() <= 0)
    return parser.emitError(parser.getNameLoc())
           << " expects a positive number of outputs";

  // `types` already contains the type for the data, add an i32 for the
  // output_index, and then the optional control inputs.
  auto builder = parser.getBuilder();
  types.push_back(RankedTensorType::get({}, builder.getIntegerType(32)));
  Type control_type = ControlType::get(builder.getContext());
  types.append(op_infos.size() - 2, control_type);

  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  // Output result types is a replication `num_outs` times the data input type.
  result.types.append(num_outs.getInt(), types[0]);
  result.types.push_back(control_type);

  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.Merge
//===----------------------------------------------------------------------===//

LogicalResult MergeOp::verify() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_21(mht_21_v, 801, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "MergeOp::verify");

  MergeOp merge = *this;
  if (!merge.getNumOperands())
    return merge.emitOpError() << "expects at least one operand";

  Type data_type = merge.getOperand(0).getType();
  if (data_type.isa<ControlType>())
    return merge.emitOpError() << "expects a non-control input";

  // Check that each operand can be individually broadcasted to the output type.
  Type output_type = merge.output().getType();
  TensorType output_tensor_ty = output_type.dyn_cast<TensorType>();
  if (!output_tensor_ty) {
    return merge.emitOpError()
           << "expects output to have tensor type but got " << output_type;
  }
  bool is_output_ref =
      output_tensor_ty.getElementType().isa<tf_type::TensorFlowRefType>();
  for (Type operand_type : merge.getOperandTypes()) {
    if (operand_type.isa<ControlType>()) break;

    // TODO(hinsu): Update ControlOperandsAfterAllData trait to verify this
    // constraint.
    TensorType operand_tensor_ty = operand_type.dyn_cast<TensorType>();
    if (!operand_tensor_ty)
      return merge.emitOpError()
             << "expects data operands to have tensor type but got "
             << operand_type;

    // If output type is a ref type then all operand types should also be of the
    // same ref type. However, if the output type is a non-ref type T, operands
    // can be tensor of type T or T_REF.
    if (is_output_ref &&
        !operand_tensor_ty.getElementType().isa<tf_type::TensorFlowRefType>()) {
      return merge.emitOpError()
             << "expects same operand and output element type but got "
             << operand_tensor_ty << " vs " << output_tensor_ty;
    }
    Type broadcasted_type = OpTrait::util::getBroadcastedType(
        tf_type::DropRefAndSubTypes(output_tensor_ty),
        tf_type::DropRefAndSubTypes(operand_tensor_ty));
    if (!broadcasted_type)
      return merge.emitOpError()
             << "expects all operands to be broadcastable with output type"
             << " but got " << operand_tensor_ty << " vs " << output_tensor_ty;
  }
  return success();
}

void MergeOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_22(mht_22_v, 853, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "MergeOp::print");

  // Use short form only when there are exactly two data operands and their
  // type matches the output type. Otherwise, use the generic printer.
  bool use_short_form = true;
  int num_data_operands = 0;

  Type output_type = output().getType();
  for (Type operand_type : getOperandTypes()) {
    if (operand_type.isa<ControlType>()) break;
    num_data_operands++;

    if (operand_type != output_type) {
      use_short_form = false;
      break;
    }
  }

  p << ' ';
  p.printOperands(getOperands());

  // Print the type signature of the operation.
  p << " : ";
  if (!use_short_form || num_data_operands != 2) {
    p.printFunctionalType(getOperation());
  } else {
    p << output_type;
  }

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult MergeOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_23(mht_23_v, 887, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "MergeOp::parse");

  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  SmallVector<Type, 1> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOperandList(op_infos) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 1)
    return parser.emitError(parser.getNameLoc())
           << " expects only a single data type";

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // inputs and the output are all using this type).
  if (FunctionType type = types.front().dyn_cast<FunctionType>()) {
    result.types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    // In case of the short form, use the parsed type for both the operands and
    // the remaining operands are expected to be control inputs.
    types.push_back(Type(types.front()));
    Type control_type = ControlType::get(parser.getBuilder().getContext());
    types.append(op_infos.size() - 2, control_type);

    RankedTensorType i32_tensor =
        RankedTensorType::get({}, parser.getBuilder().getIntegerType(32));
    result.types = {types.front(), i32_tensor, control_type};
  }

  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.Enter
//===----------------------------------------------------------------------===//

// Default number for the parallel_iterations attributes on Enter nodes.
static constexpr int kDefaultParallelIterations = 10;

void EnterOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_24(mht_24_v, 931, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "EnterOp::print");

  p << ' ';
  p.printOperands(getOperands());

  p << " frame \"";
  printEscapedString(frame_name(), p.getStream());
  p << "\"";
  if (parallel_iterations() != kDefaultParallelIterations)
    p << " parallel_iterations " << parallel_iterations();
  if (is_constant()) p << " constant ";

  // If the types aren't perfectly matching, print the functional type syntax
  // else print the shorter single type.
  p << " : ";
  if (data().getType() != output().getType()) {
    p.printFunctionalType(getOperation());
  } else {
    p << getType(0);
  }

  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {"frame_name", "parallel_iterations", "is_constant"});
}

ParseResult EnterOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_25(mht_25_v, 958, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "EnterOp::parse");

  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  llvm::SMLoc loc = parser.getCurrentLocation();
  MLIRContext *context = parser.getBuilder().getContext();
  if (parser.parseOperandList(op_infos)) return failure();
  if (op_infos.empty())
    return parser.emitError(loc) << " expects at least one data operand";

  Attribute frame;
  if (parser.parseKeyword("frame") ||
      parser.parseAttribute(frame, NoneType::get(context), "frame_name",
                            result.attributes))
    return failure();

  Type i64 = parser.getBuilder().getIntegerType(64);
  if (parser.parseOptionalKeyword("parallel_iterations")) {
    result.addAttribute("parallel_iterations",
                        IntegerAttr::get(i64, kDefaultParallelIterations));
  } else {
    IntegerAttr parallel_iterations;
    if (parser.parseAttribute(parallel_iterations, i64, "parallel_iterations",
                              result.attributes))
      return failure();
  }
  bool has_constant = succeeded(parser.parseOptionalKeyword("constant"));
  result.addAttribute("is_constant", BoolAttr::get(context, has_constant));

  SmallVector<Type, 1> types;
  if (parser.parseColonTypeList(types)) return failure();
  if (types.size() != 1)
    return parser.emitError(loc) << " expects only a single data type";

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // input and the outputs are all using this type).
  if (FunctionType type = types.front().dyn_cast<FunctionType>()) {
    // One data input, and any number of control inputs.
    if (type.getNumInputs() >= 1) {
      result.types.assign(type.getResults().begin(), type.getResults().end());
      types.assign(type.getInputs().begin(), type.getInputs().end());
    } else {
      return parser.emitError(parser.getNameLoc()) << " expects a data input";
    }
  } else {
    Type control_type = ControlType::get(context);
    types.append(op_infos.size() - 1, control_type);
    result.addTypes({types.front(), control_type});
  }

  // Extra operands are expected to be control inputs.

  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.NextIteration.Source
//===----------------------------------------------------------------------===//

LogicalResult NextIterationSourceOp::verify() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_26(mht_26_v, 1022, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "NextIterationSourceOp::verify");

  NextIterationSourceOp source = *this;
  Value token = source.token();
  if (!token.hasOneUse())
    return source.emitOpError() << "expects a single user for produced token";
  if (!isa<NextIterationSinkOp>(*token.user_begin()))
    return source.emitOpError() << "token should be consumed by a sink op";
  return success();
}

//===----------------------------------------------------------------------===//
// tf_executor.NextIteration.Sink
//===----------------------------------------------------------------------===//

LogicalResult NextIterationSinkOp::verify() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_27(mht_27_v, 1039, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "NextIterationSinkOp::verify");

  NextIterationSinkOp sink = *this;
  Value token = sink.token();
  Operation *definingOp = token.getDefiningOp();
  if (!definingOp)
    return sink.emitOpError() << "expects a token directly produced by a "
                                 "tf_executor.NextIteration.Source op: ";
  auto source = dyn_cast<NextIterationSourceOp>(definingOp);
  if (!source)
    return sink.emitOpError() << "expects a token produced by a "
                                 "tf_executor.NextIteration.Source op: ";
  if (source.output().getType() != sink.input().getType())
    return sink.emitOpError()
           << "input type " << sink.input().getType()
           << " mismatch the tf_executor.NextIteration.Source output type: "
           << source.output().getType();
  return success();
}

NextIterationSourceOp NextIterationSinkOp::GetSource() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_28(mht_28_v, 1061, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "NextIterationSinkOp::GetSource");

  return cast<NextIterationSourceOp>(token().getDefiningOp());
}

//===----------------------------------------------------------------------===//
// tf_executor.Exit
//===----------------------------------------------------------------------===//

void ExitOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_29(mht_29_v, 1072, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "ExitOp::print");

  p << ' ';
  p.printOperands(getOperands());
  p << " : " << getType(0);
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult ExitOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_30(mht_30_v, 1082, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "ExitOp::parse");

  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;
  SmallVector<Type, 1> types;

  if (parser.parseOperandList(op_infos) || parser.parseColonTypeList(types))
    return failure();

  llvm::SMLoc loc = parser.getCurrentLocation();
  Type control_type = ControlType::get(parser.getBuilder().getContext());
  types.append(op_infos.size() - 1, control_type);
  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  result.addTypes({types.front(), control_type});
  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// tf_executor.ControlTrigger
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_executor.LoopCond
//===----------------------------------------------------------------------===//

void LoopCondOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_31(mht_31_v, 1110, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "LoopCondOp::print");

  p << ' ';
  p.printOperands(getOperands());

  // If the types aren't matching (broadcast), print the functional type syntax.
  if (input().getType() != output().getType()) {
    p << " : ";
    p.printFunctionalType(getOperation());
  } else {
    p << " : " << input().getType();
  }

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult LoopCondOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_32(mht_32_v, 1128, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "LoopCondOp::parse");

  SmallVector<OpAsmParser::UnresolvedOperand, 2> op_infos;

  if (parser.parseOperandList(op_infos)) return failure();
  if (op_infos.empty())
    return parser.emitError(parser.getNameLoc())
           << "expects at least one operand";

  SmallVector<Type, 1> types;
  if (parser.parseColonTypeList(types)) return failure();

  // Support parsing either a functional type (in which case all the types are
  // fully qualified) or a short form with a single type (in which case the data
  // input and the outputs are all using this type).
  Type control_type = ControlType::get(parser.getBuilder().getContext());
  if (FunctionType type = types.front().dyn_cast<FunctionType>()) {
    if (llvm::count_if(type.getInputs(),
                       [=](Type type) { return type != control_type; }) != 1)
      return parser.emitError(parser.getNameLoc())
             << " expects a single data type";
    result.types.assign(type.getResults().begin(), type.getResults().end());
    types.assign(type.getInputs().begin(), type.getInputs().end());
  } else {
    if (types.size() != 1)
      return parser.emitError(parser.getNameLoc())
             << " expects a single data type";
    types.append(op_infos.size() - 1, control_type);
    result.addTypes({types.front(), control_type});
  }

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.resolveOperands(op_infos, types, loc, result.operands))
    return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

// TODO(lyandy): Add canonicalization for dedupping control inputs.

//===----------------------------------------------------------------------===//
// tf_executor.graph
//===----------------------------------------------------------------------===//

namespace {
// Finds in a block if the op of type `InnerOpT` is the first operation and
// optionally followed by a terminator.
template <typename InnerOpT>
bool HasSingleOpInBlock(Block *block) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_33(mht_33_v, 1182, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "HasSingleOpInBlock");

  if (block->empty()) return false;
  if (!llvm::isa<InnerOpT>(block->front())) return false;
  // Either InnerOpT is the only instruction in the block, or there is a
  // possible terminator.
  return std::next(block->begin()) == block->end() ||
         std::next(block->begin(), 2) == block->end();
}

// This pattern matches GraphOps with only one FetchOp (empty) and remaps the
// results of the GraphOp to the operands of the FetchOp.
struct DropEmptyGraph : public OpRewritePattern<GraphOp> {
  using OpRewritePattern<GraphOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GraphOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_34(mht_34_v, 1200, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "matchAndRewrite");

    Block &block = op.GetBody();
    // Check if graph only has one fetch.
    if (&block.front() != &block.back()) return failure();

    // Map graph results to fetch operands.
    rewriter.replaceOp(op, op.GetFetch().fetches());

    return success();
  }
};

// This pattern matches GraphOps with only one island, pulls out all inner ops
// of the island to the block containing the GraphOp, and then removes the
// GraphOp.
struct HoistInnerOpsSingleIslandGraph : public OpRewritePattern<GraphOp> {
  using OpRewritePattern<GraphOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GraphOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_35(mht_35_v, 1222, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "matchAndRewrite");

    Block &block = op.GetBody();
    // Check if graph only has one island.
    if (!HasSingleOpInBlock<IslandOp>(&block)) return failure();

    FetchOp fetch_op = op.GetFetch();
    auto island_op = llvm::cast<IslandOp>(block.front());
    YieldOp yield_op = island_op.GetYield();

    // Map graph results to inner ops results of single island.
    llvm::SmallVector<Value, 8> new_rets;
    for (Value operand : fetch_op.fetches()) {
      // Control results should not be propagated out.
      if (operand.getType().isa<ControlType>()) break;

      if (operand.getDefiningOp() != island_op) {
        // Operand is not from island, simply propagate it out.
        new_rets.push_back(operand);
      } else {
        // Lookup yield operand in island for inner op result.
        auto result = operand.cast<OpResult>();
        new_rets.push_back(yield_op.getOperand(result.getResultNumber()));
      }
    }

    // Move inner ops from island to block containing graph.
    auto &island_body = island_op.GetBody().getOperations();
    Operation *operation = op.getOperation();
    operation->getBlock()->getOperations().splice(
        operation->getIterator(), island_body, island_body.begin(),
        std::prev(island_body.end()));
    rewriter.replaceOp(op, new_rets);

    return success();
  }
};
}  // anonymous namespace

void GraphOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_36(mht_36_v, 1264, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "GraphOp::getCanonicalizationPatterns");

  results.add<DropEmptyGraph, HoistInnerOpsSingleIslandGraph>(context);
}

//===----------------------------------------------------------------------===//
// tf_executor.island
//===----------------------------------------------------------------------===//

namespace {
// This pattern matches and removes IslandOps with no inner ops, no control
// operands and no data results. Control result users will have their relevant
// operands removed.
struct DropEmptyIslandNoOperandNoDataResult
    : public OpRewritePattern<IslandOp> {
  using OpRewritePattern<IslandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IslandOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_37(mht_37_v, 1284, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "matchAndRewrite");

    if (op.getNumOperands() != 0 || op.getNumResults() != 1 ||
        !HasSingleOpInBlock<YieldOp>(&op.GetBody()))
      return failure();

    for (auto &use : llvm::make_early_inc_range(op.control().getUses()))
      use.getOwner()->eraseOperand(use.getOperandNumber());

    rewriter.eraseOp(op);

    return success();
  }
};

// This pattern matches and removes IslandOps with no inner ops, no control
// operands, one data result and no control result user. The single data result
// (from YieldOps first operand) is forwarded to the IslandOp single data result
// users.
struct DropEmptyIslandNoOperandOneDataResult
    : public OpRewritePattern<IslandOp> {
  using OpRewritePattern<IslandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IslandOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_38(mht_38_v, 1310, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "matchAndRewrite");

    if (op.getNumOperands() != 0 || op.getNumResults() != 2 ||
        !op.control().use_empty() ||
        !HasSingleOpInBlock<YieldOp>(&op.GetBody()))
      return failure();

    rewriter.replaceOp(op, {op.GetYield().getOperand(0), nullptr});

    return success();
  }
};

// TODO(lyandy): Add canonicalization for empty IslandOps with more than one
// control operand and no data results.

}  // anonymous namespace

void IslandOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_39(mht_39_v, 1331, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "IslandOp::getCanonicalizationPatterns");

  results.add<DropEmptyIslandNoOperandNoDataResult,
              DropEmptyIslandNoOperandOneDataResult>(context);
}

//===----------------------------------------------------------------------===//
// tf_executor.ControlTrigger
//===----------------------------------------------------------------------===//

namespace {
// This pattern matches and removes ControlTriggerOps with no control operands.
// Control result users will have their relevant operands removed.
struct DropEmptyControlTrigger : public OpRewritePattern<ControlTriggerOp> {
  using OpRewritePattern<ControlTriggerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ControlTriggerOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_40(mht_40_v, 1350, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "matchAndRewrite");

    if (op.getNumOperands() != 0) return failure();

    for (auto &use : llvm::make_early_inc_range(op.control().getUses()))
      use.getOwner()->eraseOperand(use.getOperandNumber());

    rewriter.eraseOp(op);

    return success();
  }
};
}  // anonymous namespace

void ControlTriggerOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_41(mht_41_v, 1367, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "ControlTriggerOp::getCanonicalizationPatterns");

  results.add<DropEmptyControlTrigger>(context);
}

//===----------------------------------------------------------------------===//
// Folders
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_executor.island
//===----------------------------------------------------------------------===//

LogicalResult IslandOp::fold(llvm::ArrayRef<Attribute> operands,
                             llvm::SmallVectorImpl<OpFoldResult> &results) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_executorDTcc mht_42(mht_42_v, 1383, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc", "IslandOp::fold");

  // This folds IslandOps with no inner ops, one control operand and no data
  // results. The single control operand is forwarded to the IslandOp control
  // result users.
  if (getNumOperands() != 1 || getNumResults() != 1 ||
      !HasSingleOpInBlock<YieldOp>(&GetBody()))
    return failure();

  results.emplace_back(getOperand(0));

  return success();
}

}  // namespace tf_executor
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.cc.inc"
