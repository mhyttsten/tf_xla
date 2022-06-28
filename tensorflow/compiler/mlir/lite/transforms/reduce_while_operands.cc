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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc() {
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

// This is a pass to reduce operands without changing the outcome.

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

struct ReduceWhileOperandsPass
    : public PassWrapper<ReduceWhileOperandsPass, OperationPass<FuncOp>> {
 public:
  StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_0(mht_0_v, 227, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "getArgument");
 return "tfl-reduce-while"; }
  StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_1(mht_1_v, 231, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "getDescription");

    // TODO(b/200919263): Declare Reduce While Operands Pass in Table-Gen
    return "Reduce the number of operands and results of a whlieOp.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_2(mht_2_v, 239, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "getDependentDialects");

    registry.insert<TFL::TensorFlowLiteDialect, TF::TensorFlowDialect>();
  }
  void runOnOperation() override;
};

LogicalResult FindImplicityProducers(
    const std::vector<uint64_t> &explicitly_consumed_ids,
    std::vector<bool> &is_consumed_id,
    const std::vector<std::vector<uint64_t>> &dependency_graph) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_3(mht_3_v, 251, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "FindImplicityProducers");

  std::vector<uint64_t> queue;
  queue.reserve(is_consumed_id.size());
  for (auto id : explicitly_consumed_ids) {
    is_consumed_id[id] = true;
    queue.push_back(id);
  }
  while (!queue.empty()) {
    auto i = queue.back();
    queue.pop_back();

    // If there is a consumer which cannot be found in dependency graph, return
    // false.
    if (i >= dependency_graph.size()) {
      return failure();
    }

    for (auto j : dependency_graph.at(i)) {
      if (is_consumed_id[j]) continue;
      queue.push_back(j);
      is_consumed_id[j] = true;
    }
  }

  return success();
}

void FindProducers(Value start_node, std::vector<uint64_t> &neighbors) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_4(mht_4_v, 281, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "FindProducers");

  llvm::DenseSet<Value> visited;
  std::vector<Value> queue;
  queue.push_back(start_node);
  visited.insert(start_node);
  while (!queue.empty()) {
    auto node = queue.back();
    queue.pop_back();
    if (auto arg = node.dyn_cast_or_null<BlockArgument>()) {
      neighbors.push_back(arg.getArgNumber());
      continue;
    }
    if (!node.getDefiningOp()) continue;
    for (Value operand : node.getDefiningOp()->getOperands()) {
      if (visited.contains(operand)) continue;
      queue.push_back(operand);
      visited.insert(operand);
    }
  }
}

void FindConsumedOp(Operation *start_op,
                    llvm::DenseSet<Operation *> &consumed_ops) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_5(mht_5_v, 306, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "FindConsumedOp");

  if (consumed_ops.contains(start_op)) return;
  std::vector<Operation *> queue;
  queue.push_back(start_op);
  consumed_ops.insert(start_op);
  while (!queue.empty()) {
    auto op = queue.back();
    queue.pop_back();
    for (Value operand : op->getOperands()) {
      if (!operand.getDefiningOp()) continue;
      auto def_op = operand.getDefiningOp();
      if (consumed_ops.contains(def_op)) continue;
      queue.push_back(def_op);
      consumed_ops.insert(def_op);
    }
  }
}

inline bool IsConstant(Operation *op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_6(mht_6_v, 327, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "IsConstant");
 return matchPattern(op, m_Constant()); }

bool AllOperationSafe(Block &block) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_7(mht_7_v, 332, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "AllOperationSafe");

  auto walk_result = block.walk([&](Operation *op) {
    // op has SideEffect.
    if (!isa_and_nonnull<TFL::WhileOp>(op) &&
        !op->hasTrait<OpTrait::IsTerminator>() &&
        !MemoryEffectOpInterface::hasNoEffect(op)) {
      return WalkResult::interrupt();
    }
    // op has implict arguments not listed in operands.
    // Fact: if every op's operands are defined in the same block as op,
    //       then no operation has implicit arugments (constant doesn't count).
    for (auto operand : op->getOperands()) {
      if (operand.dyn_cast_or_null<BlockArgument>()) continue;
      auto operand_op = operand.getDefiningOp();
      if (IsConstant(operand_op)) continue;
      if (operand_op->getBlock() != op->getBlock()) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return !walk_result.wasInterrupted();
}

// It reduces the following pattern:
//
// S = (0, 0, 0)
// while S[2] < 3:
//  a0 = S[0] * 2
//  a1 = a0 + S[1]
//  a2 = S[2] + 1
//  S = (a0, a1, a2)
// return S[0]
//
// the 2nd operand (i = 1) as well as its related op (a1 = a0 + S[1])
// can be removed since only S[0] is returned.
// It cannot be removed by loop-invariant-code-motion pass since every value
// is used and changed in the while loop.

// Moreover, we require
// 1. no implicit argument: For every operation in whileOp, all dependent values
//    (except for constant) are explicitly passed in.
// 2. no side effect: Every operation inside whileOp can be safely
//    remove when it is useEmpty().
// 3. no call func inside while.
bool ReduceWhileOperands(TFL::WhileOp while_op) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_8(mht_8_v, 380, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "ReduceWhileOperands");

  std::vector<uint64_t> explicitly_consumed_ids;
  Block &cond = while_op.cond().front();
  Block &body = while_op.body().front();

  auto n = while_op.getNumOperands();
  if (!AllOperationSafe(cond) || !AllOperationSafe(body)) return false;

  // Find all Consumed indices.
  // i is consumed element if result(i) is used outside whileOp or
  // arugment(i) is used in whileOp.cond().
  for (auto i = 0; i < n; ++i) {
    if (!while_op.getResult(i).use_empty() ||
        !cond.getArgument(i).use_empty()) {
      explicitly_consumed_ids.push_back(i);
    }
  }
  // Empty consumed_element_ids implies none of results is used.
  if (explicitly_consumed_ids.empty()) {
    while_op.erase();
    return true;
  }
  // If every element is consumed, one can't reduce any operand.
  if (explicitly_consumed_ids.size() == n) {
    return false;
  }

  // Build the dependency graph.
  // If result(i) is depend on argument(j) in While.body(), then we put
  // directed edge (i->j) into the graph.
  std::vector<std::vector<uint64_t>> dependency_graph;
  dependency_graph.reserve(n);

  Operation &yield_op = body.back();
  auto results = yield_op.getOperands();
  for (auto i = 0; i < n; ++i) {
    std::vector<uint64_t> neighbors;
    neighbors.reserve(n);
    FindProducers(results[i], neighbors);
    dependency_graph.push_back(neighbors);
  }

  std::vector<bool> is_consumed_id(n, false);
  if (failed(FindImplicityProducers(explicitly_consumed_ids, is_consumed_id,
                                    dependency_graph))) {
    return false;
  }

  // Find all consumed operations in while body.
  llvm::DenseSet<Operation *> consumed_ops;
  // We'll pass in the erase_indices to erase several operands simultaneously.
  llvm::BitVector erase_indices(n);
  consumed_ops.insert(&yield_op);
  for (auto i = 0; i < n; ++i) {
    if (!is_consumed_id[i]) {
      erase_indices.set(i);
    } else if (results[i].getDefiningOp()) {
      FindConsumedOp(results[i].getDefiningOp(), consumed_ops);
    }
  }
  // Remove elements and operations in while_body that are not indispensable.
  yield_op.eraseOperands(erase_indices);
  // Remove ops from bottom to top.
  for (Operation &op :
       llvm::make_early_inc_range(reverse(body.getOperations())))
    // Constant will not be removed in case it is implicitly used.
    if (!consumed_ops.contains(&op) && !IsConstant(&op)) {
      op.erase();
    }
  body.eraseArguments(erase_indices);
  cond.eraseArguments(erase_indices);

  llvm::SmallVector<Value> new_operands;
  llvm::SmallVector<Type> new_result_types;
  new_operands.reserve(n - erase_indices.size());
  new_result_types.reserve(n - erase_indices.size());
  // After reducing, the number of results is decreased. The i-th result of old
  // WhileOp becomes the j-th (j<=i) result of new WhileOp. This information is
  // stored in id_map (id_map[i] = j).
  std::vector<uint64_t> id_map(n, 0);
  uint64_t j = 0;
  for (auto i = 0; i < n; ++i) {
    if (is_consumed_id[i]) {
      id_map[i] = j++;
      new_operands.push_back(while_op.getOperand(i));
      new_result_types.push_back(while_op.getResultTypes()[i]);
    }
  }

  auto new_while_op = OpBuilder(while_op).create<WhileOp>(
      while_op.getLoc(), new_result_types, new_operands, while_op->getAttrs());
  new_while_op.cond().takeBody(while_op.cond());
  new_while_op.body().takeBody(while_op.body());

  for (auto i = 0; i < n; ++i) {
    if (!while_op.getResult(i).use_empty()) {
      auto j = id_map[i];
      while_op.getResult(i).replaceAllUsesWith(new_while_op.getResult(j));
    }
  }
  while_op.erase();
  return erase_indices.any();
}

void ReduceWhileOperandsPass::runOnOperation() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSreduce_while_operandsDTcc mht_9(mht_9_v, 487, "", "./tensorflow/compiler/mlir/lite/transforms/reduce_while_operands.cc", "ReduceWhileOperandsPass::runOnOperation");

  auto fn = getOperation();
  fn.walk([&](TFL::WhileOp while_op) { ReduceWhileOperands(while_op); });
}

static PassRegistration<ReduceWhileOperandsPass> pass;
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateReduceWhileOperandsPass() {
  return std::make_unique<ReduceWhileOperandsPass>();
}

}  // namespace TFL
}  // namespace mlir
