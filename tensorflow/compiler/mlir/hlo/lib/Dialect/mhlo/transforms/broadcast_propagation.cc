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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc() {
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

#include <algorithm>
#include <utility>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

// To avoid duplicate broadcasts, we collect all the intended broadcasts ahead
// of realizing any broadcasts in the IR. These are broadcasted versions of
// values that we are interested in, and they are uniquely characterized by a
// `BroadcastIntent` value.
struct BroadcastIntent {
  RankedTensorType result_type;
  Value target_value;
  Value output_dimensions;
  Attribute broadcast_dimensions;
  bool operator==(BroadcastIntent rhs) const {
    return result_type == rhs.result_type && target_value == rhs.target_value &&
           output_dimensions == rhs.output_dimensions &&
           broadcast_dimensions == rhs.broadcast_dimensions;
  }
  bool operator!=(BroadcastIntent rhs) const { return !(*this == rhs); }
};

}  // namespace
}  // namespace mhlo
}  // namespace mlir

namespace llvm {

template <>
struct DenseMapInfo<mlir::mhlo::BroadcastIntent> {
  static mlir::mhlo::BroadcastIntent getEmptyKey() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_0(mht_0_v, 235, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "getEmptyKey");

    return {DenseMapInfo<mlir::RankedTensorType>::getEmptyKey(),
            DenseMapInfo<mlir::Value>::getEmptyKey(),
            DenseMapInfo<mlir::Value>::getEmptyKey(),
            DenseMapInfo<mlir::Attribute>::getEmptyKey()};
  }
  static mlir::mhlo::BroadcastIntent getTombstoneKey() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_1(mht_1_v, 244, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "getTombstoneKey");

    return {DenseMapInfo<mlir::RankedTensorType>::getTombstoneKey(),
            DenseMapInfo<mlir::Value>::getTombstoneKey(),
            DenseMapInfo<mlir::Value>::getTombstoneKey(),
            DenseMapInfo<mlir::Attribute>::getTombstoneKey()};
  }
  static unsigned getHashValue(const mlir::mhlo::BroadcastIntent &intent) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "getHashValue");

    return hash_combine(
        DenseMapInfo<mlir::RankedTensorType>::getHashValue(intent.result_type),
        DenseMapInfo<mlir::Value>::getHashValue(intent.target_value),
        DenseMapInfo<mlir::Value>::getHashValue(intent.output_dimensions),
        DenseMapInfo<mlir::Attribute>::getHashValue(
            intent.broadcast_dimensions));
  }
  static bool isEqual(const mlir::mhlo::BroadcastIntent &lhs,
                      const mlir::mhlo::BroadcastIntent &rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_3(mht_3_v, 265, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "isEqual");

    return lhs == rhs;
  }
};

}  // namespace llvm

namespace mlir {
namespace mhlo {
namespace {

bool AllowsForElementwiseBroadcastPropagation(Operation *op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_4(mht_4_v, 279, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "AllowsForElementwiseBroadcastPropagation");

  if (op && op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>() &&
      op->hasTrait<mlir::OpTrait::Elementwise>() && op->getNumResults() == 1) {
    return true;
  }
  if (op && op->hasTrait<mlir::mhlo::OpTrait::BroadcastingElementwise>() &&
      op->getNumResults() == 1) {
    return true;
  }
  return false;
}

bool AllowsForBroadcastPropagation(Operation *op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_5(mht_5_v, 294, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "AllowsForBroadcastPropagation");

  return llvm::isa_and_nonnull<DynamicBroadcastInDimOp>(op) ||
         AllowsForElementwiseBroadcastPropagation(op);
}

DenseIntElementsAttr ComposeBroadcastDimensionsAttr(OpBuilder &builder,
                                                    DenseIntElementsAttr a,
                                                    DenseIntElementsAttr b) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_6(mht_6_v, 304, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "ComposeBroadcastDimensionsAttr");

  SmallVector<int64_t> b_vec =
      llvm::to_vector(llvm::map_range(b, [](const APInt &it) {
        return static_cast<int64_t>(it.getLimitedValue());
      }));
  SmallVector<int64_t> composed_vec = llvm::to_vector(llvm::map_range(
      a, [&](const APInt &it) { return b_vec[it.getLimitedValue()]; }));
  return builder.getI64TensorAttr(composed_vec);
}

// Find all the broadcast intents and their dependencies. Start analyzing from
// the root an collect all broadcast intents that can help broadcast propagation
// from there.
void FindBroadcastIntents(
    DynamicBroadcastInDimOp root, Block *parent_block,
    BroadcastIntent &root_bcast_intent,
    SmallVector<BroadcastIntent> &bcast_intents,
    DenseMap<BroadcastIntent, SmallVector<BroadcastIntent>>
        &bcast_intent_dependencies) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_7(mht_7_v, 325, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "FindBroadcastIntents");

  OpBuilder builder(root.getContext());

  // Use the result vector of broadcast intents as a worklist. The set of
  // broadcast intents helps to ensure their uniqueness.
  DenseSet<BroadcastIntent> bcast_intents_set;
  auto add_to_worklist_if_new = [&](BroadcastIntent bcast_intent) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_8(mht_8_v, 334, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "lambda");

    if (!bcast_intents_set.count(bcast_intent)) {
      bcast_intents_set.insert(bcast_intent);
      bcast_intents.push_back(bcast_intent);
    }
  };

  // Derive the broadcast intent associated with the root broadcast operation.
  // Add it to the worklist to seed the analysis.
  root_bcast_intent = {root.getResult().getType().cast<RankedTensorType>(),
                       root.operand(), root.output_dimensions(),
                       root.broadcast_dimensions()};
  add_to_worklist_if_new(root_bcast_intent);

  // We use result vector of broadcast intents as a worklist, the first `i`
  // intents of which have been processed.
  for (int i = 0; i < bcast_intents.size(); ++i) {
    BroadcastIntent it = bcast_intents[i];
    Operation *producer_op = it.target_value.getDefiningOp();

    // We can propagate broadcasts over (broadcasting) element-wise operations
    // and dynamic_broadcast_in_dim ops with the restriction that they must be
    // in the same block as they may depend on assuming regions.
    if (!producer_op || producer_op->getBlock() != parent_block ||
        !AllowsForBroadcastPropagation(producer_op)) {
      continue;
    }

    // We can skip broadcasting producers (dynamic_broadcast_in_dim ops) if we
    // compose their broadcasting dimensions.
    if (auto producer_bcast_op =
            llvm::dyn_cast<DynamicBroadcastInDimOp>(producer_op)) {
      DenseIntElementsAttr composed_bcast_dims = ComposeBroadcastDimensionsAttr(
          builder, producer_bcast_op.broadcast_dimensions(),
          it.broadcast_dimensions.cast<DenseIntElementsAttr>());
      BroadcastIntent bcasted_operand_intent = {
          it.result_type, producer_bcast_op.operand(), it.output_dimensions,
          composed_bcast_dims};

      // Record dependency and "recur".
      bcast_intent_dependencies[it] = {bcasted_operand_intent};
      add_to_worklist_if_new(bcasted_operand_intent);
      continue;
    }

    // We can propagate broadcasts over (broadcasting) element-wise operations.
    // Instead of broadcasting the result of such an op, we can broadcast the
    // operands and apply the element-wise operation to them.
    assert(AllowsForElementwiseBroadcastPropagation(producer_op));
    bcast_intent_dependencies[it] = {};
    for (auto operand : producer_op->getOperands()) {
      auto operand_ty = operand.getType().cast<RankedTensorType>();
      auto operand_bcast_dims = operand_ty.getRank() == 0
                                    ? builder.getI64TensorAttr({})
                                    : it.broadcast_dimensions;
      auto bcasted_operand_ty = RankedTensorType::get(
          it.result_type.getShape(), operand_ty.getElementType());
      BroadcastIntent bcasted_operand_intent = {bcasted_operand_ty, operand,
                                                it.output_dimensions,
                                                operand_bcast_dims};

      // Record dependency and "recur".
      bcast_intent_dependencies[it].push_back(bcasted_operand_intent);
      add_to_worklist_if_new(bcasted_operand_intent);
    }
  }
}

void SortBroadcastIntentsInReverseTopologicalOrder(
    SmallVector<BroadcastIntent> &bcast_intents_vec, Block *parent_block) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_9(mht_9_v, 406, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "SortBroadcastIntentsInReverseTopologicalOrder");

  // Sort broadcast intents in reverse topological order of the producer ops. We
  // can use the positions in the block for this. All broadcast intents outside
  // the block (e.g. arguments) will be sorted towards the front.
  // This ordering is independent of the output dimensions as dependencies can
  // only occur between broadcast intents of the same output dimension.
  std::sort(bcast_intents_vec.begin(), bcast_intents_vec.end(),
            [&](const BroadcastIntent &a, const BroadcastIntent &b) {
              Operation *producer_op_a = a.target_value.getDefiningOp();
              Operation *producer_op_b = b.target_value.getDefiningOp();
              bool a_in_block = producer_op_a != nullptr &&
                                producer_op_a->getBlock() == parent_block;
              bool b_in_block = producer_op_b != nullptr &&
                                producer_op_b->getBlock() == parent_block;
              if (a_in_block && b_in_block) {
                return producer_op_a->isBeforeInBlock(producer_op_b);
              }
              return !a_in_block && b_in_block;
            });
}

void SetInsertionPointToEarliestPointWithAllValuesAvailable(
    PatternRewriter &rewriter, Block *block, ValueRange values) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_10(mht_10_v, 431, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "SetInsertionPointToEarliestPointWithAllValuesAvailable");

  Operation *last_def = nullptr;
  for (Value v : values) {
    Operation *def = v.getDefiningOp();
    if (def && def->getBlock() == block) {
      if (!last_def || last_def->isBeforeInBlock(def)) last_def = def;
    }
  }
  if (last_def) {
    rewriter.setInsertionPointAfter(last_def);
  } else {
    rewriter.setInsertionPointToStart(block);
  }
}

DenseMap<BroadcastIntent, Value> RealizeBroadcastIntents(
    SmallVector<BroadcastIntent> &sorted_bcast_intents,
    DenseMap<BroadcastIntent, SmallVector<BroadcastIntent>>
        &bcast_intent_dependencies,
    Block *parent_block, PatternRewriter &rewriter) {
  // Realize broadcast intents in order. They must be sorted so that their
  // dependencies are realized before them.
  DenseMap<BroadcastIntent, Value> realizations;
  for (auto it : sorted_bcast_intents) {
    Operation *producer_op = it.target_value.getDefiningOp();
    assert(!realizations.count(it) && "expect unrealized broadcast intent");
    auto deps = bcast_intent_dependencies.find(it);

    // If we cannot propagate broadcasts further, materialize them as a
    // dynamic_broadcast_in_dim op.
    if (!producer_op || producer_op->getBlock() != parent_block ||
        !AllowsForBroadcastPropagation(producer_op)) {
      assert(deps == bcast_intent_dependencies.end() &&
             "expect no dependencies");
      SetInsertionPointToEarliestPointWithAllValuesAvailable(
          rewriter, parent_block,
          ValueRange{it.target_value, it.output_dimensions});
      realizations[it] = rewriter.create<DynamicBroadcastInDimOp>(
          it.target_value.getLoc(), it.result_type, it.target_value,
          it.output_dimensions,
          it.broadcast_dimensions.cast<DenseIntElementsAttr>());
      continue;
    }

    // For broadcast propagation across dynamic_broadcast_in_dim ops, the
    // broadcasted value is already materialized. Forward it.
    if (auto producer_bcast_op =
            llvm::dyn_cast_or_null<DynamicBroadcastInDimOp>(producer_op)) {
      assert(deps != bcast_intent_dependencies.end() &&
             deps->second.size() == 1 && "expect one dependency");
      auto bcasted_operand = realizations.find(deps->second.front());
      assert(bcasted_operand != realizations.end());
      realizations[it] = Value(bcasted_operand->second);
      continue;
    }

    // Othwerwise, realize broadcast intent for a (broadcasting) element-wise
    // operation based on the broadcasted operands.
    assert(AllowsForElementwiseBroadcastPropagation(producer_op) &&
           "expect broadcast propagation over an (broadcasting) element-wise "
           "operation");
    assert(deps != bcast_intent_dependencies.end() &&
           deps->second.size() == producer_op->getNumOperands() &&
           "expect one dependency per operand");
    auto bcasted_operands = llvm::to_vector(
        llvm::map_range(deps->second, [&](BroadcastIntent operand_intent) {
          auto bcasted_operand = realizations.find(operand_intent);
          assert(bcasted_operand != realizations.end() &&
                 "expect dependencies to be realized earlier");
          return bcasted_operand->second;
        }));
    SetInsertionPointToEarliestPointWithAllValuesAvailable(
        rewriter, parent_block, bcasted_operands);
    OperationState new_producer_op_state(
        producer_op->getLoc(), producer_op->getName().getStringRef(),
        bcasted_operands, it.result_type, producer_op->getAttrs());
    Operation *new_producer_op = rewriter.create(new_producer_op_state);
    assert(new_producer_op->getNumResults() == 1 &&
           "expect exactly one result");
    realizations[it] = new_producer_op->getResults().front();
  }

  return realizations;
}

void TransitivelyEraseUnusedSideEffectFreeOps(Operation *root,
                                              PatternRewriter &rewriter) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_11(mht_11_v, 520, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "TransitivelyEraseUnusedSideEffectFreeOps");

  // Find ops to erase.
  SmallPtrSet<Operation *, 16> ops_to_erase_set;
  SmallVector<Operation *, 16> ops_to_erase;
  SmallVector<Operation *, 16> worklist = {root};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // Erase ops only once.
    if (ops_to_erase_set.count(op)) continue;

    // Erase only operations that are unused and free of side effects.
    if (!MemoryEffectOpInterface::hasNoEffect(op) ||
        !llvm::all_of(op->getUsers(), [&](Operation *user) {
          return ops_to_erase_set.count(user);
        })) {
      continue;
    }

    // Erase and "recur".
    ops_to_erase_set.insert(op);
    ops_to_erase.push_back(op);
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) worklist.push_back(def);
    }
  }

  // Finally, erase the ops in the order of their uses.
  for (Operation *op : ops_to_erase) rewriter.eraseOp(op);
}

LogicalResult PropagateBroadcast(DynamicBroadcastInDimOp root,
                                 Block *parent_block,
                                 PatternRewriter &rewriter) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_12(mht_12_v, 556, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "PropagateBroadcast");

  // We can move broadcasts up over (i) (broadcasting) element-wise operations
  // and (i) dynamic_broadcast_in_dim ops. This way, we propagate them through
  // the IR to perform them early. Instead of broadcasting the result of such an
  // op, we can broadcast the operands and apply the element-wise operation to
  // them.
  //
  // To avoid exponential growth of the IR, we will do this in two phases:
  //   1) First, we collect all the unique broadcast intents. These are
  //      broadcasted versions of values that we are interested in. They may
  //      later be materialized as an explicit broadcast or they can be the
  //      direct result of an operation over which a broadcast was propagated.
  //   2) Then, we fulfill every broadcast intent in reverse topological order
  //      to ensure that their dependencies (the broadcasted operands) are
  //      available.

  // Find the unique broadcast intents.
  BroadcastIntent root_bcast_intent;
  SmallVector<BroadcastIntent> bcast_intents;
  DenseMap<BroadcastIntent, SmallVector<BroadcastIntent>>
      bcast_intent_dependencies;
  FindBroadcastIntents(root, parent_block, root_bcast_intent, bcast_intents,
                       bcast_intent_dependencies);

  // Fail if there is nothing but the root intent, i.e. if there is nothing to
  // rewrite here.
  if (bcast_intents.size() <= 1) {
    assert(bcast_intents.front() == root_bcast_intent && "expect root intent");
    return failure();
  }

  // Sort the broadcast intents in reverse topological order so that they can be
  // materialized and every depency is available when needed.
  SortBroadcastIntentsInReverseTopologicalOrder(bcast_intents, parent_block);

  // Realize broadcast intents.
  DenseMap<BroadcastIntent, Value> realizations = RealizeBroadcastIntents(
      bcast_intents, bcast_intent_dependencies, parent_block, rewriter);

  // Find the operations that may become redundant after replacing the root
  // operation. This allows us to transitively erase unused side effect-free
  // operations that result from this rewrite (after the root operation is no
  // longer accessible).
  SmallVector<Operation *> possibly_unused;
  for (auto operand : root->getOperands()) {
    if (Operation *def = operand.getDefiningOp())
      possibly_unused.push_back(def);
  }

  // Replace the root operation with its broadcast intent's realization.
  rewriter.replaceOp(root, realizations[root_bcast_intent]);

  // Erase all the operations that have become redundant as a result of this
  // rewrite.
  for (Operation *op : possibly_unused) {
    TransitivelyEraseUnusedSideEffectFreeOps(op, rewriter);
  }

  return success();
}

struct BroadcastPropagationPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern<DynamicBroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_13(mht_13_v, 625, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "matchAndRewrite");

    return PropagateBroadcast(op, op->getBlock(), rewriter);
  }
};

struct BroadcastPropagationPass
    : public BroadcastPropagationPassBase<BroadcastPropagationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_14(mht_14_v, 635, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "getDependentDialects");

    registry.insert<mhlo::MhloDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSDialectPSmhloPStransformsPSbroadcast_propagationDTcc mht_15(mht_15_v, 642, "", "./tensorflow/compiler/mlir/hlo/lib/Dialect/mhlo/transforms/broadcast_propagation.cc", "runOnOperation");

    MLIRContext *ctx = &getContext();

    // Collect patterns.
    RewritePatternSet patterns(ctx);
    patterns.add<BroadcastPropagationPattern>(ctx);

    // Apply broadcast propagation in reverse order to start propagation at
    // the root of broadcast chains. This avoids duplicate work.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = false;

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createBroadcastPropagationPass() {
  return std::make_unique<BroadcastPropagationPass>();
}

}  // namespace mhlo
}  // namespace mlir
