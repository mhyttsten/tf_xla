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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc() {
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

// This file implements logic for lowering TensorFlow dialect's collective
// ops (TF/XLA) to the HLO dialect.

#include <numeric>
#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/convert_op_folder.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/hlo_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/utils.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes_detail.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {

namespace {

constexpr absl::string_view kGroupSizeAttrName =
    "tf2xla.collective_info.group_size";
constexpr absl::string_view kGroupKeyAttrName =
    "tf2xla.collective_info.group_key";

class LegalizeTFCollective
    : public LegalizeTFCollectiveBase<LegalizeTFCollective> {
 public:
  void runOnOperation() override;
};

LogicalResult SetOnceModuleAttribute(StringRef attr_name,
                                     IntegerAttr attr_value, Operation* op,
                                     ModuleOp& module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_0(mht_0_v, 232, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "SetOnceModuleAttribute");

  const auto ex_attr_value = module->getAttrOfType<IntegerAttr>(attr_name);
  if (ex_attr_value == nullptr) {
    module->setAttr(attr_name, attr_value);
    return success();
  }
  if (ex_attr_value == attr_value) {
    return success();
  }
  return op->emitOpError() << "module already contains an attribute "
                           << attr_name << "=" << ex_attr_value.getInt()
                           << ", overwritting to a new value "
                           << attr_value.getInt() << " is not allowed.";
}

LogicalResult SetCollectiveInfo(IntegerAttr group_size, IntegerAttr group_key,
                                Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_1(mht_1_v, 251, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "SetCollectiveInfo");

  ModuleOp module = op->getParentOfType<ModuleOp>();
  // The StringRef cast is necessary before cxx14.
  if (failed(SetOnceModuleAttribute(
          StringRef(kGroupSizeAttrName.data(), kGroupSizeAttrName.size()),
          group_size, op, module))) {
    return failure();
  }
  if (failed(SetOnceModuleAttribute(
          StringRef(kGroupKeyAttrName.data(), kGroupKeyAttrName.size()),
          group_key, op, module))) {
    return failure();
  }
  return success();
}

LogicalResult SetCollectiveInfo(OpBuilder& builder,
                                DenseIntElementsAttr replica_groups,
                                Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_2(mht_2_v, 272, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "SetCollectiveInfo");

  // Use special group_key 0 to represent "all available devices". This
  // shall resolve to a DeviceAssignment that includes all devices intended for
  // replica_groups.
  IntegerAttr group_size = builder.getI32IntegerAttr(replica_groups.size());
  IntegerAttr group_key = builder.getI32IntegerAttr(0);
  return SetCollectiveInfo(group_size, group_key, op);
}

LogicalResult ConvertReplicaGroups(OpBuilder& builder,
                                   Value group_assignment_value,
                                   DenseIntElementsAttr& replica_groups,
                                   Operation* op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_3(mht_3_v, 287, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "ConvertReplicaGroups");

  DenseIntElementsAttr group_assignment;
  if (!matchPattern(group_assignment_value, m_Constant(&group_assignment))) {
    return op->emitOpError() << "expects constant group_assignment";
  }
  replica_groups =
      hlo::ConvertElementsAttr(group_assignment, builder.getIntegerType(64))
          .cast<DenseIntElementsAttr>();
  if (replica_groups.getType().getRank() != 2) {
    return op->emitOpError() << "group_assignment should have rank 2, got "
                             << replica_groups.getType().getRank();
  }
  return success();
}

ChannelHandle ConvertChannel(OpBuilder& builder, int64_t channel_id,
                             StringRef mode) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_4(mht_4_v, 306, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "ConvertChannel");

  if (mode == "CrossReplica") {
    return ChannelHandle();
  }
  return ChannelHandle::get(
      /*handle=*/builder.getI64IntegerAttr(channel_id),
      /*type=*/
      builder.getI64IntegerAttr(xla::ChannelHandle::DEVICE_TO_DEVICE),
      builder.getContext());
}

LogicalResult ConvertAllReduce(OpBuilder& builder, int64_t channel_id,
                               TensorType result_type,
                               DenseIntElementsAttr replica_groups,
                               StringRef mode, Value input, StringRef merge_op,
                               StringRef final_op, Operation* op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_5(mht_5_v, 324, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "ConvertAllReduce");

  builder.setInsertionPoint(op);
  ChannelHandle channel_handle = ConvertChannel(builder, channel_id, mode);
  Location loc = op->getLoc();
  Type element_type = getElementTypeOrSelf(input.getType());
  auto all_reduce = builder.create<AllReduceOp>(loc, result_type, input,
                                                replica_groups, channel_handle);
  if (merge_op == "Add") {
    BuildReduceBody<AddOp>(element_type, &all_reduce.computation(), &builder);
  } else if (merge_op == "Mul") {
    BuildReduceBody<MulOp>(element_type, &all_reduce.computation(), &builder);
  } else if (merge_op == "Min") {
    BuildReduceBody<MinOp>(element_type, &all_reduce.computation(), &builder);
  } else if (merge_op == "Max") {
    BuildReduceBody<MaxOp>(element_type, &all_reduce.computation(), &builder);
  } else {
    return op->emitOpError() << "invalid merge_op " << merge_op
                             << ", want one of [Add, Mul, Min, Max]";
  }

  Operation* result = all_reduce;
  // For "Div" final op, divide the merge result by group size.
  if (final_op == "Div") {
    int64_t replica_group_size = replica_groups.getType().getDimSize(1);
    if (replica_group_size == 0) {
      op->emitOpError()
          << "Div final_op requires a non-empty replica_groups argument.";
    }
    auto divisor =
        GetScalarConstOfType(element_type, loc, replica_group_size, &builder);
    auto broadcast_dims = GetI64ElementsAttr({}, &builder);
    result = builder.create<chlo::BroadcastDivOp>(
        loc, all_reduce.getResult(), divisor.getResult(), broadcast_dims);
  } else if (final_op != "Id") {
    return op->emitOpError()
           << "invalid final_op " << final_op << ", want one of [Id, Div]";
  }
  op->replaceAllUsesWith(result);

  op->erase();
  return success();
}

template <typename T>
class CollectiveRewritePattern : public OpRewritePattern<T> {
 public:
  // Does not take any ownership. Caller must ensure channel_id is valid during
  // life-cylce of this object.
  CollectiveRewritePattern(MLIRContext* context, int64_t* channel_id)
      : OpRewritePattern<T>(context), channel_id_(*channel_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_6(mht_6_v, 376, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "CollectiveRewritePattern");
}

 protected:
  int64_t& channel_id_;  // A unique channel_id shared by all rewrite patterns
                         // in this pass. Not thread-safe.
};

// Converts XlaAllReduce. Not thread-safe.
class ConvertXlaAllReduce
    : public CollectiveRewritePattern<TF::XlaAllReduceOp> {
 public:
  using CollectiveRewritePattern::CollectiveRewritePattern;

  LogicalResult matchAndRewrite(TF::XlaAllReduceOp all_reduce,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_7(mht_7_v, 393, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "matchAndRewrite");

    DenseIntElementsAttr replica_groups;
    if (failed(ConvertReplicaGroups(rewriter, all_reduce.group_assignment(),
                                    replica_groups, all_reduce))) {
      return failure();
    }

    // TODO(b/226201111): Stop emitting CollectiveInfo when it is no longer
    // needed.
    if (failed(SetCollectiveInfo(rewriter, replica_groups, all_reduce))) {
      return failure();
    }

    StringRef reduce_op = all_reduce.reduce_op();

    StringRef merge_op, final_op;
    if (reduce_op == "Add") {
      merge_op = "Add";
      final_op = "Id";
    } else if (reduce_op == "Mul") {
      merge_op = "Mul";
      final_op = "Id";
    } else if (reduce_op == "Min") {
      merge_op = "Min";
      final_op = "Id";
    } else if (reduce_op == "Max") {
      merge_op = "Max";
      final_op = "Id";
    } else if (reduce_op == "Mean") {
      merge_op = "Add";
      final_op = "Div";
    } else {
      return all_reduce->emitOpError()
             << "invalid reduce_op " << reduce_op
             << ", want one of [Add, Mul, Min, Max, Mean]";
    }

    int64_t channel_id = channel_id_++;
    return ConvertAllReduce(rewriter, channel_id, all_reduce.getType(),
                            replica_groups, all_reduce.mode(),
                            all_reduce.input(), merge_op, final_op, all_reduce);
  }
};

// Converts CollectiveReduceV2, with or without a preceding
// CollectiveAssignGroupV2. Not thread-safe.
class ConvertCollectiveReduceV2
    : public CollectiveRewritePattern<TF::CollectiveReduceV2Op> {
 public:
  using CollectiveRewritePattern::CollectiveRewritePattern;

  LogicalResult matchAndRewrite(TF::CollectiveReduceV2Op all_reduce,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_8(mht_8_v, 448, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "matchAndRewrite");

    TF::CollectiveAssignGroupV2Op assign_group =
        all_reduce.group_size().getDefiningOp<TF::CollectiveAssignGroupV2Op>();

    if (assign_group) {
      // Found a group assignment. Use replica_groups to represent group
      // assignment.

      if (assign_group != all_reduce.group_key()
                              .getDefiningOp<TF::CollectiveAssignGroupV2Op>()) {
        return all_reduce->emitOpError()
               << "group_size and group_key are not from the "
                  "same CollectiveAssignGroupV2Op";
      }

      DenseIntElementsAttr replica_groups;
      if (failed(ConvertReplicaGroups(rewriter, assign_group.group_assignment(),
                                      replica_groups, all_reduce))) {
        return failure();
      }

      // TODO(b/226201111): Stop emitting CollectiveInfo when it is no longer
      // needed.
      if (failed(SetCollectiveInfo(rewriter, replica_groups, all_reduce))) {
        return failure();
      }

      int64_t channel_id = channel_id_++;
      // FIXME(b/226139061): Mode should be set to CrossReplicaAndPartition
      // in order to use XLA:GPU for more than one workers.
      // The mode is set to use CrossReplica to keep the
      // behavior on the primary user of this optimized path, because
      // CrossReplicaAndPartition triggers a conflict with the channel_id
      // allocation in the communication lowering, and the user uses both set of
      // ops are used.
      return ConvertAllReduce(rewriter, channel_id, all_reduce.getType(),
                              replica_groups, /* mode=*/"CrossReplica",
                              all_reduce.input(), all_reduce.merge_op(),
                              all_reduce.final_op(), all_reduce);
    }

    // No group assignment, use separate channels per group_key.
    DenseIntElementsAttr group_size_attr;
    if (!matchPattern(all_reduce.group_size(), m_Constant(&group_size_attr))) {
      return all_reduce.emitOpError()
             << "group_size must be a compile time constant";
    }
    if (!group_size_attr.isSplat() || group_size_attr.size() != 1) {
      return all_reduce.emitOpError() << "group_size must be a scalar";
    }
    const auto group_size = group_size_attr.getSplatValue<IntegerAttr>();

    // Create a full group assignment. Empty group assignment errors when
    // final_op = "Div"
    llvm::SmallVector<int64_t> indices(group_size.getInt());
    std::iota(indices.begin(), indices.end(), 0);

    auto replica_groups = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get({1, group_size.getInt()},
                                    rewriter.getI64Type()),
        indices);

    {
      // TODO(b/226201111): Stop emitting CollectiveInfo when it is no longer
      // needed.
      DenseIntElementsAttr group_key_attr;
      if (!matchPattern(all_reduce.group_key(), m_Constant(&group_key_attr))) {
        return all_reduce.emitOpError()
               << "group_key must be a compile time constant";
      }
      if (failed(SetCollectiveInfo(
              /* group_size=*/group_size,
              /* group_key=*/group_key_attr.getSplatValue<IntegerAttr>(),
              all_reduce))) {
        return failure();
      }
    }

    // CrossReplicaAndPartition:
    // Even though TF2XLA will setup the device assignment to include
    // devices in this group as replicas before launching this module,
    // "CrossReplica" mode (no channel) produces a deadlock when
    // not using XLA SPMD expansion.
    int64_t channel_id = channel_id_++;
    return ConvertAllReduce(
        rewriter, channel_id, all_reduce.getType(), replica_groups,
        /* mode= */ "CrossReplicaAndPartition", all_reduce.input(),
        all_reduce.merge_op(), all_reduce.final_op(), all_reduce);
  }
};

void LegalizeTFCollective::runOnOperation() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSlegalize_tf_collectiveDTcc mht_9(mht_9_v, 542, "", "./tensorflow/compiler/mlir/xla/transforms/legalize_tf_collective.cc", "LegalizeTFCollective::runOnOperation");

  // FIXME(b/226139061): Figure out a way to share the channel_id with
  // send/recv Ops.
  int64_t channel_id = 1;
  auto module = getOperation();
  MLIRContext* context = module->getContext();

  RewritePatternSet patterns(context);
  patterns.insert<ConvertCollectiveReduceV2>(context, &channel_id);
  patterns.insert<ConvertXlaAllReduce>(context, &channel_id);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFCollectivePass() {
  return std::make_unique<LegalizeTFCollective>();
}

}  // namespace mhlo
}  // namespace mlir
