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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc() {
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

#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

constexpr int kMaxIteration = 10;

mlir::Value GetIdentitySkippedInputs(mlir::Value val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_0(mht_0_v, 211, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "GetIdentitySkippedInputs");

  mlir::Value input = val;
  while (auto identity = llvm::dyn_cast_or_null<mlir::TF::IdentityOp>(
             input.getDefiningOp())) {
    input = identity.input();
  }
  return input;
}

bool IsZeroConstant(mlir::Value val) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_1(mht_1_v, 223, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "IsZeroConstant");

  auto const_input = llvm::dyn_cast_or_null<mlir::TF::ConstOp>(
      GetIdentitySkippedInputs(val).getDefiningOp());
  if (!const_input) return false;
  mlir::DenseFPElementsAttr attr =
      const_input.value().dyn_cast<mlir::DenseFPElementsAttr>();
  // This uses the fact that constant Attrs becomes splats, so we only need to
  // check one value.
  if (!attr || !attr.isSplat()) return false;
  return attr.getSplatValue<mlir::FloatAttr>().getValue().isZero();
}

// Extracts inputs/ops required for optimization and checks whether graph
// meets the criteria for reduction + sum optimization. The criterion are:
// a) All DTensorAllReduce operations must be sum operations.
// b) Group assignment of DTensorAllReduceOp must be the same
// c) All operands of Add op must be DTensorAllReduce operations.
mlir::LogicalResult CheckReduceAndSumOptimizationCriteria(
    mlir::Operation* add_op,
    llvm::SmallVectorImpl<mlir::Value>* reduction_inputs,
    llvm::SmallVectorImpl<mlir::TF::DTensorAllReduceOp>* reduction_ops,
    bool* can_be_reordered) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_2(mht_2_v, 247, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "CheckReduceAndSumOptimizationCriteria");

  for (mlir::Value operand : add_op->getOperands()) {
    if (IsZeroConstant(operand)) {
      reduction_inputs->emplace_back(operand);
      continue;
    }

    auto reduction_op = llvm::dyn_cast_or_null<mlir::TF::DTensorAllReduceOp>(
        operand.getDefiningOp());
    if (!reduction_op) {
      *can_be_reordered = false;
      return mlir::success();
    }

    reduction_ops->emplace_back(reduction_op);
  }

  llvm::SmallDenseSet<mlir::Attribute> reduction_group_assignments;
  for (mlir::TF::DTensorAllReduceOp reduction : *reduction_ops) {
    if (reduction.reduce_op().str() != kReduceOpAdd) {
      *can_be_reordered = false;
      return mlir::success();
    }

    mlir::DenseIntElementsAttr group_assignment;
    if (!matchPattern(reduction.group_assignment(),
                      m_Constant(&group_assignment))) {
      *can_be_reordered = false;
      return mlir::success();
    }

    reduction_group_assignments.insert(group_assignment);
    reduction_inputs->emplace_back(reduction.input());
  }

  *can_be_reordered = (reduction_group_assignments.size() == 1);
  return mlir::success();
}

// Applies optimization that reorders AllReduce + Add operations.
// For example:
//   %3 = DTensorAllReduce(%0)
//   %4 = DTensorAllReduce(%1)
//   %5 = Add(%3, %4)
//
// Is transformed to:
//   %2 = Add(%0, %1)
//   %3 = DTensorAllReduce(%2)
//
// Therefore reducing the number of Reduction/cross device communication.
mlir::LogicalResult OptimizeAllReduceAndSum(mlir::Operation* op,
                                            bool* changed) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_3(mht_3_v, 301, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "OptimizeAllReduceAndSum");

  bool can_be_reordered;
  llvm::SmallVector<mlir::TF::DTensorAllReduceOp, 4> reduction_ops;
  llvm::SmallVector<mlir::Value, 4> reduction_op_inputs;
  if (mlir::failed(CheckReduceAndSumOptimizationCriteria(
          op, &reduction_op_inputs, &reduction_ops, &can_be_reordered)))
    return mlir::failure();

  if (!can_be_reordered || reduction_ops.empty()) return mlir::success();

  // Forward the inputs from the DTensorAllReduce to the add op. Calling
  // getOperand(i).getDefiningOp() since CheckReduceAndSumOptimizationCriteria
  // checks that each input is fed by a DTensorAllReduce or a Zero constant.
  for (int i = 0; i < op->getNumOperands(); ++i) {
    if (mlir::isa<mlir::TF::DTensorAllReduceOp>(
            op->getOperand(i).getDefiningOp()))
      op->setOperand(i, op->getOperand(i).getDefiningOp()->getOperand(0));
  }

  mlir::TF::DTensorAllReduceOp first_reduction_op = reduction_ops.front();

  // Invoke reduction operation on locally added tensor once.
  // From above check `CheckOptimizationCriteria()`, we know that all reduction
  // operations that are fused reused the same group assignment value.
  // 1) Get mlir::Value that represents group assignment used for reduction.
  mlir::Value group_assignment = first_reduction_op.group_assignment();

  // Create a singe reduction operation that reduces the result of the locally
  // added tensor.
  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfterValue(op->getResult(0));
  mlir::TF::DTensorAllReduceOp all_reduce =
      builder.create<mlir::TF::DTensorAllReduceOp>(
          op->getLoc(), op->getResult(0).getType(), op->getResult(0),
          group_assignment, builder.getStringAttr(std::string(kReduceOpAdd)),
          builder.getStringAttr(first_reduction_op.device_type()));

  const auto layout_or_status = ExtractSingleLayoutFromOp(first_reduction_op);
  if (!layout_or_status.ok())
    return first_reduction_op->emitOpError(llvm::formatv(
        "Malformed layout specification for DTensorAllReduce op found: {0}",
        layout_or_status.status().error_message()));

  if (!layout_or_status->has_value())
    return first_reduction_op->emitOpError(
        "DTensorAllReduce op must have layout specification.");

  // Set target layout that is equivalent to original DTensorReduction op in
  // the graph. This is used during later optimization passes.
  SetSingleLayoutOnOp(all_reduce, layout_or_status->value());

  // Replace usages of original tf.Add op with newly created output of
  // `all_reduce`.
  op->getResult(0).replaceAllUsesExcept(
      all_reduce.output(),
      llvm::SmallPtrSet<mlir::Operation*, 1>{all_reduce.getOperation()});

  // TODO(hongjunchoi, bfontain): Consider adding optimization for the case when
  // `tree` of Add operations with DTensorAllReduce op as inputs exists.
  // Remove original tf.Add `op` and if reduction operation inputs to original
  // `op` is only used by the `op`, then remove the DTensorAllReduce op as well.
  for (mlir::Operation* original_reduction_op : reduction_ops) {
    if (original_reduction_op->use_empty()) original_reduction_op->erase();
  }

  *changed = true;
  return mlir::success();
}

mlir::Value SkipIdentityLikeOpsOutputs(mlir::Value val) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_4(mht_4_v, 373, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "SkipIdentityLikeOpsOutputs");

  while (val.hasOneUse() &&
         llvm::isa<mlir::TF::CastOp, mlir::TF::ReshapeOp, mlir::TF::IdentityOp>(
             *val.user_begin())) {
    val = val.user_begin()->getResult(0);
  }
  return val;
}

// TODO(hongjunchoi): Consider using tracing algorithm to virtually transform
// the IR and only apply optimizations when total number of DTensorAllReduce in
// the graph is reduced.
bool MayRemoveAllReduce(mlir::Operation* op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_5(mht_5_v, 388, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "MayRemoveAllReduce");

  mlir::Value op_output = op->getResult(0);
  mlir::Value value_after_identity_like_ops =
      SkipIdentityLikeOpsOutputs(op_output);
  if (value_after_identity_like_ops.hasOneUse() &&
      llvm::isa<mlir::TF::AddNOp, mlir::TF::AddV2Op, mlir::TF::AddOp>(
          *value_after_identity_like_ops.user_begin()))

    return true;

  return false;
}

// Moves DTensorAllReduce ops after IdentityLike Operations if the operation is
// connected to Add operation which may led to optimization.
// For example:
//
//  %0 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>}
//  %2 = "tf.Const"() {value = dense<0.0> : tensor<8192x916xbf16>}
//  %4= "tf.DTensorAllReduce"(%2, %0) {reduce_op = "Add"}
//  %5 = "tf.Cast"(%4){Truncate = false, device = ""}
//  %6 = "tf.Identity"(%5){Truncate = false, device = ""}
//  %7 = "tf.Const"() {value = dense<[916,8192]> : tensor<2xi32>}
//  %8 = "tf.Reshape"(%6, %7)
//
// Becomes :
//
//  %0 = "tf.Const"()
//  %2 = "tf.Const"()
//  %3 = "tf.Cast"(%2)
//  %4 = "tf.Identity"(%3)
//  %7 = "tf.Const"()
//  %8 = "tf.Reshape"(%4, %7)
//  %9 = "tf.DTensorAllReduce"(%8, %0) {reduce_op = "Add"}
void OptimizeIdentityLikeOps(mlir::Operation* op, bool* changed) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_6(mht_6_v, 425, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "OptimizeIdentityLikeOps");

  auto dtensor_all_reduce =
      llvm::dyn_cast_or_null<mlir::TF::DTensorAllReduceOp>(
          op->getOperand(0).getDefiningOp());
  if (!dtensor_all_reduce) return;
  // TODO(hongjunchoi, bfontain): Consider allowing pushing DTensorAllReduce op
  // with multiple usages if it can lead to performance optimization.
  if (!dtensor_all_reduce->hasOneUse()) return;
  if (!MayRemoveAllReduce(op)) return;

  dtensor_all_reduce->moveAfter(op);
  mlir::Value input = dtensor_all_reduce.input();
  op->setOperand(0, input);

  mlir::Value op_output = op->getResult(0);
  dtensor_all_reduce.setOperand(0, op_output);
  dtensor_all_reduce.input().setType(op_output.getType());
  dtensor_all_reduce.output().setType(op_output.getType());

  llvm::SmallPtrSet<mlir::Operation*, 4> exceptions{dtensor_all_reduce};
  op_output.replaceAllUsesExcept(dtensor_all_reduce.output(), exceptions);
  *changed = true;
}

bool CheckWhileLoopOptimizationCriteria(
    const int index, mlir::TF::WhileRegionOp while_op, mlir::Value while_output,
    mlir::Operation** add_op, mlir::TF::DTensorAllReduceOp* all_reduce_op,
    mlir::OpOperand** add_input) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_7(mht_7_v, 455, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "CheckWhileLoopOptimizationCriteria");

  // Loop variant input that is being optimized should not be used in loop
  // condition.
  mlir::Value loop_condition_input = while_op.cond().getArgument(index);
  if (!loop_condition_input.use_empty()) return false;

  // While loop output should be connected to add op.
  // If operand to while loop body terminator if from Identity op,
  // skip through the input identity operations.
  mlir::Value output_value = GetIdentitySkippedInputs(while_output);
  mlir::Operation* output_defining_op = output_value.getDefiningOp();
  if (!output_defining_op) return false;

  // TODO(hongjunchoi): Handle AddN op as well.
  if (!output_defining_op ||
      !llvm::isa<mlir::TF::AddV2Op, mlir::TF::AddOp>(output_defining_op)) {
    return false;
  }

  // Input operand of add operation should be
  // 1) DTensorAllReduce
  // 2) from block argument of while loop
  mlir::OpOperand& first_operand = output_defining_op->getOpOperand(0);
  mlir::OpOperand& second_operand = output_defining_op->getOpOperand(1);
  mlir::BlockArgument block_arg;
  mlir::TF::DTensorAllReduceOp all_reduce =
      llvm::dyn_cast_or_null<mlir::TF::DTensorAllReduceOp>(
          first_operand.get().getDefiningOp());
  if (all_reduce) {
    block_arg = second_operand.get().dyn_cast<mlir::BlockArgument>();
    *add_input = &second_operand;
  } else {
    all_reduce = llvm::dyn_cast_or_null<mlir::TF::DTensorAllReduceOp>(
        second_operand.get().getDefiningOp());
    block_arg = first_operand.get().dyn_cast<mlir::BlockArgument>();
    *add_input = &first_operand;
  }
  if (!block_arg || !all_reduce) return false;

  // DTensorAllReduce should calculate sum across devices and group assignment
  // must be statically known.
  mlir::Operation* group_assignment =
      all_reduce.group_assignment().getDefiningOp();
  if (!group_assignment || !llvm::isa<mlir::TF::ConstOp>(group_assignment))
    return false;

  if (all_reduce.reduce_op().str() != kReduceOpAdd) return false;

  // While loop block argument input connected to Add op should be
  // connected to constant operations with zero value.
  const int block_arg_index = block_arg.getArgNumber();
  mlir::OpOperand& while_input = while_op->getOpOperand(block_arg_index);
  if (!IsZeroConstant(while_input.get())) return false;

  // TODO(hongjunchoi): Handle the case when input is from DTensorAllReduce op.
  // If group assignment is the same, then the input DTensorAllReduce op can
  // also be optimized away.

  *add_op = output_defining_op;
  *all_reduce_op = all_reduce;
  return true;
}

// Extracts out DTensorAllReduce operation from while op if
// a) While op contains DTensorAllReduce op followed by an Add Operation
// b) Remaining operand of Add operation is a loop variant input of the while
//    operation with zero initial value.
//
// For example:
//
//  %0 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>}
//  %2 = "tf.Const"() {value = dense<0.0> : tensor<8192x916xbf16>}
//  WhileRegionOp(%2) {
//    %0 = "tf.A"(%2)
//    "tf.Yield"(%0)
//  }, {
//  ^bb0(%barg0: tensor<8192x916xbf16>):
//    ...
//    %0 = "tf.Const"()
//    %1 = "tf.Const"()
//    %2 = "tf.DTensorAllReduce"(%1, %0) {reduce_op = "Add"}
//    %3 = "tf.Add"(%2, %barg0)
//    "tf.Yield"(%3)
//  })
//
// Becomes :
//
//  %0 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>}
//  %2 = "tf.Const"() {value = dense<0.0> : tensor<8192x916xbf16>}
//  %4 = WhileRegionOp(%2) {
//    %0 = "tf.A"(%2)
//    "tf.Yield"(%0)
//  }, {
//  ^bb0(%barg0: tensor<8192x916xbf16>):
//    ...
//    %0 = "tf.Const"()
//    %1 = "tf.Const"()
//    %3 = "tf.Add"(%1, %barg0)
//    "tf.Yield"(%3)
//  })
//  "tf.DTensorAllReduce"(%4, %0) {reduce_op = "Add"}
mlir::LogicalResult ExtractAllReduceFromWhileOp(
    const int output_index, mlir::TF::DTensorAllReduceOp all_reduce,
    mlir::TF::WhileRegionOp while_op, mlir::OpOperand& add_input,
    mlir::Operation* add_op, bool* changed) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_8(mht_8_v, 562, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "ExtractAllReduceFromWhileOp");

  // Set add input to input of all reduce.
  mlir::Value all_reduce_input = all_reduce.input();
  const int replacement_add_input_index =
      add_input.getOperandNumber() == 0 ? 1 : 0;
  add_op->setOperand(replacement_add_input_index, all_reduce_input);

  mlir::OpBuilder builder(while_op);
  builder.setInsertionPointAfter(while_op);

  mlir::Value while_output = while_op.getResult(output_index);
  mlir::Operation* group_assignment_const =
      all_reduce.group_assignment().getDefiningOp();
  mlir::Operation* cloned_group_assignment =
      builder.clone(*group_assignment_const);

  // Create a singe reduction operation that reduces the result of the locally
  // added tensor.
  auto new_all_reduce = builder.create<mlir::TF::DTensorAllReduceOp>(
      all_reduce.getLoc(), while_output.getType(), while_output,
      cloned_group_assignment->getResult(0),
      builder.getStringAttr(std::string(kReduceOpAdd)),
      builder.getStringAttr(all_reduce.device_type()));

  const auto layout_or_status = ExtractSingleLayoutFromOp(all_reduce);
  if (!layout_or_status.ok())
    return all_reduce->emitOpError(llvm::formatv(
        "Malformed layout specification for DTensorAllReduce op found: {0}",
        layout_or_status.status().error_message()));

  if (!layout_or_status->has_value())
    return all_reduce->emitOpError(
        "DTensorAllReduce op must have layout specification.");

  // Set target layout that is equivalent to original DTensorReduction op in
  // the graph. This is used during later optimization passes.
  SetSingleLayoutOnOp(new_all_reduce, layout_or_status->value());

  llvm::SmallPtrSet<mlir::Operation*, 4> exceptions;
  exceptions.insert(new_all_reduce.getOperation());
  while_output.replaceAllUsesExcept(new_all_reduce.output(), exceptions);

  if (all_reduce.use_empty()) all_reduce.erase();

  *changed = true;
  return mlir::success();
}

mlir::LogicalResult OptimizeWhileLoopLazyAllReduce(
    mlir::TF::WhileRegionOp while_op, bool* changed) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_9(mht_9_v, 614, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "OptimizeWhileLoopLazyAllReduce");

  mlir::Operation* while_body_terminator =
      while_op.body().front().getTerminator();
  for (const auto& data :
       llvm::enumerate(while_body_terminator->getOpOperands())) {
    const int index = data.index();
    mlir::OpOperand& operand = data.value();

    mlir::Operation* add_op = nullptr;
    mlir::TF::DTensorAllReduceOp all_reduce;
    mlir::OpOperand* add_input = nullptr;
    if (!CheckWhileLoopOptimizationCriteria(index, while_op, operand.get(),
                                            &add_op, &all_reduce, &add_input))
      continue;

    // Perform while loop lazy all reduce optimization.
    if (mlir::failed(ExtractAllReduceFromWhileOp(index, all_reduce, while_op,
                                                 *add_input, add_op, changed)))
      return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult ApplyOptimization(
    mlir::func::FuncOp function,
    const llvm::SmallVectorImpl<mlir::Operation*>& identity_like_ops,
    const llvm::SmallVectorImpl<mlir::TF::WhileRegionOp>& while_ops,
    const llvm::SmallVectorImpl<mlir::Operation*>& add_ops, bool* changed) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_10(mht_10_v, 645, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "ApplyOptimization");

  // Collect and fold the reduction operations within the function.
  for (mlir::Operation* add_op : add_ops)
    if (mlir::failed(OptimizeAllReduceAndSum(add_op, changed)))
      return mlir::failure();

  for (mlir::Operation* op : identity_like_ops)
    OptimizeIdentityLikeOps(op, changed);

  for (mlir::TF::WhileRegionOp op : while_ops)
    if (mlir::failed(OptimizeWhileLoopLazyAllReduce(op, changed)))
      return mlir::failure();

  return mlir::success();
}

// Finds all potential ops that could lead to all reduce optimizations. Those
// are:
//   a) Identity like ops (e.g. Identity/Reshape/Cast) ops.
//   b) WhileRegion op
//   c) Add operations.
void CollectOptimizationCandidates(
    mlir::func::FuncOp func,
    llvm::SmallVectorImpl<mlir::Operation*>* identity_like_ops,
    llvm::SmallVectorImpl<mlir::Operation*>* add_ops,
    llvm::SmallVectorImpl<mlir::TF::WhileRegionOp>* while_ops) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_11(mht_11_v, 673, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "CollectOptimizationCandidates");

  func.walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::TF::IdentityOp, mlir::TF::CastOp, mlir::TF::ReshapeOp>(
            op))
      identity_like_ops->emplace_back(op);

    if (auto while_op = llvm::dyn_cast<mlir::TF::WhileRegionOp>(op))
      while_ops->emplace_back(while_op);

    if (llvm::isa<mlir::TF::AddOp, mlir::TF::AddV2Op, mlir::TF::AddNOp>(op))
      add_ops->emplace_back(op);
  });
}

// MLIR pass that folds constants that can be removed or deduplicated away.
struct DTensorAllReduceSumOptimization
    : public DTensorAllReduceSumOptimizationBase<
          DTensorAllReduceSumOptimization> {
  void runOnOperation() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_sum_optimizationDTcc mht_12(mht_12_v, 694, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_sum_optimization.cc", "runOnOperation");

    mlir::func::FuncOp function = getOperation();
    bool changed = true;
    int iteration = 0;

    llvm::SmallVector<mlir::Operation*, 4> identity_like_ops;
    llvm::SmallVector<mlir::Operation*, 4> add_ops;
    llvm::SmallVector<mlir::TF::WhileRegionOp, 4> while_ops;
    CollectOptimizationCandidates(function, &identity_like_ops, &add_ops,
                                  &while_ops);
    bool is_optimized = false;
    while (changed && iteration < kMaxIteration) {
      changed = false;
      if (mlir::failed(ApplyOptimization(function, identity_like_ops, while_ops,
                                         add_ops, &changed)))
        return signalPassFailure();
      iteration++;
      if (changed) is_optimized = true;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorAllReduceSumOptimization() {
  return std::make_unique<DTensorAllReduceSumOptimization>();
}

}  // namespace dtensor
}  // namespace tensorflow
