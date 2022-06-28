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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfunctional_control_flow_to_regionsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfunctional_control_flow_to_regionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfunctional_control_flow_to_regionsDTcc() {
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

// This transformation pass transforms functional control flow operations in the
// TensorFlow dialect to their region based counterparts, i.e.,
// tf.If -> tf.IfRegion and tf.While -> tf.WhileRegion

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

#define DEBUG_TYPE "tf-functional-cf-to-region"

namespace mlir {
namespace TF {

namespace {

struct FunctionalControlFlowToRegions
    : public TF::FunctionalControlFlowToRegionsPassBase<
          FunctionalControlFlowToRegions> {
  void runOnOperation() override;
};

// Creates a call to function `func` in region `caller_region`. Use `args` as
// the call arguments, and terminate the region with a yield. The arguments are
// cast to the required type before the call. `use_region_args` control whether
// the input arguments are used as is (for IfOp) or block arguments of the same
// type as the input arguments are created and then used as call arguments (for
// While).
YieldOp CreateCall(Operation* op, FuncOp func, Region& caller_region,
                   ValueRange args, bool use_region_args) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfunctional_control_flow_to_regionsDTcc mht_0(mht_0_v, 228, "", "./tensorflow/compiler/mlir/tensorflow/transforms/functional_control_flow_to_regions.cc", "CreateCall");

  assert(caller_region.empty() &&
         "Expected empty region for newly created ops");
  OpBuilder builder(caller_region);
  Block* entry = builder.createBlock(&caller_region);

  auto loc = op->getLoc();
  if (use_region_args) {
    auto inputs = func.getFunctionType().getInputs();
    entry->addArguments(inputs, SmallVector<Location>(inputs.size(), loc));
    args = entry->getArguments();
  }
  llvm::SmallVector<Value, 4> casted_args;
  casted_args.reserve(func.getNumArguments());
  for (const auto& ArgAndType : zip(args, func.getFunctionType().getInputs())) {
    Value arg = std::get<0>(ArgAndType);
    Type expected_type = std::get<1>(ArgAndType);
    if (arg.getType() != expected_type) {
      arg = builder.create<CastOp>(loc, expected_type, arg,
                                   /*Truncate=*/builder.getBoolAttr(false));
    }
    casted_args.push_back(arg);
  }
  auto call = builder.create<func::CallOp>(loc, func, casted_args);
  return builder.create<YieldOp>(loc, call.getResults());
}

// Converts the condition for an IfOp/WhileOp to a boolean value.
Value ConvertConditionToBoolean(Operation* op, Value cond) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfunctional_control_flow_to_regionsDTcc mht_1(mht_1_v, 259, "", "./tensorflow/compiler/mlir/tensorflow/transforms/functional_control_flow_to_regions.cc", "ConvertConditionToBoolean");

  if (auto ranked_type = cond.getType().dyn_cast<RankedTensorType>())
    if (ranked_type.getRank() == 0 &&
        ranked_type.getElementType().isSignlessInteger(1))
      return cond;

  OpBuilder builder(op);
  return builder.create<TF::ToBoolOp>(op->getLoc(), cond);
}

// Transform a functional IfOp to a region based IfRegionOp.
LogicalResult ConvertIfOp(IfOp if_op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfunctional_control_flow_to_regionsDTcc mht_2(mht_2_v, 273, "", "./tensorflow/compiler/mlir/tensorflow/transforms/functional_control_flow_to_regions.cc", "ConvertIfOp");

  Value cond = ConvertConditionToBoolean(if_op, if_op.cond());
  OpBuilder builder(if_op);
  auto if_region = builder.create<TF::IfRegionOp>(
      if_op.getLoc(), if_op.getResultTypes(), cond, if_op.is_stateless(),
      builder.getStringAttr(if_op.then_function().getName()),
      builder.getStringAttr(if_op.else_function().getName()));
  CopyDeviceAndUnderscoredAttributes(if_op, if_region);

  CreateCall(if_op, if_op.then_function(),
             /*caller_region=*/if_region.then_branch(), if_op.input(),
             /*use_region_args=*/false);
  CreateCall(if_op, if_op.else_function(),
             /*caller_region=*/if_region.else_branch(), if_op.input(),
             /*use_region_args=*/false);
  if_op.replaceAllUsesWith(if_region.getResults());
  if_op.erase();
  return success();
}

LogicalResult ConvertWhileOp(WhileOp while_op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfunctional_control_flow_to_regionsDTcc mht_3(mht_3_v, 296, "", "./tensorflow/compiler/mlir/tensorflow/transforms/functional_control_flow_to_regions.cc", "ConvertWhileOp");

  auto while_region = OpBuilder(while_op).create<TF::WhileRegionOp>(
      while_op.getLoc(), while_op.getResultTypes(), while_op.input(),
      while_op.parallel_iterations(), while_op.is_stateless(),
      while_op.shape_invariant());
  CopyDeviceAndUnderscoredAttributes(while_op, while_region);

  YieldOp cond_yield =
      CreateCall(while_op, while_op.cond_function(),
                 /*caller_region=*/while_region.cond(), while_op.input(),
                 /*use_region_args=*/true);
  Value i1_cond =
      ConvertConditionToBoolean(cond_yield, cond_yield.getOperand(0));
  cond_yield.setOperand(0, i1_cond);

  CreateCall(while_op, while_op.body_function(),
             /*caller_region=*/while_region.body(), while_op.input(),
             /*use_region_args=*/true);
  while_op.replaceAllUsesWith(while_region.getResults());
  while_op.erase();
  return success();
}

void FunctionalControlFlowToRegions::runOnOperation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSfunctional_control_flow_to_regionsDTcc mht_4(mht_4_v, 322, "", "./tensorflow/compiler/mlir/tensorflow/transforms/functional_control_flow_to_regions.cc", "FunctionalControlFlowToRegions::runOnOperation");

  ModuleOp module = getOperation();
  auto result = module.walk([](Operation* op) {
    if (IfOp if_op = llvm::dyn_cast<IfOp>(op)) {
      if (failed(ConvertIfOp(if_op))) {
        op->emitOpError() << "failed to convert to region form";
        return WalkResult::interrupt();
      }
    } else if (auto while_op = llvm::dyn_cast<WhileOp>(op)) {
      if (failed(ConvertWhileOp(while_op))) {
        op->emitOpError() << "failed to convert to region form";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return signalPassFailure();
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFFunctionalControlFlowToRegions() {
  return std::make_unique<FunctionalControlFlowToRegions>();
}

}  // namespace TF
}  // namespace mlir
