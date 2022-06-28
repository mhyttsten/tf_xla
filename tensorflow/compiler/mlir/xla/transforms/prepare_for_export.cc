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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSprepare_for_exportDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSprepare_for_exportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSprepare_for_exportDTcc() {
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

// This file implements logic for some optimizations to reduce size on export.

#include <cstdint>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes_detail.h"

#define DEBUG_TYPE "xla-prepare-for-export"

namespace mlir {
namespace mhlo {
namespace {

// Prepare module for export to XLA HLO.
struct PrepareForExportPass
    : public PrepareForExportPassBase<PrepareForExportPass> {
  void runOnOperation() override;
};

}  // end namespace

// Materializes some splat before export because it may be more efficient in
// HLOInstruction.
void prepareConstantOp(Operation *op, SplatElementsAttr attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSprepare_for_exportDTcc mht_0(mht_0_v, 223, "", "./tensorflow/compiler/mlir/xla/transforms/prepare_for_export.cc", "prepareConstantOp");

  // Arbitrarialy chosen "small" number. This could be chosen based on the
  // proto size too.
  if (attr.getNumElements() < 32) return;
  ShapedType return_type = op->getResultTypes().front().cast<ShapedType>();
  ImplicitLocOpBuilder b(op->getLoc(), op);
  ConstOp cst;
  if (auto complexTy = return_type.getElementType().dyn_cast<ComplexType>()) {
    auto tensorType = RankedTensorType::get({}, return_type.getElementType());
    assert(complexTy.getElementType().isa<FloatType>() &&
           "unexpected int complex in MHLO");
    auto complexVal = attr.getSplatValue<std::complex<APFloat>>();
    cst = b.create<ConstOp>(DenseElementsAttr::get(tensorType, complexVal));
  } else {
    cst = b.create<ConstOp>(attr.getSplatValue<Attribute>());
  }
  auto broadcast =
      b.create<BroadcastInDimOp>(return_type, cst, b.getI64TensorAttr({}));
  op->replaceAllUsesWith(broadcast);
  op->erase();
}

// Ensure that there aren't any implicit capture before exporting.
void prepareWhileOp(WhileOp while_op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSprepare_for_exportDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/mlir/xla/transforms/prepare_for_export.cc", "prepareWhileOp");

  llvm::SetVector<Value> implicit_inputs;
  getUsedValuesDefinedAbove(while_op->getRegions(), implicit_inputs);
  // Each captured value has to be passed as operand to the while, become then
  // an operand to the condition region and the body region, and an extra
  // operand to the return op in the body. It also becomes an extra result for
  // the while operation, even if it is unused.
  // We'll process the captured values one at a time and patch the body and
  // condition regions as we go, but we'll accumulate the new operands and
  // result type and recreate a new while op to replace the existing one at the
  // end.
  SmallVector<Type> returned_types(while_op->getResultTypes().begin(),
                                   while_op->getResultTypes().end());
  SmallVector<Value> operands(while_op->getOperands().begin(),
                              while_op->getOperands().end());
  Region &cond_region = while_op.cond();
  Region &body_region = while_op.body();

  for (Value input : implicit_inputs) {
    returned_types.push_back(input.getType());
    operands.push_back(input);

    Value cond_arg =
        cond_region.front().addArgument(input.getType(), input.getLoc());
    Value body_arg =
        body_region.front().addArgument(input.getType(), input.getLoc());
    for (OpOperand &operand : llvm::make_early_inc_range(input.getUses())) {
      if (cond_region.isAncestor(operand.getOwner()->getParentRegion()))
        operand.set(cond_arg);
      else if (body_region.isAncestor(operand.getOwner()->getParentRegion()))
        operand.set(body_arg);
    }
    auto return_op = cast<mhlo::ReturnOp>(body_region.front().back());
    return_op->insertOperands(return_op->getNumOperands(), body_arg);
  }
  OpBuilder builder(while_op);
  auto new_while_op = builder.create<mhlo::WhileOp>(while_op.getLoc(),
                                                    returned_types, operands);
  new_while_op.cond().getBlocks().clear();
  new_while_op.cond().takeBody(while_op.cond());
  new_while_op.body().getBlocks().clear();
  new_while_op.body().takeBody(while_op.body());
  for (auto zipped_results :
       llvm::zip_first(while_op.getResults(), new_while_op.getResults()))
    std::get<0>(zipped_results).replaceAllUsesWith(std::get<1>(zipped_results));
  while_op->erase();
}

void prepareBroadcastInDim(BroadcastInDimOp bcast) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSprepare_for_exportDTcc mht_2(mht_2_v, 300, "", "./tensorflow/compiler/mlir/xla/transforms/prepare_for_export.cc", "prepareBroadcastInDim");

  DenseIntElementsAttr dims = bcast.broadcast_dimensions();
  // If dimensions aren't sorted, there is a transpose fused into the op, which
  // XLA Builder does not support, we unfuse here.
  if (llvm::is_sorted(dims.getValues<int64_t>())) return;

  // We need to compute a permutation that sorts the dimension before the
  // broadcast.
  // If the dims are [2, 4, 1], we create an array of indices: [0, 1, 2] and we
  // sort it using the values of the first array to produce [2, 0, 1] which
  // gives us the operand for the transpose.
  SmallVector<int64_t> transposedDim =
      to_vector(llvm::seq<int64_t>(0, dims.size()));
  auto rawDims = dims.getValues<int64_t>();
  llvm::sort(transposedDim, [&](int64_t lhs, int64_t rhs) {
    return rawDims[lhs] < rawDims[rhs];
  });
  OpBuilder builder(bcast);
  bcast.setOperand(builder.create<TransposeOp>(
      bcast.getLoc(), bcast.operand(),
      DenseIntElementsAttr::get(dims.getType(), transposedDim)));
  // Now reuse the original broadcast_dimensions and sort it.
  transposedDim.assign(rawDims.begin(), rawDims.end());
  llvm::sort(transposedDim);
  bcast.broadcast_dimensionsAttr(
      DenseIntElementsAttr::get(dims.getType(), transposedDim));
}

void PrepareForExportPass::runOnOperation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStransformsPSprepare_for_exportDTcc mht_3(mht_3_v, 331, "", "./tensorflow/compiler/mlir/xla/transforms/prepare_for_export.cc", "PrepareForExportPass::runOnOperation");

  getOperation().walk([&](Operation *op) {
    mlir::SplatElementsAttr attr;
    if (matchPattern(op, m_Constant(&attr))) return prepareConstantOp(op, attr);

    if (auto whileOp = dyn_cast<WhileOp>(op)) return prepareWhileOp(whileOp);
    if (auto bcastOp = dyn_cast<BroadcastInDimOp>(op))
      return prepareBroadcastInDim(bcastOp);
  });
}

std::unique_ptr<OperationPass<FuncOp>> CreatePrepareForExport() {
  return std::make_unique<PrepareForExportPass>();
}

}  // end namespace mhlo
}  // end namespace mlir
