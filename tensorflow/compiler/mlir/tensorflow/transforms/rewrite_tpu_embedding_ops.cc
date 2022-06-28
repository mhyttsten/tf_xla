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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSrewrite_tpu_embedding_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSrewrite_tpu_embedding_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSrewrite_tpu_embedding_opsDTcc() {
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

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TF {

namespace {

// Rewrites RecvTPUEmbeddingActivationsOp and SendTPUEmbeddingGradients ops to
// internal variants by introducing _RecvTPUEmbeddingDeduplicationData op.
struct RewriteTPUEmbeddingOps
    : public RewriteTPUEmbeddingOpsPassBase<RewriteTPUEmbeddingOps> {
  void runOnOperation() override;
};

// Rewrites the given op to `OpT` op after adding the given operand at the end.
template <typename OpT>
OpT AddOperandAndRewriteAs(Operation* op, Value operand, OpBuilder* builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSrewrite_tpu_embedding_opsDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/mlir/tensorflow/transforms/rewrite_tpu_embedding_ops.cc", "AddOperandAndRewriteAs");

  builder->setInsertionPoint(op);
  auto operands = llvm::to_vector<4>(op->getOperands());
  operands.push_back(operand);
  auto new_op = builder->create<OpT>(op->getLoc(), op->getResultTypes(),
                                     operands, op->getAttrs());
  op->replaceAllUsesWith(new_op.getOperation()->getResults());
  op->erase();
  return new_op;
}

// Returns success if the function has at most one op of the template type and
// assigns it to `result`, if present. If there are multiple such ops, returns
// failure.
template <typename OpT>
LogicalResult GetOp(Region* region, OpT* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSrewrite_tpu_embedding_opsDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/mlir/tensorflow/transforms/rewrite_tpu_embedding_ops.cc", "GetOp");

  *result = {};
  for (auto op : region->getOps<OpT>()) {
    if (*result) return op.emitError("should be unique within a function");
    *result = op;
  }
  return success();
}

LogicalResult RunOnRegion(Region* region) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSrewrite_tpu_embedding_opsDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/mlir/tensorflow/transforms/rewrite_tpu_embedding_ops.cc", "RunOnRegion");

  RecvTPUEmbeddingActivationsOp recv_op;
  if (failed(GetOp(region, &recv_op))) return failure();

  SendTPUEmbeddingGradientsOp send_op;
  if (failed(GetOp(region, &send_op))) return failure();

  // No TPU embedding ops.
  if (!recv_op && !send_op) return success();

  Location loc = recv_op ? recv_op.getLoc() : send_op.getLoc();
  StringRef config = recv_op ? recv_op.config() : send_op.config();

  // Create _RecvTPUEmbeddingDeduplicationData op.
  OpBuilder builder(region);
  auto output_ty =
      RankedTensorType::get({}, VariantType::get(region->getContext()));
  auto dedup_op = builder.create<_RecvTPUEmbeddingDeduplicationDataOp>(
      loc, output_ty, config);

  // Rewrite RecvTPUEmbeddingActivations op to the corresponding internal op.
  if (recv_op)
    AddOperandAndRewriteAs<_RecvTPUEmbeddingActivationsOp>(recv_op, dedup_op,
                                                           &builder);

  // Rewrite SendTPUEmbeddingGradients op to the corresponding internal op and
  // then update the OperandSegmentSize attribute.
  if (send_op) {
    int32_t operand_sizes[] = {static_cast<int32_t>(send_op.N()),
                               static_cast<int32_t>(send_op.NN()), 1};
    auto attr_ty = VectorType::get(3, builder.getI32Type());
    auto operand_size_attr = DenseIntElementsAttr::get(attr_ty, operand_sizes);

    auto new_send_op = AddOperandAndRewriteAs<_SendTPUEmbeddingGradientsOp>(
        send_op, dedup_op, &builder);
    new_send_op->setAttr(new_send_op.getOperandSegmentSizeAttr(),
                         operand_size_attr);
  }
  return success();
}

void RewriteTPUEmbeddingOps::runOnOperation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSrewrite_tpu_embedding_opsDTcc mht_3(mht_3_v, 281, "", "./tensorflow/compiler/mlir/tensorflow/transforms/rewrite_tpu_embedding_ops.cc", "RewriteTPUEmbeddingOps::runOnOperation");

  FuncOp func = getOperation();
  if (failed(RunOnRegion(&func.getBody()))) return signalPassFailure();

  func.walk([&](Operation* op) {
    for (Region& region : op->getRegions()) {
      if (failed(RunOnRegion(&region))) return signalPassFailure();
    }
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> CreateRewriteTPUEmbeddingOpsPass() {
  return std::make_unique<RewriteTPUEmbeddingOps>();
}

}  // namespace TF
}  // namespace mlir
