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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_update_embedding_enqueue_op_inputsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_update_embedding_enqueue_op_inputsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_update_embedding_enqueue_op_inputsDTcc() {
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
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";
constexpr char kTPUEmbeddingAttr[] = "_tpu_embedding_layer";

struct TPUUpdateEmbeddingEnqueueOpInputsPass
    : public TF::TPUUpdateEmbeddingEnqueueOpInputsPassBase<
          TPUUpdateEmbeddingEnqueueOpInputsPass> {
  void runOnOperation() override;
};

// Extracts `_tpu_embedding_layer` attribute from TPU embedding ops and
// clear the attribute from the operation. This ensures that future optimization
// passes does not trigger additional logic due to presence of this attribute.
LogicalResult ExtractEmbeddingAttribute(
    Operation* op, llvm::StringMap<Operation*>* embedding_op_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_update_embedding_enqueue_op_inputsDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_update_embedding_enqueue_op_inputs.cc", "ExtractEmbeddingAttribute");

  auto embedding_attr = op->getAttrOfType<StringAttr>(kTPUEmbeddingAttr);
  if (!embedding_attr) return mlir::success();

  if (!embedding_op_map->insert({embedding_attr.getValue(), op}).second)
    return op->emitOpError(
        "found duplicate TPU embedding ops potentially from multiple "
        "TPUEmbedding layers");

  op->removeAttr(kTPUEmbeddingAttr);
  return success();
}

LogicalResult FindTPUEmbeddingOps(
    FuncOp func_op, llvm::StringMap<Operation*>* enqueue_op_map,
    llvm::StringMap<Operation*>* recv_activation_op_map,
    llvm::StringMap<Operation*>* send_gradient_op_map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_update_embedding_enqueue_op_inputsDTcc mht_1(mht_1_v, 240, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_update_embedding_enqueue_op_inputs.cc", "FindTPUEmbeddingOps");

  auto walk_result = func_op.walk([&](Operation* op) {
    if (llvm::isa<TF::RecvTPUEmbeddingActivationsOp>(op))
      if (failed(ExtractEmbeddingAttribute(op, recv_activation_op_map)))
        return WalkResult::interrupt();

    if (llvm::isa<TF::SendTPUEmbeddingGradientsOp>(op))
      if (failed(ExtractEmbeddingAttribute(op, send_gradient_op_map)))
        return WalkResult::interrupt();

    if (llvm::isa<TF::EnqueueTPUEmbeddingSparseTensorBatchOp,
                  TF::EnqueueTPUEmbeddingRaggedTensorBatchOp,
                  TF::EnqueueTPUEmbeddingArbitraryTensorBatchOp>(op))
      if (failed(ExtractEmbeddingAttribute(op, enqueue_op_map)))
        return WalkResult::interrupt();

    return WalkResult::advance();
  });
  return failure(walk_result.wasInterrupted());
}

// Updates the operand of TPU embedding enqueue ops depending on whether
// the graph is in training mode or in non-training mode.
// If SendTPUEmbeddingGradients op is present, this means that graph is in
// training mode. As so, correctly feed in `then` branch value of SelectV2
// operand as inputs to the TPU embedding enqueue ops.
LogicalResult UpdateEmbeddingEnqueueOpInput(
    const llvm::StringMap<Operation*>& enqueue_op_map,
    const llvm::StringMap<Operation*>& recv_activation_op_map,
    const llvm::StringMap<Operation*>& send_gradient_op_map,
    OpBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_update_embedding_enqueue_op_inputsDTcc mht_2(mht_2_v, 273, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_update_embedding_enqueue_op_inputs.cc", "UpdateEmbeddingEnqueueOpInput");

  for (const auto& it : enqueue_op_map) {
    const auto& embedding_attr = it.getKey();
    Operation* embedding_op = it.second;
    if (!recv_activation_op_map.count(embedding_attr))
      return embedding_op->emitOpError()
             << "must have a corresponding '"
             << TF::RecvTPUEmbeddingActivationsOp::getOperationName() << "' op";

    // TPU Embedding enqueue ops take different inputs depending on whether
    // graph is in training mode or in eval/prediction mode. During training,
    // the mode parameter for TPUEmbeddingEnqueue op must be `train` and for
    // evaluation or prediction, mode must be set to `inference`.
    // If SendTPUEmbeddingGradients op exists in the graph, then graph is
    // in training mode, so create a const op with value `train` use the
    // output value of the constant as an operand to the TPU embedding
    // enqueue op.
    bool is_training = send_gradient_op_map.count(embedding_attr);

    // The last operand of TPUEmbeddingEnqueue ops is the mode which
    // represents whether graph is in training mode or in evaluation mode.
    auto& mode_enqueue_operand =
        embedding_op->getOpOperand(embedding_op->getNumOperands() - 1);

    llvm::SmallVector<StringRef, 1> mode_string_value;
    mode_string_value.emplace_back(is_training ? "train" : "inference");
    builder->setInsertionPoint(embedding_op);
    auto enqueue_mode = builder->create<TF::ConstOp>(
        embedding_op->getLoc(),
        DenseStringElementsAttr::get(
            RankedTensorType::get({}, builder->getType<TF::StringType>()),
            mode_string_value));

    auto outside_compilation_attr =
        embedding_op->getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr);
    if (outside_compilation_attr)
      enqueue_mode->setAttr(kXlaOutsideCompilationAttr,
                            outside_compilation_attr);

    mode_enqueue_operand.set(enqueue_mode);
  }

  return success();
}

void TPUUpdateEmbeddingEnqueueOpInputsPass::runOnOperation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_update_embedding_enqueue_op_inputsDTcc mht_3(mht_3_v, 321, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_update_embedding_enqueue_op_inputs.cc", "TPUUpdateEmbeddingEnqueueOpInputsPass::runOnOperation");

  OpBuilder builder(&getContext());
  auto func_op = getOperation();

  // All TPU embedding layer related ops are annotated with
  // `_tpu_embedding_layer` attribute along with corresponding string attribute.
  // Store all tpu embedding layer related ops with value of
  // `_tpu_embedding_layer` attribute as map key.
  llvm::StringMap<Operation*> enqueue_op_map;
  llvm::StringMap<Operation*> recv_activation_op_map;
  llvm::StringMap<Operation*> send_gradient_op_map;
  if (failed(FindTPUEmbeddingOps(func_op, &enqueue_op_map,
                                 &recv_activation_op_map,
                                 &send_gradient_op_map)))
    return signalPassFailure();

  if (enqueue_op_map.size() != recv_activation_op_map.size()) {
    func_op.emitError() << "expects the number of embedding enqueue ops to "
                           "match the number of '"
                        << TF::RecvTPUEmbeddingActivationsOp::getOperationName()
                        << "' ops";
    return signalPassFailure();
  }

  if (failed(UpdateEmbeddingEnqueueOpInput(enqueue_op_map,
                                           recv_activation_op_map,
                                           send_gradient_op_map, &builder)))
    return signalPassFailure();
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUUpdateEmbeddingEnqueueOpInputsPass() {
  return std::make_unique<TPUUpdateEmbeddingEnqueueOpInputsPass>();
}

}  // namespace TFTPU
}  // namespace mlir
