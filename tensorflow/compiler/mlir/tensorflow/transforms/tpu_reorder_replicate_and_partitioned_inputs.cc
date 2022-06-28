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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_reorder_replicate_and_partitioned_inputsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_reorder_replicate_and_partitioned_inputsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_reorder_replicate_and_partitioned_inputsDTcc() {
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

#include <cstddef>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFTPU {
namespace {

struct TPUReorderReplicateAndPartitionedInputsPass
    : public TF::TPUReorderReplicateAndPartitionedInputsPassBase<
          TPUReorderReplicateAndPartitionedInputsPass> {
  void runOnOperation() override;
};

LogicalResult ReorderReplicateAndPartitionedInputs(
    TF::TPUReplicatedInputOp replicated_input) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_reorder_replicate_and_partitioned_inputsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_reorder_replicate_and_partitioned_inputs.cc", "ReorderReplicateAndPartitionedInputs");

  if (!llvm::all_of(replicated_input.inputs(), [](Value input) {
        return llvm::isa_and_nonnull<TF::TPUPartitionedInputOp>(
            input.getDefiningOp());
      }))
    return replicated_input.emitOpError()
           << "expects all inputs from 'tf.TPUPartitionedInput' ops";

  if (replicated_input.index() != -1)
    return replicated_input->emitOpError()
           << "unsupported index = " << replicated_input.index();

  auto first_partitioned_input = llvm::cast<TF::TPUPartitionedInputOp>(
      replicated_input.getOperand(0).getDefiningOp());
  llvm::Optional<::llvm::StringRef> xla_sharding =
      first_partitioned_input._XlaSharding();
  int64_t partition_dim = first_partitioned_input.partition_dim();
  size_t num_cores_per_replica = first_partitioned_input.getNumOperands();

  for (auto operand : replicated_input.inputs().drop_front()) {
    auto partitioned_input =
        llvm::cast<TF::TPUPartitionedInputOp>(operand.getDefiningOp());
    llvm::Optional<::llvm::StringRef> op_xla_sharding =
        partitioned_input._XlaSharding();
    int64_t op_partition_dim = partitioned_input.partition_dim();
    // Abort if TPUPartitionedInput(s) do not have the same attributes.
    if (partition_dim != op_partition_dim)
      return partitioned_input->emitOpError()
             << "expects partition_dim = " << partition_dim << " but found "
             << op_partition_dim;
    if (partitioned_input.getNumOperands() != num_cores_per_replica)
      return partitioned_input->emitOpError()
             << "expects " << num_cores_per_replica << " operands but found "
             << partitioned_input.getNumOperands();
    if (xla_sharding != op_xla_sharding)
      return replicated_input.emitOpError()
             << "expects all inputs from 'tf.TPUPartitionedInput' ops to have "
                "identical XLA sharding";
  }

  // 2D Matrix to store per core per replica operands. The matrix dimensions are
  // num_cores_per_replica x num_replicas. i-th row holds the operands for i-th
  // core. j-th column holds the operands for j-th replica.
  llvm::SmallVector<llvm::SmallVector<Value, 4>, 4>
      operands_per_replica_per_core;
  operands_per_replica_per_core.resize(num_cores_per_replica);

  // Collect all operands in the 2D matrix.
  for (auto operand : replicated_input.inputs()) {
    auto pi = llvm::cast<TF::TPUPartitionedInputOp>(operand.getDefiningOp());
    for (auto& pi_operand : pi->getOpOperands()) {
      unsigned core_id = pi_operand.getOperandNumber();
      operands_per_replica_per_core[core_id].push_back(pi_operand.get());
    }
  }

  // Create new `tf.TPUReplicatedInput` ops feeding into one
  // `tf.TPUPartitionedInput` op.
  OpBuilder builder(replicated_input);
  llvm::SmallVector<Value, 4> operands_per_core;
  for (const auto& operands_per_replica : operands_per_replica_per_core) {
    auto replicate_op = builder.create<TF::TPUReplicatedInputOp>(
        replicated_input.getLoc(), replicated_input.getType(),
        operands_per_replica, replicated_input->getAttrs());
    operands_per_core.push_back(replicate_op);
  }

  auto pi = builder.create<TF::TPUPartitionedInputOp>(
      first_partitioned_input.getLoc(), replicated_input.getType(),
      operands_per_core, first_partitioned_input->getAttrs());
  replicated_input.replaceAllUsesWith(pi.output());
  return success();
}

void TPUReorderReplicateAndPartitionedInputsPass::runOnOperation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_reorder_replicate_and_partitioned_inputsDTcc mht_1(mht_1_v, 280, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_reorder_replicate_and_partitioned_inputs.cc", "TPUReorderReplicateAndPartitionedInputsPass::runOnOperation");

  auto result =
      getOperation()->walk([](TF::TPUReplicatedInputOp replicated_input) {
        if (llvm::none_of(replicated_input.inputs(), [](Value input) {
              return llvm::isa_and_nonnull<TF::TPUPartitionedInputOp>(
                  input.getDefiningOp());
            }))
          return WalkResult::advance();
        if (failed(ReorderReplicateAndPartitionedInputs(replicated_input)))
          return WalkResult::interrupt();

        assert(replicated_input->use_empty());
        replicated_input->erase();
        return WalkResult::advance();
      });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  getOperation()->walk([](TF::TPUPartitionedInputOp partitioned_input) {
    if (partitioned_input->use_empty()) partitioned_input->erase();
  });
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUReorderReplicateAndPartitionedInputsPass() {
  return std::make_unique<TPUReorderReplicateAndPartitionedInputsPass>();
}

}  // namespace TFTPU
}  // namespace mlir
