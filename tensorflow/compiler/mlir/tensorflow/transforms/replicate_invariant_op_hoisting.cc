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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplicate_invariant_op_hoistingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplicate_invariant_op_hoistingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplicate_invariant_op_hoistingDTcc() {
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

// This pass hoists replicate invariant ops, or ops that yield the same
// result(s) regardless of replication, out of their respective replicate.

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";

struct ReplicateInvariantOpHoistingPass
    : public TF::ReplicateInvariantOpHoistingPassBase<
          ReplicateInvariantOpHoistingPass> {
  void runOnOperation() override;
};

void MakeShapeOpInvariant(tf_device::ReplicateOp replicate_op, int num_replicas,
                          Block* replicate_block, TF::ShapeOp shape_op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplicate_invariant_op_hoistingDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/mlir/tensorflow/transforms/replicate_invariant_op_hoisting.cc", "MakeShapeOpInvariant");

  Value input = shape_op.input();
  // If ShapeOp operand is replicate tensor block argument, replace with the
  // associated first replica operand.
  if (auto block_arg = input.dyn_cast<BlockArgument>()) {
    if (block_arg.getOwner() != replicate_block) return;

    shape_op.setOperand(replicate_op.GetReplicaOperandForBlockArgument(
        block_arg, /*replica=*/0));

    return;
  }

  Operation* input_def = input.getDefiningOp();

  // If ShapeOp operand is a ReadVariableOp result where the ReadVariableOp
  // operand is a replicate resource block argument, replace ShapeOp with
  // VariableShapeOp and use the associated first replica operand as its
  // operand.
  auto read_var_op = llvm::dyn_cast<TF::ReadVariableOp>(input_def);
  if (!read_var_op) return;

  // TODO(lyandy): Check if resource (first replica or replicate block arg)
  // shape has not changed in replicate prior to read. Currently after both
  // ResourceOpLiftingPass and TPURewritePass, there should not be any updates
  // to resources prior to their respective ReadVariableOp.
  if (auto block_arg = read_var_op.resource().dyn_cast<BlockArgument>()) {
    if (block_arg.getOwner() != replicate_block) return;

    OpBuilder builder(shape_op);
    auto new_shape_op = builder.create<TF::VariableShapeOp>(
        shape_op.getLoc(), shape_op.getType(),
        replicate_op.GetReplicaOperandForBlockArgument(block_arg,
                                                       /*replica=*/0));
    shape_op.replaceAllUsesWith(new_shape_op.getOperation());
    shape_op.erase();
  }
}

// Check if op uses a device from a list of virtual devices.
bool UsesVirtualDevice(const Optional<DictionaryAttr>& virtual_devices,
                       Operation* operation) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplicate_invariant_op_hoistingDTcc mht_1(mht_1_v, 260, "", "./tensorflow/compiler/mlir/tensorflow/transforms/replicate_invariant_op_hoisting.cc", "UsesVirtualDevice");

  if (!virtual_devices.hasValue()) return false;

  auto result = operation->walk([&](Operation* op) {
    StringAttr op_device = op->getAttrOfType<StringAttr>(kDeviceAttr);
    if (!op_device) return WalkResult::advance();

    if (virtual_devices.getValue().get(op_device.getValue()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Checks if op and inner op operands are all replicate invariant.
bool IsOpReplicateInvariant(Region* replicate_region, Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplicate_invariant_op_hoistingDTcc mht_2(mht_2_v, 278, "", "./tensorflow/compiler/mlir/tensorflow/transforms/replicate_invariant_op_hoisting.cc", "IsOpReplicateInvariant");

  auto ancestor_of_replicate = [&](Region* region) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplicate_invariant_op_hoistingDTcc mht_3(mht_3_v, 282, "", "./tensorflow/compiler/mlir/tensorflow/transforms/replicate_invariant_op_hoisting.cc", "lambda");

    return region && region->isProperAncestor(replicate_region);
  };

  for (Value operand : op->getOperands())
    if (!ancestor_of_replicate(operand.getParentRegion())) return false;

  bool has_replicate_operands = false;
  visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand* operand) {
    if (!ancestor_of_replicate(operand->get().getParentRegion()))
      has_replicate_operands = true;
  });

  return !has_replicate_operands;
}

// Hoists replicate invariant ops out of associated `tf_device.replicate` op.
// Ops to be hoisted are determined by if all of their operands are replicate
// invariant. Shape ops are rewritten to be invariant when possible, prior to
// hoisting ops.
void HoistReplicateInvariantOps(tf_device::ReplicateOp replicate_op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplicate_invariant_op_hoistingDTcc mht_4(mht_4_v, 305, "", "./tensorflow/compiler/mlir/tensorflow/transforms/replicate_invariant_op_hoisting.cc", "HoistReplicateInvariantOps");

  const int num_replicas = replicate_op.n();
  Block* replicate_block = &replicate_op.GetBody();

  replicate_op.walk([&](TF::ShapeOp shape_op) {
    MakeShapeOpInvariant(replicate_op, num_replicas, replicate_block, shape_op);
  });

  Region* replicate_region = &replicate_op.body();
  Optional<DictionaryAttr> virtual_device_list = replicate_op.devices();
  for (Operation& inner_op :
       llvm::make_early_inc_range(replicate_op.GetBody())) {
    if (llvm::isa<tf_device::ReturnOp>(inner_op)) continue;
    // Skip hoisting if the inner op device attribute is a virtual device
    // defined by tf_device.replicate.
    if (UsesVirtualDevice(virtual_device_list, &inner_op)) continue;

    if (IsOpReplicateInvariant(replicate_region, &inner_op))
      inner_op.moveBefore(replicate_op);
  }
}

void ReplicateInvariantOpHoistingPass::runOnOperation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplicate_invariant_op_hoistingDTcc mht_5(mht_5_v, 330, "", "./tensorflow/compiler/mlir/tensorflow/transforms/replicate_invariant_op_hoisting.cc", "ReplicateInvariantOpHoistingPass::runOnOperation");

  getOperation().walk(
      [](tf_device::ReplicateOp op) { HoistReplicateInvariantOps(op); });
}
}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateReplicateInvariantOpHoistingPass() {
  return std::make_unique<ReplicateInvariantOpHoistingPass>();
}

}  // namespace TFDevice
}  // namespace mlir
