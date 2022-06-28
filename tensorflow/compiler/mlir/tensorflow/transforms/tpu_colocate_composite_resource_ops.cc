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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_colocate_composite_resource_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_colocate_composite_resource_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_colocate_composite_resource_opsDTcc() {
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

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {
namespace {

// Pass that co-locates resource ops that use composite device resources
// (packed tensors) with the underlying physical TPU device.
struct TPUColocateCompositeResourceOps
    : public TF::TPUColocateCompositeResourceOpsPassBase<
          TPUColocateCompositeResourceOps> {
  void runOnOperation() override;
};

// Wraps single op in `tf_device.launch` for explicit device assignment.
void WrapOpInLaunch(OpBuilder* builder, Location loc, Operation* op,
                    llvm::StringRef device) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_colocate_composite_resource_opsDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_colocate_composite_resource_ops.cc", "WrapOpInLaunch");

  builder->setInsertionPoint(op);
  auto launch = builder->create<tf_device::LaunchOp>(
      loc, builder->getStringAttr(device), op->getResultTypes());
  launch.body().push_back(new Block);
  op->replaceAllUsesWith(launch);

  builder->setInsertionPointToEnd(&launch.GetBody());
  builder->create<tf_device::ReturnOp>(loc, op->getResults());

  // Move op inside cluster.
  op->moveBefore(launch.GetBody().getTerminator());
}

llvm::SmallVector<Operation*, 4> GetResourceOpsUsingCompositeArgsInReplicate(
    tf_device::ReplicateOp replicate) {
  llvm::SmallVector<Operation*, 4> resource_users;
  const auto add_resource_op_to_list = [&resource_users](Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_colocate_composite_resource_opsDTcc mht_1(mht_1_v, 234, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_colocate_composite_resource_ops.cc", "lambda");

    if (!llvm::isa<TF::AssignVariableOp, TF::ReadVariableOp>(op)) return;

    resource_users.emplace_back(op);
  };

  llvm::SmallVector<Operation*, 4> resource_users_to_visit;
  for (auto composite_arguments : replicate.GetPackedBlockArguments()) {
    for (auto resource_user : composite_arguments.getUsers())
      resource_users_to_visit.emplace_back(resource_user);
  }

  while (!resource_users_to_visit.empty()) {
    llvm::SmallVector<Operation*, 4> new_resource_users;

    for (auto resource_user : resource_users_to_visit) {
      add_resource_op_to_list(resource_user);

      // Account for pass-through identity ops.
      if (auto pass_through_identity =
              llvm::dyn_cast<TF::IdentityOp>(resource_user)) {
        for (auto identity_user : pass_through_identity.output().getUsers()) {
          new_resource_users.emplace_back(identity_user);
        }
      }
    }
    resource_users_to_visit.swap(new_resource_users);
  }

  return resource_users;
}

void ColocateCompositeResourceOpsInReplicate(
    tf_device::ReplicateOp replicate_op, OpBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_colocate_composite_resource_opsDTcc mht_2(mht_2_v, 270, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_colocate_composite_resource_ops.cc", "ColocateCompositeResourceOpsInReplicate");

  auto devices = replicate_op.devices();
  if (!devices) return;
  if (!devices.getValue().get(tensorflow::GetDeviceAliasForLogicalCore(0)))
    return;

  const auto composite_resource_users =
      GetResourceOpsUsingCompositeArgsInReplicate(replicate_op);
  for (auto resource_user : composite_resource_users) {
    WrapOpInLaunch(builder, resource_user->getLoc(), resource_user,
                   tensorflow::GetDeviceAliasForLogicalCore(0));
  }
}

void TPUColocateCompositeResourceOps::runOnOperation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_colocate_composite_resource_opsDTcc mht_3(mht_3_v, 287, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_colocate_composite_resource_ops.cc", "TPUColocateCompositeResourceOps::runOnOperation");

  // Find all the executes first, since we will mutate the nodes around each
  // execute in the same tf_device.replicate op.
  llvm::SmallVector<tf_device::LaunchOp, 8> execute_launches;
  getOperation().walk([&](tf_device::LaunchOp op) {
    if (op.WrapsSingleOp() &&
        llvm::isa<TF::TPUExecuteOp, TF::TPUExecuteAndUpdateVariablesOp>(
            op.GetBody().front()))
      execute_launches.push_back(op);
  });

  OpBuilder builder(&getContext());
  for (auto execute_launch : execute_launches) {
    auto replicate = execute_launch->getParentOfType<tf_device::ReplicateOp>();
    if (!replicate) continue;

    ColocateCompositeResourceOpsInReplicate(replicate, &builder);
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTPUColocateCompositeResourceOps() {
  return std::make_unique<TPUColocateCompositeResourceOps>();
}

}  // namespace TFTPU
}  // namespace mlir
