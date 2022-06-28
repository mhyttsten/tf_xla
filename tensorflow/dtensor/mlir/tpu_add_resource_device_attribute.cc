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
class MHTracer_DTPStensorflowPSdtensorPSmlirPStpu_add_resource_device_attributeDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPStpu_add_resource_device_attributeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPStpu_add_resource_device_attributeDTcc() {
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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"

namespace tensorflow {
namespace dtensor {
namespace {

constexpr char kFuncDeviceAttr[] = "tf.device";

// Returns whether `val` is of resource type.
bool IsResourceType(mlir::Value val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPStpu_add_resource_device_attributeDTcc mht_0(mht_0_v, 212, "", "./tensorflow/dtensor/mlir/tpu_add_resource_device_attribute.cc", "IsResourceType");

  return val.isa<mlir::BlockArgument>() && val.getType()
                                               .cast<mlir::TensorType>()
                                               .getElementType()
                                               .isa<mlir::TF::ResourceType>();
}

// Adds device attribute to `arg` with the device placement of `execute_op`
void AddPlaceholderDeviceAttributeToResource(
    mlir::BlockArgument arg, mlir::TF::TPUExecuteOp execute_op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPStpu_add_resource_device_attributeDTcc mht_1(mht_1_v, 224, "", "./tensorflow/dtensor/mlir/tpu_add_resource_device_attribute.cc", "AddPlaceholderDeviceAttributeToResource");

  // TPUExecute op is wrapped inside tf_device.Launch op for device assignment.
  auto tpu_execute_device_launch =
      execute_op->getParentOfType<mlir::tf_device::LaunchOp>();
  mlir::StringRef tpu_device_attr = tpu_execute_device_launch.device();

  auto function = execute_op->getParentOfType<mlir::func::FuncOp>();
  mlir::OpBuilder builder(execute_op);
  function.setArgAttr(arg.getArgNumber(), kFuncDeviceAttr,
                      builder.getStringAttr(tpu_device_attr));
}

// Returns AssignVariableOp that consumes output of `val`. `val` is a output
// from TPUExecute op which is wrapped inside a single tf_device.Launch
// operation. As so, output of parent launch op is queried to identify connected
// AssignVariable op.
mlir::Operation* IdentifyConnectedAssignVariableOp(mlir::Value val) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPStpu_add_resource_device_attributeDTcc mht_2(mht_2_v, 243, "", "./tensorflow/dtensor/mlir/tpu_add_resource_device_attribute.cc", "IdentifyConnectedAssignVariableOp");

  for (mlir::OpOperand& use : val.getUses()) {
    auto return_op = llvm::dyn_cast<mlir::tf_device::ReturnOp>(use.getOwner());
    if (!return_op) continue;

    auto parent_launch =
        val.getDefiningOp()->getParentOfType<mlir::tf_device::LaunchOp>();
    mlir::Value launch_output = parent_launch.getResult(use.getOperandNumber());
    for (mlir::Operation* user : launch_output.getUsers()) {
      auto assign_variable = llvm::dyn_cast<mlir::TF::AssignVariableOp>(user);
      if (!assign_variable) continue;

      return assign_variable;
    }
  }
  return nullptr;
}

struct DTensorTpuAddResourceDeviceAttribute
    : public DTensorTpuAddResourceDeviceAttributeBase<
          DTensorTpuAddResourceDeviceAttribute> {
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPStpu_add_resource_device_attributeDTcc mht_3(mht_3_v, 267, "", "./tensorflow/dtensor/mlir/tpu_add_resource_device_attribute.cc", "runOnOperation");

    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder op_builder(&context);
    mlir::ModuleOp module = getOperation();
    // For each resource value that is input or that is consumed by TPUExecute
    // op, add placeholder device attribute to the resource argument.
    mlir::WalkResult walk_result =
        module.walk([](mlir::TF::TPUExecuteOp tpu_execute) {
          for (mlir::Value tpu_input : tpu_execute.getOperands()) {
            if (IsResourceType(tpu_input))
              AddPlaceholderDeviceAttributeToResource(
                  tpu_input.cast<mlir::BlockArgument>(), tpu_execute);

            mlir::Operation* input_op = tpu_input.getDefiningOp();
            auto read_variable_op =
                llvm::dyn_cast_or_null<mlir::TF::ReadVariableOp>(input_op);
            if (!read_variable_op) continue;

            AddPlaceholderDeviceAttributeToResource(
                read_variable_op.resource().cast<mlir::BlockArgument>(),
                tpu_execute);
          }

          for (mlir::Value result : tpu_execute.getResults()) {
            mlir::Operation* assign_variable =
                IdentifyConnectedAssignVariableOp(result);
            if (assign_variable == nullptr) continue;

            AddPlaceholderDeviceAttributeToResource(
                llvm::cast<mlir::TF::AssignVariableOp>(assign_variable)
                    .resource()
                    .cast<mlir::BlockArgument>(),
                tpu_execute);
          }

          return mlir::WalkResult::advance();
        });

    if (walk_result.wasInterrupted()) return signalPassFailure();
  };
};

}  // namespace

// Adds placeholder device attributes to resource arguments of TPU functions.
// Device attribute added is consistent with device placement of TPUExecute op.
// This is required for enabling CreateTPUMergeVariablesWithExecutePass as the
// pass checks that all resources must have consistent device placement with
// TPUExecute op in order to enable buffer aliasing.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorTpuAddResourceDeviceAttribute() {
  return std::make_unique<DTensorTpuAddResourceDeviceAttribute>();
}

}  // namespace dtensor
}  // namespace tensorflow
