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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStensor_device_copy_conversionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStensor_device_copy_conversionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStensor_device_copy_conversionDTcc() {
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

// This pass folds the tf.Identity op if the operation has the same device as
// its operand.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"

namespace mlir {
namespace TF {

namespace {

constexpr const char *kDeviceAttr = "device";
constexpr const char *kTFDeviceAttr = "tf.device";

struct TensorDeviceCopyConversionPass
    : public TensorDeviceCopyConversionPassBase<
          TensorDeviceCopyConversionPass> {
  void runOnOperation() override;
};

// Folds tf.IdentityOp and tf.IdentityNOp if op device and the argument devices
// from the defining ops match.
void TensorDeviceCopyConversionPass::runOnOperation() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStensor_device_copy_conversionDTcc mht_0(mht_0_v, 217, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tensor_device_copy_conversion.cc", "TensorDeviceCopyConversionPass::runOnOperation");

  FuncOp func_op = getOperation();

  auto should_fold_op_func = [&func_op](const Value &arg,
                                        const StringAttr &op_device) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStensor_device_copy_conversionDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tensor_device_copy_conversion.cc", "lambda");

    // In TFRT TPU, tensor transfer is handled specifically by D2H and
    // H2D transfer kernels. So fold the tf.Identity op if:
    // * the identity op is placed on TPU, and
    // * the arg to the identity op is produced by a TPUExecuteOp.
    if (op_device && op_device.getValue().contains("TPU")) {
      return true;
    }

    Operation *def_op = arg.getDefiningOp();
    // If the arg to this identity op is the arg of a function, there's no
    // defining op.
    if (def_op != nullptr &&
        (isa<TF::TPUExecuteOp, TF::TPUExecuteAndUpdateVariablesOp>(def_op))) {
      return true;
    }
    if (BlockArgument block_arg = arg.dyn_cast<BlockArgument>()) {
      // Skip the folding logic if the block argument is not from the function
      // arguments. This can happen when the argument is from a while loop.
      if (block_arg.getParentRegion() != &func_op.getRegion()) {
        return false;
      }
      if (StringAttr attr = func_op.getArgAttrOfType<StringAttr>(
              block_arg.getArgNumber(), kTFDeviceAttr)) {
        return op_device == attr;
      }
    } else if (StringAttr attr = arg.getDefiningOp()->getAttrOfType<StringAttr>(
                   kDeviceAttr)) {
      return op_device == attr;
    }
    // Fold tf.Identity when arg device is not defined.
    return true;
  };

  func_op.walk([&should_fold_op_func](TF::IdentityOp op) {
    StringAttr op_device = op->getAttrOfType<StringAttr>(kDeviceAttr);
    if (should_fold_op_func(op.getOperand(), op_device)) {
      op.replaceAllUsesWith(op.getOperand());
      op.erase();
    }
    return WalkResult::advance();
  });

  func_op.walk([&should_fold_op_func](TF::IdentityNOp op) {
    StringAttr op_device = op->getAttrOfType<StringAttr>(kDeviceAttr);
    bool should_fold = llvm::all_of(
        op.getOperands(), [&op_device, &should_fold_op_func](const Value &arg) {
          return should_fold_op_func(arg, op_device);
        });
    if (should_fold) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTensorDeviceCopyConversionPass() {
  return std::make_unique<TensorDeviceCopyConversionPass>();
}

}  // namespace TF
}  // namespace mlir
