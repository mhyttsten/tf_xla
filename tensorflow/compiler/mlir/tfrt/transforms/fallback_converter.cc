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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfallback_converterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfallback_converterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfallback_converterDTcc() {
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
#include "tensorflow/compiler/mlir/tfrt/transforms/fallback_converter.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_compiler {

FallbackConverter::FallbackConverter(mlir::MLIRContext *context)
    : builder_(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfallback_converterDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/mlir/tfrt/transforms/fallback_converter.cc", "FallbackConverter::FallbackConverter");

  addConversion([](tfrt::compiler::ChainType type) { return type; });
  addConversion([](tfrt::fallback::TFTensorType type) { return type; });
  addConversion([=](mlir::TensorType type) -> llvm::Optional<mlir::Type> {
    // Ref types are not supported in both compiler and runtime.
    if (type.getElementType().isa<mlir::TF::TensorFlowRefType>()) {
      return llvm::None;
    }

    return builder_.getType<tfrt::fallback::TFTensorType>();
  });
  addConversion([=](mlir::Type type) -> llvm::Optional<mlir::Type> {
    if (type == builder_.getI1Type()) return type;
    return llvm::None;
  });
}

mlir::Value ConvertCoreRTTensorHandleToFallbackTensor(
    mlir::Location loc, llvm::StringRef device, mlir::Value value,
    mlir::ConversionPatternRewriter &rewriter) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfallback_converterDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/mlir/tfrt/transforms/fallback_converter.cc", "ConvertCoreRTTensorHandleToFallbackTensor");

  if (value.getType().isa<tfrt::fallback::TFTensorType>()) return value;

  if (!value.getType().isa<tfrt::corert::TensorHandleType>()) return {};

  mlir::OpBuilder::InsertionGuard guard(rewriter);

  if (device.endswith("CPU:0") && !device.startswith("/job:")) {
    // Canonicalize CPU device name. This is needed as corert library only uses
    // the default CPU device name (i.e.
    // "/job:localhost/replica:0/task:0/device:CPU:0") and cannot recoganize
    // other legal variants (e.g. "/device:CPU:0").
    //
    // Note that we don't want to make change to the device name if it is
    // already canonicalized by users.
    // e.g. "/job:tpu_worker/replica:0/task:x/device:CPU:0".
    // TODO(tfrt-devs): to make the canonicalization more robust we should
    // introduce a util to check each component of the TF device name.
    device = GetDefaultCpuDeviceName();
  }

  auto *def = value.getDefiningOp();
  if (def) {
    rewriter.setInsertionPointAfter(def);
  } else {
    rewriter.setInsertionPointToStart(value.getParentBlock());
  }

  return rewriter
      .create<tfrt::fallback_async::CoreRTTensorHandleToFallbackTensorOp>(
          loc, rewriter.getType<tfrt::fallback::TFTensorType>(), value, device)
      .getResult(0);
}

mlir::Value ConvertFallbackTensorToCoreRTTensorHandle(
    mlir::Location loc, mlir::Value value,
    mlir::ConversionPatternRewriter &rewriter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfallback_converterDTcc mht_2(mht_2_v, 257, "", "./tensorflow/compiler/mlir/tfrt/transforms/fallback_converter.cc", "ConvertFallbackTensorToCoreRTTensorHandle");

  if (value.getType().isa<tfrt::corert::TensorHandleType>()) return value;

  if (!value.getType().isa<tfrt::fallback::TFTensorType>()) return {};

  // Use CPU device by default if no device is specified.
  llvm::StringRef device = GetDefaultCpuDeviceName();
  if (auto *def = value.getDefiningOp()) {
    if (auto device_attr = def->getAttrOfType<mlir::StringAttr>("device")) {
      // NOTE: The TPU_SYSTEM check is just a short term workaround. The long
      // term solution should be checking the HostMemory annotation of the
      // defining op (it should be defined in TF OpKernel). If HostMemory
      // annotation is set for an output tensor, we should use CPU device here.
      // TODO(b/200896904): Support HostMemory annotation.
      if (!device_attr.getValue().endswith("TPU_SYSTEM:0")) {
        device = device_attr.getValue();
      }
    }
  }

  return rewriter
      .create<tfrt::fallback_async::FallbackTensorToCoreRTTensorHandleOp>(
          loc, rewriter.getType<tfrt::corert::TensorHandleType>(), value,
          device)
      .getResult(0);
}

mlir::LogicalResult ConvertCoreRTOperands(
    mlir::Operation *op, mlir::ValueRange operands,
    llvm::SmallVectorImpl<mlir::Value> *new_operands,
    mlir::ConversionPatternRewriter &rewriter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfallback_converterDTcc mht_3(mht_3_v, 290, "", "./tensorflow/compiler/mlir/tfrt/transforms/fallback_converter.cc", "ConvertCoreRTOperands");

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  // Insert before the current op.
  rewriter.setInsertionPoint(op);

  for (auto operand : operands) {
    auto value = ConvertFallbackTensorToCoreRTTensorHandle(op->getLoc(),
                                                           operand, rewriter);
    if (!value) {
      return op->emitWarning("failed to convert to !corert.tensorhandle")
             << operand.getType();
    }

    new_operands->push_back(value);
  }
  return success();
}

mlir::LogicalResult ConvertFallbackOperands(
    mlir::Operation *op, llvm::StringRef device, mlir::ValueRange operands,
    llvm::SmallVectorImpl<mlir::Value> *new_operands,
    mlir::ConversionPatternRewriter &rewriter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSfallback_converterDTcc mht_4(mht_4_v, 314, "", "./tensorflow/compiler/mlir/tfrt/transforms/fallback_converter.cc", "ConvertFallbackOperands");

  for (auto operand : operands) {
    if (!operand.getType().isa<tfrt::fallback::TFTensorType>()) {
      auto new_operand = ConvertCoreRTTensorHandleToFallbackTensor(
          op->getLoc(), device, operand, rewriter);
      if (!new_operand)
        return op->emitWarning(
            "failed to convert the operand to fallback tensor.");
      new_operands->push_back(new_operand);
    } else {
      new_operands->push_back(operand);
    }
  }
  return success();
}

}  // namespace tfrt_compiler
}  // namespace tensorflow
