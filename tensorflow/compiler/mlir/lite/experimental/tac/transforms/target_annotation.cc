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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc() {
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

#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/tac_pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

class TargetAnnotationPass : public TacFunctionPass<TargetAnnotationPass> {
 public:
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/target_annotation.cc", "getArgument");
 return "tfl-target-annotation"; }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/target_annotation.cc", "getDescription");

    return "Add user specified target annotations to the TFL operations given "
           "operation capabilities, will default to CPU.";
  }
  // using TacFunctionPass::TacFunctionPass;
  TargetAnnotationPass() : TacFunctionPass(nullptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/target_annotation.cc", "TargetAnnotationPass");
}
  TargetAnnotationPass(const TargetAnnotationPass& copy)
      : TacFunctionPass(copy.module_) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc mht_3(mht_3_v, 230, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/target_annotation.cc", "TargetAnnotationPass");
}
  explicit TargetAnnotationPass(llvm::ArrayRef<std::string> device_specs)
      : TacFunctionPass(nullptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc mht_4(mht_4_v, 235, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/target_annotation.cc", "TargetAnnotationPass");

    device_specs_flag_ = device_specs;
  }

  explicit TargetAnnotationPass(const TacModule* module)
      : TacFunctionPass(module) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc mht_5(mht_5_v, 243, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/target_annotation.cc", "TargetAnnotationPass");
}

 private:
  void runOnFunction() override;
  void SetTargetAnnotation(Operation* op,
                           llvm::ArrayRef<std::string> device_specs,
                           OpBuilder* builder);

  ListOption<std::string> device_specs_flag_{
      *this, "device-specs",
      llvm::cl::desc(
          "comma separated list of device specs, like CPU, GPU, Hexagon."),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};

void SetAnnotation(Operation* op, std::string attribute, std::string annotation,
                   OpBuilder* builder) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("attribute: \"" + attribute + "\"");
   mht_6_v.push_back("annotation: \"" + annotation + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc mht_6(mht_6_v, 264, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/target_annotation.cc", "SetAnnotation");

  // TODO(karimnosseir): Maybe set device capabilities to allow us to have
  // more flexbility when raise the subgraphs.
  auto default_target = builder->getStringAttr(annotation);
  op->setAttr(attribute, default_target);
}

void TargetAnnotationPass::SetTargetAnnotation(
    Operation* op, llvm::ArrayRef<std::string> device_specs,
    OpBuilder* builder) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc mht_7(mht_7_v, 276, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/target_annotation.cc", "TargetAnnotationPass::SetTargetAnnotation");

  const InferenceType inference_type = GetInferenceType(op);
  const std::string inference_type_str = GetInferenceString(inference_type);
  SetAnnotation(op, kInferenceType, inference_type_str, builder);
  bool device_is_set = false;
  // TODO(b/177376459): Remove the usage of device_specs.
  // TODO(b/177376459): Update if needed to make testing easy.
  if (!module_) {
    for (const auto& device : device_specs) {
      auto* hardware = this->GetTargetHardware(device);
      if (hardware == nullptr) continue;
      if (hardware->IsOpSupported(op)) {
        SetAnnotation(op, kDevice, device, builder);
        device_is_set = true;
        break;
      }
    }
  } else {
    for (const auto* hardware : module_->GetAvailableHardwares()) {
      if (hardware == nullptr) continue;
      if (hardware->IsOpSupported(op)) {
        SetAnnotation(op, kDevice, GetHardwareName(hardware), builder);
        device_is_set = true;
        break;
      }
    }
  }
  // default to CPU
  if (!device_is_set) {
    if (IsNonConstOp(op) && !IsTerminatorOp(op) &&
        !llvm::isa<func::ReturnOp, FuncOp, CallableOpInterface>(op)) {
      SetAnnotation(op, kDevice, "CPU", builder);
      device_is_set = true;
    }
  }
  if (!device_is_set) {
    op->emitError("cannot set target device for this ops");
  }
}

void TargetAnnotationPass::runOnFunction() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPStarget_annotationDTcc mht_8(mht_8_v, 319, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/target_annotation.cc", "TargetAnnotationPass::runOnFunction");

  auto func = getFunction();
  OpBuilder builder(func);

  func.walk([&](Operation* op) {
    // We only care about TFL dialect.
    if (IsNonConstOp(op) && NotTFLQuantDequantizeOp(op) &&
        !IsTerminatorOp(op) &&
        !llvm::isa<func::ReturnOp, FuncOp, CallOpInterface>(op)) {
      SetTargetAnnotation(op, device_specs_flag_, &builder);
    }
  });
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTargetAnnotationPass(
    llvm::ArrayRef<std::string> device_specs) {
  return std::make_unique<TargetAnnotationPass>(device_specs);
}

std::unique_ptr<OperationPass<FuncOp>> CreateTargetAnnotationPass(
    const TacModule* module) {
  return std::make_unique<TargetAnnotationPass>(module);
}

static PassRegistration<TargetAnnotationPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
