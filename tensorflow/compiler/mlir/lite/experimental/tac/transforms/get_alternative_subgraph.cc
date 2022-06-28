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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc() {
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

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/subgraph.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// Given the function interface name and the InferenceDeviceType, return the
// new function name.
std::string GetFunctionImplName(
    std::string interface_name,
    const InferenceDeviceType& device_inference_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("interface_name: \"" + interface_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_0(mht_0_v, 228, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "GetFunctionImplName");

  return absl::StrCat(interface_name, "_", device_inference_type.hardware, "_",
                      GetInferenceString(device_inference_type.inference_type));
}

// For every device, we will do the following:
// If the inference type is quantized, we will try the float alternative.
// If it's float, we will just keep it as it is.
std::vector<InferenceDeviceType> GetAllAlternativeInferenceDeviceType(
    InferenceType inference_type, ArrayRef<std::string> devices) {
  std::vector<InferenceDeviceType> all_device_inference_types;
  for (const auto& device : devices) {
    if (inference_type == QUANTIZED_INT8) {
      all_device_inference_types.push_back({device, QUANTIZED_INT8});
    } else if (inference_type == QUANTIZED_UINT8) {
      all_device_inference_types.push_back({device, QUANTIZED_UINT8});
    }

    // We will alway enable float.
    all_device_inference_types.push_back({device, FLOAT});
  }

  return all_device_inference_types;
}

// This pass will try to get alternative subgraph:
// Say a subgraph is annotated with CPU (it probably means the ops it contains
// cannot be run on other deviecs):
//
// We will try:
// 1) If we can do some mathmatically equaivalent transformation so this
//   subgraph can be run on other devices.
// 2) We will other apply device-specifics optimizations as well, that includes
//   maybe tensor layout transformation, device specific fusion, etc.
class AlternativeSubgraphPass
    : public mlir::PassWrapper<AlternativeSubgraphPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_1(mht_1_v, 269, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "getArgument");

    return "tfl-get-alternative-subgraph";
  }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_2(mht_2_v, 275, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "getDescription");

    return "Get alternative subgraph representation (if appliable) for all the "
           "given devices, will by default include the cpu implementation.";
  }
  AlternativeSubgraphPass() = default;
  AlternativeSubgraphPass(const AlternativeSubgraphPass&) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_3(mht_3_v, 283, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "AlternativeSubgraphPass");
}
  explicit AlternativeSubgraphPass(llvm::ArrayRef<std::string> device_specs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_4(mht_4_v, 287, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "AlternativeSubgraphPass");

    device_specs_flag_ = device_specs;
  }

 private:
  void runOnOperation() override;

  // Given a func and targeted devices, we will try to clonse the func &
  // transform/optimize for those devices.
  // This will only happen if the whole subgraph can be supported by the target
  // or can be supported after some transformations.
  void GetAlternativeGraphForFunc(ArrayRef<std::string> devices, FuncOp func,
                                  ModuleOp module, OpBuilder* builder);

  // If all ops in the func op is able to be represented in the hardware, we
  // will return true, else will be false.
  // This is basically all or nothing.
  bool IsAllSupportedbySpec(FuncOp func,
                            const InferenceDeviceType& inference_type);

  // Given a func and a targeted device, we will try to clonse the func &
  // transform/optimize for that device.
  // It's simply clone the FuncOp and hardware specific transformations.
  FuncOp GetAlternativeViewForSpec(
      FuncOp func, const InferenceDeviceType& current_device_inference_type,
      const InferenceDeviceType& target_device_inference_type, ModuleOp module,
      OpBuilder* builder);

  // Apply any device-specific optimizations.
  void Optimize(FuncOp func, const std::string& hardware);

  ListOption<std::string> device_specs_flag_{
      *this, "device-specs",
      llvm::cl::desc(
          "comma separated list of device specs, like CPU, GPU, DPS."),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};

void AlternativeSubgraphPass::GetAlternativeGraphForFunc(
    ArrayRef<std::string> devices, FuncOp func, ModuleOp module,
    OpBuilder* builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_5(mht_5_v, 330, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "AlternativeSubgraphPass::GetAlternativeGraphForFunc");

  auto current_device = GetTargetAnnotation(func);
  if (current_device->empty()) {
    func.emitError(
        "cannot find target annotation or unknown device specified for current "
        "function");
    return;
  }

  auto current_inference_type = GetInferenceTypeAnnotation(func);
  if (!current_inference_type.hasValue() || current_inference_type == UNKNOWN) {
    func.emitError(
        "cannot find inference type annotation or unknown inference type "
        "specified for current "
        "function");
    return;
  }

  const InferenceDeviceType current_device_type(
      {current_device.getValue(), current_inference_type.getValue()});

  const std::vector<InferenceDeviceType>& all_inference_device_type =
      GetAllAlternativeInferenceDeviceType(current_inference_type.getValue(),
                                           devices);

  for (const auto& device_inference_type : all_inference_device_type) {
    if (device_inference_type != current_device_type) {
      FuncOp cloned_func = GetAlternativeViewForSpec(
          func, current_device_type, device_inference_type, module, builder);
      // If we found unsupported ops, we will just go ahead and remove this
      // function.
      // TODO(b/160284136): currently we check if the ops are supported then
      // see if we need to erase the func op.
      // Ideally it would be nice if we can utilize dynamic illegal op to do
      // the job.
      if (!IsAllSupportedbySpec(cloned_func, device_inference_type)) {
        cloned_func.erase();
      }
    }
  }

  // Perform the device-specific optimization last.
  // We need to run the optimization for the current device last because we
  // need to avoid any changes made the current graph polluting other
  // alternative graph views.
  Optimize(func, current_device.getValue());
}

bool AlternativeSubgraphPass::IsAllSupportedbySpec(
    FuncOp func, const InferenceDeviceType& device_inference_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_6(mht_6_v, 382, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "AlternativeSubgraphPass::IsAllSupportedbySpec");

  bool found_unsupported = false;
  func.walk([&](Operation* op) {
    if (IsNonConstOp(op) && !IsTerminatorOp(op) &&
        NotTFLQuantDequantizeOp(op) &&
        !llvm::isa<func::ReturnOp, FuncOp, CallOpInterface>(op) &&
        !IsSupported(op, device_inference_type.hardware)) {
      found_unsupported = true;
    }
  });
  return !found_unsupported;
}

void AlternativeSubgraphPass::Optimize(FuncOp func,
                                       const std::string& hardware) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("hardware: \"" + hardware + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_7(mht_7_v, 400, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "AlternativeSubgraphPass::Optimize");

  auto* ctx = &getContext();
  RewritePatternSet patterns = GetHardwareRewritePatterns(ctx, hardware);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

// Get the alternative view of the func for the given device_inference_type.
// It's possible the transformed func can still contain unsupported ops for the
// given device_inference_type.
FuncOp AlternativeSubgraphPass::GetAlternativeViewForSpec(
    FuncOp func, const InferenceDeviceType& current_device_inference_type,
    const InferenceDeviceType& target_device_inference_type, ModuleOp module,
    OpBuilder* builder) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_8(mht_8_v, 415, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "AlternativeSubgraphPass::GetAlternativeViewForSpec");

  FuncOp cloned_func = func.clone();
  cloned_func.setPrivate();
  auto interface_name = GetInterFaceName(func);
  if (!interface_name.hasValue()) {
    func.emitError("the func op does not have interface_name");
    return nullptr;
  }

  cloned_func->setAttr(
      kDevice, builder->getStringAttr(target_device_inference_type.hardware));
  cloned_func->setAttr(kInferenceType,
                       builder->getStringAttr(GetInferenceString(
                           target_device_inference_type.inference_type)));
  std::string new_function_name = GetFunctionImplName(
      interface_name.getValue(), target_device_inference_type);
  cloned_func.setName(new_function_name);

  // If it's quantized -> float, we need to wrap all the ops around with dequant
  // and quant.
  if ((current_device_inference_type.inference_type == QUANTIZED_UINT8 ||
       current_device_inference_type.inference_type == QUANTIZED_INT8) &&
      target_device_inference_type.inference_type == FLOAT) {
    OpBuilder cloned_func_builder(cloned_func);
    ConvertQuantizedOpToFloat(cloned_func, &cloned_func_builder);
    OptimizeQuantizedOpToFloat(cloned_func, &getContext());
  }

  Optimize(cloned_func, target_device_inference_type.hardware);

  // Set device for each op.
  cloned_func.walk([&](Operation* op) {
    if (IsNonConstOp(op) && !IsTerminatorOp(op) &&
        !llvm::isa<func::ReturnOp, FuncOp, CallableOpInterface>(op)) {
      op->setAttr(kDevice, builder->getStringAttr(
                               target_device_inference_type.hardware));
      op->setAttr(kInferenceType,
                  builder->getStringAttr(GetInferenceString(
                      target_device_inference_type.inference_type)));
    }
  });

  module.push_back(cloned_func);
  return cloned_func;
}

void AlternativeSubgraphPass::runOnOperation() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSget_alternative_subgraphDTcc mht_9(mht_9_v, 464, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/get_alternative_subgraph.cc", "AlternativeSubgraphPass::runOnOperation");

  auto module = getOperation();

  // Process devices specs.
  if (device_specs_flag_.empty()) {
    module.emitError("no device specs specified");
    signalPassFailure();
  }

  std::vector<std::string> device_specs;
  if (!ProcessTargetDevices(device_specs_flag_, &device_specs)) {
    module.emitError("unknown devices specified");
    signalPassFailure();
  }

  SmallVector<FuncOp, 25> funcs_to_be_processed;
  // We only process if func has device annotations.
  for (auto func : module.getOps<FuncOp>()) {
    auto device_attr = func->getAttrOfType<StringAttr>(kDevice);
    if (device_attr != nullptr) funcs_to_be_processed.push_back(func);
  }

  OpBuilder builder(module);
  // Go head to process those funcs.
  // We don't process in the previous loop is we're adding new funcs,
  // this is to avoid unnecessary processing.
  for (auto func : funcs_to_be_processed) {
    GetAlternativeGraphForFunc(device_specs, func, module, &builder);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateAlternativeSubgraphPass(
    llvm::ArrayRef<std::string> device_specs) {
  return std::make_unique<AlternativeSubgraphPass>(device_specs);
}

static PassRegistration<AlternativeSubgraphPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
