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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc() {
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

#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.h"

#include <algorithm>
#include <cstdint>
#include <memory>

#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/cost.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// These are just fake costs.
constexpr float kDequantCost = 2.0;
constexpr float kQuantCost = 2.0;
constexpr float kRequantCost = 2.0;

// TODO(renjieliu): Ideally this should consider different kinds of SOCs as
// well.

// Get total bytes transferred.
int64_t GetTransferredTensorBytes(func::CallOp from_graph,
                                  func::CallOp to_graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.cc", "GetTransferredTensorBytes");

  int64_t total_size_transferred = 0;
  for (auto input : to_graph.getOperands()) {
    Operation* input_op = input.getDefiningOp();
    if (input_op && input_op == from_graph.getOperation()) {
      auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
      if (input_type == nullptr || !input_type.hasStaticShape()) continue;
      // Quantized type does not support getSizeInBits.
      if (IsQUI8Type(input_type) || IsQI8Type(input_type)) {
        total_size_transferred += input_type.getNumElements() * 8;
      } else {
        total_size_transferred += input_type.cast<ShapedType>().getSizeInBits();
      }
    }
  }
  return total_size_transferred;
}

// Get total tensor element size transferred.
int64_t GetTransferredElementCount(func::CallOp from_graph,
                                   func::CallOp to_graph) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc mht_1(mht_1_v, 248, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.cc", "GetTransferredElementCount");

  int64_t total_element_count = 0;
  for (auto input : to_graph.getOperands()) {
    Operation* input_op = input.getDefiningOp();
    if (input_op && input_op == from_graph.getOperation()) {
      auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
      if (input_type == nullptr || !input_type.hasStaticShape()) continue;
      total_element_count += input_type.getNumElements();
    }
  }
  return total_element_count;
}

struct GetOpCostPass : mlir::PassWrapper<GetOpCostPass, OperationPass<FuncOp>> {
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc mht_2(mht_2_v, 265, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.cc", "getArgument");
 return "tfl-get-op-cost"; }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc mht_3(mht_3_v, 269, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.cc", "getDescription");

    return "Get cost for every op";
  }
  void runOnOperation() override;
};

void GetOpCostPass::runOnOperation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc mht_4(mht_4_v, 278, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.cc", "GetOpCostPass::runOnOperation");

  auto func = getOperation();
  OpBuilder builder(func);
  func.walk([&](Operation* op) {
    if (IsNonConstOp(op) && !IsTerminatorOp(op) &&
        !llvm::isa<func::ReturnOp, FuncOp, CallOpInterface>(op)) {
      auto hardware = GetTargetAnnotation(op);
      if (!hardware) return;
      float cost = GetCostForOp(op, hardware.getValue());
      UpdateCost(op, cost, &builder);
    }
  });
}

}  // namespace

float GetCostForOp(Operation* op, const std::string& hardware) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("hardware: \"" + hardware + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc mht_5(mht_5_v, 298, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.cc", "GetCostForOp");

  auto* device_hardware = GetTargetHardware(hardware);
  if (device_hardware == nullptr) {
    return kDefaultFixedValuedCost;
  }

  return device_hardware->GetOpCost(op);
}

float GetCostForFunc(FuncOp* func, const std::string& hardware) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("hardware: \"" + hardware + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc mht_6(mht_6_v, 311, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.cc", "GetCostForFunc");

  auto* device_hardware = GetTargetHardware(hardware);
  if (device_hardware == nullptr) {
    return kDefaultFixedValuedCost;
  }

  return device_hardware->GetFuncCost(func);
}

float GetTransferCost(const std::string& from_hardware_str,
                      const std::string& to_hardware_str,
                      func::CallOp from_graph, func::CallOp to_graph) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("from_hardware_str: \"" + from_hardware_str + "\"");
   mht_7_v.push_back("to_hardware_str: \"" + to_hardware_str + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc mht_7(mht_7_v, 327, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.cc", "GetTransferCost");

  auto from_hardware = GetTargetHardware(from_hardware_str);
  auto to_hardware = GetTargetHardware(to_hardware_str);
  if (from_hardware == nullptr) {
    from_graph.emitError(absl::StrCat(
        "we cannot find the registered hardware: ", from_hardware_str));
  }

  if (to_hardware == nullptr) {
    to_graph.emitError(absl::StrCat("we cannot find the registered hardware: ",
                                    to_hardware_str));
  }

  const int64_t total_size_transferred =
      GetTransferredTensorBytes(from_graph, to_graph);
  return to_hardware->GetHardwareSwitchingCost(from_hardware,
                                               total_size_transferred);
}

float GetQuantDequantCost(InferenceType from_inference_type,
                          InferenceType to_inference_type,
                          func::CallOp from_graph, func::CallOp to_graph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPScost_modelDTcc mht_8(mht_8_v, 351, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.cc", "GetQuantDequantCost");

  // Same inference type, no dequant/quant happens.
  if (from_inference_type == to_inference_type) return 0;

  const int64_t total_element_count_transferred =
      GetTransferredElementCount(from_graph, to_graph);

  if (from_inference_type == FLOAT || from_inference_type == HYBRID) {
    // FLOAT <-> HYBRID will have no quant/dequant as well.
    if (to_inference_type == FLOAT || to_inference_type == HYBRID) {
      return 0;
    } else if (to_inference_type == QUANTIZED_INT8 ||
               to_inference_type == QUANTIZED_UINT8) {
      // QUANT path.
      return kQuantCost * total_element_count_transferred;
    }
  }

  if (from_inference_type == QUANTIZED_INT8 ||
      from_inference_type == QUANTIZED_UINT8) {
    // Dequant path.
    if (to_inference_type == FLOAT || to_inference_type == HYBRID) {
      return kDequantCost * total_element_count_transferred;
    } else if (to_inference_type == QUANTIZED_INT8 ||
               to_inference_type == QUANTIZED_UINT8) {
      // Requant path.
      return kRequantCost * total_element_count_transferred;
    }
  }

  // Default quant/dequant/requant cost.
  return kDefaultFixedValuedCost;
}

std::unique_ptr<OperationPass<FuncOp>> CreateGetOpCostPass() {
  return std::make_unique<GetOpCostPass>();
}

static PassRegistration<GetOpCostPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
