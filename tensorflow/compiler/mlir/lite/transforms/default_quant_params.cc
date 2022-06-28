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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc() {
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "absl/memory/memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/prepare_quantize_helper.h"

//===----------------------------------------------------------------------===//
// The Pass to add default quantization parameters for the activations which
// don't have quantization information. These default parameters are usually
// not from real measurement, so this pass is only for test purpose.

namespace mlir {
namespace TFL {
// Includs an auto-generated function, which can retrieve the quantization
// specification for an TFL operation. The signature of the function is
//   std::unique_pointer<OpQuantSpec> TFL::GetOpQuantSpec(Operation *)
#include "tensorflow/compiler/mlir/lite/utils/generated_op_quant_spec_getters.inc"

namespace {
class DefaultQuantParamsPass
    : public PassWrapper<DefaultQuantParamsPass, OperationPass<FuncOp>> {
 public:
  explicit DefaultQuantParamsPass(double default_min, double default_max,
                                  bool is_signed)
      : default_min_(default_min),
        default_max_(default_max),
        is_signed_(is_signed) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc mht_0(mht_0_v, 223, "", "./tensorflow/compiler/mlir/lite/transforms/default_quant_params.cc", "DefaultQuantParamsPass");
}

  void runOnOperation() override;

  StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/mlir/lite/transforms/default_quant_params.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-default-quant";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc mht_2(mht_2_v, 238, "", "./tensorflow/compiler/mlir/lite/transforms/default_quant_params.cc", "getDescription");

    // This is a brief description of the pass.
    return "Apply quantization with default quantization parameter";
  }

 private:
  // Whether the value is used as a bias input of another op. Here we assume
  // bias is used immediately by the user. This assumption is always correct
  // after constant folding.
  bool UsedAsBias(Value value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc mht_3(mht_3_v, 250, "", "./tensorflow/compiler/mlir/lite/transforms/default_quant_params.cc", "UsedAsBias");

    for (auto &use : value.getUses()) {
      auto biases = TFL::GetOpQuantSpec(use.getOwner())->biases_params;
      if (biases.find(use.getOperandNumber()) != biases.end()) return true;
    }
    return false;
  }

  // Uses `quant_params` to quantize `value` and inserting a pair of
  // tfl.quantize and tfl.dequantize ops for this `value`.
  void QuantizeValue(OpBuilder builder, Value value,
                     quant::QuantParams quant_params);

  // If the value hasn't been quantized, the functions adds it to `values`.
  void AddToWorkListIfUnquantized(Value value, std::vector<Value> *values);

  // Converts the default min/max to the default quantization parameters.
  quant::QuantParams GetDefaultQuantParams(Builder builder);

  // Gets the quantization parameters for the bias of an operation by using the
  // quantization parameters from the non-biases operands.
  quant::QuantParams GetQuantParamsForBias(Operation *op, int bias,
                                           const std::vector<int> &non_biases,
                                           quant::AccumulatorScaleFunc func);

  double default_min_;
  double default_max_;
  bool is_signed_;
  quant::QuantParams default_quant_params_;
};
}  // namespace

void DefaultQuantParamsPass::runOnOperation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc mht_4(mht_4_v, 285, "", "./tensorflow/compiler/mlir/lite/transforms/default_quant_params.cc", "DefaultQuantParamsPass::runOnOperation");

  FuncOp func = getOperation();
  OpBuilder builder(func);

  std::vector<Value> activation_values;
  std::vector<Value> bias_values;

  // First of all, collect all the values (block arguments and op results) which
  // are required to be quantized.
  for (auto arg : func.getBody().begin()->getArguments()) {
    if (UsedAsBias(arg)) {
      AddToWorkListIfUnquantized(arg, &bias_values);
    } else {
      AddToWorkListIfUnquantized(arg, &activation_values);
    }
  }

  func.walk([&](Operation *op) {
    if (quant::IsOpNotQuantizable(op) ||
        op->getParentOfType<TFL::CustomTfOp>()) {
      return;
    }

    for (auto res : op->getResults()) {
      if (UsedAsBias(res)) {
        AddToWorkListIfUnquantized(res, &bias_values);
      } else {
        AddToWorkListIfUnquantized(res, &activation_values);
      }
    }
  });

  // Apply the default quantization parameters for these activation values.
  quant::QuantParams default_params = GetDefaultQuantParams(builder);
  for (Value value : activation_values) {
    QuantizeValue(builder, value, default_params);
  }

  // Since all the non-biases operands have quantization parameters now, we
  // should be able to propagate them to the bias operand.
  for (Value bias : bias_values) {
    Operation *op = *bias.user_begin();
    auto spec = TFL::GetOpQuantSpec(op);
    for (auto &it : spec->biases_params) {
      quant::QuantParams bias_params = GetQuantParamsForBias(
          op, it.first, it.second.first, it.second.second);
      if (!bias_params) continue;
      QuantizeValue(builder, bias, bias_params);
    }
  }
}

void DefaultQuantParamsPass::AddToWorkListIfUnquantized(
    Value value, std::vector<Value> *values) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc mht_5(mht_5_v, 341, "", "./tensorflow/compiler/mlir/lite/transforms/default_quant_params.cc", "DefaultQuantParamsPass::AddToWorkListIfUnquantized");

  // If the result isn't with float type, this result is an integer tensor and
  // doesn't require quantization.
  auto tensor_type = value.getType().dyn_cast<TensorType>();
  if (!tensor_type) {
    // There are none type values.
    return;
  }
  if (!tensor_type.getElementType().isF32()) return;

  // If the result is consumed by a quantize op, it has been quantized.
  if (value.hasOneUse() &&
      llvm::isa<TFL::QuantizeOp>(*value.getUsers().begin()))
    return;

  // Add this result to the list to apply the default value.
  values->push_back(value);
}

void DefaultQuantParamsPass::QuantizeValue(OpBuilder builder, Value value,
                                           quant::QuantParams quant_params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc mht_6(mht_6_v, 364, "", "./tensorflow/compiler/mlir/lite/transforms/default_quant_params.cc", "DefaultQuantParamsPass::QuantizeValue");

  Type expressed_type = value.getType();
  Type new_type = quant_params.castFromExpressedType(expressed_type);
  // This value isn't an expressed type (float), skip.
  if (!new_type) return;

  Block &block = value.getParentRegion()->front();
  Operation *op = value.getDefiningOp();
  if (op) {
    builder.setInsertionPoint(&block, ++Block::iterator(op));
  } else {
    builder.setInsertionPointToStart(&block);
  }
  TypeAttr type_attr = TypeAttr::get(new_type);
  auto quantize = builder.create<TFL::QuantizeOp>(value.getLoc(), new_type,
                                                  value, type_attr);
  auto dequantize = builder.create<TFL::DequantizeOp>(
      value.getLoc(), expressed_type, quantize.output());
  value.replaceAllUsesWith(dequantize);

  // `quantize` is using `dequantize` now, so we should set its operand to
  // `value`.
  quantize.getOperation()->replaceUsesOfWith(dequantize, value);
}

quant::QuantParams DefaultQuantParamsPass::GetQuantParamsForBias(
    Operation *op, int bias, const std::vector<int> &non_biases,
    quant::AccumulatorScaleFunc func) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc mht_7(mht_7_v, 394, "", "./tensorflow/compiler/mlir/lite/transforms/default_quant_params.cc", "DefaultQuantParamsPass::GetQuantParamsForBias");

  std::vector<quant::QuantizedType> non_bias_types;
  non_bias_types.reserve(non_biases.size());
  for (int non_bias : non_biases) {
    Operation *non_bias_define = op->getOperand(non_bias).getDefiningOp();
    if (auto dequant = llvm::dyn_cast<TFL::DequantizeOp>(non_bias_define)) {
      auto non_bias_type = dequant.input().getType().cast<TensorType>();
      auto non_bias_ele_type =
          non_bias_type.getElementType().cast<quant::QuantizedType>();
      non_bias_types.push_back(non_bias_ele_type);
    } else {
      // The non-bias hasn't been quantized, let's skip this bias.
      break;
    }
  }
  // The non-bias hasn't been quantized, let's skip this bias.
  if (non_bias_types.size() != non_biases.size()) return {};

  return func(non_bias_types, false);
}

quant::QuantParams DefaultQuantParamsPass::GetDefaultQuantParams(
    Builder builder) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSdefault_quant_paramsDTcc mht_8(mht_8_v, 419, "", "./tensorflow/compiler/mlir/lite/transforms/default_quant_params.cc", "DefaultQuantParamsPass::GetDefaultQuantParams");

  if (!default_quant_params_) {
    default_quant_params_ = quant::fakeQuantAttrsToType(
        builder.getUnknownLoc(),
        /*numBits=*/8, default_min_, default_max_, /*narrowRange=*/false,
        builder.getF32Type(), is_signed_);
  }
  return default_quant_params_;
}

// Creates an instance of the default quant parameters pass.
std::unique_ptr<OperationPass<FuncOp>> CreateDefaultQuantParamsPass(
    double default_min, double default_max, bool is_signed) {
  return absl::make_unique<DefaultQuantParamsPass>(default_min, default_max,
                                                   is_signed);
}

// Registers this pass with default values, only for test
static PassRegistration<DefaultQuantParamsPass> pass([] {
  return CreateDefaultQuantParamsPass(/*default_min=*/-1.0,
                                      /*default_max=*/1.0,
                                      /*is_signed=*/false);
});

}  // namespace TFL
}  // namespace mlir
