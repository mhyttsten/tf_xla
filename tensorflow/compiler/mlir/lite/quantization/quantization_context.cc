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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc() {
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

#include "tensorflow/compiler/mlir/lite/quantization/quantization_context.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/device_target.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"

#define DEBUG_TYPE "quantization-context"

namespace mlir {
namespace quant {

QuantizeContext::QuantizeContext(FuncOp func, const DeviceTarget &spec)
    : func_(func), target_spec_(spec) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::QuantizeContext");

  llvm::DenseMap<Value, int> value_to_state;
  func.walk([&](quant::QuantizeRegionOp op) {
    for (int i = 0, e = op.getNumOperands(); i != e; ++i) {
      states_manager_.InitializeOperandState(op, i, &value_to_state);
    }

    for (int res = 0, e = op.getNumResults(); res != e; ++res) {
      states_manager_.InitializeResultState(op, res, &value_to_state);
    }
  });
}

std::vector<quant::QuantizeRegionOp> QuantizeContext::GetAllOps() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::GetAllOps");

  std::vector<quant::QuantizeRegionOp> all_ops;
  all_ops.reserve(128);
  func_.walk([&](quant::QuantizeRegionOp op) { all_ops.push_back(op); });
  return all_ops;
}

KernelSpecs::Signature QuantizeContext::GetSignature(QuantizeRegionOp op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_2(mht_2_v, 242, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::GetSignature");

  KernelSpecs::Signature signature;
  signature.reserve(op.input_specs().size() + op.output_specs().size());
  for (int i = 0; i < op.getNumOperands(); ++i) {
    DeviceTarget::AppendToSignature(GetOperandParams(op, i), &signature);
  }
  for (int i = 0; i < op.getNumResults(); ++i) {
    DeviceTarget::AppendToSignature(GetResultParams(op, i), &signature);
  }
  return signature;
}

LogicalResult QuantizeContext::Handle(
    quant::QuantizeRegionOp op, llvm::SmallVectorImpl<Operation *> *new_items,
    bool *changed) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_3(mht_3_v, 259, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::Handle");

  auto signature = GetSignature(op);
  auto spec = target_spec_.GetKernelSpec(op.logical_kernel(), signature);
  if (!spec.hasValue()) {
    op.emitWarning(
        "Couldn't find kernel from the registration for quantization.");
    return success();
  }
  switch (spec->type) {
    case ScaleConstraintType::OutputInputFreeScale: {
      // no propagation.
      *changed |= false;
      break;
    }
    case ScaleConstraintType::CustomScale: {
      if (failed(spec->scale_fn(this, op, new_items, changed))) {
        return failure();
      }
      break;
    }
    case ScaleConstraintType::OutputInputSameScale: {
      auto params = GetQuantParamsForSameScaleConstraint(op);
      if (EmptyParams(params)) {
        *changed |= false;
        break;
      }
      // propagate this params to all the quantizable ports.
      if (failed(PropagateQuantParams(op, params, new_items, changed))) {
        return failure();
      }
      break;
    }
    default: {
      // TODO(fengliuai): implement the other types.
      llvm_unreachable("no implementation.");
      return failure();
    }
  }
  return success();
}

LogicalResult QuantizeContext::Finalize() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_4(mht_4_v, 303, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::Finalize");

  MLIRContext *context = func_.getContext();
  func_.walk([&](quant::QuantizeRegionOp op) {
    llvm::SmallVector<Attribute, 4> input_specs;
    auto original_input_specs = op.input_specs().getValue();
    for (int i = 0, e = op.getNumOperands(); i != e; ++i) {
      auto &state = states_manager_.GetOperandQuantState(op, i);
      auto &requantize = states_manager_.GetOperandRequantizeState(op, i);
      if (state.IsEmpty() && requantize.pos == RequantizeState::NO_REQUANTIZE) {
        input_specs.push_back(original_input_specs[i]);
      } else if (requantize.pos == RequantizeState::ON_OUTPUT) {
        input_specs.push_back(TypeAttr::get(requantize.params));
      } else {
        input_specs.push_back(TypeAttr::get(state.params));
      }
    }
    op->setAttr("input_specs", ArrayAttr::get(context, input_specs));

    llvm::SmallVector<Attribute, 4> output_specs;
    auto original_output_specs = op.output_specs().getValue();
    for (int res = 0, e = op.getNumResults(); res != e; ++res) {
      auto &state = states_manager_.GetResultQuantState(op, res);
      auto &requantize = states_manager_.GetResultRequantizeState(op, res);
      if (state.IsEmpty() && requantize.pos == RequantizeState::NO_REQUANTIZE) {
        output_specs.push_back(original_output_specs[res]);
      } else if (requantize.pos == RequantizeState::ON_INPUT) {
        output_specs.push_back(TypeAttr::get(requantize.params));
      } else {
        output_specs.push_back(TypeAttr::get(state.params));
      }
    }
    op->setAttr("output_specs", ArrayAttr::get(context, output_specs));
  });
  return success();
}

void QuantizeContext::DumpStates(QuantizeRegionOp current_op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_5(mht_5_v, 342, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::DumpStates");

  if (current_op) {
    llvm::errs() << "\n\n\n" << current_op.logical_kernel() << "\n";
  }
  func_.walk([&](QuantizeRegionOp op) {
    if (current_op == op) llvm::errs() << "===>>>";
    llvm::errs() << op.logical_kernel() << " : (";
    for (auto i = 0; i < op.getNumOperands(); ++i) {
      if (auto params = GetOperandParams(op, i))
        params.print(llvm::errs());
      else
        llvm::errs() << "_";
      llvm::errs() << ",";
    }
    llvm::errs() << ") -> (";
    for (auto i = 0; i < op.getNumResults(); ++i) {
      if (auto params = GetResultParams(op, i))
        params.print(llvm::errs());
      else
        llvm::errs() << "_";
      llvm::errs() << ",";
    }
    llvm::errs() << ")\n";
  });
}

// A heuristic to get quantization parameters satisfies the same scale
// constraints:
// - If there are immutable states,
//   - use the single input, or,
//   - use the single output, or,
//   - use the first one in the collection,
// - use the single input if it is ready, or,
// - use the single output if it is ready, or,
// - use the first ready one in the collection.
QuantParams QuantizeContext::GetQuantParamsForSameScaleConstraint(
    Operation *op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_6(mht_6_v, 381, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::GetQuantParamsForSameScaleConstraint");

  // Two vector to collect Non-empty operands and results states.
  std::vector<quant::QuantState *> mutable_states, immutable_states;
  for (int i = 0, e = op->getNumOperands(); i != e; ++i) {
    auto &state = states_manager_.GetOperandQuantState(op, i);
    if (state.immutable) {
      immutable_states.push_back(&state);
    } else if (!state.IsEmpty()) {
      mutable_states.push_back(&state);
    }
  }

  int immutable_operands_num = immutable_states.size();
  int mutable_operands_num = mutable_states.size();
  // Use the operand's state if it is immutable and it is the only one
  // operand.
  if (op->getNumOperands() == 1 && immutable_operands_num == 1) {
    return immutable_states.front()->params;
  }

  for (int i = 0, e = op->getNumResults(); i != e; ++i) {
    auto &state = states_manager_.GetResultQuantState(op, i);
    if (state.immutable) {
      immutable_states.push_back(&state);
    } else if (!state.IsEmpty()) {
      mutable_states.push_back(&state);
    }
  }

  int immutable_results_num = immutable_states.size() - immutable_operands_num;
  int mutable_results_num = mutable_states.size() - mutable_operands_num;
  // Use the result's state if it is immutable and it is the only one result.
  if (op->getNumResults() == 1 && immutable_results_num == 1) {
    return immutable_states.back()->params;
  }

  LLVM_DEBUG(llvm::dbgs()
             << "Quantization parameters are not collected in an ideal place. "
                "Has to fallback values which might introduce errors.\n");

  // Use the first immutable state to quantize the rest operands and results.
  if (!immutable_states.empty()) return immutable_states.front()->params;

  // If there are no immutable states, use the operand's state if it is the
  // only one operand and has parameters propagated.
  if (op->getNumOperands() == 1 && mutable_operands_num == 1) {
    return mutable_states.front()->params;
  }

  // If there are no immutable states, use the result's state if it is the
  // only one result and has parameters propagated.
  if (op->getNumResults() == 1 && mutable_results_num == 1) {
    return mutable_states.back()->params;
  }

  // Use the first propagated state to quantize the rest operands and results.
  if (!mutable_states.empty()) return mutable_states.front()->params;

  // None operands/results have parameters propagated, skip this node for now.
  return {};
}

LogicalResult QuantizeContext::PropagateQuantParams(
    Operation *op, const QuantParams params,
    quant::AdjacentOperations *new_items, bool *changed) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_7(mht_7_v, 448, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::PropagateQuantParams");

  // Use the final state to set all the operands' parameters.
  for (int i = 0, e = op->getNumOperands(); i != e; ++i) {
    auto ele = op->getOperand(i).getType().cast<ShapedType>().getElementType();
    if (ele.isa<FloatType>() && SetOperandParams(op, i, params)) {
      *changed |= true;
      new_items->push_back(op->getOperand(i).getDefiningOp());
    }
  }

  // Use the final state to set all the results' parameters.
  for (int res = 0, e = op->getNumResults(); res != e; ++res) {
    auto ele = op->getResult(res).getType().cast<ShapedType>().getElementType();
    if (ele.isa<FloatType>() && SetResultParams(op, res, params)) {
      auto users = op->getResult(res).getUsers();
      *changed |= !users.empty();
      new_items->append(users.begin(), users.end());
    }
  }
  return success();
}

int QuantizeContext::StatesManager::InitializeState(quant::QuantizeRegionOp op,
                                                    int index, bool as_result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_8(mht_8_v, 474, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::StatesManager::InitializeState");

  Attribute params_attr;
  if (as_result) {
    params_attr = op.output_specs()[index];
  } else {
    params_attr = op.input_specs()[index];
  }
  QuantParams params =
      params_attr.cast<TypeAttr>().getValue().dyn_cast<QuantParams>();
  bool immutable = !EmptyParams(params);
  int next_state_index = states_.size();
  states_.push_back({params, immutable});
  if (as_result) {
    result_states_.insert({{op, index}, next_state_index});
  } else {
    operand_states_.insert({{op, index}, next_state_index});
  }
  return next_state_index;
}

void QuantizeContext::StatesManager::InitializeOperandState(
    quant::QuantizeRegionOp op, int index, llvm::DenseMap<Value, int> *cache) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_9(mht_9_v, 498, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::StatesManager::InitializeOperandState");

  Value in = op.getOperand(index);
  auto cached = cache->insert({in, 0});
  if (!cached.second) {
    operand_states_.insert({{op, index}, cached.first->second});
    return;
  }
  cached.first->second = InitializeState(op, index, /*as_result=*/false);
}

void QuantizeContext::StatesManager::InitializeResultState(
    quant::QuantizeRegionOp op, int index, llvm::DenseMap<Value, int> *cache) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_10(mht_10_v, 512, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::StatesManager::InitializeResultState");

  auto res = op.getResult(index);
  auto cached = cache->insert({res, 0});
  if (!cached.second) {
    result_states_.insert({{op, index}, cached.first->second});
    return;
  }
  cached.first->second = InitializeState(op, index, /*as_result=*/true);
}

bool QuantizeContext::StatesManager::SetConstantResultParams(Operation *op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_11(mht_11_v, 525, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::StatesManager::SetConstantResultParams");

  llvm_unreachable("no implementation.");
  return false;
}

bool QuantizeContext::StatesManager::SetResultParams(Operation *op,
                                                     int res_index,
                                                     QuantParams params) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_12(mht_12_v, 535, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::StatesManager::SetResultParams");

  auto &state = GetResultQuantState(op, res_index);
  if (state.params == params) {
    return false;
  }
  if (!state.IsEmpty()) {
    auto &rescale = GetResultRequantizeState(op, res_index);
    rescale.params = params;
    rescale.pos = RequantizeState::ON_INPUT;
    return false;
  }
  state.params = params;
  return true;
}

bool QuantizeContext::StatesManager::SetOperandParams(Operation *op, int index,
                                                      QuantParams params) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_contextDTcc mht_13(mht_13_v, 554, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_context.cc", "QuantizeContext::StatesManager::SetOperandParams");

  auto &state = GetOperandQuantState(op, index);
  if (state.params == params) {
    return false;
  }

  if (!state.IsEmpty()) {
    auto &rescale = GetOperandRequantizeState(op, index);
    rescale.params = params;
    rescale.pos = RequantizeState::ON_OUTPUT;
    return false;
  }
  state.params = params;
  return true;
}
}  //  namespace quant
}  // namespace mlir
