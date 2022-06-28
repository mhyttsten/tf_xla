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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc() {
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
#include <unordered_map>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/subgraph.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// Subgraph here is actually an intermediate data structure holder for the ops:
// The ops within share the same "target", they're topologically sorted.
// The subgraph here will be later populated to generate func ops.
// All the subgraphs should not create cyclic dependencies:
// So we should not have:
//     subgraph1
//             \
//            subgraph2
//            /
//       subgraph1
struct Subgraph {
  // All ops must be inserted in it's topological order.
  llvm::SetVector<Operation*> all_ops;
  int subgraph_id;
  InferenceDeviceType inference_device_type;
};

// This will exclude arguments & consts & quantize/dequantize ops.
inline bool IsNonConstQuantizeOp(Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_0(mht_0_v, 238, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "IsNonConstQuantizeOp");

  return IsNonConstOp(op) && NotTFLQuantDequantizeOp(op) && !IsTerminatorOp(op);
}

// This pass will group those ops (non-const TFL dialect ops) have the same
// target together and raise them as FuncOps.
// See the following Example:
//
//     op1 (GPU)
//       \       op2 (GPU)
//       \        |
//        \      op3 (GPU)
//         \     /
//         op4 (CPU)
//
// will be raised as 3 subgraphs:
// Subgraph 1: {op1}, GPU -> Func_1_GPU
// Subgraph 2: {op2, op3}, GPU -> Func_2_GPU
// Subgraph 3: {op4} CPU -> Func_3_CPU
//
// MainFunc:
//   %0 = call @Func_1_GPU
//   %1 = call @Func_2_GPU
//   %2 = call @Func_3_CPU(%0, %1)
class RaiseTargetSubgraphsPass
    : public mlir::PassWrapper<RaiseTargetSubgraphsPass,
                               mlir::OperationPass<ModuleOp>> {
 private:
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_1(mht_1_v, 269, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "getArgument");

    return "tfl-raise-target-subgraphs";
  }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_2(mht_2_v, 275, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "getDescription");

    return "This pass will merge those have target-annotated TFL IRs together "
           "& raise them as a function.";
  }
  void runOnOperation() override;

  void RaiseTargetSubgraphsForBlock(Block* block, OpBuilder* builder,
                                    ModuleOp module);

  void ExtractSubgraphToFunc(Subgraph* subgraph, OpBuilder* builder,
                             ModuleOp module);

  FuncOp BuildFuncOp(Subgraph* subgraph, OpBuilder* builder, ModuleOp module_op,
                     SmallVector<Value, 4>* inputs,
                     SmallVector<Value, 4>* outputs,
                     InferenceDeviceType* inference_device_type);

  int subgraph_count_ = 0;
};

// This is to collect input arguments for the given set of ops.
// See the example:
//
//   value1  value2
//    \     /
//      op1
//        \     value3
//        \   /
//         op2
//         |
//         op3
//
//  Then the arguments will be {value1, value2, value3}
void CollectInputs(const llvm::SetVector<Operation*>& all_ops,
                   SmallVector<Value, 4>* inputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_3(mht_3_v, 312, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "CollectInputs");

  for (Operation* op : all_ops) {
    for (Value input : op->getOperands()) {
      Operation* input_op = input.getDefiningOp();
      const bool input_within_subgraph =
          (input_op && all_ops.count(input_op) == 1);
      if (!input_within_subgraph) {
        inputs->push_back(input);
      }
    }
  }
}

// This is to collect outputs arguments for the given set of ops.
// See the example:
//
//      op1
//      /    \
//   value1   \
//           op2
//           |  \
//         op3  value2
//         |
//       value3
//
//  Then the arguments will be {value1, value2, value3}
void CollectOutputs(const llvm::SetVector<Operation*>& all_ops,
                    SmallVector<Value, 4>* outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_4(mht_4_v, 342, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "CollectOutputs");

  for (Operation* op : all_ops) {
    for (Value output : op->getResults()) {
      bool output_consumed_outside_subgraph = false;
      for (Operation* consumer : output.getUsers()) {
        if (all_ops.count(consumer) == 0) {
          output_consumed_outside_subgraph = true;
        }
      }
      if (output_consumed_outside_subgraph) {
        outputs->push_back(output);
      }
    }
  }
}

void BuildTypes(const SmallVector<Value, 4>& values,
                SmallVector<Type, 4>* types) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_5(mht_5_v, 362, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "BuildTypes");

  for (auto value : values) {
    types->push_back(value.getType());
  }
}

void GetFunctionName(const Subgraph& subgrpah, std::string* function_name,
                     std::string* interface_name) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_6(mht_6_v, 372, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "GetFunctionName");

  *interface_name = absl::StrCat("func_", std::to_string(subgrpah.subgraph_id));
  *function_name = absl::StrCat(
      (*interface_name), "_", subgrpah.inference_device_type.hardware, "_",
      GetInferenceString(subgrpah.inference_device_type.inference_type));
}

FuncOp RaiseTargetSubgraphsPass::BuildFuncOp(
    Subgraph* subgraph, OpBuilder* builder, ModuleOp module_op,
    SmallVector<Value, 4>* inputs, SmallVector<Value, 4>* outputs,
    InferenceDeviceType* inference_device_type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_7(mht_7_v, 385, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "RaiseTargetSubgraphsPass::BuildFuncOp");

  CollectInputs(subgraph->all_ops, inputs);
  CollectOutputs(subgraph->all_ops, outputs);

  SmallVector<Type, 4> input_types;
  SmallVector<Type, 4> return_types;

  BuildTypes(*inputs, &input_types);
  BuildTypes(*outputs, &return_types);

  FunctionType function_type =
      builder->getFunctionType(input_types, return_types);

  SmallVector<NamedAttribute, 4> attrs;
  // Function name.
  std::string function_name;
  std::string interface_name;
  GetFunctionName(*subgraph, &function_name, &interface_name);
  attrs.push_back(builder->getNamedAttr(
      kInterfaceNameAttr, builder->getStringAttr(interface_name)));

  // Inference Device type.
  attrs.push_back(builder->getNamedAttr(
      kDevice,
      builder->getStringAttr(subgraph->inference_device_type.hardware)));
  attrs.push_back(builder->getNamedAttr(
      kInferenceType, builder->getStringAttr(GetInferenceString(
                          subgraph->inference_device_type.inference_type))));
  *inference_device_type = subgraph->inference_device_type;

  FuncOp new_func = FuncOp::create(builder->getUnknownLoc(), function_name,
                                   function_type, llvm::makeArrayRef(attrs));
  new_func.setPrivate();

  new_func.addEntryBlock();

  // Function argument mapping.
  llvm::DenseMap<Value, int> function_argument_mapping;
  for (int i = 0; i < inputs->size(); ++i) {
    function_argument_mapping.insert({(*inputs)[i], i});
  }

  OpBuilder function_builder(new_func.getBody());

  llvm::DenseMap<Operation*, Operation*> op_cloned_op_mapping;
  llvm::DenseMap<Value, Value> output_cloned_op_output_mapping;
  for (Operation* op : subgraph->all_ops) {
    Operation* cloned_op = function_builder.clone(*op);
    op_cloned_op_mapping.insert({op, cloned_op});
    for (int i = 0; i < op->getNumResults(); ++i) {
      Value op_output = op->getResult(i);
      Value cloned_op_output = cloned_op->getResult(i);
      output_cloned_op_output_mapping.insert({op_output, cloned_op_output});
    }
  }

  for (Operation* op : subgraph->all_ops) {
    Operation* cloned_op = op_cloned_op_mapping.find(op)->second;
    for (int i = 0; i < op->getNumOperands(); ++i) {
      Value input = op->getOperand(i);
      Value cloned_op_input;
      // If the input is actually a function argument.
      if (function_argument_mapping.count(input) > 0) {
        int function_argument = function_argument_mapping.find(input)->second;
        cloned_op_input = new_func.getArgument(function_argument);
      } else {
        // The input is actually with in the subgraph.
        cloned_op_input = output_cloned_op_output_mapping.find(input)->second;
      }
      cloned_op->setOperand(i, cloned_op_input);
    }
  }

  SmallVector<Value, 4> final_outputs;
  for (auto output : *outputs) {
    auto cloned_output = output_cloned_op_output_mapping.find(output)->second;
    final_outputs.push_back(cloned_output);
  }
  function_builder.create<mlir::func::ReturnOp>(new_func.getLoc(),
                                                final_outputs);

  module_op.push_back(new_func);
  return new_func;
}

void RaiseTargetSubgraphsPass::ExtractSubgraphToFunc(Subgraph* subgraph,
                                                     OpBuilder* builder,
                                                     ModuleOp module) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_8(mht_8_v, 475, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "RaiseTargetSubgraphsPass::ExtractSubgraphToFunc");

  SmallVector<Value, 4> func_inputs;
  SmallVector<Value, 4> func_outputs;

  InferenceDeviceType inference_device_type;
  FuncOp func = BuildFuncOp(subgraph, builder, module, &func_inputs,
                            &func_outputs, &inference_device_type);

  // We just use the location of the last ops in the subgraph as the location
  // for the call_op.
  Operation* last_output = subgraph->all_ops.back();

  // TODO(renjieliu): we should add func attributes to the call op.
  builder->setInsertionPoint(last_output);
  auto call_op =
      builder->create<func::CallOp>(last_output->getLoc(), func, func_inputs);

  auto interface_name = GetInterFaceName(func);

  // Set call op attribute: interface_name, hardware.
  call_op->setAttr(kInterfaceNameAttr,
                   builder->getStringAttr(interface_name.getValue()));
  call_op->setAttr(kDevice,
                   builder->getStringAttr(inference_device_type.hardware));
  call_op->setAttr(kInferenceType, builder->getStringAttr(GetInferenceString(
                                       inference_device_type.inference_type)));

  // Rewire the outputs.
  if (call_op.getNumResults() != func_outputs.size()) {
    module.emitError("the constructed func op has mismatched returns");
    signalPassFailure();
  }

  for (int i = 0; i < func_outputs.size(); ++i) {
    Value output = func_outputs[i];
    output.replaceAllUsesWith(call_op.getResult(i));
  }

  // Clear the subgraph.
  // Those ops should be removed.
  for (auto* op : subgraph->all_ops) {
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  }
}

// TODO(renjieliu): We may need to consider about side effect ops: we may leave
// those ops alone when building the subgraph.
void RaiseTargetSubgraphsPass::RaiseTargetSubgraphsForBlock(Block* block,
                                                            OpBuilder* builder,
                                                            ModuleOp module) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_9(mht_9_v, 529, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "RaiseTargetSubgraphsPass::RaiseTargetSubgraphsForBlock");

  // This is a very naive implementation:
  // It will greedily group adjacent ops that have the same inference type to a
  // subgraph.
  llvm::DenseMap<int, Subgraph> all_subgraphs;
  llvm::Optional<InferenceDeviceType> previous_device_type = llvm::None;
  int current_subgraph_id = -1;
  for (auto& op : *block) {
    if (IsNonConstQuantizeOp(&op) && !IsTerminatorOp(&op) &&
        !llvm::isa<func::ReturnOp, FuncOp, CallOpInterface>(op)) {
      auto current_device_type = GetInferenceDeviceTypeForOp(&op);
      if (!(current_device_type.hasValue() &&
            current_device_type == previous_device_type)) {
        // We should start a new subgraph.
        Subgraph new_subgraph;
        new_subgraph.inference_device_type = current_device_type.getValue();
        new_subgraph.subgraph_id = subgraph_count_++;
        all_subgraphs.insert({new_subgraph.subgraph_id, new_subgraph});
        current_subgraph_id = new_subgraph.subgraph_id;
      }
      previous_device_type = current_device_type;
      all_subgraphs.find(current_subgraph_id)->second.all_ops.insert(&op);
    }
  }

  // Create FuncOp & replace with current uses based on those subgraphs.
  for (auto& subgraph : all_subgraphs) {
    ExtractSubgraphToFunc(&subgraph.second, builder, module);
  }
}

void RaiseTargetSubgraphsPass::runOnOperation() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSraise_target_subgraphsDTcc mht_10(mht_10_v, 563, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/raise_target_subgraphs.cc", "RaiseTargetSubgraphsPass::runOnOperation");

  auto module = getOperation();
  SmallVector<FuncOp, 16> funcs(module.getOps<FuncOp>());
  for (auto func : funcs) {
    for (auto& block : func) {
      auto builder = OpBuilder::atBlockBegin(&block);
      RaiseTargetSubgraphsForBlock(&block, &builder, module);
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateRaiseTargetSubgraphsPass() {
  return std::make_unique<RaiseTargetSubgraphsPass>();
}

static PassRegistration<RaiseTargetSubgraphsPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
