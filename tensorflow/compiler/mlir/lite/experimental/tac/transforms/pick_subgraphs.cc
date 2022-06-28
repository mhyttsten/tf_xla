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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc() {
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

#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/cost.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/subgraph.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/cost_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

// This pass is used to "determine" the best combinations of the whole "graph".
//
// Assume we have the graph looks like below:
//    subgraph1 (CPU/GPU)    subgraph2 (CPU)
//      \                     /
//      subgraph3 (CPU/GPU)     subgraph4 (CPU/GPU)
//         |                  /
//      subgraph5 (CPU/GPU)
//         |
//      subgraph6 (CPU)
//
//  We want to evaluate the possible options and minize the overall costs to
// produce a graph like below:
//
//    subgraph1 (GPU)   subgraph2(CPU)
//       \              /
//     subgraph3 (GPU)      subgraph4(GPU)
//         |             /
//      subgraph5 (GPU)
//         |
//      subgraph6 (CPU)
//
// The overall workflow of the pick subgraphs pass:
//  1) Build subgraphs
//    1.1) Collect output subgraphs.
//    1.2) Build `Subgraph` and their "alternative view" from FuncOp.
//  2) Pick subgraphs
//    2.1) Populate the "dp table" for (subgraph, hardware).
//    2.2) Make decisions based on the populated dp table.
//    2.3) Rewire the whole graph based on the desicions.
//
namespace mlir {
namespace TFL {
namespace tac {
namespace {

// GrapView is used to hold the aggregated cost for the given hardware
// view.
struct GraphView {
  float total_cost;
  std::unordered_map<Operation*, InferenceDeviceType> input_subgraph_plans;
};

// Subgraph is to hold the "conceptual" subgraph.
// A subgraph may associate with 1...n FuncOp, and each FuncOp may correspond
// with different hardwares.
struct Subgraph {
  // The call can be thought as an "API".
  func::CallOp call;

  // available_choces can be viewed as "real implementation" assosicated with
  // the hardware.
  std::unordered_map<InferenceDeviceType, FuncOp,
                     InferenceDeviceType::inference_device_type_hash>
      available_choices;

  // This will include self (the subgraph itself).
  // subgraphn
  //    |
  // current_subgraph   <- aggregated cost
  std::unordered_map<InferenceDeviceType, GraphView,
                     InferenceDeviceType::inference_device_type_hash>
      aggregated_cost_with_decisions;
};

// If the output is produced by a callop, will return the callop, otherwise,
// will return nullptr.
inline func::CallOp GetProducerCallOpOrNull(Value output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_0(mht_0_v, 286, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "GetProducerCallOpOrNull");

  Operation* output_op = output.getDefiningOp();
  if (output_op != nullptr && llvm::isa<func::CallOp>(output_op)) {
    return llvm::cast<func::CallOp>(output_op);
  }
  return nullptr;
}

class PickSubgraphsPass
    : public mlir::PassWrapper<PickSubgraphsPass,
                               mlir::OperationPass<ModuleOp>> {
  llvm::StringRef getArgument() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_1(mht_1_v, 300, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "getArgument");
 return "tfl-pick-subgraphs"; }
  llvm::StringRef getDescription() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_2(mht_2_v, 304, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "getDescription");

    return "Pick the best subgraphs to minimize the overall total costs.";
  }
  void runOnOperation() override;

  std::unordered_map<std::string, std::vector<FuncOp>> CollectSubgraphFuncs(
      ModuleOp module);

  void BuildSubgraphs(
      FuncOp main_fn,
      const std::unordered_map<std::string, std::vector<FuncOp>>& func_impls,
      llvm::SetVector<Operation*>* unprocessed_subgraphs,
      SmallVector<func::CallOp, 4>* output_subgraphs);

  void ProcessSubgraph(func::CallOp current_graph,
                       llvm::SetVector<Operation*>* unprocessed_subgraphs);

  bool PickSubgraphs(llvm::SetVector<Operation*>* all_subgraphs,
                     ArrayRef<func::CallOp> output_subgraphs,
                     const std::unordered_map<std::string, std::vector<FuncOp>>&
                         collected_impl_funcs,
                     OpBuilder* builder);

  // Make the decisions based on the subgraphs.
  // It may be the case we cannot decide the best scenarios for the user,
  // in this case, we just return false.
  bool MakeDecisions(ArrayRef<func::CallOp> output_subgraphs);

  // Rewire the subgraphs based on the decisions made.
  // If we cannot make a decisions, we just don't do anything.
  // TODO(renjieliu): we may change the vector to a map of hardware with
  // corresponding ipml.
  void RewireSubgraphs(
      const std::unordered_map<std::string, std::vector<FuncOp>>&
          collected_impl_funcs,
      OpBuilder* builder);

  float GetCostOrFail(FuncOp func);

  llvm::DenseMap<Operation*, Subgraph> subgraphs_;

  llvm::DenseMap<Operation*, InferenceDeviceType> decisions_;
};

float PickSubgraphsPass::GetCostOrFail(FuncOp func) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_3(mht_3_v, 351, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "PickSubgraphsPass::GetCostOrFail");

  float self_cost;
  if (!GetCostOnOp(func, &self_cost)) {
    func.emitError("we cannot find cost for this func");
    signalPassFailure();
  }
  return self_cost;
}

// Here we choose to do a greedy dynamic programming based algorithm for
// simplicity.
//
// See the following graph:
//
//    input_subgraph_1      ....      input_subgraph_n
//              \                          /
//               \                        /
//                   current_subgraph
//                      /     |      \
//
// Assume all the input subgraphs of the current subgraph are independent.
// If we already got optimal results for all the input subgraphs.
// Then the current_subgraph's aggregated optimal costs with regards to target
// perspective is simply:
//     for target in current_subgraph.supported_targets:
//       total_cost = 0
//       for input_subgraph in current_subgraph.input_subgraphs:
//         input_cost = kInfinity
//         for input_target in input_subgraphs.upported_targets:
//           # cost = aggregated cost for input_subgraph with transfer cost.
//           input_cost = min(input_cost, cost)
//         total_cost += input_cost
//       total_cost += current_subgraph.get_computation_cost(target)
//
// Note: for input subgraphs are not independent case, the dp case it a little
// bit complicated to handle. A potential thought is resolve only where
// conflict "happened".
//
// The above mentioned thought should probably be revisited for better thought
// or expanded more for more careful design.
// TODO(renjieliu): We may revisit this later.
void PickSubgraphsPass::ProcessSubgraph(
    func::CallOp current_graph_call,
    llvm::SetVector<Operation*>* unprocessed_subgraphs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_4(mht_4_v, 397, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "PickSubgraphsPass::ProcessSubgraph");

  Subgraph& current_subgraph = subgraphs_.find(current_graph_call)->second;

  std::vector<Subgraph*> input_subgraphs;
  for (auto input : current_graph_call.getOperands()) {
    func::CallOp input_call = GetProducerCallOpOrNull(input);
    // If the input subgraph is not processed yet, we just go ahead and process
    // that one first.
    if (input_call == nullptr) continue;

    if (unprocessed_subgraphs->count(input_call) > 0) {
      unprocessed_subgraphs->remove(input_call);
      ProcessSubgraph(input_call, unprocessed_subgraphs);
    }
    Subgraph& input_subgraph = subgraphs_.find(input_call)->second;
    input_subgraphs.push_back(&input_subgraph);
  }

  // Find the best plan for the current subgraph.
  for (const auto& kv : current_subgraph.available_choices) {
    const auto& current_inference_device_type = kv.first;
    FuncOp impl_target = kv.second;
    float self_compute_cost = GetCostOrFail(impl_target);

    GraphView current_graph_view;
    auto& input_subgraph_plans = current_graph_view.input_subgraph_plans;

    float inputs_total_costs = 0.0;
    for (Subgraph* input_subgraph : input_subgraphs) {
      float input_total_cost = std::numeric_limits<float>::max();
      for (const auto& input_kv : input_subgraph->available_choices) {
        const auto& input_inference_device_type = input_kv.first;
        FuncOp input_impl_target = input_kv.second;
        float input_compute_cost = GetCostOrFail(input_impl_target);

        float transfer_cost =
            GetTransferCost(input_inference_device_type.hardware,
                            current_inference_device_type.hardware,
                            input_subgraph->call, current_graph_call);
        float quant_dequant_cost =
            GetQuantDequantCost(input_inference_device_type.inference_type,
                                current_inference_device_type.inference_type,
                                input_subgraph->call, current_graph_call);
        float summed_cost =
            transfer_cost + quant_dequant_cost + input_compute_cost;

        if (summed_cost < input_total_cost) {
          // Looks this hardware is better for this input_subgraph, let's change
          // it.
          input_total_cost = summed_cost;
          input_subgraph_plans[input_subgraph->call] =
              input_inference_device_type;
        }
      }  // for every hardware of input_subgraph
      inputs_total_costs += input_total_cost;
    }  // for every input_subgraph
    current_graph_view.total_cost = inputs_total_costs + self_compute_cost;
    current_subgraph
        .aggregated_cost_with_decisions[current_inference_device_type] =
        current_graph_view;
  }  // for every subgraph
}

void PickSubgraphsPass::BuildSubgraphs(
    FuncOp fn,
    const std::unordered_map<std::string, std::vector<FuncOp>>& func_impls,
    llvm::SetVector<Operation*>* unprocessed_subgraphs,
    SmallVector<func::CallOp, 4>* output_subgraphs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_5(mht_5_v, 467, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "PickSubgraphsPass::BuildSubgraphs");

  llvm::DenseSet<Operation*> returned_call_op_set;
  // Collect all returns first from the main function.
  // all the outputs of the main function are actually the final outputs.
  // main_func:
  //  %output1 = call @subgraph_1...
  //   ...
  //  %output2 = call @subgraph_m...
  //   ...
  //  %outputn = call @subgraph_k...
  //  return %output1, output2, ..., outputn.
  fn.walk([&](func::ReturnOp return_op) {
    for (auto output : return_op.getOperands()) {
      func::CallOp output_call = GetProducerCallOpOrNull(output);
      if (output_call != nullptr) {
        returned_call_op_set.insert(output_call);
      }
    }
  });

  // Each call op actually is the entry of the subgraph.
  fn.walk([&](func::CallOp call_op) {
    auto interface_name = GetInterFaceName(call_op);
    // we only need to care about the call ops those have interface_name.
    if (!interface_name.hasValue()) return;

    unprocessed_subgraphs->insert(call_op);

    // Build the subgraph.
    Subgraph subgraph;
    subgraph.call = call_op;
    auto impl_iter = func_impls.find(interface_name.getValue());
    if (impl_iter == func_impls.end()) {
      call_op.emitError(
          "we cannot find corresponding implementation for this call op");
      signalPassFailure();
    }

    for (auto impl : impl_iter->second) {
      auto inference_device_type = GetInferenceDeviceTypeForOp(impl);
      if (!inference_device_type.hasValue()) {
        impl.emitError("we cannot find inference device type for this func");
        signalPassFailure();
      }
      subgraph.available_choices.emplace(inference_device_type.getValue(),
                                         impl);
    }

    // Insert in the subgraphs.
    subgraphs_.try_emplace(call_op, subgraph);

    // If it's an output subgraph, we will add to the output_subgraphs.
    if (returned_call_op_set.find(call_op) != returned_call_op_set.end()) {
      output_subgraphs->push_back(call_op);
    }
  });
}

// Collect all the subgraphs (and their alternatives) in the module.
std::unordered_map<std::string, std::vector<FuncOp>>
PickSubgraphsPass::CollectSubgraphFuncs(ModuleOp module) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_6(mht_6_v, 530, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "PickSubgraphsPass::CollectSubgraphFuncs");

  std::unordered_map<std::string, std::vector<FuncOp>> func_impls;
  for (auto func : module.getOps<FuncOp>()) {
    auto interface_name = GetInterFaceName(func);
    if (interface_name.hasValue()) {
      auto impls_iter = func_impls.find(interface_name.getValue());
      if (impls_iter == func_impls.end())
        impls_iter =
            func_impls.emplace(interface_name.getValue(), std::vector<FuncOp>())
                .first;
      impls_iter->second.push_back(func);
    }
  }
  return func_impls;
}

// Given the final outputs, evaluate on the overall costs and pick the best
// plan, if we cannot make a decision, nothing would change, just fallback
// to the original plan.
bool PickSubgraphsPass::MakeDecisions(ArrayRef<func::CallOp> output_subgraphs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_7(mht_7_v, 552, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "PickSubgraphsPass::MakeDecisions");

  // BFS to make decisions.
  std::queue<const GraphView*> processing_queue;
  for (func::CallOp output : output_subgraphs) {
    const GraphView* preferred_graph_view;
    float minimum_cost = std::numeric_limits<float>::max();

    const Subgraph& subgraph = subgraphs_.find(output)->second;
    for (const auto& kv : subgraph.aggregated_cost_with_decisions) {
      if (minimum_cost > kv.second.total_cost) {
        minimum_cost = kv.second.total_cost;
        preferred_graph_view = &kv.second;
        decisions_[output] = kv.first;
      }
    }

    processing_queue.push(preferred_graph_view);
  }

  // If we see conflict, we will just abort.
  while (!processing_queue.empty()) {
    const GraphView* current = processing_queue.front();
    processing_queue.pop();
    for (const auto& input_with_plans : current->input_subgraph_plans) {
      func::CallOp input = llvm::cast<func::CallOp>(input_with_plans.first);
      const InferenceDeviceType& input_decision = input_with_plans.second;
      auto made_input_decision_it = decisions_.find(input);
      if (made_input_decision_it == decisions_.end()) {
        // Input is not processed.
        // Let's process it, also push it to the queue.
        decisions_[input] = input_decision;
        const Subgraph& input_subgraph = subgraphs_.find(input)->second;
        const GraphView& input_subgraph_view =
            input_subgraph.aggregated_cost_with_decisions.find(input_decision)
                ->second;
        processing_queue.push(&input_subgraph_view);
      } else if (made_input_decision_it->second != input_decision) {
        // We see confliction, we need to abort.
        return false;
      }
    }
  }
  return true;
}

// This rewire subgraph is essentially "hook" the call op with the "best" choice
// (subgraph).
void PickSubgraphsPass::RewireSubgraphs(
    const std::unordered_map<std::string, std::vector<FuncOp>>&
        collected_impl_funcs,
    OpBuilder* builder) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_8(mht_8_v, 605, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "PickSubgraphsPass::RewireSubgraphs");

  for (auto& kv : decisions_) {
    func::CallOp call = llvm::cast<func::CallOp>(kv.first);

    const InferenceDeviceType& preferred_inference_device_type = kv.second;

    // We need to rewire the call.
    std::string interface_name = GetInterFaceName(call).getValue();
    for (auto impl : collected_impl_funcs.find(interface_name)->second) {
      const auto& impl_inference_device_type =
          GetInferenceDeviceTypeForOp(impl);
      if (impl_inference_device_type.getValue() ==
          preferred_inference_device_type) {
        if (call.getCallee() != impl.getName()) {
          // We need to rebuild the call op. :(
          builder->setInsertionPoint(call);
          auto new_call = builder->create<func::CallOp>(call.getLoc(), impl,
                                                        call.getOperands());

          // Set interface_name & target to the call_op as well.
          new_call->setAttr(kInterfaceNameAttr,
                            builder->getStringAttr(interface_name));
          new_call->setAttr(
              kDevice,
              builder->getStringAttr(preferred_inference_device_type.hardware));
          new_call->setAttr(
              kInferenceType,
              builder->getStringAttr(GetInferenceString(
                  preferred_inference_device_type.inference_type)));

          call.replaceAllUsesWith(new_call.getResults());
          call.erase();
        }
      }
    }
  }
}

bool PickSubgraphsPass::PickSubgraphs(
    llvm::SetVector<Operation*>* all_subgraphs,
    ArrayRef<func::CallOp> output_subgraphs,
    const std::unordered_map<std::string, std::vector<FuncOp>>&
        collected_impl_funcs,
    OpBuilder* builder) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_9(mht_9_v, 651, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "PickSubgraphsPass::PickSubgraphs");

  // Process those collected unprocessed subgraphs.
  //
  // Algorithm complexity for this:
  // This Complexity should be O(edge * specs ^ 2).
  // We should expect the specs to be a small number.
  // In future, the spesc can be Hardwares x inference_types
  // The Hardware can be {CPU, GPU, DSP, EDGE_TPU}
  // The inference_types can be {float, Q_INT8, float16}.
  // But still, we should expect the specs to be a small number.
  //
  // The process is essentially evaluating the accumulated cost for the dp table
  // for all the subgraphs (and their alternatives).
  while (!all_subgraphs->empty()) {
    func::CallOp current_subgraph =
        llvm::cast<func::CallOp>(all_subgraphs->front());
    all_subgraphs->remove(current_subgraph);
    ProcessSubgraph(current_subgraph, all_subgraphs);
  }

  // Make decisions given the "outputs" and the populated dp table.
  // This is hoping to achieve a global minimum.
  if (!MakeDecisions(output_subgraphs)) {
    return false;
  }

  // Once the design has been made.
  // Start from the outputs and go back and checkout the plan.
  RewireSubgraphs(collected_impl_funcs, builder);

  return true;
}

void PickSubgraphsPass::runOnOperation() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPStransformsPSpick_subgraphsDTcc mht_10(mht_10_v, 687, "", "./tensorflow/compiler/mlir/lite/experimental/tac/transforms/pick_subgraphs.cc", "PickSubgraphsPass::runOnOperation");

  auto module = getOperation();
  // Collect & build the subgraphs.
  // Also collect the output subgraphs.
  // Output subgraphs are essentially those subgraphs pointed by the return
  // op.
  const std::unordered_map<std::string, std::vector<FuncOp>> func_impls =
      CollectSubgraphFuncs(module);
  llvm::SetVector<Operation*> unprocessed_subgraphs;
  SmallVector<func::CallOp, 4> output_subgraphs;

  for (auto fn : module.getOps<FuncOp>()) {
    BuildSubgraphs(fn, func_impls, &unprocessed_subgraphs, &output_subgraphs);
  }
  OpBuilder builder(module);
  if (!PickSubgraphs(&unprocessed_subgraphs, output_subgraphs, func_impls,
                     &builder)) {
    module.emitWarning(
        "we cannot find the best scenarios for your case, so we just use "
        "your original model plans");
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreatePickSubgraphsPass() {
  return std::make_unique<PickSubgraphsPass>();
}

static PassRegistration<PickSubgraphsPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
