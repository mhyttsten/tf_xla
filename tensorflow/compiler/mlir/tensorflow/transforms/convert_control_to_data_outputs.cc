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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc() {
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

#include <algorithm>
#include <queue>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verify_suitable_for_graph_export.h"

namespace mlir {
namespace tf_executor {
namespace {

using TF::ResourceId;
static constexpr ResourceId kUnknownResourceId =
    TF::detail::ResourceAliasAnalysisInfo::kUnknownResourceId;
static constexpr ResourceId kInvalidResourceId =
    TF::detail::ResourceAliasAnalysisInfo::kInvalidResourceId;
using OperationSetTy = SmallPtrSet<Operation*, 4>;
using ResourceToOpsMapTy = DenseMap<ResourceId, OperationSetTy>;

class ConvertControlToDataOutputsPass
    : public TF::ExecutorConvertControlToDataOutputsPassBase<
          ConvertControlToDataOutputsPass> {
 public:
  void runOnOperation() override;
};

// Returns a vector of all tf.WhileOp(s) which use func as while body. If any of
// the uses is as a while condition, an empty vector is returned.
SmallVector<TF::WhileOp> GetWhileCallers(FuncOp func,
                                         SymbolUserMap& symbol_map) {
  SmallVector<TF::WhileOp> while_callers;
  for (auto user : symbol_map.getUsers(func)) {
    if (auto while_caller = dyn_cast<TF::WhileOp>(user)) {
      // If used as while conditional anywhere, then skip optimizing this
      // function. Return empty vector.
      if (while_caller.cond_function() == func) return {};
      assert(while_caller.body_function() == func);
      while_callers.push_back(while_caller);
    }
  }
  return while_callers;
}

// Populates `chain_resource_to_ops_map`, the map from all resources that need
// to be chained to the set of operations that access the resource, and
// `resource_equivalence_classes`. Resources are equivalent if they are accessed
// by a common op, and equivalent resources will be assigned to the same chain.
void CollectChainResources(
    FuncOp func, ResourceToOpsMapTy& chain_resource_to_ops_map,
    llvm::EquivalenceClasses<ResourceId>& resource_equivalence_classes,
    const TF::SideEffectAnalysis::Info& side_effect_analysis) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_0(mht_0_v, 251, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "CollectChainResources");

  auto graph_op = cast<GraphOp>(func.front().front());

  // For each op in the graph, get the resources it uses and update the access
  // information for them.
  graph_op.walk([&](IslandOp island) {
    // This pass assumes that all functions are suitable for export i.e., each
    // function has a single tf_executor.graph op and all islands wrap the
    // internal op perfectly. Hence this assertion should never fail.
    assert(island.WrapsSingleOp());
    Operation& op = island.GetBody().front();

    ResourceId prev_resource_id = kInvalidResourceId;
    for (auto resource_id_read_only_pair :
         side_effect_analysis.GetResourceIds(&op)) {
      ResourceId resource_id = resource_id_read_only_pair.first;
      // If the resource was allocated by an op with `UniqueResourceAllocation`
      // trait, then we don't need to chain resource ops accessing this resource
      // between iterations: Every iteration will create a new independent
      // resource. This enables more parallelism across iterations.
      if (!side_effect_analysis.IsUniqueResourceAllocationId(resource_id)) {
        chain_resource_to_ops_map[resource_id].insert(&op);
        if (prev_resource_id != kInvalidResourceId) {
          // Merge class of current ID with class of previous ID since both
          // resources are accessed by `op`.
          resource_equivalence_classes.unionSets(prev_resource_id, resource_id);
        } else {
          resource_equivalence_classes.insert(resource_id);
        }
        prev_resource_id = resource_id;
      }
    }
  });
}

// tf.NoOp islands are used to combine multiple control dependencies into one.
// These islands have a single tf.NoOp inside them and consume multiple control
// outputs to generate a single control output.
//
// For example,
// ```
// %merged_control = "tf_executor.island"(%control_a, %control_b) ({
//   "tf.NoOp"() : () -> ()
//   "tf_executor.yield"() : () -> ()
// }) : (!tf_executor.control, !tf_executor.control) -> (!tf_executor.control)
// ```
//
// `%merged_control` is a NoOp control barrier in this case.
//
// Checks if the value `control` is a NoOp control barrier.
bool IsNoOpControlBarrier(Value control) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_1(mht_1_v, 304, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "IsNoOpControlBarrier");

  if (!control.getType().isa<ControlType>()) return false;

  auto control_island = dyn_cast_or_null<IslandOp>(control.getDefiningOp());
  if (!control_island) return false;

  // All islands perfectly wrap a single op is an invariant of this pass and
  // is checked at the very beginning of the pass.
  assert(control_island.WrapsSingleOp());
  return control_island.outputs().empty() &&
         isa<TF::NoOp>(control_island.GetBody().front());
}

// Remove all control outputs of the function. Traverses NoOp control barrier
// chains from FetchOp to all NoOp control barriers. Returns true
// iff at least one control output is deleted.
bool RemoveAllControlOutputs(FuncOp func) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_2(mht_2_v, 323, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "RemoveAllControlOutputs");

  auto graph_op = cast<GraphOp>(func.front().front());

  FetchOp fetch = graph_op.GetFetch();
  // Return early if no control outputs exist.
  if (fetch.getNumOperands() == graph_op->getNumResults()) return false;

  std::queue<Value> control_barrier_worklist;
  for (Value control_output :
       fetch.fetches().drop_front(graph_op->getNumResults())) {
    if (IsNoOpControlBarrier(control_output))
      control_barrier_worklist.push(control_output);
  }

  // Erase all control outputs at the end from fetch.
  fetch.fetchesMutable().erase(
      graph_op.getNumResults(),
      fetch.getNumOperands() - graph_op.getNumResults());

  // Iterate the worklist to remove all NoOp control barriers at the end of the
  // function body that are used to merge two or more control dependencies.
  while (!control_barrier_worklist.empty()) {
    Value control_barrier = control_barrier_worklist.front();
    control_barrier_worklist.pop();

    // We can only erase control barriers whose uses have been erased as well.
    if (!control_barrier.use_empty()) continue;

    // Only values defined by IslandOp were inserted in the worklist.
    IslandOp current_island = cast<IslandOp>(control_barrier.getDefiningOp());

    for (auto control_input : current_island.controlInputs()) {
      if (IsNoOpControlBarrier(control_input))
        control_barrier_worklist.push(control_input);
    }
    current_island.erase();
  }
  return true;
}

// Appends function arguments with `num_resources` number of arguments of
// requested type.
void AppendFunctionArguments(FuncOp func, int num_resources,
                             ShapedType chaining_data_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_3(mht_3_v, 369, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "AppendFunctionArguments");

  for (int i = 0; i < num_resources; ++i) {
    func.getRegion().addArgument(chaining_data_type, func.getLoc());
  }

  FunctionType ftype =
      FunctionType::get(func.getContext(), func.getBody().getArgumentTypes(),
                        func.getFunctionType().getResults());
  func.setType(ftype);
}

// Appends function results with `num_resources` number of results of requested
// type.
void AppendFunctionResults(FuncOp func, int num_resources,
                           ShapedType chaining_data_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_4(mht_4_v, 386, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "AppendFunctionResults");

  Block& block = func.front();
  auto graph_op = cast<GraphOp>(block.front());
  // Note that func result types are same as the result types of
  // GraphOp in the function `func`.
  assert(std::equal(func->getResultTypes().begin(),
                    func->getResultTypes().end(),
                    graph_op->getResultTypes().begin()));
  auto new_result_types =
      llvm::to_vector<4>(func.getFunctionType().getResults());
  for (int i = 0; i < num_resources; ++i) {
    new_result_types.push_back(chaining_data_type);
  }
  FunctionType ftype = FunctionType::get(
      func.getContext(), func.getArgumentTypes(), new_result_types);
  func.setType(ftype);

  // Rewrite GraphOp to have same number of results as the
  // function.
  OpBuilder builder(graph_op);
  auto new_graph_op =
      builder.create<GraphOp>(graph_op.getLoc(), new_result_types);
  new_graph_op.getRegion().takeBody(graph_op.getRegion());
  graph_op->replaceAllUsesWith(
      new_graph_op->getResults().drop_back(num_resources));
  graph_op.erase();
  func::ReturnOp return_op = cast<func::ReturnOp>(block.getTerminator());
  int num_old_arguments = return_op.getNumOperands();
  for (int i = 0; i < num_resources; ++i) {
    return_op.operandsMutable().append(
        new_graph_op.getResult(num_old_arguments + i));
  }
}

// Creates a wrapper island enclosing the `sub_op` dependent on
// `control_inputs`.
IslandOp CreateIsland(Operation* sub_op, ValueRange control_inputs,
                      OpBuilder builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_5(mht_5_v, 426, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "CreateIsland");

  assert(sub_op);
  auto control_type = ControlType::get(builder.getContext());
  auto island = builder.create<IslandOp>(
      sub_op->getLoc(), sub_op->getResultTypes(), control_type, control_inputs);
  island.body().push_back(new Block);
  Block* block = &island.body().back();
  builder.setInsertionPointToEnd(block);
  sub_op->replaceAllUsesWith(island.outputs());
  sub_op->moveBefore(block, block->begin());
  builder.create<YieldOp>(sub_op->getLoc(), sub_op->getResults());
  return island;
}

// Adds control dependencies from/to chain arguments/results. It adds two
// identity ops, chain_src and chain_sink, per resource equivalence class.
// Using the resource to operations map, it adds (1) a control dependency
// from chain_src to all the operations that read/write to a resource of the
// equivalence class, and (2) a control dependency from all the operations that
// read/write to a resource of the class to the chain_sink operation.
void ChainResourceOps(
    FuncOp func, ResourceToOpsMapTy& chain_resource_to_ops_map,
    llvm::EquivalenceClasses<ResourceId>& resource_equivalence_classes,
    int num_old_outputs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_6(mht_6_v, 452, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "ChainResourceOps");

  assert(num_old_outputs + resource_equivalence_classes.getNumClasses() ==
         func.getNumArguments());
  auto graph_op = cast<GraphOp>(func.front().front());

  auto fetch = graph_op.GetFetch();
  OpBuilder builder_chain_src(fetch);
  builder_chain_src.setInsertionPointToStart(fetch->getBlock());

  OpBuilder builder_chain_sink(fetch);
  int chain_index = num_old_outputs;

  // Iterate over all equivalence classes.
  for (auto class_iter = resource_equivalence_classes.begin();
       class_iter != resource_equivalence_classes.end(); ++class_iter) {
    // Only visit one element per class, the leader.
    if (!class_iter->isLeader()) continue;

    // Create chain source and sink identity islands for current equivalence
    // class.
    auto chain_arg = func.getArgument(chain_index++);
    auto src_identity = builder_chain_src.create<TF::IdentityOp>(
        chain_arg.getLoc(), chain_arg.getType(), chain_arg);
    auto chain_src_island = CreateIsland(src_identity, {}, builder_chain_src);

    auto sink_identity = builder_chain_sink.create<TF::IdentityOp>(
        chain_arg.getLoc(), chain_arg.getType(), chain_arg);
    auto chain_sink_island =
        CreateIsland(sink_identity, {}, builder_chain_sink);

    // Add the chain sink data output to fetch.
    fetch.fetchesMutable().append(chain_sink_island.outputs().front());

    // Iterate over all members of the current equivalence class (represented
    // by `class_iter`). Keep track of ops that have already been processed.
    llvm::SmallDenseSet<Operation*> processed_ops;
    for (auto member_iter =
             resource_equivalence_classes.member_begin(class_iter);
         member_iter != resource_equivalence_classes.member_end();
         ++member_iter) {
      ResourceId resource_id = *member_iter;
      auto map_iter = chain_resource_to_ops_map.find(resource_id);
      if (map_iter == chain_resource_to_ops_map.end()) continue;
      OperationSetTy& resource_ops = map_iter->getSecond();

      // Add dependencies between all ops that access current resource and chain
      // source and sink.
      for (Operation* op : resource_ops) {
        if (processed_ops.contains(op)) continue;

        IslandOp wrapper = op->getParentOfType<IslandOp>();
        assert(wrapper);
        wrapper.controlInputsMutable().append(chain_src_island.control());
        chain_sink_island.controlInputsMutable().append(wrapper.control());
        processed_ops.insert(op);
      }
    }
  }
  VLOG(2) << "Added " << resource_equivalence_classes.getNumClasses()
          << " chains for " << chain_resource_to_ops_map.size() << " resources";
}

// Generate a dummy constant island of requested type.
IslandOp GetDummyConstant(OpBuilder builder, ShapedType const_type,
                          Location loc) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_7(mht_7_v, 519, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "GetDummyConstant");

  DenseIntElementsAttr val = DenseIntElementsAttr::get(const_type, 1);
  auto const_op = builder.create<TF::ConstOp>(loc, val);
  auto const_island = CreateIsland(const_op, {}, builder);
  return const_island;
}

// Rewrites the while op with extra chaining operands and results. Uses a
// dummy constant of requested type as argument to all the new chaining
// operands.
TF::WhileOp RewriteWhileOp(TF::WhileOp while_op, int num_resource_inputs,
                           ShapedType const_type) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_8(mht_8_v, 533, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "RewriteWhileOp");

  IslandOp while_wrapper = while_op->getParentOfType<IslandOp>();
  assert(while_wrapper && "While op is expected to be wrapped in a IslandOp");

  // Get the dummy constant.
  OpBuilder builder(while_wrapper);
  auto loc = NameLoc::get(
      builder.getStringAttr("chain_control_outputs@" + while_op.body()));
  IslandOp const_wrapper = GetDummyConstant(builder, const_type, loc);

  // Get new operand and result types.
  auto new_operands = llvm::to_vector<4>(while_op->getOperands());
  auto new_result_types = llvm::to_vector<4>(while_op->getResultTypes());
  Value const_output = const_wrapper.outputs()[0];
  for (int i = 0; i < num_resource_inputs; ++i) {
    new_operands.push_back(const_output);
    new_result_types.push_back(const_output.getType());
  }

  // Replace old while op with new while op.
  auto new_while_op = builder.create<TF::WhileOp>(
      while_op.getLoc(), new_result_types, new_operands, while_op->getAttrs());
  auto new_while_wrapper =
      CreateIsland(new_while_op, while_wrapper.controlInputs(), builder);
  for (auto result : while_wrapper.outputs()) {
    result.replaceAllUsesWith(
        new_while_wrapper.outputs()[result.getResultNumber()]);
  }
  while_wrapper.control().replaceAllUsesWith(new_while_wrapper.control());
  while_wrapper.erase();
  return new_while_op;
}

// Converts the control outputs of the while body to data outputs, thus
// removing control barrier at the end of while loop body.
void ConvertControlToDataOutputs(
    FuncOp while_body, SmallVectorImpl<TF::WhileOp>& while_callers,
    OperationSetTy& recompute_analysis_for_funcs,
    const TF::SideEffectAnalysis::Info& side_effect_analysis) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_9(mht_9_v, 574, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "ConvertControlToDataOutputs");

  if (while_callers.empty()) return;

  // Collect access information for each resource in the while body that needs
  // to be chained, along with equivalence classes (resources in one class will
  // use the same chain).
  ResourceToOpsMapTy chain_resource_to_ops_map;
  llvm::EquivalenceClasses<ResourceId> resource_equivalence_classes;
  CollectChainResources(while_body, chain_resource_to_ops_map,
                        resource_equivalence_classes, side_effect_analysis);

  // Check for presence of unknown side-effecting ops within the while loop
  // body. These ops act as barriers and the optimization would not yield much
  // inter iteration parallelism for this while loop body. So return with
  // warning.
  if (chain_resource_to_ops_map.count(kUnknownResourceId) > 0) {
    std::set<std::string> blocking_ops;
    for (Operation* op : chain_resource_to_ops_map[kUnknownResourceId]) {
      std::string op_name = op->getName().getStringRef().str();
      if (blocking_ops.insert(op_name).second) {
        LOG(INFO) << "[`tf-executor-convert-control-to-data-outputs` disabled] "
                     "Op type '"
                  << op_name
                  << "' has unknown side effects and blocks inter iteration "
                     "parallelism for the while loop. Consider modeling side "
                     "effects of this op.";
      }
    }
    return;
  }

  // First remove all control outputs of while loop body.
  bool changed = RemoveAllControlOutputs(while_body);

  // If there was no control output to be removed, return early.
  if (!changed) return;

  int num_chains = resource_equivalence_classes.getNumClasses();
  RankedTensorType chaining_data_type =
      RankedTensorType::get({}, OpBuilder(while_body).getI32Type());
  // Create new while body
  int num_old_outputs = while_body.getNumResults();
  AppendFunctionArguments(while_body, num_chains, chaining_data_type);
  AppendFunctionResults(while_body, num_chains, chaining_data_type);

  // Insert identity ops with control dep
  ChainResourceOps(while_body, chain_resource_to_ops_map,
                   resource_equivalence_classes, num_old_outputs);
  // Modify all the while ops referencing the body function and the
  // corresponding while condition functions. Note that each while condition
  // needs to be modified only once.
  OperationSetTy visited;
  for (TF::WhileOp while_op : while_callers) {
    // If the while callers are modified as part of the optimization, then the
    // side effect analysis of their parent functions are invalidated. They
    // need to be recomputed.
    recompute_analysis_for_funcs.insert(while_op->getParentOfType<FuncOp>());
    FuncOp while_cond = while_op.cond_function();
    // Rewrite while op with extra chaining arguments and results.
    while_op = RewriteWhileOp(while_op, num_chains, chaining_data_type);
    bool first_visit = visited.insert(while_cond).second;
    if (!first_visit) continue;
    // Modify while condition function with extra chaining arguments.
    AppendFunctionArguments(while_cond, num_chains, chaining_data_type);
  }
}

void ConvertControlToDataOutputsPass::runOnOperation() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSconvert_control_to_data_outputsDTcc mht_10(mht_10_v, 644, "", "./tensorflow/compiler/mlir/tensorflow/transforms/convert_control_to_data_outputs.cc", "ConvertControlToDataOutputsPass::runOnOperation");

  ModuleOp module = getOperation();
  // This pass assumes that all functions are suitable for export i.e., each
  // function has a single tf_executor.graph op and all islands wrap the
  // internal op perfectly. Verify that in the beginning once.
  if (failed(tensorflow::VerifyExportSuitable(module))) {
    signalPassFailure();
    return;
  }
  TF::SideEffectAnalysis side_effect_analysis(module);

  SymbolTableCollection table;
  SymbolUserMap symbol_map(table, module);
  llvm::SmallDenseMap<FuncOp, SmallVector<TF::WhileOp>>
      while_body_func_to_while_ops;

  // Get all the while body functions and the corresponding while ops first
  // because the symbol user map is invalidated once we start deleting while
  // ops.
  for (auto func : module.getOps<FuncOp>()) {
    if (func.isExternal()) continue;
    SmallVector<TF::WhileOp> while_callers = GetWhileCallers(func, symbol_map);
    if (while_callers.empty()) continue;
    while_body_func_to_while_ops[func] = while_callers;
  }
  // Keep track of functions whose side effect analysis is invalidated because
  // of modifications to that function.
  OperationSetTy recompute_analysis_for_funcs;

  for (auto& entry : while_body_func_to_while_ops) {
    FuncOp while_body = entry.getFirst();
    SmallVector<TF::WhileOp>& while_callers = entry.getSecond();
    if (recompute_analysis_for_funcs.contains(while_body)) {
      // TODO(b/202540801): Recomputing side effect analysis for the entire
      // module is wasteful. It would be better to just recompute analysis for
      // specific functions but the current side effect analysis interface
      // does not allow that.
      side_effect_analysis = TF::SideEffectAnalysis(module);
    }
    ConvertControlToDataOutputs(
        while_body, while_callers, recompute_analysis_for_funcs,
        side_effect_analysis.GetAnalysisForFunc(while_body));
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorConvertControlToDataOutputsPass() {
  return std::make_unique<ConvertControlToDataOutputsPass>();
}

}  // namespace tf_executor
}  // namespace mlir
