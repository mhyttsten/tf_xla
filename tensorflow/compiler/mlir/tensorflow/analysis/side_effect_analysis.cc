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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"

#include <bitset>
#include <string>

#include "absl/container/node_hash_map.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace {

constexpr ResourceId kUnknownResourceId =
    ResourceAliasAnalysis::Info::kUnknownResourceId;
static_assert(kUnknownResourceId < 0, "kUnknownResourceId must be < 0");

// A collection of Resource IDs. Note that `kUnknownResourceId` is smaller than
// all other resource IDs which are nonnegative (see check above) so it will
// always be the first element of a `ResourceIdSet` (we make use of this).
using ResourceIdSet = llvm::SmallSet<ResourceId, 8>;

// Note that we cannot simply define a `static const llvm::SmallSet` here
// because of missing `initializer_list` support for `llvm::SmallSet`.
const ResourceIdSet& UnknownResourceSet() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_0(mht_0_v, 229, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "UnknownResourceSet");

  // clang-format off
  static auto* id_set = new ResourceIdSet();
  id_set->insert(kUnknownResourceId);
  return *id_set;
}

// Helper function to avoid frequent checks for unknown IDs.
const ResourceIdSet& GetResourceUniqueIdsOrUnknown(
    Value value,
    const ResourceAliasAnalysis::Info& alias_analysis) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_1(mht_1_v, 242, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "GetResourceUniqueIdsOrUnknown");

  if (!getElementTypeOrSelf(value.getType()).isa<TF::ResourceType>() ||
      alias_analysis.IsUnknownResource(value)) return UnknownResourceSet();
  return alias_analysis.GetResourceUniqueIds(value);
}

// Helper class for a collection of side effects for one resource.
class SideEffects {
  enum Type {
    kAlloc = 0,
    kFree = 1,
    kRead = 2,
    kWrite = 3
  };

 public:
  bool IsAlloc() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_2(mht_2_v, 261, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "IsAlloc");
 return effects_.test(kAlloc); }
  bool IsFree() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_3(mht_3_v, 265, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "IsFree");
 return effects_.test(kFree); }
  bool IsRead() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_4(mht_4_v, 269, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "IsRead");
 return effects_.test(kRead); }
  bool IsWrite() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_5(mht_5_v, 273, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "IsWrite");
 return effects_.test(kWrite); }
  bool IsAllocOnly() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_6(mht_6_v, 277, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "IsAllocOnly");
 return IsAlloc() && effects_.count() == 1; }
  bool IsReadOnly() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_7(mht_7_v, 281, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "IsReadOnly");
 return IsRead() && effects_.count() == 1; }
  ResourceId GetResourceId() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_8(mht_8_v, 285, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "GetResourceId");
 return resource_id_; }

  void SetAlloc() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_9(mht_9_v, 290, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SetAlloc");
 effects_.set(kAlloc); }
  void SetFree() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_10(mht_10_v, 294, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SetFree");
 effects_.set(kFree); }
  void SetRead() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_11(mht_11_v, 298, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SetRead");
 effects_.set(kRead); }
  void SetWrite() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_12(mht_12_v, 302, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SetWrite");
 effects_.set(kWrite); }
  void SetUnknownEffect() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_13(mht_13_v, 306, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SetUnknownEffect");
 effects_.set(); }
  void SetResourceId(ResourceId resource_id) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_14(mht_14_v, 310, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SetResourceId");
 resource_id_ = resource_id; }
  void AddEffects(const SideEffects& other_effects) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_15(mht_15_v, 314, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "AddEffects");

    effects_ |= other_effects.effects_;
  }

 private:
  std::bitset<4> effects_ = 0;
  ResourceId resource_id_ = kUnknownResourceId;
};

// We use `std::map` here because we rely on the order of elements.
using SideEffectsByResourceId = std::map<ResourceId, SideEffects>;

// We use `std::unordered_map` here for pointer stability reasons.
// Note: If memory usage ever becomes a bottleneck here (not expected) we could
// use a Trie-like data structure to avoid storing side effects in both parent
// op and all its child ops (recursively), at the expense of lookup time.
using OpSideEffectMap = std::unordered_map<Operation*, SideEffectsByResourceId>;

// Update `side_effects_by_resource_id` with `side_effects`.
void UpdateSideEffectsByResourceId(
    const SideEffects& side_effects,
    SideEffectsByResourceId& side_effects_by_resource_id) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_16(mht_16_v, 338, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "UpdateSideEffectsByResourceId");

  ResourceId id = side_effects.GetResourceId();
  auto iter = side_effects_by_resource_id.find(id);
  if (iter == side_effects_by_resource_id.end()) {
    side_effects_by_resource_id[id] = side_effects;
  } else {
    iter->second.AddEffects(side_effects);
  }
}

bool MayHaveSideEffect(Operation* op) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_17(mht_17_v, 351, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "MayHaveSideEffect");

  if (isa_and_nonnull<TF::TensorFlowDialect>(op->getDialect()))
    return TensorFlowDialect::CanHaveSideEffects(op);

  if (mlir::MemoryEffectOpInterface::hasNoEffect(op)) return false;
  // Conservatively assume that there can be side effects.
  return true;
}

bool ShouldUseResourceAliasAnalysis(
    const MemoryEffects::EffectInstance& effect) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_18(mht_18_v, 364, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "ShouldUseResourceAliasAnalysis");

  Value value = effect.getValue();
  if (value && getElementTypeOrSelf(value.getType()).isa<ResourceType>()) {
    // For value-based effects on resource values we can use resource alias
    // analysis.
    return true;
  }
  // For all other effects don't rely on resource alias analysis. Note that
  // non-resource values are not processed in resource alias analysis.
  return false;
}

//===----------------------------------------------------------------------===//
// SideEffectAnalysisInfo helper functions.
//===----------------------------------------------------------------------===//

SideEffects GetSideEffectsFromEffectInstance(
    const MemoryEffects::EffectInstance& effect_instance, Operation* op) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_19(mht_19_v, 384, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "GetSideEffectsFromEffectInstance");

  mlir::SideEffects::Effect* effect = effect_instance.getEffect();
  SideEffects side_effects;
  if (isa<MemoryEffects::Allocate>(effect)) {
    side_effects.SetAlloc();
  } else if (isa<MemoryEffects::Free>(effect)) {
    side_effects.SetFree();
  } else if (isa<MemoryEffects::Read>(effect)) {
    side_effects.SetRead();
  } else if (isa<MemoryEffects::Write>(effect)) {
    side_effects.SetWrite();
  } else {
    LOG(WARNING) << "Unsupported effect for op "
                 << op->getName().getStringRef().str();
    side_effects.SetUnknownEffect();
  }
  return side_effects;
}

// Collects all op-based and value-based side effects for `op` per resource ID.
SideEffectsByResourceId CollectSideEffectsByResourceId(
    Operation* op,
    const SideEffectsByResourceId& op_side_effects,
    const TF::ResourceAliasAnalysis::Info& alias_analysis) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_20(mht_20_v, 410, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "CollectSideEffectsByResourceId");

  SideEffectsByResourceId side_effects_by_resource_id;
  if (!MayHaveSideEffect(op)) return side_effects_by_resource_id;

  // Copy op-based side effects.
  bool found_any_effect = !op_side_effects.empty();
  side_effects_by_resource_id = op_side_effects;

  // Collect value-based side effects from op interface.
  llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (interface) interface.getEffects(effects);

  llvm::SmallDenseSet<Value, 8> processed_values;
  for (const auto& effect : effects) {
    Value value = effect.getValue();
    found_any_effect = true;
    if (value) processed_values.insert(value);

    // We only collect value-based side effects here for which we can use
    // resource alias analysis. Other side effects are treated as op-based
    // side effects.
    if (!ShouldUseResourceAliasAnalysis(effect)) continue;

    // Add side effects for every potentially accessed resource ID.
    SideEffects side_effects(GetSideEffectsFromEffectInstance(effect, op));
    const auto& ids = GetResourceUniqueIdsOrUnknown(value, alias_analysis);
    for (ResourceId id : ids) {
      side_effects.SetResourceId(id);
      UpdateSideEffectsByResourceId(side_effects, side_effects_by_resource_id);
    }
  }

  auto add_remaining_effects = [&](auto resource_values) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_21(mht_21_v, 446, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "lambda");

    for (Value resource_value : resource_values) {
      // If we already processed this value before, skip it.
      if (processed_values.count(resource_value) > 0) continue;
      found_any_effect = true;

      // Conservatively set unknown effect.
      SideEffects unknown_effect;
      unknown_effect.SetUnknownEffect();

      // Add side effects for every potentially accessed resource ID.
      const auto& ids =
          GetResourceUniqueIdsOrUnknown(resource_value, alias_analysis);
      for (ResourceId id : ids) {
        unknown_effect.SetResourceId(id);
        UpdateSideEffectsByResourceId(unknown_effect,
                                      side_effects_by_resource_id);
      }
    }
  };
  // Add value-based side effects for resource values which are not covered by
  // any side effect so far, for example, resource values being passed to
  // `tf.While` or `tf.If` ops which are not part of the op definition but
  // appear in a variadic input list.
  add_remaining_effects(filter_resources(op->getOperands()));
  add_remaining_effects(filter_resources(op->getResults()));

  if (!found_any_effect) {
    // We haven't collected any side effect but the op is potentially
    // side-effecting (otherwise we would have returned), therefore we have an
    // unknown side effect for an unknown resource.
    SideEffects unknown_effect;
    unknown_effect.SetUnknownEffect();
    unknown_effect.SetResourceId(kUnknownResourceId);
    UpdateSideEffectsByResourceId(unknown_effect,
                                  side_effects_by_resource_id);
  }
  return side_effects_by_resource_id;
}

}  // namespace

namespace detail {

// Class for propagating op-based side effects bottom-up and collecting them
// per op, by resource ID.
class OpSideEffectCollector {
 public:
  // Recursively collects op-based side effects for all ops in module and
  // populates `op_side_effect_map_`.
  explicit OpSideEffectCollector(ModuleOp module) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_22(mht_22_v, 499, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "OpSideEffectCollector");

    symbol_table_collection_.getSymbolTable(module);
    for (auto func : module.getOps<FuncOp>()) {
      CollectOpSideEffects(func);
    }
  }

  // Returns op-based side effects by resource ID for `op`.
  const SideEffectsByResourceId& GetSideEffectsForOp(Operation* op) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_23(mht_23_v, 510, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "GetSideEffectsForOp");

    auto iter = op_side_effect_map_.find(op);
    if (iter != op_side_effect_map_.end()) return iter->second;
    return empty_side_effects_map_;
  }

 private:
  // Adds op-based side effects from all ops in `region` to `op` side effects.
  // Collects side effects for ops that weren't visited before.
  void AddRegionSideEffectsForOp(Region& region, Operation* op) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_24(mht_24_v, 522, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "AddRegionSideEffectsForOp");

    for (Block& block : region) {
      for (Operation& curr_op : block) {
        if (op_side_effect_map_.count(&curr_op) == 0) {
          CollectOpSideEffects(&curr_op);
        }
        for (const auto& entry : op_side_effect_map_[&curr_op]) {
          UpdateSideEffectsByResourceId(entry.second, op_side_effect_map_[op]);
        }
      }
    }
  }

  // Collects op-based side effects for `op` in `op_side_effect_map_[op]`.
  void CollectOpSideEffects(Operation* op) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_25(mht_25_v, 539, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "CollectOpSideEffects");

    if (!MayHaveSideEffect(op)) return;
    // Skip following ops to avoid that every island, graph and function is
    // classified as unknown side-effecting.
    if (isa<tf_executor::YieldOp, tf_executor::FetchOp, mlir::func::ReturnOp>(op))
      return;

    // Propagate side effects from regions or functions attached to `op` for
    // some special cases.
    if (auto func = llvm::dyn_cast<FuncOp>(op)) {
      AddRegionSideEffectsForOp(func.getBody(), op);
    } else if (auto call = llvm::dyn_cast<CallOpInterface>(op)) {
      FuncOp func_op =
          dyn_cast<FuncOp>(call.resolveCallable(&symbol_table_collection_));
      if (func_op) {
        AddRegionSideEffectsForOp(func_op.getBody(), op);
      }
    } else if (auto if_op = llvm::dyn_cast<IfOp>(op)) {
      AddRegionSideEffectsForOp(if_op.then_function().getBody(), op);
      AddRegionSideEffectsForOp(if_op.else_function().getBody(), op);
    } else if (auto while_op = dyn_cast<WhileOp>(op)) {
      AddRegionSideEffectsForOp(while_op.body_function().getBody(), op);
    } else if (auto while_region_op = dyn_cast<WhileRegionOp>(op)) {
      AddRegionSideEffectsForOp(while_region_op.body(), op);
    } else if (auto case_op = dyn_cast<CaseOp>(op)) {
      llvm::SmallVector<FuncOp, 4> branch_funcs;
      case_op.get_branch_functions(branch_funcs);
      for (auto branch_func : branch_funcs) {
        AddRegionSideEffectsForOp(branch_func.getBody(), op);
      }
    } else if (isa<tf_device::LaunchOp, tf_device::ClusterOp,
                   tf_executor::IslandOp, tf_executor::GraphOp, IfRegionOp,
                   CaseRegionOp>(op)) {
      for (Region& region : op->getRegions()) {
        AddRegionSideEffectsForOp(region, op);
      }
    } else {
      // Now handle all other ops.
      auto& side_effects_by_resource_id = op_side_effect_map_[op];
      llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
      auto interface = dyn_cast<MemoryEffectOpInterface>(op);
      if (interface) interface.getEffects(effects);
      if (effects.empty()) {
        // The op is potentially side-effecting and doesn't have any effect
        // assigned, treat it as unknown side effect.
        SideEffects side_effects;
        side_effects.SetResourceId(kUnknownResourceId);
        side_effects.SetUnknownEffect();
        UpdateSideEffectsByResourceId(side_effects,
                                      side_effects_by_resource_id);
        // An unknown side effect dominates other side effects so we don't have
        // to add them and can return here.
        return;
      }
      // Add op-based side effects from regions (if any).
      for (Region& region : op->getRegions()) {
        AddRegionSideEffectsForOp(region, op);
      }
      // Add op-based side effects for the op itself.
      for (const auto& effect : effects) {
        // We handle value-based side effects for which we can use resource
        // alias analysis at a different place, skip here.
        if (ShouldUseResourceAliasAnalysis(effect)) continue;
        if (llvm::isa<ResourceEffects::MustExecute>(effect.getResource()))
          // We have this fake resource to avoid that certain ops are considered
          // dead or get pruned, ignore it for side effect analysis.
          continue;

        // Add side effects for op resource ID.
        std::string instance_str = "";
        SideEffects side_effects(GetSideEffectsFromEffectInstance(effect, op));
        if (auto resource_instance_op =
            dyn_cast<GetResourceInstanceInterface>(op)) {
          instance_str = resource_instance_op.GetResourceInstanceStr();
        }
        ResourceId resource_id = GetOpResourceId(
            effect.getResource()->getResourceID(), instance_str);
        side_effects.SetResourceId(resource_id);
        UpdateSideEffectsByResourceId(side_effects,
                                      side_effects_by_resource_id);
      }
    }
  }

  // Get internal op resource ID from MLIR type ID and instance ID.
  ResourceId GetOpResourceId(TypeID type_id, std::string instance_str) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("instance_str: \"" + instance_str + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_26(mht_26_v, 628, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "GetOpResourceId");

    auto emplace_result = type_instance_str_to_op_resource_id_.try_emplace(
        std::make_pair(type_id.getAsOpaquePointer(), instance_str),
        next_op_resource_id_);
    // Increment type ID if we have encountered a new resource type.
    if (emplace_result.second) ++next_op_resource_id_;
    return emplace_result.first->second;
  }

  // We use [0, kMaxResourceId] for resource IDs returned by resource alias
  // analysis and [kMaxResourceId + 1, ...] for resource IDs which we generate
  // for op-based side effects.
  const ResourceId kMaxResourceId =
      std::numeric_limits<ResourceId>::max() / 2;
  // Next available ID for op-based resources (resources not handled by resource
  // alias analysis).
  ResourceId next_op_resource_id_ = kMaxResourceId + 1;
  // Maps (type ID, instance ID) pairs to internal IDs for op-based resources.
  // Also see comment above. Instead of using TypeID directly we use its opaque
  // pointer.
  absl::node_hash_map<std::pair<const void*, std::string>, ResourceId>
    type_instance_str_to_op_resource_id_;
  // Used for faster callable resolution.
  SymbolTableCollection symbol_table_collection_;
  // Collect all op-based side effects here.
  OpSideEffectMap op_side_effect_map_;
  const SideEffectsByResourceId empty_side_effects_map_;
};


//===----------------------------------------------------------------------===//
// SideEffectAnalysisInfo
//===----------------------------------------------------------------------===//

void SideEffectAnalysisInfo::AddPredecessorsForAccess(ResourceId resource_id,
                                                      Operation* op,
                                                      bool read_only) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_27(mht_27_v, 667, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SideEffectAnalysisInfo::AddPredecessorsForAccess");

  VLOG(2) << "    Adding predecessors for resource " << resource_id;
  auto it = per_resource_access_info_.find(resource_id);
  if (it == per_resource_access_info_.end()) return;
  const auto& access_info = it->getSecond();

  auto& control_predecessors = control_predecessors_[op];
  bool is_last_write_indirectly_tracked = false;
  if (!read_only) {
    // Add reads after last write as predecessors.
    control_predecessors.insert(access_info.reads_since_last_write.begin(),
                                access_info.reads_since_last_write.end());
    // Last write is indirectly tracked by any read predecessor we added.
    is_last_write_indirectly_tracked =
        !access_info.reads_since_last_write.empty();
  }
  if (access_info.last_write && !is_last_write_indirectly_tracked) {
    // Add last write as predecessor.
    control_predecessors.insert(access_info.last_write);
  }
}

void SideEffectAnalysisInfo::UpdateAccess(ResourceId resource_id,
                                          Operation* op,
                                          bool read_only) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_28(mht_28_v, 694, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SideEffectAnalysisInfo::UpdateAccess");

  VLOG(2) << "    Updating access for resource " << resource_id;
  op_to_resource_ids_[op].push_back({resource_id, read_only});
  if (resource_id == kUnknownResourceId) {
    if (read_only) {
      // New unknown read is not tracked by any known resource access.
      for (auto& entry : per_resource_access_info_) {
        entry.getSecond().are_last_unknown_reads_tracked = false;
      }
    } else {
      // Unknown write can clear all other tracked information, since it acts
      // like a barrier.
      per_resource_access_info_.clear();
    }
  }
  auto& access_info = per_resource_access_info_[resource_id];
  if (read_only) {
    access_info.reads_since_last_write.push_back(op);
    // Last unknown write is indirectly tracked by this read (we have added the
    // write as a predecessor for `op` before).
    access_info.is_last_unknown_write_tracked = true;
  } else {
    access_info.last_write = op;
    access_info.reads_since_last_write.clear();
    // Last unknown read(s) and write are indirectly tracked by this write (we
    // have added the read(s) and write as predecessors for `op` before).
    access_info.are_last_unknown_reads_tracked = true;
    access_info.is_last_unknown_write_tracked = true;
    access_info.is_last_unknown_write_tracked_by_write = true;
  }
}

void SideEffectAnalysisInfo::AnalyzeFunction(FuncOp func_op) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_29(mht_29_v, 729, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SideEffectAnalysisInfo::AnalyzeFunction");

  // AnalyzeRegion() recursively analyzes the function body, and only populates
  // control_predecessors_.
  AnalyzeRegion(&func_op.getBody());
  // Populate sorted_control_predecessors_ and sorted_control_successors_ based
  // on control_predecessors.
  for (auto& entry : control_predecessors_) {
    auto op = entry.getFirst();
    auto& predecessors = entry.getSecond();
    auto& sorted_predecessors = sorted_control_predecessors_[op];
    for (Operation* predecessor : predecessors) {
      sorted_predecessors.push_back(predecessor);
      sorted_control_successors_[predecessor].push_back(op);
    }
  }
  control_predecessors_.clear();
  for (auto& entry : sorted_control_predecessors_) {
    llvm::sort(entry.getSecond(), [](Operation* a, Operation* b) {
      return a->isBeforeInBlock(b);
    });
  }
  for (auto& entry : sorted_control_successors_) {
    llvm::sort(entry.getSecond(), [](Operation* a, Operation* b) {
      return a->isBeforeInBlock(b);
    });
  }

  // Populate the control sinks (i.e. side-effecting ops with no control
  // successors) in the top level block.
  for (const auto& entry : sorted_control_predecessors_) {
    auto* op = entry.getFirst();
    if (op->getBlock() == &func_op.front() &&
        sorted_control_successors_.count(op) == 0) {
      sorted_control_sinks_.push_back(op);
    }
  }
  llvm::sort(sorted_control_sinks_, [](Operation* a, Operation* b) {
    return a->isBeforeInBlock(b);
  });
}

void SideEffectAnalysisInfo::AnalyzeRegion(Region* region) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_30(mht_30_v, 773, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SideEffectAnalysisInfo::AnalyzeRegion");

  // We explicitly iterate through the regions and blocks in order to handle
  // different nested regions separately.
  for (Block& block : *region) {
    for (Operation& op : block) {
      for (Region& child_region : op.getRegions()) {
        SideEffectAnalysisInfo child_analysis(
            &child_region, op_side_effect_collector_, alias_analysis_);
        // Move data from `child_analysis` to current region.
        for (auto& entry : child_analysis.control_predecessors_)
          control_predecessors_[entry.first] = std::move(entry.second);
        for (auto& entry : child_analysis.op_to_resource_ids_)
          op_to_resource_ids_[entry.first] = std::move(entry.second);
      }
      AnalyzeOp(&op);
    }
  }
}

void SideEffectAnalysisInfo::AnalyzeOp(Operation* op) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_31(mht_31_v, 795, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SideEffectAnalysisInfo::AnalyzeOp");

  VLOG(2) << "Processing op " << mlir::debugString(*op);
  SideEffectsByResourceId side_effects_by_resource_id =
      CollectSideEffectsByResourceId(
          op,
          op_side_effect_collector_.GetSideEffectsForOp(op),
          alias_analysis_);

  // If the side-effecting op is a control source (i.e. it has no control
  // predecessors), then `control_predecessors_` won't be updated below.
  // However, we still want to track this op as it may have side effects visible
  // to ops outside the function.
  if (!side_effects_by_resource_id.empty()) control_predecessors_[op];

  // Traverse all resource IDs and their associated side effects.
  bool had_unknown_resource_read = false;
  for (auto pair : side_effects_by_resource_id) {
    ResourceId resource_id = pair.first;
    const SideEffects& side_effects = pair.second;
    const bool read_only = side_effects.IsReadOnly();
    VLOG(2) << "  Processing resource ID: " << resource_id
            << ", read-only effect: " << read_only;
    // An op that only allocates a resource is expected to return a handle that
    // is used by all other accesses of the same resource. That means, other ops
    // that access the same resource already have a data dependency on the
    // allocating op so it doesn't need any control predecessors or successors.
    if (side_effects.IsAllocOnly()) continue;
    // Effect is dominated by previous unknown resource read effect.
    if (read_only && had_unknown_resource_read) continue;

    // We collect all conflicting IDs except unknown resource ID which is
    // handled later.
    ResourceIdSet conflicting_ids;
    bool is_unknown_access_indirectly_tracked = false;
    if (resource_id == kUnknownResourceId) {
      for (auto& entry : per_resource_access_info_) {
        ResourceId other_id = entry.getFirst();
        if (other_id != kUnknownResourceId) conflicting_ids.insert(other_id);
      }
    } else {
      conflicting_ids.insert(resource_id);
    }
    // Add predecessors for conflicting IDs.
    for (ResourceId id : conflicting_ids) {
      AddPredecessorsForAccess(id, op, read_only);
      is_unknown_access_indirectly_tracked |=
          IsUnknownAccessIndirectlyTrackedByResource(id, read_only);
    }
    // Add predecessors for unknown resource if not already tracked.
    if (!is_unknown_access_indirectly_tracked)
      AddPredecessorsForAccess(kUnknownResourceId, op, read_only);
    // Update resource access.
    UpdateAccess(resource_id, op, read_only);

    // If this effect dominates all other possible effects, return here. Note
    // that if there is any effect for an unknown resource, then we encounter it
    // in the first iteration since `kUnknownResourceId` is smaller than all
    // other resource IDs.
    if (resource_id == kUnknownResourceId && !read_only) return;
    if (resource_id == kUnknownResourceId && read_only) {
      had_unknown_resource_read = true;
    }
  }
}

bool SideEffectAnalysisInfo::IsUnknownAccessIndirectlyTrackedByResource(
    ResourceId resource_id, bool read_only) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_32(mht_32_v, 864, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SideEffectAnalysisInfo::IsUnknownAccessIndirectlyTrackedByResource");

  auto it = per_resource_access_info_.find(resource_id);
  if (it == per_resource_access_info_.end()) return false;
  auto access_info = it->getSecond();

  auto unknown_it = per_resource_access_info_.find(kUnknownResourceId);
  if (unknown_it == per_resource_access_info_.end()) return true;
  auto unknown_access_info = unknown_it->getSecond();

  bool no_unknown_read = unknown_access_info.reads_since_last_write.empty();
  bool no_unknown_write = (unknown_access_info.last_write == nullptr);

  // For the read-only case we only need that the last unknown write is already
  // tracked by the last `resource` write since we don't have dependencies to
  // any other read accesses.
  // Otherwise, we need that the last unknown read(s) and write are already
  // tracked by any read or write accesses of `resource`.
  bool is_tracked = read_only ?
      no_unknown_write || access_info.is_last_unknown_write_tracked_by_write :
      (no_unknown_write || access_info.is_last_unknown_write_tracked) &&
      (no_unknown_read || access_info.are_last_unknown_reads_tracked);
  if (is_tracked) {
    VLOG(2) << "      Unknown access indirectly tracked by resource "
            << resource_id;
  }
  return is_tracked;
}

llvm::SmallVector<Operation*, 4>
SideEffectAnalysisInfo::DirectControlPredecessors(
    Operation* op, llvm::function_ref<bool(Operation*)> filter) const {
  llvm::SmallVector<Operation*, 4> result;
  auto it = sorted_control_predecessors_.find(op);
  if (it == sorted_control_predecessors_.end()) return result;
  result.reserve(it->getSecond().size());
  for (auto predecessor : it->getSecond()) {
    if (!filter || filter(predecessor)) result.push_back(predecessor);
  }
  return result;
}

llvm::SmallVector<Operation*, 4>
SideEffectAnalysisInfo::DirectControlSuccessors(
    Operation* op, llvm::function_ref<bool(Operation*)> filter) const {
  llvm::SmallVector<Operation*, 4> result;
  auto it = sorted_control_successors_.find(op);
  if (it == sorted_control_successors_.end()) return result;
  result.reserve(it->getSecond().size());
  for (auto successor : it->getSecond()) {
    if (!filter || filter(successor)) result.push_back(successor);
  }
  return result;
}

const llvm::SmallVector<std::pair<ResourceId, bool>>&
SideEffectAnalysisInfo::GetResourceIds(Operation* op) const {
  auto it = op_to_resource_ids_.find(op);
  if (it == op_to_resource_ids_.end()) return empty_resource_ids_;
  return it->getSecond();
}

}  // namespace detail

SideEffectAnalysis::SideEffectAnalysis(ModuleOp module)
    : alias_analysis_(module) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSside_effect_analysisDTcc mht_33(mht_33_v, 931, "", "./tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.cc", "SideEffectAnalysis::SideEffectAnalysis");

  // Analyze entire module for alias analysis info.
  detail::OpSideEffectCollector op_side_effect_collector(module);

  // Analyze all functions.
  for (auto func : module.getOps<FuncOp>())
    this->info_map_.try_emplace(func, func,
                                op_side_effect_collector,
                                alias_analysis_.GetAnalysisForFunc(func));
}

}  // namespace TF
}  // namespace mlir
