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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc() {
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

#include "mlir-hlo/Analysis/userange_analysis.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace {

class CopyRemoval : bufferization::BufferPlacementTransformationBase {
 public:
  explicit CopyRemoval(Operation *op)
      : BufferPlacementTransformationBase(op),
        userange_(op, allocs, aliases),
        dominators_(op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/copy_removal.cc", "CopyRemoval");
}

  void removeCopy() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc mht_1(mht_1_v, 206, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/copy_removal.cc", "removeCopy");

    // A set with the copy Operations to process.
    llvm::SetVector<Operation *> toProcess;
    fillProcessSet(toProcess);

    DenseMap<Value, UseInterval::Vector> updatedUserange;
    DenseMap<Value, UserangeAnalysis::UsePositionList> updatedUsepositions;

    // Lambda expression to update the userange interval.
    auto lambdaUserangeUpdate = [&](Value v,
                                    DenseMap<Value, UseInterval::Vector> &map)
        -> UseInterval::Vector & { return insertUserangeInterval(v, map); };
    // Lambda expression to update the use-position.
    auto lambdaUsePosUpdate =
        [&](Value v, DenseMap<Value, UserangeAnalysis::UsePositionList> &map)
        -> UserangeAnalysis::UsePositionList & {
      return insertUserangePositions(v, map);
    };

    // A set containing copy operations that can be erased.
    SmallPtrSet<Operation *, 16> toErase;
    while (!toProcess.empty()) {
      Operation *currentOp = toProcess.pop_back_val();

      // Cast the Operation and get the Source and Target.
      auto copyOpInterface = dyn_cast<CopyOpInterface>(currentOp);
      Value copySource = copyOpInterface.getSource();
      Value copyTarget = copyOpInterface.getTarget();

      // Get the UserangeIntervals.
      UseInterval::Vector sourceInterval =
          getOrInsert(copySource, updatedUserange, lambdaUserangeUpdate);
      UseInterval::Vector targetInterval =
          getOrInsert(copyTarget, updatedUserange, lambdaUserangeUpdate);

      UseInterval::Vector intersect = sourceInterval;

      // Compute the intersection.
      UseInterval::intervalIntersect(intersect, targetInterval);

      // If the sourceInterval contains more than one UseInterval, there are
      // multiple operations that intersect. The sourceInterval must have at
      // least one UseInterval that contains the copyOp.
      if (intersect.size() != 1) continue;

      // Check if all Operations inside the intersection are part of the copyOp.
      if (!checkAncestor(currentOp, *intersect.begin())) continue;
      UserangeAnalysis::UsePositionList targetUsePosList =
          getOrInsert(copyTarget, updatedUsepositions, lambdaUsePosUpdate);

      // Check if the currentOp dominates all uses of the copyTarget.
      if (!checkDominance(currentOp, targetUsePosList, toErase)) continue;

      // Merge the Useranges.
      UseInterval::intervalMerge(sourceInterval, targetInterval);

      // Merge the UsePositions.
      UserangeAnalysis::UsePositionList sourceUsePosList =
          getOrInsert(copySource, updatedUsepositions, lambdaUsePosUpdate);

      userange_.mergeUsePositions(sourceUsePosList, targetUsePosList);

      // Replace all uses of the target with the source.
      copyTarget.replaceAllUsesWith(copySource);
      toErase.insert(currentOp);
    }
    // Erase the copy operations.
    for (auto *eraseOp : toErase) eraseOp->erase();

    // Erase all allocs without uses.
    for (const bufferization::BufferPlacementAllocs::AllocEntry &entry :
         allocs) {
      Value alloc = std::get<0>(entry);
      if (alloc.use_empty()) alloc.getDefiningOp()->erase();
    }
  }

 private:
  /// Iterate over all allocs and their aliases and add their uses to the
  /// process set that implement a CopyOpInterface, where the alloc or alias is
  /// the source of the CopyOpInterface.
  void fillProcessSet(llvm::SetVector<Operation *> &toProcess) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc mht_2(mht_2_v, 290, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/copy_removal.cc", "fillProcessSet");

    // A Set that contains the already processed aliases.
    SmallPtrSet<Value, 16U> processedAliases;

    // Iterate over the allocs.
    for (const bufferization::BufferPlacementAllocs::AllocEntry &entry :
         allocs) {
      Value allocValue = std::get<0>(entry);

      // Resolve the aliases of the current alloc and iterate over them.
      const ValueSetT &aliasSet = aliases.resolve(allocValue);
      for (Value alias : aliasSet) {
        // If the alias is already processed, continue.
        if (!processedAliases.insert(alias).second) continue;

        // If the value has no uses/ empty userange, continue.
        auto userangeInterval = userange_.getUserangeInterval(alias);
        if (!userangeInterval) continue;

        // Iterate over the UseIntervals and check if the last Operation in the
        // UseInterval implements a CopyOpInterface.
        for (const UseInterval &interval : *userangeInterval.getValue()) {
          Operation *currentLastUse = userange_.getOperation(interval.end);
          auto copyOpInterface = dyn_cast<CopyOpInterface>(currentLastUse);
          if (!copyOpInterface) continue;

          // Check if the source is the alias.
          if (copyOpInterface.getSource() != alias) continue;

          toProcess.insert(currentLastUse);
        }
      }
    }
  }

  /// Find the given Value in the DenseMap and return the pointer. If the given
  /// Value is not in the Map, insert a copy of the given original to the
  /// DenseMap using the pased update function and return a pointer to that
  /// element.
  template <typename T, typename TFunc>
  T &getOrInsert(Value v, DenseMap<Value, T> &updateMap,
                 const TFunc &updateFunc) {
    auto iter = updateMap.find(v);
    if (iter != updateMap.end()) return iter->second;
    return updateFunc(v, updateMap);
  }

  /// Insert the original userange intervals of the operation in the map.
  UseInterval::Vector &insertUserangeInterval(
      Value v, DenseMap<Value, UseInterval::Vector> &updateMap) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc mht_3(mht_3_v, 342, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/copy_removal.cc", "insertUserangeInterval");

    const auto *original = userange_.getUserangeInterval(v).getValue();
    auto &entry = updateMap[v];
    entry = *original;
    return entry;
  }

  /// Insert the original use positions of the operation in the map.
  UserangeAnalysis::UsePositionList &insertUserangePositions(
      Value v, DenseMap<Value, UserangeAnalysis::UsePositionList> &updateMap) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc mht_4(mht_4_v, 354, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/copy_removal.cc", "insertUserangePositions");

    const auto *original = userange_.getUserangePositions(v).getValue();
    auto &entry = updateMap[v];
    entry = *original;
    return entry;
  }

  /// Check if all uses of the target Value are dominated by given Operation.
  /// Note: The target has always at least one use which is the copy operation.
  bool checkDominance(Operation *useOp,
                      const UserangeAnalysis::UsePositionList &usePosList,
                      SmallPtrSet<Operation *, 16> &ignoreSet) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc mht_5(mht_5_v, 368, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/copy_removal.cc", "checkDominance");

    Block *useBlock = useOp->getBlock();
    // Check if any use of the target is not dominated by the useOp. Erased
    // operations are ignored as uses.
    return llvm::all_of(
        usePosList, [=](const UserangeAnalysis::UsePosition usePos) {
          Operation *use = usePos.second;
          return ignoreSet.count(use) ||
                 dominators_.dominates(useBlock, use->getBlock());
        });
  }

  /// Check if the given Operation is an ancestor of the operations inside the
  /// UseInterval.
  bool checkAncestor(Operation *op, UseInterval &interval) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc mht_6(mht_6_v, 385, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/copy_removal.cc", "checkAncestor");

    // Divide the start and end by two to remove read/write properties.
    for (int id = interval.start / 2, e = interval.end / 2; id <= e; ++id) {
      // Get the operation from the id. Multiply the id by 2, because the
      // userange operates on doubled ids. Return false if the operation is not
      // an ancestor.
      if (!op->isAncestor(userange_.getOperation(id * 2))) return false;
    }
    return true;
  }

  /// The current userange info.
  UserangeAnalysis userange_;

  /// The current dominance info.
  DominanceInfo dominators_;
};

struct CopyRemovalPass : public CopyRemovalBase<CopyRemovalPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPScopy_removalDTcc mht_7(mht_7_v, 407, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/copy_removal.cc", "runOnOperation");

    Operation *funcOp = getOperation();
    CopyRemoval removal(funcOp);
    removal.removeCopy();
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createCopyRemovalPass() {
  return std::make_unique<CopyRemovalPass>();
}

}  // namespace mlir
