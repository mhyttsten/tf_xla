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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_ALIAS_ANALYSIS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_ALIAS_ANALYSIS_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_alias_analysisDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_alias_analysisDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_alias_analysisDTh() {
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


#include <cstddef>
#include <cstdint>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/per_function_aggregate_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {
namespace detail {
class BacktrackAnalysis;
class BacktrackAnalysisInfo;

// Resource alias analysis information for a single function.
class ResourceAliasAnalysisInfo {
 public:
  // Constructs analysis info by analyzing the given function.
  ResourceAliasAnalysisInfo(FuncOp func,
                            const BacktrackAnalysis& backtrack_analysis,
                            SymbolTableCollection& symbol_table_collection);

  ResourceAliasAnalysisInfo(ResourceAliasAnalysisInfo&&) = default;

  // Returns if the analysis fails to resolve a resource-type value.
  bool IsUnknownResource(Value resource) const;

  // Returns the set of unique IDs which `resource` could alias. Requires that
  // IsUnknownResource(resource) == false.
  const llvm::SmallSet<int64_t, 8>& GetResourceUniqueIds(Value resource) const;

  // Returns the set of values that are potentially aliases of `value`. Requires
  // `IsUnknownResource(resource) == false`.
  llvm::SmallSetVector<Value, 8> GetResourceAliases(Value resource) const;

  // Returns true iff given resource is allocated by op with
  // `UniqueResourceAllocation` trait. This can be utilized for while-loop
  // parallelization.
  bool IsUniqueResourceAllocationId(int64_t resource_id) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_alias_analysisDTh mht_0(mht_0_v, 234, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h", "IsUniqueResourceAllocationId");

    return unique_resource_allocation_ids_.contains(resource_id);
  }

 private:
  // Maps resource value to unique ID and vice-versa. Returns true if the
  // mapping has changed.
  bool AddValueUniqueIDMapping(Value value, int64_t id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSanalysisPSresource_alias_analysisDTh mht_1(mht_1_v, 244, "", "./tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h", "AddValueUniqueIDMapping");

    resource_value_to_ids_[value].insert(id);
    return id_to_resource_values_[id].insert(value);
  }

  // Returns the set unique Values which map to `id`.
  const llvm::SmallSetVector<Value, 8>& GetUniqueIdResources(int64_t id) const;

  // Propagates the resource IDs from an input operand to a result. Returns
  // true of the mapping has changed.
  bool PropagateInputToOutput(const Value& operand, const OpResult& result);

  // Analyzes while loops to compute resource IDs for the loop results.
  // `body_info` is the backtrack analysis info for the loop body.
  void AnalyzeWhileLoop(Operation* while_op,
                        const BacktrackAnalysisInfo& body_info);

  // Analyzes tf.Case/tf.If ops to compute resource IDs.
  template <class CaseOrIfOp>
  void AnalyzeFunctionalCaseOrIfOp(CaseOrIfOp case_or_if_op,
                                   llvm::ArrayRef<FuncOp> functions,
                                   const BacktrackAnalysis& backtrack_analysis);

  // Analyzes tf.CaseRegion/tf.IfRegion ops to compute resource IDs.
  void AnalyzeRegionCaseOrIfOp(Operation* case_or_if_op,
                               const BacktrackAnalysis& backtrack_analysis);

  // Maps each resource-type value to a set of unique IDs that it could alias.
  llvm::SmallDenseMap<Value, llvm::SmallSet<int64_t, 8>, 8>
      resource_value_to_ids_;

  // Maps each unique ID to a set of resource-type values that could alias to
  // it. This is inverse of `resource_value_to_ids_` map.
  llvm::SmallDenseMap<int64_t, llvm::SmallSetVector<Value, 8>, 8>
      id_to_resource_values_;

  // Maps MLIR type IDs for resource types to internal resource type IDs.
  llvm::SmallDenseMap<TypeID, int64_t> type_id_to_internal_type_id_;

  // Contains IDs of all resources that are allocated by ops with
  // `UniqueResourceAllocation` trait.
  llvm::SmallDenseSet<int64_t, 32> unique_resource_allocation_ids_;

 public:
  // Resource IDs have the following semantics:
  // a) -1 represents an unknown resource (both instance and type unknown)
  // b) IDs in range [0,kMaxResourceTypeId] represent resource type IDs; we use
  //    such IDs when we know the resource type but not the instance
  // c) IDs > kMaxResourceTypeId represent resource instance IDs (i.e., we know
  //    the specific resource instance)
  //
  // Note: In general, there can be different ops allocating a resource of the
  // same type, for one we might assign a resource type ID and for the other
  // a resource instance ID. That means, they will be treated as non-aliasing.
  // This is correct for all current cases. A problematic case could be if we
  // had two ops A and B, A has the `ResourceHandleAllocatorInterface` and B has
  // not, and both ops might return a handle to the same resource (depending on
  // attributes). In this case, the return value of A would get a different ID
  // than the return value of B although both could point to the same resource.
  // It seems highly unlikely to encounter such a case but, to be safe, this
  // should be revisited for new resource-allocators that might potentially
  // break our currently guaranteed correctness.
  // For context, we are very conservative here compared to
  // `auto_control_deps.py` where it is assumed that allocated resource values
  // NEVER alias. We should align our assumptions in the future.
  static constexpr int64_t kUnknownResourceId = -1;
  static constexpr int64_t kInvalidResourceId = -2;
  static constexpr int64_t kMaxResourceTypeId = 9999;
};

}  // namespace detail

// An analysis that runs on a module and maps each resource-type value to a
// set of unique IDs representing the possible resources it could alias.
//
// Note that this is not an inter-procedural or inter-regional analysis, i.e.,
// each function and region are handled separately and cross-function or cross-
// region aliasing cannot be checked by this analysis.
class ResourceAliasAnalysis : public detail::PerFunctionAggregateAnalysis<
                                  detail::ResourceAliasAnalysisInfo> {
 public:
  // Constructs analysis by analyzing the given module operation.
  explicit ResourceAliasAnalysis(ModuleOp module);
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_RESOURCE_ALIAS_ANALYSIS_H_
