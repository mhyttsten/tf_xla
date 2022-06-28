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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clustering_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clustering_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clustering_passDTcc() {
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

#include "llvm/ADT/STLExtras.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/cluster_ops_by_policy.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using llvm::ArrayRef;

using mlir::TF::ConstOp;
using mlir::TF::HashTableV2Op;
using mlir::TF::ReadVariableOp;

using mlir::TFDevice::Cluster;
using mlir::TFDevice::ClusteringPolicySet;
using mlir::TFDevice::CreateClusterOp;
using mlir::TFDevice::FindClustersInTheBlock;

// -------------------------------------------------------------------------- //
// Cluster operations based on the TF JitRt clustering policy.
// -------------------------------------------------------------------------- //
struct ClusteringPass : public ClusteringBase<ClusteringPass> {
  ClusteringPass() = default;
  ClusteringPass(ArrayRef<std::string> cluster_oplist, int cluster_min_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clustering_passDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering_pass.cc", "ClusteringPass");

    oplist = cluster_oplist;
    min_cluster_size = cluster_min_size;
  }

  void runOnOperation() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clustering_passDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering_pass.cc", "runOnOperation");

    ClusteringPolicySet policies;

    // Parse clustering tier and operations filter from the oplist.
    llvm::DenseSet<llvm::StringRef> opset;
    llvm::Optional<JitRtClusteringTier> tier;

    for (const auto& op : oplist) {
      if (op == "tier0") {
        tier = JitRtClusteringTier::kTier0;
      } else if (op == "tier1") {
        tier = JitRtClusteringTier::kTier1;
      } else if (op == "tier1metadata") {
        tier = JitRtClusteringTier::kTier1Metadata;
      } else if (op == "tier1reductions") {
        tier = JitRtClusteringTier::kTier1Reductions;
      } else if (op == "all") {
        tier = JitRtClusteringTier::kAll;
      } else {
        opset.insert(op);
      }
    }

    // Run clustering only if the clustering tier or supported operations are
    // explicitly defined by the oplist.
    if (!tier.hasValue() && opset.empty()) return;

    // If the clustering tier is not defined, it means that the opset will later
    // filter supported operations, so it's ok to use `all` tier.
    populateTfJitRtClusteringPolicies(
        policies, tier.getValueOr(JitRtClusteringTier::kAll));

    // If opset is not empty restrict operations that are enabled for
    // clustering.
    auto opset_filter = [&](mlir::Operation* op) -> bool {
      return opset.empty() || opset.contains(op->getName().getStringRef());
    };

    // Find operations that could be hoisted from the function body into the
    // TFRT resource initialization function. Currently it is an approximation
    // of hoisting rules in the TFRT, we just find all the operations that
    // depend only on ConstOp, ReadVariableOp or HashTableV2Op operations. We
    // don't do any side effects analysis and conservatively can mark as
    // hoistable operations that will not be hoisted by TFRT because of side
    // effect dependencies.
    //
    // TODO(ezhulenev): This should be shared with TFRT hoisting implementation.

    // Initialize a set of operations that we assume we will hoist.
    llvm::DenseSet<mlir::Operation*> hoisted_ops;
    getOperation().walk([&](mlir::Operation* op) {
      if (mlir::isa<ReadVariableOp, ConstOp, HashTableV2Op>(op))
        hoisted_ops.insert(op);
    });

    // Initialize work list with users of ReadVariableOp results.
    llvm::SmallVector<mlir::Operation*> work_list;
    for (mlir::Operation* hoisted : hoisted_ops)
      work_list.append(hoisted->user_begin(), hoisted->user_end());

    // Traverse all users until we find all operations that could be hoisted.
    while (!work_list.empty()) {
      mlir::Operation* op = work_list.pop_back_val();

      // Skip operations that are already in the hoisted set.
      if (hoisted_ops.contains(op)) continue;

      // Add operation to hoisted ops set if all operands can be hoisted.
      bool all_operands_hoisted =
          llvm::all_of(op->getOperands(), [&](mlir::Value value) {
            return hoisted_ops.contains(value.getDefiningOp());
          });
      if (!all_operands_hoisted) continue;

      hoisted_ops.insert(op);
      work_list.append(op->user_begin(), op->user_end());
    }

    auto hoist_filter = [&](mlir::Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPStransformsPStf_jitrt_clustering_passDTcc mht_2(mht_2_v, 307, "", "./tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering_pass.cc", "lambda");

      return !hoisted_ops.contains(op);
    };

    // Combine together opset and hoist filters.
    auto filter = [&](mlir::Operation* op) -> bool {
      return opset_filter(op) && hoist_filter(op);
    };

    // Annotate all formed clusters with an attribute.
    auto policy = mlir::StringAttr::get(&getContext(), "tfrt.auto-fusion");

    getOperation().walk([&](mlir::Block* block) {
      for (Cluster& cluster : FindClustersInTheBlock(block, policies, filter)) {
        // Do not create too small clusters.
        if (cluster.operations.size() < min_cluster_size) continue;
        // Verify that JIT runtime can compile the cluster.
        if (failed(VerifyCluster(cluster))) continue;

        CreateClusterOp(cluster, policy);
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfJitRtClusteringPass() {
  return std::make_unique<ClusteringPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfJitRtClusteringPass(llvm::ArrayRef<std::string> oplist,
                            int min_cluster_size) {
  return std::make_unique<ClusteringPass>(oplist, min_cluster_size);
}

}  // namespace tensorflow
