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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_cluster_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_cluster_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_cluster_utilDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {

namespace {
mlir::LogicalResult WalkReachableFromTpuCluster(
    bool pass_host_device, ModuleOp module,
    std::function<WalkResult(Operation*, tf_device::ClusterOp,
                             absl::optional<std::string>)>
        callback) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_cluster_utilDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_cluster_util.cc", "WalkReachableFromTpuCluster");

  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(module, &devices))) return failure();
  const CallGraph call_graph(module);
  // symbol_table caches callees in the CallGraph.
  SymbolTableCollection symbol_table;
  // List pending nodes to traverse with their root TPU cluster.
  llvm::SmallVector<std::pair<CallGraphNode*, tf_device::ClusterOp>>
      pending_call_nodes;
  // Cache the host device for each TPU cluster.
  std::unordered_map<Operation*, absl::optional<std::string>> cluster_to_host;

  auto insert_pending_op = [&](Operation* op,
                               tf_device::ClusterOp tpu_cluster) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_cluster_utilDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_cluster_util.cc", "lambda");

    // Add callee nodes to pending_call_nodes.
    if (CallOpInterface call = dyn_cast<CallOpInterface>(op)) {
      CallGraphNode* node = call_graph.resolveCallable(call, symbol_table);
      pending_call_nodes.emplace_back(node, tpu_cluster);
    }
  };

  // Traverse ops in each TPU cluster.
  auto result = module.walk([&](tf_device::ClusterOp tpu_cluster) {
    absl::optional<std::string> host_device;
    if (pass_host_device && !tensorflow::HasModelParallelism(tpu_cluster)) {
      std::string host_device_value;
      if (failed(tensorflow::GetHostDeviceOutsideComputation(
              devices, tpu_cluster, &host_device_value)))
        return WalkResult::interrupt();
      host_device = host_device_value;
    }
    cluster_to_host[tpu_cluster.getOperation()] = host_device;
    return tpu_cluster.walk([&](Operation* op) {
      insert_pending_op(op, tpu_cluster);
      return callback(op, tpu_cluster,
                      cluster_to_host[tpu_cluster.getOperation()]);
    });
  });
  if (result.wasInterrupted()) return failure();

  // Traverse ops that are reachable from some TPU cluster.
  // node_to_cluster is used to avoid traversing the same node twice, and to
  // check that no node is reachable from multiple TPU clusters.
  std::unordered_map<CallGraphNode*, tf_device::ClusterOp> node_to_cluster;
  while (!pending_call_nodes.empty()) {
    auto pair = pending_call_nodes.back();
    pending_call_nodes.pop_back();
    CallGraphNode* node = pair.first;
    tf_device::ClusterOp tpu_cluster = pair.second;
    if (node_to_cluster.count(node)) {
      if (node_to_cluster[node].getOperation() != tpu_cluster) {
        node->getCallableRegion()->getParentOp()->emitOpError(
            "The same function is reachable from multiple TPU Clusters.");
      }
    } else {
      node_to_cluster[node] = tpu_cluster;
      auto result = node->getCallableRegion()->walk([&](Operation* op) {
        insert_pending_op(op, tpu_cluster);
        return callback(op, tpu_cluster,
                        cluster_to_host[tpu_cluster.getOperation()]);
      });
      if (result.wasInterrupted()) return failure();
    }
  }

  return success();
}
}  // namespace

mlir::LogicalResult WalkReachableFromTpuCluster(
    ModuleOp module, std::function<WalkResult(Operation*, tf_device::ClusterOp,
                                              absl::optional<std::string>)>
                         callback) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_cluster_utilDTcc mht_2(mht_2_v, 275, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_cluster_util.cc", "WalkReachableFromTpuCluster");

  return WalkReachableFromTpuCluster(true, module, callback);
}

mlir::LogicalResult WalkReachableFromTpuCluster(
    ModuleOp module,
    std::function<WalkResult(Operation*, tf_device::ClusterOp)> callback) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_cluster_utilDTcc mht_3(mht_3_v, 284, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_cluster_util.cc", "WalkReachableFromTpuCluster");

  auto with_host_op = [&](Operation* op, tf_device::ClusterOp tpu_cluster,
                          absl::optional<std::string> host_device) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPStpu_cluster_utilDTcc mht_4(mht_4_v, 289, "", "./tensorflow/compiler/mlir/tensorflow/utils/tpu_cluster_util.cc", "lambda");

    return callback(op, tpu_cluster);
  };
  return WalkReachableFromTpuCluster(false, module, with_host_op);
}

}  // namespace TFTPU
}  // namespace mlir
