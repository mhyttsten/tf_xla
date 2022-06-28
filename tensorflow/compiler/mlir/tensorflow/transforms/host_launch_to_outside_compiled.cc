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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShost_launch_to_outside_compiledDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShost_launch_to_outside_compiledDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShost_launch_to_outside_compiledDTcc() {
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Analysis/CallGraph.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_cluster_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

struct HostLaunchToOutsideCompiledPass
    : public HostLaunchToOutsideCompiledPassBase<
          HostLaunchToOutsideCompiledPass> {
  void runOnOperation() override;
};

// Assign all ops in region with _xla_outside_compilation attribute.
void MarkOutsideCompiledInRegion(Region& region) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShost_launch_to_outside_compiledDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/mlir/tensorflow/transforms/host_launch_to_outside_compiled.cc", "MarkOutsideCompiledInRegion");

  region.walk([&](Operation* op) {
    op->setAttr(kXlaOutsideCompilationAttr,
                StringAttr::get(op->getContext(), "from_launch"));
  });
}

void HoistOpsAndAnnotateWithOutsideCompilation(tf_device::LaunchOp launch) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShost_launch_to_outside_compiledDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/mlir/tensorflow/transforms/host_launch_to_outside_compiled.cc", "HoistOpsAndAnnotateWithOutsideCompilation");

  // Forward launch inner op results to launch op results.
  launch.replaceAllUsesWith(launch.GetBody().getTerminator()->getOperands());

  // For all inner ops, assign the launch device as a `device` attribute.
  MarkOutsideCompiledInRegion(launch.body());

  // Move all inner ops of the launch to the block containing the launch.
  auto body = launch.GetBody().without_terminator();
  Operation* launch_op = launch.getOperation();
  launch_op->getBlock()->getOperations().splice(
      launch_op->getIterator(), launch.GetBody().getOperations(), body.begin(),
      body.end());

  launch.erase();
}

void HostLaunchToOutsideCompiledPass::runOnOperation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShost_launch_to_outside_compiledDTcc mht_2(mht_2_v, 243, "", "./tensorflow/compiler/mlir/tensorflow/transforms/host_launch_to_outside_compiled.cc", "HostLaunchToOutsideCompiledPass::runOnOperation");

  auto traverse_op = [&](Operation* op, tf_device::ClusterOp tpu_cluster,
                         absl::optional<std::string> host_device) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPShost_launch_to_outside_compiledDTcc mht_3(mht_3_v, 248, "", "./tensorflow/compiler/mlir/tensorflow/transforms/host_launch_to_outside_compiled.cc", "lambda");

    // Hoist launch.
    if (tf_device::LaunchOp launch = dyn_cast<tf_device::LaunchOp>(op)) {
      StringAttr device_attr = launch->getAttrOfType<StringAttr>(kDeviceAttr);
      if (host_device && device_attr &&
          device_attr.getValue().equals(*host_device))
        HoistOpsAndAnnotateWithOutsideCompilation(launch);
    }
    return WalkResult::advance();
  };

  ModuleOp module = getOperation();
  // If there is model parallelism, we return early since
  // GetHostDeviceOutsideComputation will fail and an error should have been
  // returned in an earlier pass.
  // TODO(b/186420116): Remove this check once outside compilation and model
  // parallelism work together.
  auto result = module.walk([&](tf_device::ClusterOp tpu_cluster) {
    if (tensorflow::HasModelParallelism(tpu_cluster)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return;

  if (failed(TFTPU::WalkReachableFromTpuCluster(module, traverse_op)))
    return signalPassFailure();
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateHostLaunchToOutsideCompiledPass() {
  return std::make_unique<HostLaunchToOutsideCompiledPass>();
}

}  // namespace TFDevice
}  // namespace mlir
