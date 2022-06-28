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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplica_id_to_device_ordinalDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplica_id_to_device_ordinalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplica_id_to_device_ordinalDTcc() {
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

// This pass sets the device ordinal attribute of the required op using
// the replica id attribute.

#include <memory>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"

namespace mlir {
namespace TFDevice {
namespace {
constexpr char kReplicaIdAttr[] = "_xla_replica_id";
constexpr char kDeviceOrdinalAttr[] = "device_ordinal";

struct ReplicaIDToDeviceOrdinalPass
    : public TF::ReplicaIDToDeviceOrdinalPassBase<
          ReplicaIDToDeviceOrdinalPass> {
  void runOnOperation() override;
};

// Returns whether op requires `device_ordinal` attribute.
bool RequiresDeviceOrdinalAttribute(Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplica_id_to_device_ordinalDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/mlir/tensorflow/transforms/replica_id_to_device_ordinal.cc", "RequiresDeviceOrdinalAttribute");

  return (llvm::isa<TF::EnqueueTPUEmbeddingSparseTensorBatchOp,
                    TF::EnqueueTPUEmbeddingRaggedTensorBatchOp,
                    TF::EnqueueTPUEmbeddingArbitraryTensorBatchOp>(op) &&
          op->hasAttr(kDeviceOrdinalAttr) && op->hasAttr(kReplicaIdAttr));
}

void ReplicaIDToDeviceOrdinalPass::runOnOperation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPSreplica_id_to_device_ordinalDTcc mht_1(mht_1_v, 221, "", "./tensorflow/compiler/mlir/tensorflow/transforms/replica_id_to_device_ordinal.cc", "ReplicaIDToDeviceOrdinalPass::runOnOperation");

  const Dialect* tf_dialect = getContext().getLoadedDialect("tf");
  if (!tf_dialect) {
    getOperation().emitError() << "'tf' dialect is not registered";
    return signalPassFailure();
  }

  // Get the number of devices per host.
  int device_num = 0;
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(
          getOperation()->getParentOfType<ModuleOp>(), &devices)))
    return signalPassFailure();
  for (const auto& device_name : devices.device_names()) {
    if (device_name.has_type && device_name.type == "TPU") ++device_num;
  }

  if (device_num == 0) return;

  llvm::SmallVector<Operation*, 4> require_device_ordinal_ops;
  getOperation().walk([&](Operation* op) {
    if (RequiresDeviceOrdinalAttribute(op)) {
      require_device_ordinal_ops.push_back(op);
    }
  });

  if (require_device_ordinal_ops.size() == 1) {
    // If there is only one op which requires the device ordinal being set,
    // set the device ordinal to 0. Note: This is for single device use case
    // (eg. pf megacore) for which `_xla_replica_id` isn't set via the
    // replicate_to_islands pass.
    Operation* op = require_device_ordinal_ops.front();
    if (op->getAttrOfType<IntegerAttr>(kDeviceOrdinalAttr).getInt() == -1) {
      OpBuilder builder(op);
      op->setAttr(kDeviceOrdinalAttr, builder.getI64IntegerAttr(0));
    }
  } else {
    // If the device ordinal attribute is -1, set it with the replica id
    // attribute modulo the number of TPU cores in the system.
    for (auto op : require_device_ordinal_ops) {
      if (op->getAttrOfType<IntegerAttr>(kDeviceOrdinalAttr).getInt() == -1) {
        OpBuilder builder(op);
        int device_ordinal =
            op->getAttrOfType<IntegerAttr>(kReplicaIdAttr).getInt() %
            device_num;
        op->setAttr(kDeviceOrdinalAttr,
                    builder.getI64IntegerAttr(device_ordinal));
      }
    }
  }
}
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateReplicaIDToDeviceOrdinalPass() {
  return std::make_unique<ReplicaIDToDeviceOrdinalPass>();
}

}  // namespace TFDevice
}  // namespace mlir
