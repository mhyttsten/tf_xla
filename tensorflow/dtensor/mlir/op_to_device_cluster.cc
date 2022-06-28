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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSop_to_device_clusterDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSop_to_device_clusterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSop_to_device_clusterDTcc() {
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

#include <string>

#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"

namespace tensorflow {
namespace dtensor {

namespace {

// Extracts mesh config from the Op.
// We currently hard extract mesh information from all the args and assume they
// are the same. This should not be the case when we have multiple functions.
mlir::LogicalResult WrapDeviceCluster(mlir::OpBuilder *builder,
                                      mlir::Operation *op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSop_to_device_clusterDTcc mht_0(mht_0_v, 220, "", "./tensorflow/dtensor/mlir/op_to_device_cluster.cc", "WrapDeviceCluster");

  // Create new tf_device.cluster op wrapping a single operation.
  builder->setInsertionPoint(op);
  auto cluster = builder->create<mlir::tf_device::ClusterOp>(
      op->getLoc(), op->getResultTypes());
  if (auto layout_op = llvm::dyn_cast<mlir::TF::DTensorLayout>(op)) {
    cluster->setAttr(kMeshAttr, builder->getStringAttr(
                                    layout_op.layout().mesh().ToString()));
  } else if (auto copy_to_mesh = llvm::dyn_cast<mlir::TF::CopyToMeshOp>(op)) {
    const std::string layout_string = copy_to_mesh.layout().str();
    auto layout_or = Layout::FromString(layout_string);
    if (!layout_or.ok())
      return op->emitOpError(
          llvm::formatv("Found tf.CopyToMesh Op with unparsable layout : {0}",
                        layout_string));

    cluster->setAttr(kMeshAttr,
                     builder->getStringAttr(layout_or->mesh().ToString()));
  } else {
    // If mesh configuration can be inferred from the op directly, use the mesh
    // information from op attribute directly. If op is not annotated with mesh
    // information, then mesh will be inferred in following
    // DTensorMeshPropagation pass and will be inferred from consumers or
    // operands.
    auto status_or_mesh = ExtractDeviceMeshFromOp(op);

    if (!status_or_mesh.ok())
      return op->emitOpError(
          llvm::formatv("failed to wrap to device cluster. {0}",
                        status_or_mesh.status().error_message()));

    const auto mesh_config = status_or_mesh.ValueOrDie();
    if (mesh_config)
      cluster->setAttr(kMeshAttr,
                       builder->getStringAttr(mesh_config->ToString()));
  }

  op->replaceAllUsesWith(cluster);

  cluster.body().push_back(new mlir::Block);

  builder->setInsertionPointToEnd(&cluster.GetBody());
  builder->create<mlir::tf_device::ReturnOp>(op->getLoc(), op->getResults());

  // Move `op` inside newly created `ClusterOp`.
  op->moveBefore(cluster.GetBody().getTerminator());

  return mlir::success();
}

// MLIR pass that wraps tf_device.cluster op to every TF op.
struct DTensorOpToDeviceClusterPass
    : public DTensorOpToDeviceClusterBase<DTensorOpToDeviceClusterPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSop_to_device_clusterDTcc mht_1(mht_1_v, 276, "", "./tensorflow/dtensor/mlir/op_to_device_cluster.cc", "getDependentDialects");

    registry.insert<mlir::dtensor::DTensorDialect>();
    registry.insert<mlir::tf_device::TensorFlowDeviceDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSop_to_device_clusterDTcc mht_2(mht_2_v, 284, "", "./tensorflow/dtensor/mlir/op_to_device_cluster.cc", "runOnOperation");

    mlir::MLIRContext &context = getContext();
    mlir::OpBuilder op_builder(&context);
    mlir::Dialect *tf =
        getContext().getLoadedDialect<mlir::TF::TensorFlowDialect>();

    auto walk_result = getOperation().walk([&](mlir::Operation *operation) {
      const auto op_dialect = operation->getDialect();
      // Only TF dialects are supported for layout propagation.
      if (op_dialect != tf) return mlir::WalkResult::advance();

      // For control flow operations, tf.yield ops exists and should not be
      // wrapped to tf_device.cluster as the op does not need to be transformed
      // in SPMD expansion and tf.If/tf.While op require all ops to terminate
      // with tf.Yield op. Wrapping yield op in tf_device.cluster invalidates
      // this invariant.
      if (llvm::isa<mlir::TF::YieldOp>(operation))
        return mlir::WalkResult::advance();

      if (mlir::failed(WrapDeviceCluster(&op_builder, operation)))
        return mlir::WalkResult::interrupt();
      return mlir::WalkResult::advance();
    });

    if (walk_result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorOpToDeviceClusterPass() {
  return std::make_unique<DTensorOpToDeviceClusterPass>();
}

}  // namespace dtensor
}  // namespace tensorflow
