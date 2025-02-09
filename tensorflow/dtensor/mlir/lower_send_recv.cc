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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSlower_send_recvDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlower_send_recvDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSlower_send_recvDTcc() {
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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/dtensor_send_recv.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

constexpr char kMissingMeshErrorMsg[] =
    "Failed to extract mesh for DTensorMergeCluster pass. "
    "All clusters must have specified mesh.";

// Extracts mesh from `cluster`.
mlir::LogicalResult ExtractMeshFromCluster(mlir::tf_device::ClusterOp cluster,
                                           Mesh* mesh_output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlower_send_recvDTcc mht_0(mht_0_v, 219, "", "./tensorflow/dtensor/mlir/lower_send_recv.cc", "ExtractMeshFromCluster");

  auto mesh_or_status = ExtractDeviceMeshFromOp(cluster);
  if (!mesh_or_status.ok()) return cluster.emitOpError(kMissingMeshErrorMsg);

  const absl::optional<Mesh>& mesh_or_null = *mesh_or_status;
  if (!mesh_or_null.has_value())
    return cluster.emitOpError(kMissingMeshErrorMsg);

  *mesh_output = mesh_or_null.value();
  return mlir::success();
}

// Find all DTesorSend/Recv ops and lower into TF/XLA Send/Recv operations with
// execution kernels.
mlir::LogicalResult LowerDTensorSendRecvsOps(mlir::ModuleOp module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlower_send_recvDTcc mht_1(mht_1_v, 236, "", "./tensorflow/dtensor/mlir/lower_send_recv.cc", "LowerDTensorSendRecvsOps");

  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::TF::DTensorSend send_op) {
    if (mlir::failed(result)) return;

    auto recv_op = GetCorrespondingDTensorSendRecvOp<mlir::TF::DTensorSend>(
        module, send_op);
    if (!recv_op.ok()) {
      result = send_op.emitOpError(recv_op.status().error_message());
      return;
    }
    auto dtensor_recv = llvm::dyn_cast<mlir::TF::DTensorRecv>(*recv_op);
    if (!dtensor_recv) {
      result = send_op.emitOpError(
          "Cannot find a matching DTensorRecv op for this DTensorSend op");
      return;
    }
    const Mesh recv_mesh = dtensor_recv.layout().mesh();

    Mesh send_mesh;
    if (mlir::failed(ExtractMeshFromCluster(
            send_op->getParentOfType<mlir::tf_device::ClusterOp>(),
            &send_mesh))) {
      result = mlir::failure();
      return;
    }

    if (!send_mesh.is_tpu_mesh() && !recv_mesh.is_tpu_mesh()) {
      result = send_op->emitOpError(
          "Multi-mesh tensor transfer between non-xla devices are not yet "
          "supported.");
      return;
    }

    const Layout recv_layout =
        Layout::ReplicatedOnMesh(recv_mesh, ValueRank(dtensor_recv.output()));
    const Layout send_input_layout =
        Layout::ReplicatedOnMesh(send_mesh, ValueRank(send_op.input()));

    StatusOr<mlir::Operation*> lowered_recv =
        LowerDTensorRecvToXlaOp(dtensor_recv);
    if (!lowered_recv.ok()) {
      result = dtensor_recv->emitOpError(lowered_recv.status().error_message());
      return;
    }
    dtensor_recv->replaceAllUsesWith(*lowered_recv);
    dtensor_recv.erase();

    auto lowered_send_or =
        LowerDTensorSendToXlaOp(send_input_layout, send_op.input(), send_op,
                                /*from_spmd_expander=*/false);
    if (!lowered_send_or.ok()) {
      result = send_op->emitOpError(lowered_send_or.status().error_message());
      return;
    }
  });
  return result;
}

// Adds Identity Op that uses device_id argument as inputs for clusters that
// does not have device id usages. When send/recv operations exists in
// tf_device.Clusters to transfer data across mesh clusters, device_id argument
// is required. However, mlir::func::FuncOp's created by transforming
// tf_device.Cluster to tf_device.ClusterFunc during ClusterOutlining pass will
// **not** include device_id as input argument if there are no usages within the
// cluster op body. As so, add Identity op that uses device_id argument from
// main function in all tf_device.Clusters so that device_id argument can be
// retained when converting tf_device.Cluster to functions.
void PropagateDeviceIdToClusters(mlir::ModuleOp module) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlower_send_recvDTcc mht_2(mht_2_v, 307, "", "./tensorflow/dtensor/mlir/lower_send_recv.cc", "PropagateDeviceIdToClusters");

  mlir::WalkResult result = module.walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::TF::_XlaSendFromHostOp, mlir::TF::_XlaRecvAtHostV2Op,
                  mlir::TF::XlaSendToHostOp, mlir::TF::XlaRecvFromHostOp,
                  mlir::TF::_HostSendOp, mlir::TF::_HostRecvOp,
                  mlir::TF::SendOp, mlir::TF::RecvOp>(op))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  const bool has_cross_mesh_send_recv = result.wasInterrupted();
  if (!has_cross_mesh_send_recv) return;

  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  auto device_id = DeviceId(main_func);

  module.walk([&](mlir::tf_device::ClusterOp op) {
    mlir::OpBuilder builder(&op.GetBody().front());
    builder.create<mlir::TF::IdentityOp>(main_func.getLoc(),
                                         device_id->getType(), *device_id);
  });
}

// Pass that merges multiple tf_device.Cluster ops for multi-mesh computation
// into a single cluster. After this pass, exactly one tf_device.Cluster op
// exists for each device mesh.
struct DTensorLowerSendRecv
    : public DTensorLowerSendRecvBase<DTensorLowerSendRecv> {
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSlower_send_recvDTcc mht_3(mht_3_v, 339, "", "./tensorflow/dtensor/mlir/lower_send_recv.cc", "runOnOperation");

    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder op_builder(&context);
    auto module = getOperation();

    // Merging clusters and decomposing control flow may have created new
    // DTensorSend/DTensorRecv ops. Lower DTensorSend/DTensorRecv ops added by
    // above transformations.
    if (mlir::failed(LowerDTensorSendRecvsOps(module)))
      return signalPassFailure();

    // Ensure that all mesh clusters have at least one usages of device_id
    // argument from main function to guarantee that device_id argument is
    // retained after ClusterOutlinging.
    PropagateDeviceIdToClusters(module);
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorLowerSendRecv() {
  return std::make_unique<DTensorLowerSendRecv>();
}

}  // namespace dtensor
}  // namespace tensorflow
