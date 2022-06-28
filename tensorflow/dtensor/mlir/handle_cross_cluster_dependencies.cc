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
class MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc() {
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

constexpr char kMissingMeshErrorMsg[] =
    "Failed to extract mesh for DTensorHandleCrossClusterDependencies pass. "
    "All clusters must have specified mesh.";

constexpr char kInvalidTensorTransferErrorMsg[] =
    "CopyToMeshOp must be used to send data across mesh.";

constexpr char kInvalidLayoutMsg[] =
    "found CopyToMesh with invalid layout. Found layout {0}.";

// Extracts mesh from `cluster`.
mlir::LogicalResult ExtractMeshFromCluster(mlir::tf_device::ClusterOp cluster,
                                           Mesh* mesh_output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc mht_0(mht_0_v, 225, "", "./tensorflow/dtensor/mlir/handle_cross_cluster_dependencies.cc", "ExtractMeshFromCluster");

  auto mesh_or_status = ExtractDeviceMeshFromOp(cluster);
  if (!mesh_or_status.ok()) return cluster.emitOpError(kMissingMeshErrorMsg);

  const auto& mesh_or_null = mesh_or_status.ValueOrDie();
  if (!mesh_or_null.has_value())
    return cluster.emitOpError(kMissingMeshErrorMsg);

  *mesh_output = mesh_or_null.value();
  return mlir::success();
}

// Returns const op if `op` is a const op or DTensorLayoutOp with Const op as
// input.
mlir::Operation* GetConstOp(mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc mht_1(mht_1_v, 242, "", "./tensorflow/dtensor/mlir/handle_cross_cluster_dependencies.cc", "GetConstOp");

  if (llvm::isa<mlir::TF::ConstOp>(op)) return op;

  if (auto layout = llvm::dyn_cast<mlir::TF::DTensorLayout>(op)) {
    mlir::Operation* input_op = layout.input().getDefiningOp();
    if (input_op && llvm::isa<mlir::TF::ConstOp>(input_op)) return input_op;
  }
  return nullptr;
}

// Creates a clone of `const_op` at the beginning of `cluster` body region and
// set the output value of cloned op replace output of CopyToMesh op within
// `cluster`.
mlir::LogicalResult CloneOpToCluster(mlir::Operation* const_op,
                                     mlir::tf_device::ClusterOp cluster,
                                     mlir::OpOperand* operand) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc mht_2(mht_2_v, 260, "", "./tensorflow/dtensor/mlir/handle_cross_cluster_dependencies.cc", "CloneOpToCluster");

  auto copy_to_mesh =
      llvm::dyn_cast<mlir::TF::CopyToMeshOp>(operand->getOwner());
  assert(copy_to_mesh);
  const std::string layout_attr = copy_to_mesh.layout().str();
  StatusOr<Layout> layout = Layout::FromString(layout_attr);
  if (!layout.ok())
    return copy_to_mesh.emitOpError(
        llvm::formatv(kInvalidLayoutMsg, layout_attr));

  mlir::OpBuilder builder(&cluster.GetBody().front());
  mlir::Operation* cloned_op = builder.clone(*const_op);
  mlir::TensorType type =
      cloned_op->getResult(0).getType().cast<mlir::TensorType>();
  auto layout_op = builder.create<mlir::TF::DTensorLayout>(
      const_op->getLoc(), cloned_op->getResult(0),
      mlir::dtensor::LayoutAttr::get(builder.getContext(), *layout),
      mlir::TF::ShapeAttr::get(builder.getContext(), type));

  copy_to_mesh.output().replaceUsesWithIf(
      layout_op.output(), [&](mlir::OpOperand& operand) {
        return cluster.getOperation()->isProperAncestor(operand.getOwner());
      });

  if (copy_to_mesh->getUsers().empty()) copy_to_mesh.erase();

  return mlir::success();
}

mlir::LogicalResult GetInputProducingValue(mlir::OpOperand& operand,
                                           mlir::Value* val_output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc mht_3(mht_3_v, 293, "", "./tensorflow/dtensor/mlir/handle_cross_cluster_dependencies.cc", "GetInputProducingValue");

  auto input_value = operand.get().dyn_cast<mlir::OpResult>();
  if (!input_value) return mlir::success();

  auto input_cluster =
      llvm::dyn_cast<mlir::tf_device::ClusterOp>(input_value.getOwner());
  if (input_cluster) {
    // If value is from another tf_device.cluster output, then query into
    // the terminator of the input cluster to get mlir::Value from Tensorflow
    // operation that is producing the value.
    *val_output = input_cluster.GetBody().getTerminator()->getOperand(
        input_value.getResultNumber());
  } else {
    *val_output = input_value;
  }
  return mlir::success();
}

// Copies constant operation to mesh clusters if there are multiple usages of
// constants across multiple mesh computations. This is needed for 2 reasons.
// a) Cloning constants across mesh can reduce send/recvs during execution.
// b) DTensor SPMD Expansion for some ops (like tf.reduce_sum) requires inputs
//    to computation to be constants.
mlir::LogicalResult CloneConstantsAcrossMesh(
    mlir::tf_device::ClusterOp cluster) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc mht_4(mht_4_v, 320, "", "./tensorflow/dtensor/mlir/handle_cross_cluster_dependencies.cc", "CloneConstantsAcrossMesh");

  auto& body_region = cluster.body();
  Mesh mesh;
  if (mlir::failed(ExtractMeshFromCluster(cluster, &mesh)))
    return mlir::failure();

  mlir::LogicalResult result(mlir::success());
  mlir::visitUsedValuesDefinedAbove(
      body_region, body_region, [&](mlir::OpOperand* operand) {
        if (mlir::failed(result)) return;

        mlir::Value input_value;
        result = GetInputProducingValue(*operand, &input_value);
        if (mlir::failed(result) || !input_value) return;

        auto input_cluster =
            input_value.getDefiningOp()
                ->getParentOfType<mlir::tf_device::ClusterOp>();
        Mesh input_mesh;
        if (mlir::failed(ExtractMeshFromCluster(input_cluster, &input_mesh))) {
          result = mlir::failure();
          return;
        }

        if (input_mesh == mesh) return;
        if (!llvm::isa<mlir::TF::CopyToMeshOp>(operand->getOwner())) {
          result =
              operand->getOwner()->emitOpError(kInvalidTensorTransferErrorMsg);
          return;
        }

        mlir::Operation* const_op = GetConstOp(input_value.getDefiningOp());
        if (const_op) result = CloneOpToCluster(const_op, cluster, operand);
      });

  return result;
}

// Transforms CopyToMesh op to a pair of DTensorSend/DTensorRecv operations.
mlir::LogicalResult LowerToSendRecv(mlir::TF::CopyToMeshOp copy_to_mesh,
                                    mlir::MLIRContext* context,
                                    int* send_recv_counter) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc mht_5(mht_5_v, 364, "", "./tensorflow/dtensor/mlir/handle_cross_cluster_dependencies.cc", "LowerToSendRecv");

  const mlir::OpResult copied_value =
      copy_to_mesh.input().cast<mlir::OpResult>();
  const int result_index = copied_value.getResultNumber();
  auto src_cluster =
      llvm::cast<mlir::tf_device::ClusterOp>(copied_value.getDefiningOp());
  mlir::Value value_to_send =
      src_cluster.GetBody().getTerminator()->getOperand(result_index);

  // Create DTensorSend op that sends `value_to_send` across mesh cluster.
  mlir::OpBuilder builder(value_to_send.getParentBlock()->getTerminator());

  const std::string op_key =
      llvm::formatv("communication_key_{0}_{1}", copy_to_mesh.layout(),
                    *send_recv_counter)
          .str();
  const std::string layout_attr = copy_to_mesh.layout().str();
  auto layout_or_status = Layout::FromString(layout_attr);
  if (!layout_or_status.ok())
    return copy_to_mesh.emitOpError(
        llvm::formatv(kInvalidLayoutMsg, layout_attr));

  // Create send op that sends data from input cluster to target cluster.
  const Layout& target_layout = layout_or_status.ValueOrDie();
  builder.create<mlir::TF::DTensorSend>(
      copy_to_mesh.getLoc(), value_to_send, builder.getStringAttr(op_key),
      mlir::dtensor::LayoutAttr::get(context, target_layout));

  // Create recv op that recvs data from send op.
  auto tensor_type = value_to_send.getType().dyn_cast<mlir::TensorType>();
  if (!tensor_type)
    return copy_to_mesh.emitOpError(
        "found CopyToMesh sending value with unknown shape. Inputs to "
        "CopyToMesh op must have static shape.");

  builder.setInsertionPoint(copy_to_mesh);
  auto recv_op = builder.create<mlir::TF::DTensorRecv>(
      copy_to_mesh.getLoc(), value_to_send.getType(),
      builder.getStringAttr(op_key),
      mlir::TF::ShapeAttr::get(context, tensor_type),
      mlir::dtensor::LayoutAttr::get(context, target_layout));

  // Replace value for recv ops for all usages of `copy_to_mesh` op.
  copy_to_mesh.replaceAllUsesWith(recv_op.output());

  // Remove copy to mesh op.
  copy_to_mesh.erase();

  *send_recv_counter += 1;

  return mlir::success();
}

// Lowers tf.CopyToMesh to a pair of DTensorSend/DTensorRecv operations.
//
// For example:
//    %0 = "tf_device.cluster"() ({
//      %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
//      tf_device.return %1 : tensor<i32>
//    }) {_mesh="mesh:CPU,x=2,y=2"} : () -> (tensor<i32>)
//
//    %2 = "tf_device.cluster"() ({
//      %3 = "tf.CopyToMesh"(%0)
//          { layout ="mesh:TPU,x=2,y=2 layout:x,replicated" } :
//                      (tensor<i32>) -> (tensor<i32>)
//      %4 = "tf.Neg"(%3) : (tensor<i32>) -> tensor<i32>
//      tf_device.return %4 : tensor<i32>
//    }) {_mesh="mesh:TPU,x=2,y=2"} : () -> (tensor<i32>)
//    return
// }
//
// Is transformed to:
//
//    %0 = "tf_device.cluster"() ({
//      %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
//      "tf.DTensorSend"(%1) {...} : (tensor<i32>) -> ()
//      tf_device.return %1 : tensor<i32>
//    }) {_mesh="mesh:CPU,x=2,y=2"} : () -> (tensor<i32>)
//
//    %2 = "tf_device.cluster"() ({
//      %3 = "tf.DTensorRecv"() {...} : () -> (tensor<i32>)
//      %4 = "tf.Neg"(%3) : (tensor<i32>) -> tensor<i32>
//      tf_device.return %4 : tensor<i32>
//    }) {_mesh="mesh:TPU,x=2,y=2"} : () -> (tensor<i32>)
//    return
// }
mlir::LogicalResult ReplaceCopyToMeshWithVirtualSendRecv(
    mlir::tf_device::ClusterOp cluster, mlir::MLIRContext* context,
    int* send_recv_counter) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc mht_6(mht_6_v, 455, "", "./tensorflow/dtensor/mlir/handle_cross_cluster_dependencies.cc", "ReplaceCopyToMeshWithVirtualSendRecv");

  Mesh current_mesh;
  if (mlir::failed(ExtractMeshFromCluster(cluster, &current_mesh)))
    return mlir::failure();

  mlir::Region& cluster_region = cluster.body();
  mlir::LogicalResult result = mlir::success();

  mlir::visitUsedValuesDefinedAbove(
      cluster_region, cluster_region, [&](mlir::OpOperand* operand) {
        mlir::Value input_value;
        if (mlir::failed(GetInputProducingValue(*operand, &input_value))) {
          result = mlir::failure();
          return;
        }
        if (!input_value) return;

        auto input_cluster =
            input_value.getDefiningOp()
                ->getParentOfType<mlir::tf_device::ClusterOp>();
        Mesh input_mesh;
        if (mlir::failed(ExtractMeshFromCluster(input_cluster, &input_mesh))) {
          result = mlir::failure();
          return;
        }

        if (current_mesh == input_mesh) return;

        // Check that values that cross mesh boundaries go through CopyToMesh
        // op.
        mlir::Operation* input_op = operand->getOwner();
        mlir::TF::CopyToMeshOp copy_to_mesh =
            llvm::dyn_cast<mlir::TF::CopyToMeshOp>(input_op);
        if (!copy_to_mesh) {
          result =
              operand->getOwner()->emitOpError(kInvalidTensorTransferErrorMsg);
          return;
        }

        // Lower CopyToMesh op to a pair of virtual Send/Recv op.
        if (mlir::failed(
                LowerToSendRecv(copy_to_mesh, context, send_recv_counter))) {
          result = mlir::failure();
          return;
        }
      });
  return result;
}

struct DTensorHandleCrossClusterDependencies
    : public DTensorHandleCrossClusterDependenciesBase<
          DTensorHandleCrossClusterDependencies> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc mht_7(mht_7_v, 510, "", "./tensorflow/dtensor/mlir/handle_cross_cluster_dependencies.cc", "getDependentDialects");

    registry.insert<mlir::dtensor::DTensorDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPShandle_cross_cluster_dependenciesDTcc mht_8(mht_8_v, 517, "", "./tensorflow/dtensor/mlir/handle_cross_cluster_dependencies.cc", "runOnOperation");

    mlir::MLIRContext& context = getContext();
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<mlir::tf_device::ClusterOp, 4> clusters;
    module.walk([&](mlir::tf_device::ClusterOp cluster) {
      clusters.emplace_back(cluster);
    });

    int send_recv_counter = 0;
    for (auto cluster : clusters) {
      if (mlir::failed(CloneConstantsAcrossMesh(cluster)))
        return signalPassFailure();

      if (mlir::failed(ReplaceCopyToMeshWithVirtualSendRecv(
              cluster, &context, &send_recv_counter)))
        return signalPassFailure();
    }

    // Once CopyToMesh has been lowered to DTensorSend/Recv operations,
    // tf_device.Cluster may now have dangling/unused result values. Remove all
    // such return values.
    for (auto cluster : clusters) RemoveUnusedClusterResults(cluster);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorHandleCrossClusterDependencies() {
  return std::make_unique<DTensorHandleCrossClusterDependencies>();
}

}  // namespace dtensor
}  // namespace tensorflow
