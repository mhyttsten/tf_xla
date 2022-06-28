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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSutilsPSupdate_tpu_metadataDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSutilsPSupdate_tpu_metadataDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSutilsPSupdate_tpu_metadataDTcc() {
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";

// Removes explicit device assignment on TPUExecute and _TPUCompileMlir ops.
// As TPU execution replication logic is delegated to DTensorDevice,
// DTensorDevice should handle replication and Placer would assign devices.
void UpdateTPUDeviceAssignment(mlir::func::FuncOp function,
                               mlir::OpBuilder* builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSutilsPSupdate_tpu_metadataDTcc mht_0(mht_0_v, 221, "", "./tensorflow/dtensor/mlir/utils/update_tpu_metadata.cc", "UpdateTPUDeviceAssignment");

  function.walk([&](mlir::Operation* op) {
    if (!llvm::isa<
            mlir::TF::TPUExecuteOp, mlir::TF::TPUExecuteAndUpdateVariablesOp,
            mlir::TF::_TPUCompileMlirOp, mlir::TF::TPUCompileSucceededAssertOp>(
            op))
      return;

    assert(!op->getAttrOfType<mlir::StringAttr>(kDeviceAttr));

    auto enclosing_launch = op->getParentOfType<mlir::tf_device::LaunchOp>();
    if (!enclosing_launch) return;

    enclosing_launch.deviceAttr(builder->getStringAttr(""));

    // Remove placeholder device attributes of resource arguments to TPU
    // computation.
    for (int i = 0; i < function.getNumArguments(); ++i)
      function.removeArgAttr(i, builder->getStringAttr(kFuncDeviceAttr));
  });
}

// Updates `num_replicas` section of TPUCompileMetadataProto to number of
// devices set by DTensor device.
mlir::LogicalResult UpdateTPUCompileMetadata(const Mesh& mesh_config,
                                             mlir::func::FuncOp function,
                                             mlir::OpBuilder* builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSutilsPSupdate_tpu_metadataDTcc mht_1(mht_1_v, 250, "", "./tensorflow/dtensor/mlir/utils/update_tpu_metadata.cc", "UpdateTPUCompileMetadata");

  auto result = function.walk([&](mlir::TF::_TPUCompileMlirOp compile) {
    auto original_metadata = compile.metadata();
    tpu::TPUCompileMetadataProto metadata_proto;
    if (!metadata_proto.ParseFromString(original_metadata.str())) {
      compile.emitOpError("unable to parse TPUCompileMetadata");
      return mlir::WalkResult::interrupt();
    }

    int num_replicas = mesh_config.num_devices();
    metadata_proto.set_num_replicas(num_replicas);

    // We keep DTensor mesh global device IDs equal to XLA replica IDs, both
    // sequentially increasing over mesh dimensions. Collective lowering has
    // generated `replica_groups` using these IDs.
    //
    // We need to set appropriate XLA replica ID-to-core ID mappings here to get
    // correct results, by being consistent with what the user Python program
    // gets and assumes. There are three kinds of mesh:
    //
    // 1. The first mesh getting here is a one-of-a-kind mesh for merging core
    //    IDs across hosts during TPU initialization. This mesh doesn't need any
    //    mapping to be set. Mesh::tpu_core_ids() is empty when this happens.
    // 2. Users can manually create meshes, with empty or non-empty names. These
    //    meshes have global device IDs equal to TF task-device ordinals, and
    //    they do not place any entry in Mesh::tpu_core_ids(). The default entry
    //    in Mesh::tpu_core_ids(), stored under an empty name key by the mesh
    //    computation in 1, works on these meshes.
    // 3. Users can create ring reduction-optimized meshes using provided
    //    helpers. These meshes must have non-empty names and store an entry in
    //    Mesh::tpu_core_ids() when they are created, using their name as key.
    //
    // For any user-defined mesh, if users have manually specified device
    // assignment, always respect that.
    if (!Mesh::tpu_core_ids().empty() &&
        !metadata_proto.has_device_assignment()) {
      std::string mesh_name = mesh_config.name();
      if (Mesh::tpu_core_ids().count(mesh_name) == 0) {
        // This can happen only for manually created meshes (2 above) with
        // non-empty names. This mesh should use the default mapping.
        VLOG(1) << "mesh_name " << mesh_name << " not found, using empty name";
        mesh_name = "";
      }
      const std::vector<int>& tpu_core_ids = Mesh::tpu_core_ids()[mesh_name];
      VLOG(1) << "tpu_core_ids: " << str_util::Join(tpu_core_ids, ", ");

      xla::DeviceAssignmentProto device_assignment;
      device_assignment.set_replica_count(num_replicas);
      device_assignment.set_computation_count(1);
      auto* computation_device = device_assignment.add_computation_devices();
      // TODO(b/188076080): Clean up device id.
      const int64_t start_device_id = mesh_config.min_global_device_id();
      for (int i = 0; i < num_replicas; ++i) {
        int tpu_core_id_index = i + start_device_id;
        computation_device->add_replica_device_ids(
            tpu_core_ids[tpu_core_id_index]);
      }
      *metadata_proto.mutable_device_assignment() = device_assignment;
    }

    compile.metadataAttr(
        builder->getStringAttr(metadata_proto.SerializeAsString()));
    return mlir::WalkResult::advance();
  });
  return mlir::failure(result.wasInterrupted());
}

// Pass that updates TPU specific metadata including `num_replicas` and device
// assignment of TPUCompileMlirOp and TPUExecute ops.
struct DTensorUpdateTPUMetadata
    : public DTensorUpdateTPUMetadataBase<DTensorUpdateTPUMetadata> {
  void runOnOperation() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSutilsPSupdate_tpu_metadataDTcc mht_2(mht_2_v, 324, "", "./tensorflow/dtensor/mlir/utils/update_tpu_metadata.cc", "runOnOperation");

    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);
    auto module = getOperation();
    mlir::func::FuncOp main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main_func) return;

    auto result = main_func.walk([&](mlir::TF::StatefulPartitionedCallOp op) {
      auto call_config = op.config();
      auto mesh_or_status = Mesh::FromString(call_config.str());
      if (!mesh_or_status.ok()) return mlir::WalkResult::advance();

      const auto mesh = mesh_or_status.ValueOrDie();
      if (!mesh.is_tpu_mesh()) return mlir::WalkResult::advance();

      auto function = MaybeFindFunction(op);
      if (!function) {
        op.emitOpError(
            "Could not find function definition for "
            "StatefulPartitionedCall op running on TPU.");
        return mlir::WalkResult::interrupt();
      }

      if (mlir::failed(UpdateTPUCompileMetadata(mesh, *function, &builder)))
        return mlir::WalkResult::interrupt();

      UpdateTPUDeviceAssignment(*function, &builder);
      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted()) return signalPassFailure();
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorUpdateTPUMetadata() {
  return std::make_unique<DTensorUpdateTPUMetadata>();
}

}  // namespace dtensor
}  // namespace tensorflow
