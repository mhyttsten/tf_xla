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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_send_recvDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_send_recvDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_send_recvDTcc() {
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

#include "tensorflow/dtensor/mlir/dtensor_send_recv.h"

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Returns compilation key placeholder. This placeholder will be replaced with
// output of TPUCompile op during TPURewrite pass. Program key (output of
// TPUCompile op) is used to differentiate TPU computation from which to receive
// data.
mlir::Value GetOrCreateCompilationKey(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_send_recvDTcc mht_0(mht_0_v, 205, "", "./tensorflow/dtensor/mlir/dtensor_send_recv.cc", "GetOrCreateCompilationKey");

  mlir::Value key;
  auto cluster = op->getParentOfType<mlir::tf_device::ClusterOp>();
  assert(cluster);
  cluster.walk(
      [&](mlir::TF::_TPUCompileMlirPlaceholderProgramKeyOp compilation_key) {
        key = compilation_key.program();
      });
  if (key) return key;

  mlir::OpBuilder builder(&cluster.GetBody().front());
  auto result_type =
      mlir::RankedTensorType::get({3}, builder.getType<mlir::TF::StringType>());
  auto new_compilation_key =
      builder.create<mlir::TF::_TPUCompileMlirPlaceholderProgramKeyOp>(
          cluster.getLoc(), /*program=*/result_type,
          llvm::ArrayRef<mlir::Value>{});
  return new_compilation_key.program();
}

}  // namespace

StatusOr<mlir::Value> GetDeviceOrdinal(const Mesh& mesh,
                                       const mlir::Location& loc,
                                       mlir::func::FuncOp function,
                                       mlir::OpBuilder* builder,
                                       bool return_int64_type) {
  // Create as many entries as the number of devices in the entire mesh.
  llvm::SmallVector<int32, 4> device_id_to_ordinal(mesh.num_devices(), 0);
  // Only fill in entries with indices equal to local device IDs. For TPUs,
  // there are usually 8 local devices.
  for (int i = 0; i < mesh.local_device_ids().size(); ++i) {
    device_id_to_ordinal[mesh.local_device_ids()[i]] = i;
  }
  // Slice out the device ordinal using the device ID as index.
  TF_ASSIGN_OR_RETURN(mlir::Value device_id, DeviceId(function));
  mlir::TF::SliceOp device_ordinal = builder->create<mlir::TF::SliceOp>(
      loc,
      /*output=*/EffectivelyScalarR1Type(builder->getIntegerType(32)),
      /*input=*/IntConst(*builder, loc, device_id_to_ordinal),
      /*begin=*/
      mlir::TF::collection_ops_util::ReshapeScalarToSizeType(*builder,
                                                             device_id, loc),
      /*size=*/IntConst(*builder, loc, {1}));
  mlir::Value device_ordinal_scalar =
      ReshapeSizeTypeToScalar(*builder, loc, device_ordinal);
  if (return_int64_type) {
    device_ordinal_scalar = builder->create<mlir::TF::CastOp>(
        loc, mlir::RankedTensorType::get({}, builder->getI64Type()),
        device_ordinal_scalar);
  }
  return device_ordinal_scalar;
}

// Lowers DTensorSend Op to either one of XlaSendFromHost op or XlaSendToHost,
// depending on the src mesh cluster.
StatusOr<mlir::Operation*> LowerDTensorSendToXlaOp(
    const Layout& send_input_layout, mlir::Value send_input,
    mlir::TF::DTensorSend dtensor_send, bool send_from_device_zero) {
  const bool send_from_cpu = !send_input_layout.mesh().is_tpu_mesh();
  mlir::OpBuilder builder(dtensor_send);

  mlir::Location loc = dtensor_send.getLoc();
  mlir::Operation* lowered_send_op;
  if (send_from_cpu) {
    llvm::SmallVector<mlir::Value, 4> value_to_send{send_input};
    mlir::OpBuilder::InsertPoint insertion_point = builder.saveInsertionPoint();
    mlir::Value program_key = GetOrCreateCompilationKey(dtensor_send);
    builder.restoreInsertionPoint(insertion_point);

    mlir::Value device_ordinal;
    if (send_from_device_zero) {
      // For CopyToMesh, we currently only support sending from host device 0
      // to target TPUs.
      device_ordinal = CreateIntScalarConst(0, builder, loc);
    } else {
      // For special topologies, always send from CPU device i to TPU device i.
      auto send_cluster =
          dtensor_send->getParentOfType<mlir::tf_device::ClusterOp>();
      if (!send_cluster) {
        return errors::InvalidArgument("DTensorSend is not inside a ClusterOp");
      }
      auto send_func = send_cluster->getParentOfType<mlir::func::FuncOp>();
      if (!send_func) {
        return errors::InvalidArgument("DTensorSend is not inside a FuncOp");
      }
      TF_ASSIGN_OR_RETURN(
          device_ordinal,
          GetDeviceOrdinal(send_input_layout.mesh(), loc, send_func, &builder));
    }
    // Create XlaSendFromHostV2 op
    lowered_send_op = builder.create<mlir::TF::_XlaSendFromHostV2Op>(
        loc, value_to_send, program_key, device_ordinal, dtensor_send.key());
  } else {
    // Note that for ops running in XLA/TPU, device ordinal input is not needed.
    lowered_send_op = builder.create<mlir::TF::XlaSendToHostOp>(
        loc, send_input, dtensor_send.key());
  }

  dtensor_send.erase();
  return lowered_send_op;
}

// Lowers DTensorRecv op to either one of XlaRecvAtHost or XlaRecvFromHost,
// depending on src mesh cluster configuration.
StatusOr<mlir::Operation*> LowerDTensorRecvToXlaOp(
    mlir::TF::DTensorRecv dtensor_recv) {
  return LowerDTensorRecvToXlaOp(dtensor_recv, dtensor_recv.getType());
}

// Lowers DTensorRecv op to either one of XlaRecvAtHost or XlaRecvFromHost,
// depending on src mesh cluster configuration. `output_type` can be set to the
// specific local tensor type needed, if different from the Recv op output type.
StatusOr<mlir::Operation*> LowerDTensorRecvToXlaOp(
    mlir::TF::DTensorRecv dtensor_recv, mlir::Type output_type) {
  const bool recv_at_cpu = dtensor_recv.layout().mesh().is_cpu_mesh();
  mlir::Operation* recv_xla_op = nullptr;
  mlir::OpBuilder builder(dtensor_recv);

  if (recv_at_cpu) {
    // Create XlaRecvAtHostV2 op.
    llvm::SmallVector<mlir::Type, 4> output_types{output_type};
    auto recv_cluster =
        dtensor_recv->getParentOfType<mlir::tf_device::ClusterOp>();

    TF_ASSIGN_OR_RETURN(absl::optional<Mesh> mesh,
                        ExtractDeviceMeshFromOp(recv_cluster));
    if (!mesh.has_value())
      return errors::InvalidArgument(
          "failed to get device ordinal as mesh for operation is not "
          "specified.");

    mlir::OpBuilder builder(&recv_cluster.GetBody().front());
    TF_ASSIGN_OR_RETURN(
        mlir::Value device_ordinal,
        GetDeviceOrdinal(*mesh, recv_cluster.getLoc(),
                         recv_cluster->getParentOfType<mlir::func::FuncOp>(),
                         &builder));

    auto program_key = GetOrCreateCompilationKey(dtensor_recv);
    builder.setInsertionPoint(dtensor_recv);
    recv_xla_op = builder.create<mlir::TF::_XlaRecvAtHostV2Op>(
        dtensor_recv.getLoc(), output_types,
        /*dynamic_key=*/program_key, device_ordinal, dtensor_recv.keyAttr());
  } else {
    // Create XlaRecvFromHost op.
    recv_xla_op = builder.create<mlir::TF::XlaRecvFromHostOp>(
        dtensor_recv.getLoc(), output_type,
        ConvertTypeToTensorShapeAttr(dtensor_recv.getType()),
        dtensor_recv.keyAttr());
  }

  assert(recv_xla_op);

  // TODO(hongjunchoi): After receiving tensor, convert tensor to requested
  // layout with EmitRelayout.
  return recv_xla_op;
}

// Lowers a DTensorSend Op from a CPU to a TF Send op.
StatusOr<mlir::Operation*> LowerDTensorSendFromCPUToTFOp(
    const Layout& send_input_layout, mlir::Value send_input,
    mlir::TF::DTensorSend dtensor_send) {
  mlir::OpBuilder builder(dtensor_send);
  builder.setInsertionPointAfter(send_input.getDefiningOp());

  llvm::SmallVector<mlir::Value, 4> value_to_send{send_input};

  // Create multiple send from host. There should be #number of local
  // devices(in target mesh) number of sends.
  absl::Span<const std::string> sending_devices =
      send_input_layout.mesh().local_devices();

  Layout target_layout = dtensor_send.target_layout();
  absl::Span<const std::string> receiving_devices =
      target_layout.mesh().local_devices();

  std::string tensor_name = dtensor_send.key().str();

  mlir::Operation* lowered_send_op;
  for (size_t i = 0; i < receiving_devices.size(); ++i)
    lowered_send_op = builder.create<mlir::TF::_HostSendOp>(
        send_input.getLoc(), dtensor_send.input(), tensor_name,
        sending_devices[0],
        /*send_device_incarnation=*/0, receiving_devices[i]);

  dtensor_send.erase();
  return lowered_send_op;
}

// Lowers DTensorRecv op to TF Recv Op.
StatusOr<mlir::Operation*> LowerDTensorRecvFromCPUToTFOp(
    const Mesh& send_mesh, mlir::TF::DTensorRecv dtensor_recv) {
  const Layout& recv_layout = dtensor_recv.layout();

  auto recv_cluster =
      dtensor_recv->getParentOfType<mlir::tf_device::ClusterOp>();

  mlir::OpBuilder builder(&recv_cluster.GetBody().front());
  llvm::SmallVector<mlir::Type, 4> output_types{dtensor_recv.getType()};
  builder.setInsertionPoint(dtensor_recv);
  std::string tensor_name = dtensor_recv.key().str();
  absl::Span<const std::string> sending_devices = send_mesh.local_devices();
  absl::Span<const std::string> receiving_devices =
      recv_layout.mesh().local_devices();

  mlir::Operation* lowered_recv_op;
  mlir::Location loc = dtensor_recv.getLoc();
  for (size_t i = 0; i < receiving_devices.size(); ++i)
    lowered_recv_op = builder.create<mlir::TF::_HostRecvOp>(
        loc, dtensor_recv.getType(), tensor_name, sending_devices[0],
        /*send_device_incarnation=*/0, receiving_devices[i]);

  // Replace dtensor_recv with newly created recv op and remove DTensorRecv op.
  assert(lowered_recv_op);
  dtensor_recv.replaceAllUsesWith(lowered_recv_op);
  dtensor_recv.erase();
  return lowered_recv_op;
}

}  // namespace dtensor
}  // namespace tensorflow
