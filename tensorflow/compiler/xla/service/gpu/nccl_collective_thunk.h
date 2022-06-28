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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTh() {
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


#include "absl/synchronization/mutex.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#if XLA_ENABLE_XCCL
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#endif  // XLA_ENABLE_XCCL

struct ncclComm;
using ncclComm_t = ncclComm*;

namespace xla {
namespace gpu {

class NcclClique;

struct NcclCollectiveConfig {
  NcclCollectiveConfig();
  NcclCollectiveConfig(NcclCollectiveConfig&&);
  ~NcclCollectiveConfig();

  NcclCollectiveConfig& operator=(NcclCollectiveConfig&&);

  int64_t operand_count;
  std::vector<PrimitiveType> operand_element_type;
  std::vector<ReplicaGroup> replica_groups;
  RendezvousKey::CollectiveOpKind collective_op_kind;
  int64_t op_id;
  CollectiveOpGroupMode group_mode;

  template <typename OpT>
  void SetCollectiveOpKindAndID(OpT op);
  bool IsDegenerate(int64_t replica_count, int64_t partition_count) const;
};

template <typename OpT>
void NcclCollectiveConfig::SetCollectiveOpKindAndID(OpT op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTh mht_0(mht_0_v, 231, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h", "NcclCollectiveConfig::SetCollectiveOpKindAndID");

  if (op.channel_id()) {
    collective_op_kind = RendezvousKey::kCrossModule;
    op_id = static_cast<int64_t>(op.channel_id()->handle().getInt());
  } else {
    collective_op_kind = RendezvousKey::kCrossReplica;
    mlir::ModuleOp parent = op->template getParentOfType<mlir::ModuleOp>();
    mlir::IntegerAttr unique_id =
        parent->getAttrOfType<mlir::IntegerAttr>("hlo.unique_id");
    op_id = static_cast<int64_t>(unique_id.getInt());
  }
}

template <typename OpT>
NcclCollectiveConfig GetNcclCollectiveConfigForMlir(
    OpT op, absl::optional<bool> use_global_device_ids) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_collective_thunkDTh mht_1(mht_1_v, 249, "", "./tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h", "GetNcclCollectiveConfigForMlir");

  NcclCollectiveConfig config;
  config.operand_count = op.operands().size();
  config.operand_element_type.reserve(config.operand_count);
  for (int i = 0; i < config.operand_count; i++) {
    const Shape shape = GetShape(op.operands()[i]);
    config.operand_element_type.push_back(shape.element_type());
  }
  config.replica_groups =
      ConvertReplicaGroups(op.replica_groups()).ValueOrDie();
  config.SetCollectiveOpKindAndID(op);
  config.group_mode = GetCollectiveOpGroupMode(op.channel_id().hasValue(),
                                               use_global_device_ids)
                          .ValueOrDie();
  return config;
}

// Thunk base class for NCCL collective operations.
class NcclCollectiveThunk : public Thunk {
 public:
  using Thunk::Thunk;

  struct Buffer {
    int64_t element_count;
    BufferAllocation::Slice source_buffer;
    BufferAllocation::Slice destination_buffer;
  };

  // Returns whether NCCL operations appear possible to perform; e.g. if we
  // haven't done a build with the CUDA compiler enabled, we can't compile the
  // NCCL header, and thus this will be false.
  //
  // When this is false, the ExecuteOnStream() call will simply return a status
  // error.
  static bool NcclIsEnabled();

  Status ExecuteOnStream(const ExecuteParams& params) override;

 protected:
  virtual Status RunNcclCollective(const ExecuteParams& params,
                                   ncclComm_t comm) = 0;
  virtual const NcclCollectiveConfig& config() const = 0;

  // Logging support.
  std::string GetDeviceString(const ExecuteParams& params) const;

 private:
#if XLA_ENABLE_XCCL
  bool first_call_to_execute_ = true;
#endif  // XLA_ENABLE_XCCL
};

// Returns if the given data type is supported by NCCL.
// Note: Keep this in sync with ToNcclDataType().
bool IsTypeSupportedByNccl(PrimitiveType element_type);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_
