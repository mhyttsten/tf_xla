/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_GROUP_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_GROUP_H_
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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh() {
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


#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {
namespace tpu {

class TpuAotCompilationOptions : public xla::AotCompilationOptions {
 public:
  explicit TpuAotCompilationOptions(int64_t replica_count)
      : num_cores_(0), replica_count_(replica_count) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/tpu/kernels/tpu_program_group.h", "TpuAotCompilationOptions");
}

  // Returns the ID of the platform to which these options apply.
  se::Platform::Id PlatformId() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh mht_1(mht_1_v, 217, "", "./tensorflow/core/tpu/kernels/tpu_program_group.h", "PlatformId");

    LOG(FATAL) << "Not implemented.";
    return nullptr;
  };

  void set_num_cores(int64_t tpu_cores) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh mht_2(mht_2_v, 225, "", "./tensorflow/core/tpu/kernels/tpu_program_group.h", "set_num_cores");
 num_cores_ = tpu_cores; }
  int64_t replica_count() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh mht_3(mht_3_v, 229, "", "./tensorflow/core/tpu/kernels/tpu_program_group.h", "replica_count");
 return replica_count_; }
  int64_t num_cores() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh mht_4(mht_4_v, 233, "", "./tensorflow/core/tpu/kernels/tpu_program_group.h", "num_cores");
 return num_cores_; }

  void set_allow_separate_sharding_programs(bool allow) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh mht_5(mht_5_v, 238, "", "./tensorflow/core/tpu/kernels/tpu_program_group.h", "set_allow_separate_sharding_programs");

    allow_separate_sharding_programs_ = allow;
  }
  bool allow_separate_sharding_programs() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh mht_6(mht_6_v, 244, "", "./tensorflow/core/tpu/kernels/tpu_program_group.h", "allow_separate_sharding_programs");

    return allow_separate_sharding_programs_;
  }

  const std::vector<xla::HloModuleConfig::ShardableValueUpdatePair>
  shardable_value_update_pairs() const {
    return shardable_value_update_pairs_;
  }
  void set_shardable_value_update_pairs(
      std::vector<xla::HloModuleConfig::ShardableValueUpdatePair> pairs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTh mht_7(mht_7_v, 256, "", "./tensorflow/core/tpu/kernels/tpu_program_group.h", "set_shardable_value_update_pairs");

    shardable_value_update_pairs_ = std::move(pairs);
  }

 private:
  int64_t num_cores_;
  int64_t replica_count_;

  // Whether to allow the compiler to create separte sharding and unsharding
  // programs, and modify the original program's input/output sharded size. This
  // is used for XLA-chosen sharding on parameters without an on-device loop:
  // the caller can invoke sharding first, then (repeatedly) invoke the sharded
  // main program, and finally invoke the unsharding program when it needs the
  // full output.
  bool allow_separate_sharding_programs_ = false;

  // The list of input/output pairs in the main program that could be sharded.
  std::vector<xla::HloModuleConfig::ShardableValueUpdatePair>
      shardable_value_update_pairs_;
};

class TpuProgramGroup : public TpuProgramGroupInterface {
 public:
  using Status = ::stream_executor::port::Status;

  // Compiles Mlir or TF function computation by lowering into HLO IR and
  // returns TPU programs ready for execution.
  static Status CompileAndBuild(
      const TpuCompilationRequestProto& compilation_request,
      const XLA_TpuMeshState* mesh_state,
      TpuProgramGroupInterface* tpu_program_group_interface);

  // Compiles HLO IR and returns TPU programs ready for execution.
  static Status CompileAndBuild(
      const xrt::XLAComputation& xrt_computation_proto,
      const XLA_TpuMeshState* mesh_state,
      TpuProgramGroupInterface* tpu_program_group_interface);

  // Initializes `TpuProgramGroup` object with `xla_tpu_programs`.
  void Initialize(absl::Span<XLA_TpuProgram* const> xla_tpu_programs);

  TpuProgramGroup() = default;
  TpuProgramGroup(TpuProgramGroup&& other);
  TpuProgramGroup& operator=(TpuProgramGroup&&) = delete;

  bool has_sharding_program() const override;

  size_t program_count() const override;

  int64_t program_size() const override;

  bool LogProgramMemorySummary() override;

  void UnloadAndDestroyPrograms() override;

  const std::vector<bool>& may_modify_variables_list() const override;
  void set_may_modify_variables(const std::vector<bool>& may_modify_variables);
  bool may_modify_variables(int index) const override;

  const std::vector<std::string>& fingerprints() const;
  void set_fingerprints();

  const std::string& fingerprint(int index) const override;

  const std::vector<XLA_TpuProgram*>& tpu_programs() const;
  std::vector<XLA_TpuProgram*> tpu_programs(TpuProgramShardingType type) const;
  const XLA_TpuProgram* tpu_program(int index) const override;
  void set_tpu_programs(absl::Span<XLA_TpuProgram* const> tpu_programs);

  const TPUExecutableInfoProto& executable_info(int index) const override;

  const TPUHostTransferInfoProto& host_transfer_info(int index) const override;
  void set_hlo_metadatas(absl::Span<const xla::HloProto> hlo_metadatas);
  const xla::HloProto* hlo_metadata(int index) const;
  absl::Span<const xla::HloProto* const> hlo_metadatas() const override;

  // Deserializes `GetTpuProgramResponse` protos from remote cache.
  Status DeserializeFromRpcResponseProtos(
      const std::vector<TpuSerializedProto>& rpc_response_protos);

  // Serializes executable proto from the TPU program for the given core
  // `index`.
  Status SerializeExecutable(int index,
                             TpuExecutableSerializedProto* executable) const;

  // Serializes compiler metadata of the TPU program for the given core `index`.
  Status SerializeCompilerMetadata(
      int index, CompilerMetadataSerializedProto* compiler_metadata) const;

  // Serializes host compute metadata of the TPU program for the given core
  // `index`.
  Status SerializeHostComputeMetadata(
      int index,
      HostComputeMetadataSerializedProto* host_compute_metadata) const;

 private:
  TPUExecutableInfoProto ConstructExecutableInfo(
      const XLA_TpuProgram* tpu_program);
  TPUHostTransferInfoProto ConstructHostTransferInfo(
      const XLA_TpuProgram* tpu_program);
  xla::HloProto ConstructHloMetadata(const XLA_TpuProgram* tpu_program);

  // Update `hlo_metadatas__ptrs_` array from `hlo_metadatas_`. This needs to be
  // called on `hlo_metadatas_` change(s).
  void RefreshHloMetadatasPtrs();

  std::vector<bool> may_modify_variables_;
  std::vector<std::string> tpu_program_fingerprints_;

  std::vector<XLA_TpuProgram*> tpu_programs_;  // Not owned.
  std::vector<TPUExecutableInfoProto> executable_infos_;
  std::vector<TPUHostTransferInfoProto> host_transfer_infos_;

  // To be consistent with the TpuProgramGroupInterface::hlo_metadatas()
  // signature, we store HloProto values in hlo_metadatas_ when
  // set_hlo_metadata(...) is called, and return their pointers from
  // hlo_metadatas_ptrs_ when hlo_metadatas() is called. hlo_metadata_ptrs_ is
  // refreshed whenever hlo_metadatas_ is set or the object is moved.
  std::vector<xla::HloProto> hlo_metadatas_;  // Owned.
  std::vector<const xla::HloProto*> hlo_metadatas_ptrs_;

  TF_DISALLOW_COPY_AND_ASSIGN(TpuProgramGroup);
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_GROUP_H_
