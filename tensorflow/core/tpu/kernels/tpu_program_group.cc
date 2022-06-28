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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc() {
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
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"

#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"

namespace tensorflow {
namespace tpu {
namespace {
namespace se_tpu = ::stream_executor::tpu;
using stream_executor::port::Status;
}  // namespace

TPUExecutableInfoProto TpuProgramGroup::ConstructExecutableInfo(
    const XLA_TpuProgram* xla_tpu_program) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::ConstructExecutableInfo");

  VLOG(1) << "ConstructExecutableInfo";
  TpuSerializedProto serialized_executable_info = {};
  StatusHelper status;
  OpsApiFn()->TpuProgram_GetExecutableInfoFn(
      xla_tpu_program, &serialized_executable_info, status.c_status);
  TPUExecutableInfoProto executable_info;
  if (status.ok()) {
    executable_info = se_tpu::DeserializeProto<TPUExecutableInfoProto>(
        serialized_executable_info);
    StreamExecutor_Tpu_FreeSerializedProto(&serialized_executable_info);
  }
  return executable_info;
}

TPUHostTransferInfoProto TpuProgramGroup::ConstructHostTransferInfo(
    const XLA_TpuProgram* xla_tpu_program) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::ConstructHostTransferInfo");

  VLOG(1) << "ConstructHostTransferInfo";
  TpuSerializedProto serialized_host_transfer_info = {};
  StatusHelper status;
  OpsApiFn()->TpuProgram_GetHostTransferInfoFn(
      xla_tpu_program, &serialized_host_transfer_info, status.c_status);
  TPUHostTransferInfoProto host_transfer_info;
  if (status.ok()) {
    host_transfer_info = se_tpu::DeserializeProto<TPUHostTransferInfoProto>(
        serialized_host_transfer_info);
    StreamExecutor_Tpu_FreeSerializedProto(&serialized_host_transfer_info);
  }
  return host_transfer_info;
}

xla::HloProto TpuProgramGroup::ConstructHloMetadata(
    const XLA_TpuProgram* xla_tpu_program) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::ConstructHloMetadata");

  VLOG(1) << "ConstructHloMetadata";
  TpuSerializedProto serialized_hlo_metadata = {};
  StatusHelper status;
  OpsApiFn()->TpuProgram_GetHloMetadataFn(
      xla_tpu_program, &serialized_hlo_metadata, status.c_status);
  xla::HloProto hlo_metadata;
  if (status.ok()) {
    hlo_metadata =
        se_tpu::DeserializeProto<xla::HloProto>(serialized_hlo_metadata);
    StreamExecutor_Tpu_FreeSerializedProto(&serialized_hlo_metadata);
  }
  return hlo_metadata;
}

void TpuProgramGroup::Initialize(
    absl::Span<XLA_TpuProgram* const> xla_tpu_programs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_3(mht_3_v, 263, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::Initialize");

  CHECK_GT(xla_tpu_programs.size(), 0);
  CHECK_EQ(program_count(), 0) << "Reinitialization of an existing "
                                  "`TpuProgramGroup` instance is prohibited.";
  set_tpu_programs(xla_tpu_programs);

  CHECK_EQ(tpu_program_fingerprints_.size(), 0);
  set_fingerprints();

  std::vector<bool> may_modify_variables_array(tpu_programs_.size(), false);
  std::vector<TPUExecutableInfoProto> executable_infos(tpu_programs_.size());
  std::vector<TPUHostTransferInfoProto> host_transfer_infos(
      tpu_programs_.size());
  std::vector<xla::HloProto> hlo_metadatas(tpu_programs_.size());
  for (size_t i = 0; i < tpu_programs_.size(); ++i) {
    const XLA_TpuProgram* xla_tpu_program = tpu_programs_[i];
    bool may_modify_variables;
    OpsApiFn()->TpuProgram_GetMayModifyVariablesFn(xla_tpu_program,
                                                   &may_modify_variables);
    may_modify_variables_array[i] = may_modify_variables;
    executable_infos[i] = ConstructExecutableInfo(xla_tpu_program);
    host_transfer_infos[i] = ConstructHostTransferInfo(xla_tpu_program);
    hlo_metadatas[i] = ConstructHloMetadata(xla_tpu_program);
  }

  may_modify_variables_ = may_modify_variables_array;
  executable_infos_ = executable_infos;
  host_transfer_infos_ = host_transfer_infos;
  hlo_metadatas_ = hlo_metadatas;
  RefreshHloMetadatasPtrs();
}

bool TpuProgramGroup::has_sharding_program() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_4(mht_4_v, 298, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::has_sharding_program");

  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    if (!OpsApiFn()->TpuProgram_HasShardingFn(tpu_program)) {
      return false;
    }
  }
  return true;
}

size_t TpuProgramGroup::program_count() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_5(mht_5_v, 310, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::program_count");
 return tpu_programs_.size(); }

int64_t TpuProgramGroup::program_size() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_6(mht_6_v, 315, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::program_size");

  int64_t total_size = 0;
  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    total_size += OpsApiFn()->TpuProgram_GetProgramSizeFn(tpu_program);
  }
  return total_size;
}

bool TpuProgramGroup::LogProgramMemorySummary() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_7(mht_7_v, 326, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::LogProgramMemorySummary");

  bool success = true;
  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    success &= OpsApiFn()->TpuProgram_LogProgramMemorySummaryFn(tpu_program);
  }
  return success;
}

void TpuProgramGroup::UnloadAndDestroyPrograms() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_8(mht_8_v, 337, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::UnloadAndDestroyPrograms");

  for (XLA_TpuProgram* tpu_program : tpu_programs_) {
    StatusHelper status;
    OpsApiFn()->TpuProgram_UnloadAndDestroyFn(tpu_program, status.c_status);
    auto s = status.status();
    if (!s.ok()) {
      LOG(ERROR) << "TpuProgramGroup::UnloadPrograms(): " << s.ToString();
    }
  }
  tpu_programs_.clear();
}

TpuProgramGroup::TpuProgramGroup(TpuProgramGroup&& other)
    : may_modify_variables_(std::move(other.may_modify_variables_)),
      tpu_programs_(std::move(other.tpu_programs_)),
      executable_infos_(std::move(other.executable_infos_)),
      host_transfer_infos_(std::move(other.host_transfer_infos_)),
      hlo_metadatas_(std::move(other.hlo_metadatas_)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_9(mht_9_v, 357, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::TpuProgramGroup");

  RefreshHloMetadatasPtrs();
}

void TpuProgramGroup::set_hlo_metadatas(
    absl::Span<const xla::HloProto> hlo_metadatas) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_10(mht_10_v, 365, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::set_hlo_metadatas");

  hlo_metadatas_.resize(hlo_metadatas.size());
  for (size_t i = 0; i < hlo_metadatas.size(); ++i) {
    hlo_metadatas_[i] = hlo_metadatas[i];
  }
  RefreshHloMetadatasPtrs();
}

absl::Span<const xla::HloProto* const> TpuProgramGroup::hlo_metadatas() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_11(mht_11_v, 376, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::hlo_metadatas");

  return hlo_metadatas_ptrs_;
}

const xla::HloProto* TpuProgramGroup::hlo_metadata(int index) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_12(mht_12_v, 383, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::hlo_metadata");

  CHECK_GE(index, 0);
  CHECK_LT(index, hlo_metadatas_ptrs_.size());
  return hlo_metadatas_ptrs_[index];
}

void TpuProgramGroup::RefreshHloMetadatasPtrs() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_13(mht_13_v, 392, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::RefreshHloMetadatasPtrs");

  hlo_metadatas_ptrs_.reserve(hlo_metadatas_.size());
  for (const auto& hlo_metadata_internal_ : hlo_metadatas_) {
    hlo_metadatas_ptrs_.push_back(&hlo_metadata_internal_);
  }
}

const std::vector<bool>& TpuProgramGroup::may_modify_variables_list() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_14(mht_14_v, 402, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::may_modify_variables_list");

  return may_modify_variables_;
}

void TpuProgramGroup::set_may_modify_variables(
    const std::vector<bool>& may_modify_variables) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_15(mht_15_v, 410, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::set_may_modify_variables");

  may_modify_variables_ = may_modify_variables;
}

bool TpuProgramGroup::may_modify_variables(int index) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_16(mht_16_v, 417, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::may_modify_variables");

  CHECK_GE(index, 0);
  CHECK_LT(index, tpu_programs_.size());
  bool may_modify_variables;
  OpsApiFn()->TpuProgram_GetMayModifyVariablesFn(tpu_programs_[index],
                                                 &may_modify_variables);
  return may_modify_variables;
}

const std::vector<XLA_TpuProgram*>& TpuProgramGroup::tpu_programs() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_17(mht_17_v, 429, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::tpu_programs");

  return tpu_programs_;
}

const std::vector<std::string>& TpuProgramGroup::fingerprints() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_18(mht_18_v, 436, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::fingerprints");

  return tpu_program_fingerprints_;
}

void TpuProgramGroup::set_fingerprints() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_19(mht_19_v, 443, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::set_fingerprints");

  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    TpuProgramFingerprint fingerprint =
        OpsApiFn()->TpuProgram_GetFingerprintFn(tpu_program);
    tpu_program_fingerprints_.emplace_back(
        std::string(fingerprint.bytes, fingerprint.size));
    OpsApiFn()->TpuProgram_DestroyFingerprintFn(fingerprint);
  }
}

const std::string& TpuProgramGroup::fingerprint(int index) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_20(mht_20_v, 456, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::fingerprint");

  return fingerprints().at(index);
}

const XLA_TpuProgram* TpuProgramGroup::tpu_program(int index) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_21(mht_21_v, 463, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::tpu_program");

  CHECK_GE(index, 0);
  CHECK_LT(index, tpu_programs_.size());
  return tpu_programs_[index];
}

void TpuProgramGroup::set_tpu_programs(
    absl::Span<XLA_TpuProgram* const> tpu_programs) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_22(mht_22_v, 473, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::set_tpu_programs");

  tpu_programs_.resize(tpu_programs.size());
  for (size_t i = 0; i < tpu_programs.size(); ++i) {
    tpu_programs_[i] = tpu_programs[i];
  }
}

const TPUExecutableInfoProto& TpuProgramGroup::executable_info(
    int index) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_23(mht_23_v, 484, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::executable_info");

  CHECK_GE(index, 0);
  CHECK_LT(index, executable_infos_.size());
  return executable_infos_[index];
}

const TPUHostTransferInfoProto& TpuProgramGroup::host_transfer_info(
    int index) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_24(mht_24_v, 494, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::host_transfer_info");

  CHECK_GE(index, 0);
  CHECK_LT(index, host_transfer_infos_.size());
  return host_transfer_infos_[index];
}

/*static*/
Status TpuProgramGroup::CompileAndBuild(
    const TpuCompilationRequestProto& compilation_request,
    const XLA_TpuMeshState* mesh_state,
    TpuProgramGroupInterface* tpu_program_group_interface) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_25(mht_25_v, 507, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::CompileAndBuild");

  se_tpu::SerializedProto serialized_compilation_request =
      se_tpu::SerializeProto(compilation_request);
  auto cleanup = gtl::MakeCleanup([serialized_compilation_request] {
    se_tpu::SerializedProto_Free(serialized_compilation_request);
  });
  size_t count = 0;
  XLA_TpuProgram** xla_tpu_programs = nullptr;
  StatusHelper status;
  OpsApiFn()->TpuCompile_CompileAndBuildFn(serialized_compilation_request,
                                           mesh_state, &xla_tpu_programs,
                                           &count, status.c_status);
  if (!status.ok()) {
    VLOG(1) << "Run CompileAndBuild failed.";
    return status.status();
  }

  // SPMD could return 1 result for all partitions.
  TF_RET_CHECK(count == 1 ||
               count == compilation_request.metadata().num_cores_per_replica());

  VLOG(1) << "Initialize TpuProgramGroup.";
  TpuProgramGroup* tpu_program_group =
      tensorflow::down_cast<TpuProgramGroup*>(tpu_program_group_interface);
  tpu_program_group->Initialize(
      absl::MakeConstSpan(&xla_tpu_programs[0], count));
  OpsApiFn()->TpuProgram_FreeArrayFn(xla_tpu_programs);
  return status.status();
}

/*static*/
Status TpuProgramGroup::CompileAndBuild(
    const xrt::XLAComputation& xrt_computation_proto,
    const XLA_TpuMeshState* mesh_state,
    TpuProgramGroupInterface* tpu_program_group_interface) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_26(mht_26_v, 544, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::CompileAndBuild");

  se_tpu::SerializedProto serialized_compilation_request =
      se_tpu::SerializeProto(xrt_computation_proto);
  auto cleanup = gtl::MakeCleanup([serialized_compilation_request] {
    se_tpu::SerializedProto_Free(serialized_compilation_request);
  });
  size_t count = 0;
  XLA_TpuProgram** xla_tpu_programs = nullptr;
  StatusHelper status;
  OpsApiFn()->TpuCompile_XrtCompileAndBuildFn(serialized_compilation_request,
                                              mesh_state, &xla_tpu_programs,
                                              &count, status.c_status);
  if (!status.ok()) {
    VLOG(1) << "Run CompileAndBuild failed.";
    return status.status();
  }

  // SPMD could return 1 result for all partitions.
  int num_cores_per_replica =
      xrt_computation_proto.config().num_cores_per_replica()
          ? xrt_computation_proto.config().num_cores_per_replica()
          : 1;
  TF_RET_CHECK(count == 1 || count == num_cores_per_replica);
  VLOG(1) << "Initialize TpuProgramGroup.";
  TpuProgramGroup* tpu_program_group =
      tensorflow::down_cast<TpuProgramGroup*>(tpu_program_group_interface);
  tpu_program_group->Initialize(
      absl::MakeConstSpan(&xla_tpu_programs[0], count));
  OpsApiFn()->TpuProgram_FreeArrayFn(xla_tpu_programs);
  return status.status();
}

std::vector<XLA_TpuProgram*> TpuProgramGroup::tpu_programs(
    TpuProgramShardingType sharding_type) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_27(mht_27_v, 580, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::tpu_programs");

  std::vector<XLA_TpuProgram*> tpu_programs;
  tpu_programs.reserve(tpu_programs_.size());
  for (size_t i = 0; i < tpu_programs_.size(); ++i) {
    if (OpsApiFn()->TpuProgram_HasShardingFn(tpu_programs_[i])) {
      tpu_programs.push_back(OpsApiFn()->TpuProgram_GetTpuProgramFn(
          tpu_programs_[i], sharding_type));
      CHECK_NE(tpu_programs[i], nullptr);
    }
  }
  return tpu_programs;
}

Status TpuProgramGroup::DeserializeFromRpcResponseProtos(
    const std::vector<TpuSerializedProto>& rpc_response_protos) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_28(mht_28_v, 597, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::DeserializeFromRpcResponseProtos");

  std::vector<XLA_TpuProgram*> tpu_programs;
  tpu_programs.resize(rpc_response_protos.size());

  for (size_t i = 0; i < rpc_response_protos.size(); ++i) {
    StatusHelper status;
    auto* xla_tpu_program = OpsApiFn()->TpuProgram_NewFn();
    OpsApiFn()->TpuProgram_DeserializeFromGetTpuProgramResponseProtoFn(
        rpc_response_protos[i], xla_tpu_program, status.c_status);
    if (!status.status().ok()) {
      OpsApiFn()->TpuProgram_FreeFn(xla_tpu_program);
      return status.status();
    }
    tpu_programs[i] = xla_tpu_program;
  }

  Initialize(tpu_programs);
  return Status::OK();
}

Status TpuProgramGroup::SerializeExecutable(
    int index, TpuExecutableSerializedProto* executable) const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_29(mht_29_v, 621, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::SerializeExecutable");

  CHECK_GE(index, 0);
  CHECK_LT(index, tpu_programs_.size());
  StatusHelper status;
  OpsApiFn()->TpuProgram_SerializeTpuExecutableFn(tpu_programs_[index],
                                                  executable, status.c_status);
  return status.status();
}

Status TpuProgramGroup::SerializeCompilerMetadata(
    int index, CompilerMetadataSerializedProto* compiler_metadata) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_program_groupDTcc mht_30(mht_30_v, 634, "", "./tensorflow/core/tpu/kernels/tpu_program_group.cc", "TpuProgramGroup::SerializeCompilerMetadata");

  CHECK_GE(index, 0);
  CHECK_LT(index, tpu_programs_.size());
  StatusHelper status;
  OpsApiFn()->TpuProgram_SerializeCompilerMetadataFn(
      tpu_programs_[index], compiler_metadata, status.c_status);
  return status.status();
}
}  // namespace tpu
}  // namespace tensorflow
