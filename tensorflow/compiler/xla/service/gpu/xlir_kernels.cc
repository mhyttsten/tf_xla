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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc() {
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

// Copyright 2021 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements the XLIR kernels.

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "llvm/Support/Error.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tfrt/gpu/gpu_types.h"  // from @tf_runtime
#include "tfrt/gpu/kernels/kernels_detail.h"  // from @tf_runtime
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

#if XLA_ENABLE_XCCL
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tfrt/gpu/wrapper/ccl_wrapper.h"  // from @tf_runtime
#endif  // XLA_ENABLE_XCCL

namespace xla {
namespace gpu {

static llvm::Expected<tfrt::gpu::GpuModule> ModuleLoad(
    tfrt::Argument<tfrt::gpu::GpuContext> context,
    const tfrt::ExecutionContext& exec_ctx) {
  const GpuModuleData* gpu_module_data =
      exec_ctx.request_ctx()->GetDataIfExists<GpuModuleData>();

  if (gpu_module_data == nullptr) {
    return tfrt::MakeStringError(
        "No GpuModuleData resource found in the request context.");
  }
  llvm::ArrayRef<uint8_t> blob = gpu_module_data->blob;

  if (blob.empty())
    return tfrt::MakeStringError("blob must be null-terminated");

  auto current = tfrt::gpu::wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();

  auto module = tfrt::gpu::wrapper::ModuleLoadData(*current, blob.data());
  if (!module) return module.takeError();

  // Resolve constants.
  for (const auto& constant : gpu_module_data->constants) {
    if (constant.content.empty()) continue;

    auto global = tfrt::gpu::wrapper::ModuleGetGlobal(
        module->get(), constant.symbol_name.data());
    if (!global) return global.takeError();

    const void* constant_content =
        static_cast<const void*>(constant.content.data());
    tfrt::gpu::GpuPointer constant_content_ptr(
        const_cast<void*>(constant_content), current->platform());

    if (auto error = tfrt::gpu::wrapper::MemcpyAsync(
            *current, global->base, constant_content_ptr, global->size_bytes,
            tfrt::gpu::wrapper::Stream(nullptr, current->platform()))) {
      return error;
    }
  }
  return tfrt::gpu::GpuModule(context.ValueRef(), std::move(*module));
}

static llvm::Expected<DeviceAssignment::LogicalID> GetLogicalId(
    const tfrt::ExecutionContext& exec_ctx) {
  auto* xla_gpu_params =
      exec_ctx.request_ctx()->GetDataIfExists<XlaGpuParams>();
  if (!xla_gpu_params) {
    return tfrt::MakeStringError("Failed to get XlaGpuParams");
  }

  StatusOr<DeviceAssignment::LogicalID> current_logical_id_or =
      xla_gpu_params->device_assn->LogicalIdForDevice(
          xla_gpu_params->global_device_id);
  if (!current_logical_id_or.ok()) {
    return tfrt::MakeStringError(
        current_logical_id_or.status().error_message());
  }
  return current_logical_id_or.ValueOrDie();
}

static llvm::Error ReplicaId(const tfrt::gpu::GpuStream& stream,
                             const tfrt::gpu::GpuBuffer& output,
                             const tfrt::ExecutionContext& exec_ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc mht_0(mht_0_v, 277, "", "./tensorflow/compiler/xla/service/gpu/xlir_kernels.cc", "ReplicaId");

  auto current_logical_id = GetLogicalId(exec_ctx);
  if (!current_logical_id) return current_logical_id.takeError();

  auto current = tfrt::gpu::wrapper::CtxSetCurrent(stream.context()->get());
  if (!current) return current.takeError();

  return tfrt::gpu::wrapper::MemsetD32Async(*current, output.pointer(),
                                            current_logical_id->replica_id, 1,
                                            stream.get());
}

static llvm::Error PartitionId(const tfrt::gpu::GpuStream& stream,
                               const tfrt::gpu::GpuBuffer& output,
                               const tfrt::ExecutionContext& exec_ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc mht_1(mht_1_v, 294, "", "./tensorflow/compiler/xla/service/gpu/xlir_kernels.cc", "PartitionId");

  auto current_logical_id = GetLogicalId(exec_ctx);
  if (!current_logical_id) return current_logical_id.takeError();

  auto current = tfrt::gpu::wrapper::CtxSetCurrent(stream.context()->get());
  if (!current) return current.takeError();

  return tfrt::gpu::wrapper::MemsetD32Async(*current, output.pointer(),
                                            current_logical_id->computation_id,
                                            1, stream.get());
}

#if XLA_ENABLE_XCCL
static tfrt::AsyncValueRef<tfrt::gpu::GpuCclHandle> CclCreate(
    tfrt::Argument<tfrt::gpu::GpuContext> context,
    tfrt::Attribute<int64_t> group_mode_attr,
    tfrt::Attribute<int64_t> op_id_attr,
    tfrt::RemainingAttributes replica_groups_attrs,
    const tfrt::ExecutionContext& exec_ctx) {
  auto* xla_gpu_params =
      exec_ctx.request_ctx()->GetDataIfExists<XlaGpuParams>();
  if (!xla_gpu_params) {
    return tfrt::MakeErrorAsyncValueRef("Failed to get XlaGpuParams");
  }

  std::vector<ReplicaGroup> replica_groups;
  replica_groups.reserve(replica_groups_attrs.size());
  for (int i = 0; i < replica_groups_attrs.size(); ++i) {
    ReplicaGroup replica_group;
    for (auto replica_id :
         replica_groups_attrs.GetArrayAttribute<int64_t>(i).data()) {
      replica_group.add_replica_ids(replica_id);
    }
    replica_groups.push_back(replica_group);
  }

  StatusOr<std::vector<GlobalDeviceId>> participants_or =
      GetParticipatingDevices(
          xla_gpu_params->global_device_id, *xla_gpu_params->device_assn,
          replica_groups, static_cast<CollectiveOpGroupMode>(*group_mode_attr));
  if (!participants_or.ok()) {
    return tfrt::MakeErrorAsyncValueRef(
        participants_or.status().error_message());
  }
  std::vector<GlobalDeviceId> participants =
      std::move(participants_or.ValueOrDie());

  if (IsGlobalNcclConfig() &&
      (participants.size() != xla_gpu_params->device_assn->replica_count())) {
    return tfrt::MakeErrorAsyncValueRef(
        "Partial replica groups are not allowed when using NCCL_COMM_ID "
        "environment configuration.");
  }

  auto it = absl::c_find(participants, xla_gpu_params->global_device_id);
  if (it == participants.end()) {
    return tfrt::MakeErrorAsyncValueRef(
        "Device ID not found among participants.");
  }
  int rank = it - participants.begin();

  OpId op_id(*op_id_attr);
  size_t num_local_participants = GetNumLocalParticipants(
      participants, /*local_devices=*/xla_gpu_params->gpu_global_device_ids);

  bool is_local = participants.size() == num_local_participants;
  StatusOr<const NcclUniqueIdCallback*> unique_id_callback_or =
      GetNcclUniqueIdCallback(xla_gpu_params->nccl_unique_id_callback,
                              is_local);
  if (!unique_id_callback_or.ok()) {
    return tfrt::MakeErrorAsyncValueRef(
        unique_id_callback_or.status().error_message());
  }
  const NcclUniqueIdCallback* unique_id_callback =
      unique_id_callback_or.ValueOrDie();

  // AcquireNcclComm() blocks to wait for all participants and therefore needs
  // to run inside a blocking task.
  return tfrt::RunBlockingWork(
      exec_ctx.host(),
      tfrt::gpu::DestroyCapturesOnInvoke(
          [=, participants = std::move(participants),
           context = context.ValueRef()]() mutable
          -> llvm::Expected<tfrt::gpu::GpuCclHandle> {
            auto current = tfrt::gpu::wrapper::CtxSetCurrent(context->get());
            if (!current) return current.takeError();

            StatusOr<NcclComm::Lock> comm_or = AcquireNcclComm(
                xla_gpu_params->run_id, op_id, std::move(participants),
                num_local_participants, *unique_id_callback, rank);
            if (!comm_or.ok())
              return tfrt::MakeStringError(comm_or.status().error_message());

            auto* comm_ptr = comm_or->get();
            auto comm_deleter =
                [comm = std::move(comm_or.ValueOrDie())](
                    tfrt::gpu::wrapper::CclComm /*ccl_comm*/) mutable {
                  comm.reset();
                };

            return tfrt::gpu::GpuCclHandle(
                std::move(context),
                tfrt::gpu::wrapper::OwningCclComm(
                    {*comm_ptr, current->platform()}),
                std::move(comm_deleter));
          }));
}

static tfrt::AsyncValueRef<tfrt::Chain> CclCollectivePermute(
    tfrt::Argument<tfrt::gpu::GpuCclHandle> handle,
    tfrt::Argument<tfrt::gpu::GpuBuffer> input,
    tfrt::Argument<tfrt::gpu::GpuBuffer> output,
    // Needs to be sorted alphabetically by attribute name!
    tfrt::Attribute<int32_t> data_type_attr,
    tfrt::Attribute<int64_t> group_mode_attr, tfrt::ArrayAttr source_peers_attr,
    tfrt::ArrayAttr target_peers_attr, const tfrt::ExecutionContext& exec_ctx) {
  // NCCL 2.8.x has an issue with point-to-point communication primitives if
  // different ranks process different amounts of data. This can happen in the
  // case of a collective permute as certain nodes may not do any send or
  // receives, or do only send or only receive. Sending and receiving to self
  // as well (identity pair) causes this imbalance. NCCL 2.8.x requires the
  // use of NCCL_LAUNCH_MODE=PARALLEL to avoid these issues. See
  // https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-8-4.html#rel_2-8-4
  if (!IsNcclLaunchModeParallel()) {
    LOG(WARNING) << "NCCL based collective permute may not work correctly if "
                    "NCCL_LAUNCH_MODE is not set to PARALLEL";
  }

  auto current_logical_id = GetLogicalId(exec_ctx);
  if (!current_logical_id)
    return tfrt::MakeErrorAsyncValueRef(
        llvm::toString(current_logical_id.takeError()));

  const int64_t current_id =
      static_cast<CollectiveOpGroupMode>(*group_mode_attr) ==
              CollectiveOpGroupMode::kCrossReplica
          ? current_logical_id->replica_id
          : current_logical_id->computation_id;

  NcclCollectivePermuteConfig config;
  for (int i = 0; i < source_peers_attr.GetNumElements(); ++i) {
    int64_t source = source_peers_attr.GetValue<int64_t>()[i];
    int64_t target = target_peers_attr.GetValue<int64_t>()[i];

    config.id_to_source_target.insert({target, {}}).first->second.source =
        source;
    config.id_to_source_target.insert({source, {}}).first->second.target =
        target;
  }
  NcclCollectivePermuteConfig::SourceTargetMapEntry source_target =
      config.GetSourceTarget(current_id);
  const absl::optional<int64_t>& source_peer = source_target.source;
  const absl::optional<int64_t>& target_peer = source_target.target;

  auto type = static_cast<ncclDataType_t>(*data_type_attr);
  auto width = tfrt::gpu::wrapper::GetCclDataTypeSizeBytes(type);
  if (!width)
    return tfrt::MakeErrorAsyncValueRef(llvm::toString(width.takeError()));
  assert(*width != 0);

  if (target_peer) {
    handle->AddCallback(
        [input = input.ValueRef(), count = input->size() / *width, type,
         peer = *target_peer](tfrt::gpu::wrapper::CurrentContext current,
                              tfrt::gpu::wrapper::Stream stream,
                              tfrt::gpu::wrapper::CclComm comm) -> llvm::Error {
          return tfrt::gpu::wrapper::CclSend(current, input->pointer(), count,
                                             type, peer, comm, stream);
        });
  }

  if (source_peer) {
    handle->AddCallback(
        [output = output.ValueRef(), count = output->size() / *width, type,
         peer = *source_peer](tfrt::gpu::wrapper::CurrentContext current,
                              tfrt::gpu::wrapper::Stream stream,
                              tfrt::gpu::wrapper::CclComm comm) -> llvm::Error {
          return tfrt::gpu::wrapper::CclRecv(current, output->pointer(), count,
                                             type, peer, comm, stream);
        });
  } else {
    // If there is no source peer, i.e. no one send us any data, zero out dest
    // buffer.
    handle->AddCallback(
        [output = output.ValueRef(), count = output->size() / *width, type,
         peer = *source_peer](tfrt::gpu::wrapper::CurrentContext current,
                              tfrt::gpu::wrapper::Stream stream,
                              tfrt::gpu::wrapper::CclComm comm) -> llvm::Error {
          return tfrt::gpu::wrapper::MemsetD8Async(current, output->pointer(),
                                                   0, output->size(), stream);
        });
  }

  return tfrt::MakeAvailableAsyncValueRef<tfrt::Chain>();
}
#endif  // XLA_ENABLE_XCCL

static llvm::Error CustomCall(
    const tfrt::gpu::GpuStream& stream,
    tfrt::RepeatedArguments<tfrt::gpu::GpuBuffer> buffers_and_chain,
    // Needs to be sorted alphabetically by attribute name!
    tfrt::ArrayAttr indices, tfrt::StringAttribute opaque,
    tfrt::StringAttribute symbol) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc mht_2(mht_2_v, 499, "", "./tensorflow/compiler/xla/service/gpu/xlir_kernels.cc", "CustomCall");

  // Lookup custom call target from registry.
  auto platform = stream->platform();
  auto key = absl::AsciiStrToUpper(tfrt::StrCat(platform));  // 'ROCm' -> 'ROCM'
  auto* target = CustomCallTargetRegistry::Global()->Lookup(symbol.str(), key);
  if (!target) {
    return tfrt::MakeStringError("Custom call target '", symbol.str(),
                                 "' not registered for platform ", key);
  }

  auto current = tfrt::gpu::wrapper::CtxSetCurrent(stream.context()->get());
  if (!current) return current.takeError();

  // Create buffer pointer array argument.
  llvm::SmallVector<void*, 16> pointers;
  llvm::transform(indices.GetValue<int32_t>(), std::back_inserter(pointers),
                  [&](int32_t index) -> void* {
                    if (index < 0) return nullptr;
                    return buffers_and_chain[index].pointer().raw(platform);
                  });

  auto stream_ptr = [&]() -> void* {
    switch (platform) {
      case tfrt::gpu::wrapper::Platform::CUDA:
        return static_cast<CUstream>(stream.get());
      case tfrt::gpu::wrapper::Platform::ROCm:
        return static_cast<hipStream_t>(stream.get());
      default:
        return nullptr;
    }
  }();

  XlaCustomCallStatus status;
  using FuncPtr =
      void (*)(void*, void* const*, const char*, size_t, XlaCustomCallStatus*);
  reinterpret_cast<FuncPtr>(target)(stream_ptr, pointers.data(),
                                    opaque.get().data(), opaque.get().size(),
                                    &status);

  if (auto message = CustomCallStatusGetMessage(&status))
    return tfrt::MakeStringError("Custom call failed: ", message->data());

  return llvm::Error::success();
}

static llvm::Error CheckShapeCompatibleWithBuffer(const Shape& parent_shape,
                                                  const ShapeIndex& shape_index,
                                                  size_t buffer_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc mht_3(mht_3_v, 549, "", "./tensorflow/compiler/xla/service/gpu/xlir_kernels.cc", "CheckShapeCompatibleWithBuffer");

  const Shape& shape = ShapeUtil::GetSubshape(parent_shape, shape_index);
  if (buffer_size != ShapeUtil::ByteSizeOf(shape)) {
    return tfrt::MakeStringError(
        absl::StrCat("Mismatch between buffer shape ",
                     ShapeUtil::HumanStringWithLayout(shape),
                     " and buffer size ", buffer_size));
  }
  return llvm::Error::success();
}

static llvm::Error InfeedBlockingWork(
    tfrt::AsyncValueRef<tfrt::gpu::GpuStream> stream,
    const tfrt::RCArray<tfrt::AsyncValue> outputs,
    const XlaGpuParams* xla_gpu_params) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc mht_4(mht_4_v, 566, "", "./tensorflow/compiler/xla/service/gpu/xlir_kernels.cc", "InfeedBlockingWork");

  auto context = tfrt::gpu::wrapper::CtxSetCurrent(stream->context()->get());
  if (!context) {
    return context.takeError();
  }

  ShapeTree<se::ScopedDeviceMemory<uint8_t>> source_buffers =
      xla_gpu_params->infeed_manager->BlockingGetNextDestination();
  // Make sure that the outputs and the input buffers match.
  const size_t num_outputs = outputs.size();
  const int64_t leaf_count = source_buffers.leaf_count();
  if (num_outputs != leaf_count)
    return tfrt::MakeStringError(
        absl::StrCat("Mismatch between number of infeed outputs (", num_outputs,
                     ") and input buffers (", leaf_count, ")"));

  // TODO(bixia): Switch to use zip, may need to add operator += to
  //   RepeatedArguments to support this.
  for (const auto& source_and_index :
       llvm::enumerate(source_buffers.leaves())) {
    const auto& source = source_and_index.value();
    const Shape& parent_shape = source_buffers.shape();
    const ShapeIndex& shape_index = source.first;
    const size_t index = source_and_index.index();
    tfrt::gpu::GpuBuffer& output =
        outputs[index]->template get<tfrt::gpu::GpuBuffer>();
    const size_t buffer_size = output.size();
    if (auto error = CheckShapeCompatibleWithBuffer(parent_shape, shape_index,
                                                    buffer_size)) {
      return error;
    }

    // Enqueue the memory copy.
    const se::ScopedDeviceMemory<uint8_t>& source_buffer = source.second;
    tfrt::gpu::wrapper::Pointer<const void> source_pointer(
        source_buffer.ptr()->opaque(), context->platform());
    if (auto error = tfrt::gpu::wrapper::MemcpyAsync(
            *context, output.pointer(), source_pointer,
            source_buffer.ptr()->size(), stream->get()))
      return error;
  }

  // Wait for the memory copy operations to finish.
  return tfrt::gpu::wrapper::StreamSynchronize(stream->get());
}

static tfrt::AsyncValueRef<tfrt::Chain> Infeed(
    tfrt::Argument<tfrt::gpu::GpuStream> stream,
    tfrt::RepeatedArguments<tfrt::gpu::GpuBuffer> outputs_then_chain,
    const tfrt::ExecutionContext& exec_ctx) {
  VLOG(2) << "Infeeding to GPU";

  auto* xla_gpu_params =
      exec_ctx.request_ctx()->GetDataIfExists<XlaGpuParams>();
  if (!xla_gpu_params) {
    return tfrt::MakeErrorAsyncValueRef("Failed to get XlaGpuParams");
  }

  // Capture a reference to the output buffers to ensure that the buffers
  // won't be released before the MemcpyAsync operations complete, even if
  // the results of the memory copy aren't used.
  return tfrt::EnqueueBlockingWork(
      exec_ctx.host(),
      [=, stream = stream.ValueRef(),
       outputs = tfrt::RCArray<tfrt::AsyncValue>(
           outputs_then_chain.values().drop_back())]() mutable
      -> llvm::Expected<tfrt::Chain> {
        // Move to local so that the stream/outputs are released before
        // returning, or else, they will be released when this lambda is
        // destructed, which is after returning.
        if (auto error = InfeedBlockingWork(std::move(stream),
                                            std::move(outputs), xla_gpu_params))
          return error;
        VLOG(2) << "Infeeding to GPU complete";
        return tfrt::Chain();
      });
}

static llvm::Error OutfeedBlockingWork(
    tfrt::AsyncValueRef<tfrt::gpu::GpuStream> stream,
    const tfrt::RCArray<tfrt::AsyncValue> inputs,
    const XlaGpuParams* xla_gpu_params) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc mht_5(mht_5_v, 650, "", "./tensorflow/compiler/xla/service/gpu/xlir_kernels.cc", "OutfeedBlockingWork");

  VLOG(2) << "Outfeeding from GPU";

  auto context = tfrt::gpu::wrapper::CtxSetCurrent(stream->context()->get());
  if (!context) {
    return context.takeError();
  }

  ShapeTree<std::unique_ptr<OutfeedBuffer>>* dest_buffers =
      xla_gpu_params->outfeed_manager->BlockingGetNextDestination();
  const int64_t leaf_count = dest_buffers->leaf_count();

  // Check that the inputs and the output buffers match.
  const size_t num_inputs = inputs.size();
  if (num_inputs != leaf_count)
    return tfrt::MakeStringError(
        absl::StrCat("Mismatch between number of outfeed inputs (", num_inputs,
                     ") and output buffers (", leaf_count, ")"));

  std::vector<OutfeedBuffer*> buffers;
  for (const auto& iter_and_index : llvm::enumerate(dest_buffers->leaves())) {
    auto& leaf_iter = iter_and_index.value();
    const Shape& parent_shape = dest_buffers->shape();
    const ShapeIndex& shape_index = leaf_iter.first;
    const size_t index = iter_and_index.index();
    tfrt::gpu::GpuBuffer& input =
        inputs[index]->template get<tfrt::gpu::GpuBuffer>();
    const size_t buffer_size = input.size();
    if (auto error = CheckShapeCompatibleWithBuffer(parent_shape, shape_index,
                                                    buffer_size)) {
      return error;
    }

    // Enqueue the memory copy.
    const std::unique_ptr<OutfeedBuffer>& dest_buffer = leaf_iter.second;
    tfrt::gpu::wrapper::Pointer<void> dest_pointer(
        dest_buffer->destination()->untyped_data(), context->platform());
    if (auto error = tfrt::gpu::wrapper::MemcpyAsync(
            *context, dest_pointer, input.pointer(), dest_buffer->length(),
            stream->get()))
      tfrt::MakeErrorAsyncValueRef(tfrt::DecodedDiagnostic(error));

    // Collect the OutfeedBuffer that we will need to notify after all the
    // memory copy operations finish. The buffers are owned by the outfeed
    // manager.
    buffers.push_back(dest_buffer.get());
  }

  // Wait for the memory copy operations to finish and notify the destination
  // buffers.
  if (auto error = tfrt::gpu::wrapper::StreamSynchronize(stream->get()))
    return error;
  for (auto& buffer : buffers) buffer->Done();
  return llvm::Error::success();
}

static tfrt::AsyncValueRef<tfrt::Chain> Outfeed(
    tfrt::Argument<tfrt::gpu::GpuStream> stream,
    tfrt::RepeatedArguments<tfrt::gpu::GpuBuffer> inputs_then_chain,
    const tfrt::ExecutionContext& exec_ctx) {
  VLOG(2) << "Outfeeding from GPU";

  auto* xla_gpu_params =
      exec_ctx.request_ctx()->GetDataIfExists<XlaGpuParams>();
  if (!xla_gpu_params) {
    return tfrt::MakeErrorAsyncValueRef("Failed to get XlaGpuParams");
  }

  return tfrt::EnqueueBlockingWork(
      exec_ctx.host(),
      [=, stream = stream.ValueRef(),
       inputs = tfrt::RCArray<tfrt::AsyncValue>(
           inputs_then_chain.values().drop_back())]() mutable
      -> llvm::Expected<tfrt::Chain> {
        // Move to local so that the stream/inputs are released before
        // returning, or else, they will be released when this lambda is
        // destructed, which is after returning.
        if (auto error = OutfeedBlockingWork(std::move(stream),
                                             std::move(inputs), xla_gpu_params))
          return error;
        VLOG(2) << "Outfeeding to GPU complete";
        return tfrt::Chain();
      });
}

static void RegisterXlirKernels(tfrt::KernelRegistry* kernel_reg) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxlir_kernelsDTcc mht_6(mht_6_v, 738, "", "./tensorflow/compiler/xla/service/gpu/xlir_kernels.cc", "RegisterXlirKernels");

  kernel_reg->AddKernel("xlir.custom_call",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(CustomCall));
  // This kernel is only used for bef thunks, not bef executables.
  kernel_reg->AddKernel("xlir.module.load", TFRT_KERNEL(ModuleLoad));
  kernel_reg->AddKernel("xlir.infeed", TFRT_KERNEL(Infeed));
  kernel_reg->AddKernel("xlir.outfeed", TFRT_KERNEL(Outfeed));
  kernel_reg->AddKernel("xlir.replica_id",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(ReplicaId));
  kernel_reg->AddKernel("xlir.partition_id",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(PartitionId));
#if XLA_ENABLE_XCCL
  kernel_reg->AddKernel("xlir.ccl.create", TFRT_KERNEL(CclCreate));
  kernel_reg->AddKernel("xlir.ccl.collective_permute",
                        TFRT_KERNEL(CclCollectivePermute));
#endif  // XLA_ENABLE_XCCL
}

TFRT_STATIC_KERNEL_REGISTRATION(RegisterXlirKernels);

}  // namespace gpu
}  // namespace xla
