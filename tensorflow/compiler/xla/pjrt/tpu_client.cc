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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc() {
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

#include "tensorflow/compiler/xla/pjrt/tpu_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/pjrt/tracked_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/tpu_computation_placer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/tpu/tpu_executable.h"
#include "tensorflow/stream_executor/tpu/tpu_executable_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_stream.h"

namespace tf_tpu = tensorflow::tpu;

namespace xla {
namespace {

class TpuDeviceState : public LocalDeviceState {
 public:
  TpuDeviceState(se::StreamExecutor* executor, LocalClient* client,
                 int max_inflight_computations);

  Status ThenMemcpyDeviceToDevice(se::Stream* transfer_stream,
                                  se::Stream* dst_stream,
                                  se::DeviceMemoryBase src_buffer,
                                  se::DeviceMemoryBase dst_buffer) override;
};

TpuDeviceState::TpuDeviceState(se::StreamExecutor* executor,
                               LocalClient* client,
                               int max_inflight_computations)
    : LocalDeviceState(executor, client, LocalDeviceState::kAsynchronous,
                       max_inflight_computations,
                       /*allow_event_reuse=*/false,
                       /*use_callback_stream=*/true) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc mht_0(mht_0_v, 239, "", "./tensorflow/compiler/xla/pjrt/tpu_client.cc", "TpuDeviceState::TpuDeviceState");
}

Status TpuDeviceState::ThenMemcpyDeviceToDevice(
    se::Stream* transfer_stream, se::Stream* dst_stream,
    se::DeviceMemoryBase src_buffer, se::DeviceMemoryBase dst_buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/xla/pjrt/tpu_client.cc", "TpuDeviceState::ThenMemcpyDeviceToDevice");

  auto* transfer_tpu_stream = tensorflow::down_cast<tf_tpu::TpuStream*>(
      transfer_stream->implementation());
  TF_RETURN_IF_ERROR(transfer_tpu_stream->EnqueueOnTpuDeviceSendRecvLocal(
      src_buffer, dst_buffer));
  return Status::OK();
}

}  // namespace

PjRtTpuClient::PjRtTpuClient(
    LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index)
    : PjRtStreamExecutorClient(TpuName(), client, std::move(devices),
                               process_index,
                               /*allocator=*/nullptr,
                               /*host_memory_allocator=*/nullptr,
                               /*should_stage_host_to_device_transfers=*/false,
                               /*gpu_run_options=*/nullptr),
      platform_version_([]() {
        // Example platform version string:
        //   libtpu version 0.0.1
        //   Built on Mar 4 2021 15:25:57 (1614900357) cl/360760169
        tf_tpu::TpuPlatformInterface* platform =
            tf_tpu::TpuPlatformInterface::GetRegisteredPlatform();
        TpuRuntimeVersion version = platform->version();
        return absl::StrCat(
            "libtpu version ", absl::StrJoin(version.version, "."), "\n",
            absl::string_view(version.metadata, version.metadata_size));
      }()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc mht_2(mht_2_v, 279, "", "./tensorflow/compiler/xla/pjrt/tpu_client.cc", "PjRtTpuClient::PjRtTpuClient");

  // We always initialize the tpu client even if libtpu isn't linked in or
  // initialized.
  if (tf_tpu::ExecutorApiFn()->TpuAsyncCollectiveOffloadHelper_InitFn !=
      nullptr) {
    tf_tpu::ExecutorApiFn()->TpuAsyncCollectiveOffloadHelper_InitFn();
  }
}

PjRtTpuClient::~PjRtTpuClient() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc mht_3(mht_3_v, 291, "", "./tensorflow/compiler/xla/pjrt/tpu_client.cc", "PjRtTpuClient::~PjRtTpuClient");

  if (tf_tpu::ExecutorApiFn()->TpuAsyncCollectiveOffloadHelper_ShutdownFn !=
      nullptr) {
    tf_tpu::ExecutorApiFn()->TpuAsyncCollectiveOffloadHelper_ShutdownFn();
  }
}

StatusOr<DeviceAssignment> PjRtTpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc mht_4(mht_4_v, 302, "", "./tensorflow/compiler/xla/pjrt/tpu_client.cc", "PjRtTpuClient::GetDefaultDeviceAssignment");

  tf_tpu::TpuPlatformInterface* platform =
      tf_tpu::TpuPlatformInterface::GetRegisteredPlatform();
  tf_tpu::TpuHostLocationExternal host = platform->GetTpuHostLocation();
  int num_local_devices = host.Cores(kTensorCore).size();
  if (num_replicas * num_partitions <= num_local_devices) {
    return tf_tpu::TpuComputationPlacer::AssignLocalDevices(host, num_replicas,
                                                            num_partitions);
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                              num_partitions);
}

StatusOr<absl::optional<std::string>> PjRtTpuClient::ExecutableFingerprint(
    const PjRtExecutable& executable) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc mht_5(mht_5_v, 320, "", "./tensorflow/compiler/xla/pjrt/tpu_client.cc", "PjRtTpuClient::ExecutableFingerprint");

  if (executable.client() != this) {
    return InvalidArgument(
        "Passed executable from different client (platform '%s') to "
        "PjRtTpuClient::ExecutableFingerprint",
        executable.client()->platform_name());
  }
  if (executable.num_partitions() > 1) {
    LOG(INFO) << "ExecutableFingerprint not fully implemented for MPMD "
                 "executables, fingerprint may not be unique.";
  }
  xla::TpuExecutableInterface* tpu_executable =
      tensorflow::down_cast<xla::TpuExecutableInterface*>(
          tensorflow::down_cast<const PjRtStreamExecutorExecutable*>(
              &executable)
              ->executables()[0]
              ->executable());
  return absl::optional<std::string>(tpu_executable->fingerprint());
}

StatusOr<std::string> PjRtTpuClient::SerializeExecutable(
    const PjRtExecutable& executable) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc mht_6(mht_6_v, 344, "", "./tensorflow/compiler/xla/pjrt/tpu_client.cc", "PjRtTpuClient::SerializeExecutable");

  const PjRtStreamExecutorExecutable* se_executable =
      tensorflow::down_cast<const PjRtStreamExecutorExecutable*>(&executable);
  if (se_executable->executables().size() > 1) {
    return Unimplemented(
        "PjRtTpuClient::SerializeExecutable unimplemented for MPMD "
        "executables");
  }
  const TpuExecutable* tpu_executable =
      tensorflow::down_cast<const TpuExecutable*>(
          se_executable->executables()[0]->executable());
  return tpu_executable->Serialize();
}

StatusOr<std::unique_ptr<PjRtExecutable>> PjRtTpuClient::DeserializeExecutable(
    absl::string_view serialized, CompileOptions options) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("serialized: \"" + std::string(serialized.data(), serialized.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStpu_clientDTcc mht_7(mht_7_v, 363, "", "./tensorflow/compiler/xla/pjrt/tpu_client.cc", "PjRtTpuClient::DeserializeExecutable");

  TF_ASSIGN_OR_RETURN(std::unique_ptr<TpuExecutable> tpu_executable,
                      TpuExecutable::Deserialize(serialized));

  TF_ASSIGN_OR_RETURN(ExecutableExtras extras, GetExecutableExtras(&options));

  // TODO(skyewm): can we streamline this? e.g. removing proto serialization
  XlaComputation computation(tpu_executable->module().ToProto());
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  std::vector<const Shape*> unused_argument_layout_pointers;
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [local_client = client()](Shape shape) {
        return local_client->backend()
            .transfer_manager()
            ->ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &unused_argument_layout_pointers));

  auto local_executable = absl::make_unique<LocalExecutable>(
      std::move(tpu_executable), client_->mutable_backend(),
      options.executable_build_options);
  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.emplace_back(std::move(local_executable));

  auto pjrt_executable = absl::make_unique<PjRtStreamExecutorExecutable>(
      std::move(local_executables), options.parameter_is_tupled_arguments,
      std::move(extras.device_assignment),
      std::move(extras.addressable_device_logical_ids),
      std::move(extras.addressable_devices), this);
  TF_RETURN_IF_ERROR(
      pjrt_executable->SetUpDonation(options.parameter_is_tupled_arguments));
  return std::unique_ptr<PjRtExecutable>(std::move(pjrt_executable));
}

static StatusOr<std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>>
GetTpuDevices(
    LocalClient* client,
    std::vector<std::unique_ptr<LocalDeviceState>> local_device_states) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  tf_tpu::TpuTopologyExternal topology =
      tf_tpu::TpuPlatformInterface::GetRegisteredPlatform()->topology();

  std::map<int, int> core_id_to_device_ordinal;
  for (int i = 0; i < client->device_count(); ++i) {
    se::StreamExecutor* executor =
        client->backend().stream_executor(i).ValueOrDie();
    tf_tpu::TpuExecutorInterface* tpu_executor =
        tensorflow::down_cast<tf_tpu::TpuExecutorInterface*>(
            executor->implementation());
    core_id_to_device_ordinal[tpu_executor->GetCoreLocationExternal().Id()] = i;
  }

  for (const tf_tpu::TpuCoreLocationExternal& core :
       topology.cores(TpuCoreTypeEnum::kTensorCore)) {
    auto it = core_id_to_device_ordinal.find(core.Id());
    int device_ordinal =
        (it != core_id_to_device_ordinal.end()) ? it->second : -1;
    int process_index = topology.IdForHost(core.host_coordinates());
    const tf_tpu::TpuDimensionsExternal coords = core.chip_coordinates();
    std::array<int, 3> coords_array = {coords.x, coords.y, coords.z};
    std::unique_ptr<LocalDeviceState> local_device_state;
    if (device_ordinal >= 0) {
      local_device_state = std::move(local_device_states[device_ordinal]);
    }
    auto device = absl::make_unique<PjRtTpuDevice>(
        core, std::move(local_device_state), process_index, coords_array,
        std::string(tf_tpu::TpuVersionEnumToString(topology.version())));
    devices.push_back(std::move(device));
  }
  return devices;
}

StatusOr<std::shared_ptr<PjRtClient>> GetTpuClient(
    int max_inflight_computations, absl::Duration init_retry_timeout) {
  tf_tpu::TpuPlatformInterface* platform =
      tf_tpu::TpuPlatformInterface::GetRegisteredPlatform(
          /*initialize_platform=*/true, /*num_tries=*/1);
  if (platform == nullptr) {
    return InvalidArgument("TpuPlatform is not available.");
  }
  // NOTE: We retry in a loop since some pod failures are transient (e.g. some
  // RPCs may timeout waiting for other hosts to come up, but will succeed
  // at a later point if retried).
  auto start = absl::Now();
  while (true) {
    Status status = platform->Initialize({});
    if (status.ok()) {
      break;
    }
    // TODO(b/165870356): refactor this loop to be
    // while(!platform->Initialized()) once the Initialized() function works
    // correctly, and remove this check. The platform may already be initialized
    // when running internally.
    if (status.code() == tensorflow::error::ALREADY_EXISTS) {
      LOG(INFO) << "TpuPlatform already initialized, continuing...";
      break;
    }
    LOG(INFO) << "TPU platform initialization failed: " << status;
    if ((absl::Now() - start) >= init_retry_timeout) {
      return status;
    }
    absl::SleepFor(absl::Microseconds(10));
  }
  CHECK(platform->Initialized());
  if (platform->VisibleDeviceCount() <= 0) {
    return InvalidArgument("No TPU devices found.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));

  std::vector<std::unique_ptr<LocalDeviceState>> local_device_states;
  local_device_states.reserve(client->device_count());
  for (int i = 0; i < client->device_count(); ++i) {
    se::StreamExecutor* executor =
        client->backend().stream_executor(i).ValueOrDie();
    local_device_states.push_back(absl::make_unique<TpuDeviceState>(
        executor, client, max_inflight_computations));
  }

  TF_ASSIGN_OR_RETURN(auto devices,
                      GetTpuDevices(client, std::move(local_device_states)));
  int process_index = platform->GetTpuHostLocation().Id();

  return std::shared_ptr<PjRtClient>(absl::make_unique<PjRtTpuClient>(
      client, std::move(devices), process_index));
}

}  // namespace xla
