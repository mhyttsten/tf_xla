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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc() {
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

#include "tensorflow/compiler/xla/pjrt/gpu_device.h"

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/stream_executor/device_memory.h"

#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif  // TENSORFLOW_USE_ROCM

#ifdef NCCL_ENABLED
#include "third_party/nccl/nccl.h"
#endif  // NCCL_ENABLED
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace xla {
namespace {

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

std::string GetCudaErrorMessage(CUresult result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "GetCudaErrorMessage");

  const char* error;
  cuGetErrorString(result, &error);
  const char* name;
  cuGetErrorName(result, &name);
  return absl::StrCat("CUDA error: ", error ? error : "<unknown>", " (",
                      name ? name : "Unknown", ")");
}

// A compute-stream synchronized allocator, implemented using the CUDA async
// allocation API added in CUDA 11.2.
// TODO(phawkins): this approach does not use the full capabilities of the
// allocator. We don't need to synchronize allocations to the compute stream
// with this allocator design. However that would be a larger change to PJRT.
class CudaAsyncDeviceMemoryAllocator : public se::DeviceMemoryAllocator {
 public:
  static StatusOr<std::unique_ptr<CudaAsyncDeviceMemoryAllocator>> Create(
      se::Platform* platform, std::vector<se::Stream*> streams) {
    auto allocator = std::make_unique<CudaAsyncDeviceMemoryAllocator>(platform);
    allocator->pools_.resize(streams.size());

    for (size_t i = 0; i < streams.size(); ++i) {
      se::Stream* stream = streams[i];
      TF_RET_CHECK(stream->parent()->device_ordinal() == i);
      se::cuda::ScopedActivateExecutorContext scoped_activation{
          stream->parent()};
      int cuda_malloc_async_supported;
      if (auto status = cuDeviceGetAttribute(
              &cuda_malloc_async_supported,
              CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, i)) {
        return Unknown("Failed to get device attribute: %s",
                       GetCudaErrorMessage(status));
      }
      if (!cuda_malloc_async_supported) {
        return FailedPrecondition(
            "cuda_malloc_async isn't supported. "
            " Possible causes: device not supported, driver too old, "
            " OS not supported, CUDA version too old.");
      }
      if (auto status = cuDeviceGetDefaultMemPool(&allocator->pools_[i], i)) {
        return Unknown("Failed to get default CUDA pool: %s",
                       GetCudaErrorMessage(status));
      }
    }
    allocator->streams_ = std::move(streams);
    return allocator;
  }

  // Use Create() instead of calling this constructor.
  explicit CudaAsyncDeviceMemoryAllocator(se::Platform* platform)
      : se::DeviceMemoryAllocator(platform) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_1(mht_1_v, 274, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "CudaAsyncDeviceMemoryAllocator");
}

  StatusOr<se::OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                            bool retry_on_failure,
                                            int64_t memory_space) override {
    se::Stream* stream = streams_.at(device_ordinal);
    se::cuda::ScopedActivateExecutorContext scoped_activation{stream->parent()};
    CUstream custream = reinterpret_cast<cudaStream_t>(
        stream->implementation()->GpuStreamHack());
    void* ptr = nullptr;
    if (auto result =
            cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&ptr), size,
                                    pools_.at(device_ordinal), custream)) {
      return ResourceExhausted("CUDA allocation of %d bytes failed.", size);
    }
    return se::OwningDeviceMemory(se::DeviceMemoryBase(ptr, size),
                                  device_ordinal, this);
  }

  Status Deallocate(int device_ordinal, se::DeviceMemoryBase mem) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_2(mht_2_v, 296, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "Deallocate");

    se::Stream* stream = streams_.at(device_ordinal);
    se::cuda::ScopedActivateExecutorContext scoped_activation{stream->parent()};
    CUstream custream = reinterpret_cast<cudaStream_t>(
        stream->implementation()->GpuStreamHack());
    void* ptr = const_cast<void*>(mem.opaque());
    if (auto result = cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(ptr),
                                     custream)) {
      return Unknown("CUDA deallocation failed.");
    }
    return Status::OK();
  }

  StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    return streams_.at(device_ordinal);
  }

 private:
  std::vector<se::Stream*> streams_;
  std::vector<CUmemoryPool> pools_;
};

StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>> CreateCudaAsyncAllocator(
    se::Platform* platform,
    absl::Span<std::unique_ptr<LocalDeviceState> const> addressable_devices) {
  std::vector<se::Stream*> streams;
  streams.reserve(addressable_devices.size());
  for (const auto& device : addressable_devices) {
    streams.push_back(device->compute_stream());
  }
  return CudaAsyncDeviceMemoryAllocator::Create(platform, std::move(streams));
}

#else  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>> CreateCudaAsyncAllocator(
    se::Platform* platform,
    absl::Span<std::unique_ptr<LocalDeviceState> const> addressable_devices) {
  return FailedPrecondition("CUDA async allocator requires CUDA >= 11.2");
}

#endif  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

// A custom PjRtClient that overrides the device assignment method.
class GpuClient : public xla::PjRtStreamExecutorClient {
 public:
  using xla::PjRtStreamExecutorClient::PjRtStreamExecutorClient;

  xla::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::string_view platform_version() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_3(mht_3_v, 350, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "platform_version");

#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#if TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)  // rocm
    // TF_ROCM_VERSION fomrat may change in future. Use it
    // cautiously
    return "rocm " STRINGIFY(TF_ROCM_VERSION);
#elif GOOGLE_CUDA && defined(CUDART_VERSION)  // cuda
    return "cuda " STRINGIFY(CUDART_VERSION);
#else
    return "<unknown>";
#endif  // TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)
  }
};

xla::StatusOr<xla::DeviceAssignment> GpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_4(mht_4_v, 369, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "GpuClient::GetDefaultDeviceAssignment");

  if (num_partitions == 1 && num_replicas <= addressable_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, 1);
    for (int i = 0; i < num_replicas; ++i) {
      assignment(i, 0) = addressable_devices().at(i)->id();
    }
    return assignment;
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                              num_partitions);
}

// Builds an xla::LocalClient for the GPU platform.
StatusOr<LocalClient*> GetGpuXlaClient() {
  // "gpu" will be substitued by the default defined in platform_util.cc
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("gpu"));
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("No visible GPU devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

void EnablePeerAccess(absl::Span<se::StreamExecutor* const> executors) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_5(mht_5_v, 398, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "EnablePeerAccess");

  for (int i = 0; i < executors.size(); ++i) {
    for (int j = 0; j < executors.size(); ++j) {
      if (i == j) {
        continue;
      }
      se::StreamExecutor* from = executors[i];
      se::StreamExecutor* to = executors[j];
      if (from->CanEnablePeerAccessTo(to)) {
        Status status = from->EnablePeerAccessTo(to);
        if (!status.ok()) {
          LOG(WARNING) << "Unable to enable peer access between GPUs " << i
                       << " and " << j << "; status: " << status;
        } else {
          VLOG(2) << "Enabled peer access from GPU " << i << " to GPU " << j;
        }
      }
    }
  }
}

// Builds a LocalDeviceState for each GPU present.
StatusOr<std::vector<std::unique_ptr<LocalDeviceState>>> BuildLocalDeviceStates(
    LocalClient* xla_client, bool asynchronous) {
  std::vector<std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (int i = 0; i < xla_client->device_count(); ++i) {
    se::StreamExecutor* executor =
        xla_client->backend().stream_executor(i).ValueOrDie();
    addressable_devices.push_back(absl::make_unique<LocalDeviceState>(
        executor, xla_client, LocalDeviceState::kComputeSynchronized,
        /*max_inflight_computations=*/32,
        /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(addressable_devices);
}

// Builds a BFCAllocator for all local GPUs.
StatusOr<std::unique_ptr<se::MultiDeviceAdapter>> CreateBFCAllocator(
    absl::Span<std::unique_ptr<LocalDeviceState> const> addressable_devices,
    double memory_fraction, bool preallocate) {
  CHECK_GT(addressable_devices.size(), 0);
  const se::Platform* platform =
      addressable_devices.front()->executor()->platform();
  std::vector<se::MultiDeviceAdapter::AllocatorWithStream> allocators;
  bool enable_unified_memory;
  Status status = tensorflow::ReadBoolFromEnvVar("TF_FORCE_UNIFIED_MEMORY",
                                                 false, &enable_unified_memory);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to read TF_FORCE_UNIFIED_MEMORY: "
               << status.error_message();
  }

  for (auto& local_device : addressable_devices) {
    se::StreamExecutor* executor = local_device->executor();
    int device_ordinal = executor->device_ordinal();
    auto sub_allocator = absl::make_unique<tensorflow::DeviceMemAllocator>(
        executor, tensorflow::PlatformDeviceId(device_ordinal),
        /*use_unified_memory=*/enable_unified_memory,
        /*alloc_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>(),
        /*free_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>());

    int64_t free_memory;
    int64_t total_memory;
    if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
      return Unavailable("Failed to query available memory from device %i",
                         device_ordinal);
    }
    // To allow full GPU memory to be visible to the BFC allocator if using
    // unified memory.
    // When unified memory is enabled, allow GPU memory oversubscription by
    // setting memory_fraction > 1.
    size_t allocator_memory = enable_unified_memory
                                  ? total_memory * fmax(1.0, memory_fraction)
                                  : free_memory * memory_fraction;
    if (preallocate) {
      LOG(INFO) << "XLA backend allocating " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for BFCAllocator.";
    } else {
      LOG(INFO) << "XLA backend will use up to " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for BFCAllocator.";
    }

    tensorflow::BFCAllocator::Options opts;
    opts.allow_growth = !preallocate;
    auto gpu_bfc_allocator = absl::make_unique<tensorflow::BFCAllocator>(
        std::move(sub_allocator), allocator_memory,
        absl::StrCat("GPU_", device_ordinal, "_bfc"), opts);
    allocators.emplace_back(std::move(gpu_bfc_allocator),
                            local_device->compute_stream());
  }
  return absl::make_unique<se::MultiDeviceAdapter>(platform,
                                                   std::move(allocators));
}

// Constructs a GPU device memory allocator to use, according to the allocator
// configuration the client requested.
StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>> GetGpuDeviceAllocator(
    se::Platform* platform, const GpuAllocatorConfig& allocator_config,
    absl::Span<std::unique_ptr<LocalDeviceState> const> addressable_devices) {
  std::unique_ptr<se::DeviceMemoryAllocator> allocator;
  switch (allocator_config.kind) {
    case GpuAllocatorConfig::Kind::kCudaAsync: {
      auto allocator_or =
          CreateCudaAsyncAllocator(platform, addressable_devices);
      if (allocator_or.ok()) {
        LOG(INFO) << "Using CUDA async allocator.";
        allocator = std::move(allocator_or.ValueOrDie());
        break;
      }
      LOG(ERROR) << "Failed to initialize CUDA async allocator: "
                 << allocator_or.status() << "; falling back to BFC.";
      ABSL_FALLTHROUGH_INTENDED;
    }

    case GpuAllocatorConfig::Kind::kDefault:
    case GpuAllocatorConfig::Kind::kBFC: {
      LOG(INFO) << "Using BFC allocator.";
      TF_ASSIGN_OR_RETURN(allocator,
                          CreateBFCAllocator(addressable_devices,
                                             allocator_config.memory_fraction,
                                             allocator_config.preallocate));
      break;
    }

    case GpuAllocatorConfig::Kind::kPlatform:
      LOG(INFO) << "Using platform allocator.";
      break;
  }
  return std::move(allocator);
}

// Returns a GPU pinned host memory allocator to use when staging host->GPU
// transfers. We use a fixed 64MB pool of pinned memory.
std::unique_ptr<tensorflow::BFCAllocator> GetGpuHostAllocator(
    se::StreamExecutor* executor) {
  std::unique_ptr<tensorflow::SubAllocator> sub_allocator(
      new tensorflow::DeviceHostAllocator(executor, /*numa_node=*/0,
                                          /*alloc_visitors=*/{},
                                          /*free_visitors=*/{}));
  // TODO(phawkins): allow the user to tune this.
  const int64_t kGpuHostMemoryLimitBytes = 64 * (1LL << 30);

  tensorflow::BFCAllocator::Options opts;
  opts.allow_growth = true;
  return absl::make_unique<tensorflow::BFCAllocator>(
      std::move(sub_allocator), kGpuHostMemoryLimitBytes,
      /*name=*/"xla_gpu_host_bfc", opts);
}

// A table mapping NcclCliqueKeys to ncclUniqueId values encoded as strings.
// In a distributed setup the table of NCCL IDs is kept on the master node
// (node 0). The node of the first participating device will create the unique
// id.
class NcclIdStore {
 public:
  NcclIdStore(int node_id, std::shared_ptr<DistributedRuntimeClient> client,
              absl::flat_hash_map<GlobalDeviceId, int> device_to_node)
      : node_id_(node_id),
        client_(std::move(client)),
        device_to_node_(std::move(device_to_node)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_6(mht_6_v, 562, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "NcclIdStore");
}

  StatusOr<std::string> GetNcclUniqueId(const gpu::NcclCliqueKey& key);

 private:
  const int node_id_;
  const std::shared_ptr<DistributedRuntimeClient> client_;
  const absl::flat_hash_map<GlobalDeviceId, int> device_to_node_;

  absl::Mutex mu_;
  absl::flat_hash_map<gpu::NcclCliqueKey, std::string> cache_
      ABSL_GUARDED_BY(mu_);
};

StatusOr<std::string> NcclIdStore::GetNcclUniqueId(
    const gpu::NcclCliqueKey& key) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_7(mht_7_v, 580, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "NcclIdStore::GetNcclUniqueId");

  // The caller must ensure that threads calling this method concurrently have
  // unique keys, otherwise the global key-value store may hold the wrong value.
  {
    absl::MutexLock lock(&mu_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
  }
  std::string id_string;
  int primary_node_id = device_to_node_.at(key.devices()[0]);
  if (node_id_ == primary_node_id) {
#ifdef NCCL_ENABLED
    ncclUniqueId id;
    ncclResult_t r = ncclGetUniqueId(&id);
    TF_RET_CHECK(r == ncclSuccess);
    id_string = std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
    TF_RETURN_IF_ERROR(client_->KeyValueSet(key.ToString(), id_string));
#else
    return FailedPrecondition("NCCL support was not built into XLA binary.");
#endif
  } else {
    TF_ASSIGN_OR_RETURN(id_string, client_->BlockingKeyValueGet(
                                       key.ToString(), absl::Minutes(5)));
  }
  absl::MutexLock lock(&mu_);
  auto result = cache_.emplace(key, std::move(id_string));
  TF_RET_CHECK(result.second) << "Unique ID already in cache.";
  return result.first->second;
}

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::vector<std::unique_ptr<LocalDeviceState>> local_device_states) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& local_device : local_device_states) {
    int device_ordinal = local_device->device_ordinal();
    const se::DeviceDescription& description =
        local_device->executor()->GetDeviceDescription();
    auto device = absl::make_unique<GpuDevice>(
        device_ordinal, std::move(local_device), description.name(),
        description.device_vendor(),
        /*node_id=*/0);
    devices.push_back(std::move(device));
  }
  return devices;
}

Status BuildDistributedDevices(
    std::vector<std::unique_ptr<LocalDeviceState>> local_device_states,
    std::shared_ptr<DistributedRuntimeClient> distributed_client, int node_id,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>* devices,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_8(mht_8_v, 635, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "BuildDistributedDevices");

  LocalTopologyProto local_topology;
  local_topology.set_node_id(node_id);
  for (const auto& local_device : local_device_states) {
    const se::Platform* platform = local_device->executor()->platform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(local_device->device_ordinal()));
    TF_RET_CHECK(local_device->device_ordinal() ==
                 local_topology.devices_size());
    DeviceProto* device_proto = local_topology.add_devices();
    device_proto->set_local_device_ordinal(local_device->device_ordinal());
    device_proto->set_name(desc->name());
    device_proto->set_vendor(desc->device_vendor());
  }

  GlobalTopologyProto global_topology;
  TF_RETURN_IF_ERROR(
      distributed_client->EnumerateDevices(local_topology, &global_topology));

  std::vector<GlobalDeviceId> gpu_device_ids(local_device_states.size());
  absl::flat_hash_map<GlobalDeviceId, int> device_to_node;
  for (const LocalTopologyProto& node : global_topology.nodes()) {
    for (const DeviceProto& device_proto : node.devices()) {
      GlobalDeviceId global_device_id(device_proto.global_device_id());
      device_to_node[global_device_id] = node.node_id();
      std::unique_ptr<LocalDeviceState> local_device;
      if (node.node_id() == node_id) {
        TF_RET_CHECK(device_proto.local_device_ordinal() >= 0 &&
                     device_proto.local_device_ordinal() <
                         local_device_states.size());
        TF_RET_CHECK(local_device_states[device_proto.local_device_ordinal()] !=
                     nullptr);
        local_device =
            std::move(local_device_states[device_proto.local_device_ordinal()]);
        gpu_device_ids[device_proto.local_device_ordinal()] = global_device_id;
      }
      auto device = absl::make_unique<GpuDevice>(
          device_proto.global_device_id(), std::move(local_device),
          device_proto.name(), device_proto.vendor(), node.node_id());
      devices->push_back(std::move(device));
    }
  }
  for (const auto& device : local_device_states) {
    TF_RET_CHECK(device == nullptr);
  }
  gpu_executable_run_options->set_gpu_global_device_ids(
      std::move(gpu_device_ids));
  auto nccl_id_store = std::make_shared<NcclIdStore>(
      node_id, distributed_client, device_to_node);
  gpu_executable_run_options->set_nccl_unique_id_callback(
      [nccl_id_store](const gpu::NcclCliqueKey& key) {
        return nccl_id_store->GetNcclUniqueId(key);
      });
  return Status::OK();
}

}  // namespace

GpuDevice::GpuDevice(int id,
                     std::unique_ptr<LocalDeviceState> local_device_state,
                     std::string device_kind, std::string device_vendor,
                     int node_id)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               std::move(device_kind), node_id),
      device_vendor_(std::move(device_vendor)) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("device_kind: \"" + device_kind + "\"");
   mht_9_v.push_back("device_vendor: \"" + device_vendor + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_9(mht_9_v, 705, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "GpuDevice::GpuDevice");
}

absl::string_view GpuDevice::device_vendor() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSgpu_deviceDTcc mht_10(mht_10_v, 710, "", "./tensorflow/compiler/xla/pjrt/gpu_device.cc", "GpuDevice::device_vendor");
 return device_vendor_; }

StatusOr<std::unique_ptr<PjRtClient>> GetGpuClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config,
    std::shared_ptr<DistributedRuntimeClient> distributed_client, int node_id) {
  TF_ASSIGN_OR_RETURN(LocalClient * xla_client, GetGpuXlaClient());
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalDeviceState>> local_device_states,
      BuildLocalDeviceStates(xla_client, asynchronous));
  EnablePeerAccess(xla_client->backend().stream_executors());
  TF_ASSIGN_OR_RETURN(
      auto allocator,
      GetGpuDeviceAllocator(xla_client->platform(), allocator_config,
                            local_device_states));
  auto host_memory_allocator =
      GetGpuHostAllocator(local_device_states.front()->executor());

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  auto gpu_run_options = absl::make_unique<gpu::GpuExecutableRunOptions>();
  if (distributed_client) {
    TF_RETURN_IF_ERROR(BuildDistributedDevices(
        std::move(local_device_states), std::move(distributed_client), node_id,
        &devices, gpu_run_options.get()));
  } else {
    devices = BuildLocalDevices(std::move(local_device_states));
  }

  return std::unique_ptr<PjRtClient>(std::make_unique<GpuClient>(
      GpuName(), xla_client, std::move(devices),
      /*node_id=*/node_id, std::move(allocator),
      std::move(host_memory_allocator),
      /*should_stage_host_to_device_transfers=*/true,
      /*gpu_run_options=*/std::move(gpu_run_options)));
}

}  // namespace xla
