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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"

#include <cstring>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static bool UseCudaMallocAllocator() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "UseCudaMallocAllocator");

  const char* allocator_env = std::getenv("TF_GPU_ALLOCATOR");
  return allocator_env != nullptr &&
         std::strcmp(allocator_env, "cuda_malloc") == 0;
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static bool UseCudaMemoryGuardAllocator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "UseCudaMemoryGuardAllocator");

  const char* allocator_env = std::getenv("TF_GPU_ALLOCATOR");
  return allocator_env != nullptr &&
         std::strcmp(allocator_env, "memory_guard") == 0;
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
static bool UseCudaMallocAsyncAllocator() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "UseCudaMallocAsyncAllocator");

  const char* allocator_env = std::getenv("TF_GPU_ALLOCATOR");
  auto result = allocator_env != nullptr &&
                std::strcmp(allocator_env, "cuda_malloc_async") == 0;
#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  return result;
#else
  if (result)
    LOG(ERROR) << "TF_GPU_ALLOCATOR=cuda_malloc_async environment found, "
               << "but TensorFlow was not compiled with CUDA 11.2+.";
  return false;
#endif
}

/*static*/ GPUProcessState* GPUProcessState::singleton(GPUProcessState* ps) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_3(mht_3_v, 253, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::singleton");

  static GPUProcessState* instance = ps ? ps : new GPUProcessState;
  DCHECK((!ps) || (ps == instance))
      << "Multiple calls to GPUProcessState with non-null ps";
  return instance;
}

GPUProcessState::GPUProcessState() : gpu_device_enabled_(false) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_4(mht_4_v, 263, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::GPUProcessState");

  process_state_ = ProcessState::singleton();
}

int GPUProcessState::BusIdForGPU(TfDeviceId tf_device_id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_5(mht_5_v, 270, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::BusIdForGPU");

  // Return the NUMA node associated with the GPU's StreamExecutor.
  se::StreamExecutor* se = DeviceIdUtil::ExecutorForTfDeviceId(
                               DEVICE_GPU, GPUMachineManager(), tf_device_id)
                               .ValueOrDie();
  int numa_node = se->GetDeviceDescription().numa_node();
  // bus_id must be non-negative.  If the numa_node is not known,
  // use 0.
  return numa_node >= 0 ? numa_node : 0;
}

// NOLINTNEXTLINE: clang-tidy complains this is unused because of build flags.
static std::unique_ptr<SubAllocator> CreateSubAllocator(
    const GPUOptions& options, PlatformDeviceId platform_device_id,
    const std::vector<SubAllocator::Visitor>& alloc_visitors,
    size_t total_bytes, const std::vector<TfDeviceId>& peer_gpu_ids) {
  auto executor = DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                            platform_device_id)
                      .ValueOrDie();

  // FIXME(imintz): Observed OOM issues when using the virtual memory
  // allocators. This should be reenabled when resolved.
#if 0 && defined(GOOGLE_CUDA) && CUDA_VERSION >= 10020
  // Use the old allocator when unified memory is required.
  // TODO(imintz): Remove the cuMemAlloc capability of this allocator.
  if (options.per_process_gpu_memory_fraction() > 1.0 ||
      options.experimental().use_unified_memory()) {
    return new DeviceMemAllocator(executor, platform_device_id,
                                  /*use_unified_memory=*/true, alloc_visitors,
                                  {});
  } else {
    auto* gpu_context = reinterpret_cast<stream_executor::gpu::GpuContext*>(
        executor->implementation()->GpuContextHack());

    absl::flat_hash_set<PlatformDeviceId> platform_peer_gpu_ids;
    platform_peer_gpu_ids.reserve(peer_gpu_ids.size());
    for (const TfDeviceId tf_device_id : peer_gpu_ids) {
      PlatformDeviceId platform_device_id;
      TF_CHECK_OK(GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
      platform_peer_gpu_ids.insert(platform_device_id);
    }
    std::vector<PlatformDeviceId> platform_peer_gpu_ids_vec(
        platform_peer_gpu_ids.begin(), platform_peer_gpu_ids.end());

    // Adjust virtual address space to be slightly larger than the physical
    // address space in case the BFC allocator performs suboptimal garbage
    // collection.
    // TODO(imintz): Update BFC allocator to ensure it doesn't create holes in
    // the va space.
    return GpuVirtualMemAllocator::Create(
               alloc_visitors, {}, *gpu_context, platform_device_id,
               /*virtual_address_space_size=*/total_bytes * 2,
               platform_peer_gpu_ids_vec)
        .ValueOrDie()
        .release();
  }
#else
  return absl::WrapUnique(
      new DeviceMemAllocator(executor, platform_device_id,
                             (options.per_process_gpu_memory_fraction() > 1.0 ||
                              options.experimental().use_unified_memory()),
                             alloc_visitors, {}));
#endif
}

Allocator* GPUProcessState::GetGPUAllocator(
    const GPUOptions& options, TfDeviceId tf_device_id, size_t total_bytes,
    const std::vector<TfDeviceId>& peer_gpu_ids) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_6(mht_6_v, 340, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::GetGPUAllocator");

  CHECK(process_state_);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  const string& allocator_type = options.allocator_type();
  mutex_lock lock(mu_);
  DeviceIdUtil::CheckValidTfDeviceId(DEVICE_GPU, GPUMachineManager(),
                                     tf_device_id);

  if (tf_device_id.value() >= static_cast<int64_t>(gpu_allocators_.size())) {
    gpu_allocators_.resize(tf_device_id.value() + 1);
  }

  AllocatorParts& allocator_parts = gpu_allocators_[tf_device_id.value()];
  if (allocator_parts.allocator == nullptr) {
    // Validate allocator types.
    if (!allocator_type.empty() && allocator_type != "BFC") {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }

    PlatformDeviceId platform_device_id;
    TF_CHECK_OK(
        GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
    int bus_id = BusIdForGPU(tf_device_id);
    DCHECK_GE(bus_id, 0);
    while (bus_id >= gpu_visitors_.size()) {
      gpu_visitors_.push_back({});
    }
    std::unique_ptr<SubAllocator> sub_allocator =
        CreateSubAllocator(options, platform_device_id, gpu_visitors_[bus_id],
                           total_bytes, peer_gpu_ids);
    SubAllocator* sub_allocator_ptr = sub_allocator.get();

    auto gpu_bfc_allocator = absl::make_unique<GPUBFCAllocator>(
        std::move(sub_allocator), total_bytes,
        strings::StrCat("GPU_", tf_device_id.value(), "_bfc"), [&] {
          GPUBFCAllocator::Options o;
          o.allow_growth = options.allow_growth();
          o.allow_retry_on_failure =
              !options.experimental().disallow_retry_on_allocation_failure();
          o.fragmentation_fraction =
              options.experimental().internal_fragmentation_fraction();
          return o;
        }());
    Allocator* gpu_allocator = gpu_bfc_allocator.get();

    SharedCounter* timing_counter = nullptr;
    if (options.experimental().timestamped_allocator()) {
      timing_counter = new SharedCounter;
      gpu_bfc_allocator->SetTimingCounter(timing_counter);
    }

    // If true, checks for memory overwrites by writing
    // distinctive patterns on both ends of allocated memory.
    if (UseCudaMemoryGuardAllocator()) {
      LOG(INFO) << "Using memory guard allocator for GPU.";
      gpu_allocator = new GPUNanResetAllocator(
          new GPUDebugAllocator(gpu_allocator, platform_device_id),
          platform_device_id);
    } else if (UseCudaMallocAllocator()) {
      LOG(INFO) << "Using CUDA malloc allocator for GPU.";
      // If true, passes all allocation requests through to cudaMalloc
      // useful for doing memory debugging with tools like cuda-memcheck
      // **WARNING** probably will not work in a multi-gpu scenario
      gpu_bfc_allocator.reset();
      gpu_allocator = new GPUcudaMallocAllocator(platform_device_id);
    } else if (UseCudaMallocAsyncAllocator() ||
               options.experimental().use_cuda_malloc_async()) {
      LOG(INFO) << "Using CUDA malloc Async allocator for GPU: "
                << platform_device_id;
      // If true, passes all allocation requests through to cudaMallocAsync
      // TODO: useful for doing memory debugging with tools like
      // compute-sanitizer.
      // TODO: **WARNING** probably will not work in a multi-gpu scenario
      gpu_bfc_allocator.reset();
      gpu_allocator =
          new GpuCudaMallocAsyncAllocator(platform_device_id, total_bytes);
    }

    Allocator* recording_allocator = nullptr;
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::GPU;
      md.dev_index = platform_device_id.value();
      md.gpu_registered = false;
      md.nic_registered = true;
      recording_allocator = new internal::RecordingAllocator(
          &process_state_->mem_desc_map_, gpu_allocator, md, &mu_);
    }
    allocator_parts = {
        std::unique_ptr<Allocator>(gpu_allocator),
        std::unique_ptr<SharedCounter>(timing_counter),
        gpu_bfc_allocator.release(),
        sub_allocator_ptr,
        std::unique_ptr<Allocator>(recording_allocator),
    };
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
    return allocator_parts.recording_allocator.get();
  } else {
    return allocator_parts.allocator.get();
  }
#else
  LOG(FATAL) << "GPUAllocator unavailable. Not compiled with --config=cuda or "
                "--config=rocm.";
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

SharedCounter* GPUProcessState::GPUAllocatorCounter(TfDeviceId tf_device_id) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_7(mht_7_v, 453, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::GPUAllocatorCounter");

  DCHECK(process_state_);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  DeviceIdUtil::CheckValidTfDeviceId(DEVICE_GPU, GPUMachineManager(),
                                     tf_device_id);
  mutex_lock l(mu_);
  if (tf_device_id.value() >= static_cast<int64_t>(gpu_allocators_.size())) {
    LOG(ERROR) << "Asked for counter for GPU allocator " << tf_device_id.value()
               << " but only have " << gpu_allocators_.size();
    return nullptr;
  }

  AllocatorParts& allocator_parts = gpu_allocators_[tf_device_id.value()];
  if (allocator_parts.counter.get() == nullptr) {
    if (allocator_parts.bfc_allocator == nullptr) {
      return nullptr;
    }
    SharedCounter* timing_counter = new SharedCounter;
    allocator_parts.bfc_allocator->SetTimingCounter(timing_counter);
    allocator_parts.counter.reset(timing_counter);
  }
  return allocator_parts.counter.get();
#else
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

Allocator* GPUProcessState::GetGpuHostAllocator(int numa_node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_8(mht_8_v, 484, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::GetGpuHostAllocator");

  CHECK(process_state_);
  if (!HasGPUDevice() ||
      !process_state_->ProcessState::FLAGS_brain_mem_reg_gpu_dma) {
    return process_state_->GetCPUAllocator(numa_node);
  }
  if (numa_node == port::kNUMANoAffinity) {
    numa_node = 0;
  }
  {
    // Here we optimize the most common use case where gpu_host_allocators_
    // have already been populated and since we're only reading
    // these vectors, we can get by with a shared lock. In the slower case,
    // we take a unique lock and populate these vectors.
    tf_shared_lock lock(mu_);

    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types &&
        !gpu_host_allocators_.empty() &&
        gpu_host_allocators_[0].recording_allocator != nullptr) {
      return gpu_host_allocators_[0].recording_allocator.get();
    }
    if (static_cast<int>(gpu_host_allocators_.size()) > numa_node) {
      return gpu_host_allocators_[0].allocator.get();
    }
  }

  mutex_lock lock(mu_);
  // Find the first valid StreamExecutor to request CUDA or ROCm host memory
  // through, since any will work.
  //
  // This search isn't super clean, and it would be nice to use a
  // better source of information about which executor to use.  For
  // example, process_state could maybe save the first stream executor
  // it knows is valid.
  se::StreamExecutor* se = nullptr;
  for (int i = 0; i < static_cast<int>(gpu_allocators_.size()); ++i) {
    if (gpu_allocators_[i].allocator != nullptr) {
      se = DeviceIdUtil::ExecutorForTfDeviceId(DEVICE_GPU, GPUMachineManager(),
                                               TfDeviceId(i))
               .ValueOrDie();
      break;
    }
  }

  CHECK_NE(nullptr, se);

  while (static_cast<int>(gpu_host_allocators_.size()) <= numa_node) {
    while (gpu_host_alloc_visitors_.size() <= numa_node) {
      gpu_host_alloc_visitors_.push_back({});
    }
    while (gpu_host_free_visitors_.size() <= numa_node) {
      gpu_host_free_visitors_.push_back({});
    }
    SubAllocator* sub_allocator = new DeviceHostAllocator(
        se, numa_node, gpu_host_alloc_visitors_[numa_node],
        gpu_host_free_visitors_[numa_node]);
    // TODO(zheng-xq): evaluate whether 64GB by default is the best choice.
    int64_t gpu_host_mem_limit_in_mb = -1;
    Status status = ReadInt64FromEnvVar("TF_GPU_HOST_MEM_LIMIT_IN_MB",
                                        1LL << 16 /*64GB max by default*/,
                                        &gpu_host_mem_limit_in_mb);
    if (!status.ok()) {
      LOG(ERROR) << "GetGpuHostAllocator: " << status.error_message();
    }
    int64_t gpu_host_mem_limit = gpu_host_mem_limit_in_mb * (1LL << 20);

    BFCAllocator::Options allocator_opts;
    allocator_opts.allow_growth = true;
    Allocator* allocator =
        new BFCAllocator(absl::WrapUnique(sub_allocator), gpu_host_mem_limit,
                         /*name=*/"gpu_host_bfc", allocator_opts);

    if (LogMemory::IsEnabled() && !allocator->TracksAllocationSizes()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingAllocator(allocator, true);
    }
    gpu_host_allocators_.push_back({std::unique_ptr<Allocator>(allocator),
                                    std::unique_ptr<SharedCounter>(nullptr),
                                    nullptr, sub_allocator,
                                    std::unique_ptr<Allocator>(nullptr)});
    AllocatorParts& allocator_parts = gpu_host_allocators_.back();
    if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
      ProcessState::MemDesc md;
      md.loc = ProcessState::MemDesc::CPU;
      md.dev_index = 0;
      md.gpu_registered = true;
      md.nic_registered = false;
      allocator_parts.recording_allocator.reset(
          new internal::RecordingAllocator(&process_state_->mem_desc_map_,
                                           allocator_parts.allocator.get(), md,
                                           &mu_));
    }
  }
  if (process_state_->ProcessState::FLAGS_brain_gpu_record_mem_types) {
    return gpu_host_allocators_[0].recording_allocator.get();
  } else {
    return gpu_host_allocators_[0].allocator.get();
  }
}

void GPUProcessState::AddGPUAllocVisitor(int bus_id,
                                         const SubAllocator::Visitor& visitor) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_9(mht_9_v, 589, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::AddGPUAllocVisitor");

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  mutex_lock lock(mu_);
  CHECK(gpu_allocators_.empty())  // Crash OK
      << "AddGPUAllocVisitor must be called before "
         "first call to GetGPUAllocator.";
  DCHECK_GE(bus_id, 0);
  while (bus_id >= static_cast<int64_t>(gpu_visitors_.size())) {
    gpu_visitors_.push_back(std::vector<SubAllocator::Visitor>());
  }
  gpu_visitors_[bus_id].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::AddGpuHostAllocVisitor(
    int numa_node, const SubAllocator::Visitor& visitor) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_10(mht_10_v, 608, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::AddGpuHostAllocVisitor");

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  mutex_lock lock(mu_);
  CHECK(gpu_host_allocators_.empty())  // Crash OK
      << "AddGpuHostAllocVisitor must be called before "
         "first call to GetGpuHostAllocator.";
  while (numa_node >= static_cast<int64_t>(gpu_host_alloc_visitors_.size())) {
    gpu_host_alloc_visitors_.push_back(std::vector<SubAllocator::Visitor>());
  }
  gpu_host_alloc_visitors_[numa_node].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::AddGpuHostFreeVisitor(
    int numa_node, const SubAllocator::Visitor& visitor) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_11(mht_11_v, 626, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::AddGpuHostFreeVisitor");

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  mutex_lock lock(mu_);
  CHECK(gpu_host_allocators_.empty())  // Crash OK
      << "AddGpuHostFreeVisitor must be called before "
         "first call to GetGpuHostAllocator.";
  while (numa_node >= static_cast<int64_t>(gpu_host_free_visitors_.size())) {
    gpu_host_free_visitors_.push_back(std::vector<SubAllocator::Visitor>());
  }
  gpu_host_free_visitors_[numa_node].push_back(visitor);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

void GPUProcessState::TestOnlyReset() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTcc mht_12(mht_12_v, 643, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.cc", "GPUProcessState::TestOnlyReset");

  if (process_state_) {
    process_state_->ProcessState::TestOnlyReset();
  }
  {
    mutex_lock lock(mu_);
    gpu_device_enabled_ = false;
    gpu_allocators_.clear();
    gpu_visitors_.clear();
    gpu_host_allocators_.clear();
    gpu_host_alloc_visitors_.clear();
    gpu_host_free_visitors_.clear();
  }
}

}  // namespace tensorflow
