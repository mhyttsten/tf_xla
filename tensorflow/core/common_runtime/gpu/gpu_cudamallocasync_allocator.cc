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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/types/optional.h"
#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif  // GOOGLE_CUDA

#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

#if GOOGLE_CUDA
static std::string GetCudaErrorMessage(CUresult result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GetCudaErrorMessage");

  const char* error;
  cuGetErrorString(result, &error);
  const char* name;
  cuGetErrorName(result, &name);
  return absl::StrCat("CUDA error: ", error ? error : "<unknown>", " (",
                      name ? name : "Unknown", ")");
}
#endif  // GOOGLE_CUDA

void GpuCudaMallocAsyncAllocator::PrintAllocatorStatistics() {
  mutex_lock lock(lock_);

  std::map<size_t, int> size_map_historgram;
  std::vector<string> ptr_size_string;
  for (auto p : size_map_) {
    if (VLOG_IS_ON(8)) {
      ptr_size_string.push_back(
          absl::StrCat("(", absl::Hex(p.first), ",", p.second) + ")");
    }
    size_map_historgram[p.second]++;
  }
  LOG(ERROR) << "Histogram of current allocation: (allocation_size_in_bytes, "
             << "nb_allocation_of_that_sizes), ...;";
  for (auto p : size_map_historgram) {
    LOG(ERROR) << p.first << ", " << p.second;
  }

  VLOG(8) << "\nThe sorted list of (ptr,size):";
  VLOG(8) << absl::StrJoin(ptr_size_string, ",");

#if CUDA_VERSION >= 11030
  cuuint64_t mem_reserved_current;
  if (auto result = cuMemPoolGetAttribute(
          pool_, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, &mem_reserved_current)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << GetCudaErrorMessage(result);
  }
  cuuint64_t mem_used_current;
  if (auto result = cuMemPoolGetAttribute(
          pool_, CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &mem_used_current)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << GetCudaErrorMessage(result);
  }
  cuuint64_t mem_reserved_high;
  if (auto result = cuMemPoolGetAttribute(
          pool_, CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH, &mem_reserved_high)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << GetCudaErrorMessage(result);
  }
  cuuint64_t mem_used_high;
  if (auto result = cuMemPoolGetAttribute(pool_, CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                                          &mem_used_high)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << GetCudaErrorMessage(result);
  }
  LOG(ERROR) << "CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: "
             << mem_reserved_current;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_USED_MEM_CURRENT: " << mem_used_current;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: " << mem_reserved_high;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_USED_MEM_HIGH: " << mem_used_high;
#endif
}

std::atomic<int> GpuCudaMallocAsyncAllocator::number_instantiated_(0);

GpuCudaMallocAsyncAllocator::GpuCudaMallocAsyncAllocator(
    PlatformDeviceId platform_device_id, size_t pool_size, bool reserve_memory,
    bool compute_stats)
    : name_(absl::StrCat("gpu_async_", platform_device_id.value())),
      reserve_memory_(reserve_memory) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_1(mht_1_v, 276, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::GpuCudaMallocAsyncAllocator");

  ++number_instantiated_;

  // Stop clang from complaining about unused private fields when
  // TF_CUDA_MALLOC_ASYNC_SUPPORTED is not defined.
  (void)reserve_memory_;

#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  stream_exec_ = DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                           platform_device_id)
                     .ValueOrDie();
  // Initialized here as it only exist if compiled with a recent
  // enough CUDA.
  pool_ = nullptr;
  cuda_stream_ = nullptr;
  int driverVersion;
  cuDriverGetVersion(&driverVersion);
  VLOG(2) << "DRIVER VERSION: " << driverVersion;
  if (driverVersion < 11020) {
    LOG(FATAL)  // Crash OK.
        << "Disable cuda_malloc_async or update your CUDA driver to a version"
        << " compitible with CUDA 11.2 or higher."
        << " We detected a version compatible with: " << driverVersion;
  }

  // WAR an CUDA 11.2 driver bug for multiple-GPU. It currently
  // request that the context on GPU 0 is initialized. Which isn't the
  // case for TF+horovod.
  if (platform_device_id.value() > 0 && driverVersion < 11030) {
    CUcontext pctx;  // We loose track of it. But this is fine.
    if (auto result = cuDevicePrimaryCtxRetain(&pctx, 0))
      LOG(FATAL)  // Crash OK.
          << "Failed to retain context: " << GetCudaErrorMessage(result);
  }

  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};

  // Check the the CUDA runtime is recent enough.
  if (auto status2 = cuDriverGetVersion(&driverVersion)) {
    LOG(FATAL)  // Crash OK.
        << "Error while fetching driver version: "
        << GetCudaErrorMessage(status2);
  }

  // Check that cudaMallocAsync is supported.
  int cuda_malloc_async_supported;
  if (auto status =
          cuDeviceGetAttribute(&cuda_malloc_async_supported,
                               CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                               platform_device_id.value())) {
    LOG(FATAL)  // Crash OK.
        << "On device: " << platform_device_id.value()
        << " Current driver: " << driverVersion
        << ". Failed to get device attribute : " << GetCudaErrorMessage(status);
  }
  if (!cuda_malloc_async_supported)
    LOG(FATAL)  // Crash OK.
        << "TF_GPU_ALLOCATOR=cuda_malloc_async isn't currently supported on "
        << "GPU id " << platform_device_id.value() << ":"
        << " Possible causes: device not supported (request SM60+), driver too "
           "old, "
        << " OS not supported, CUDA version too old(request CUDA11.2+).";

  if (auto status =
          cuDeviceGetDefaultMemPool(&pool_, platform_device_id.value()))
    LOG(FATAL) <<  // Crash OK.
        "Failed to get default CUDA pool: " << GetCudaErrorMessage(status);

  VLOG(1) << Name() << " CudaMallocAsync initialized on platform: "
          << platform_device_id.value() << " with pool size of: " << pool_size
          << " this ptr: " << this;
  uint64_t pool_size_64 = pool_size;
  if (auto status = cuMemPoolSetAttribute(
          pool_, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &pool_size_64))
    LOG(FATAL) <<  // Crash OK.
        "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);

  if (compute_stats) {
    stats_ = std::make_unique<AllocatorStats>();
    stats_->bytes_limit = static_cast<int64_t>(pool_size);
  }  // If not set, it means we do not compute stats.

  // If in TF_DETERMINISTIC_ALLOCATOR is set, then make the allocator behave
  // determistically.
  bool deterministic = false;
  TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DETERMINISTIC_ALLOCATOR",
                                             /*default_val=*/false,
                                             &deterministic));
  if (deterministic) {
    int disable = 0;
    if (auto status = cuMemPoolSetAttribute(
            pool_, CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &disable)) {
      LOG(FATAL) <<  // Crash OK.
          "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);
    }
    if (auto status = cuMemPoolSetAttribute(
            pool_, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            &disable)) {
      LOG(FATAL) <<  // Crash OK.
          "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);
    }
  }

  // Set read/write access to all GPUs.
  static auto* all_pools_ = new std::vector<CUmemoryPool*>();
  static auto* all_ids_ = new std::vector<PlatformDeviceId>();
  DCHECK(all_pools_->size() == all_ids_->size());
  for (int i = 0; i < all_pools_->size(); ++i) {
    // Set the current pool access to the previous GPUs.
    CUmemAccessDesc map;
    map.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    map.location.id = (*all_ids_)[i].value();

    map.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    VLOG(2) << "Setting access of the current pool to "
            << " location id: " << map.location.id;
    int canAccessPeer;
    if (auto status = cuDeviceCanAccessPeer(
            &canAccessPeer, platform_device_id.value(), map.location.id)) {
      pool_ = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed to know if GPU id "
          << map.location.id << " can access GPU id "
          << platform_device_id.value() << ": " << GetCudaErrorMessage(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess(pool_, &map, 1)) {
        pool_ = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << i
            << " location id: " << map.location.id
            << " error: " << GetCudaErrorMessage(status);
      }
    }

    // Set the previous pools access to the current GPU.
    map.location.id = platform_device_id.value();

    VLOG(2) << "Set access to the pool id: " << i
            << " location id: " << map.location.id;
    if (auto status = cuDeviceCanAccessPeer(&canAccessPeer, i,
                                            platform_device_id.value())) {
      pool_ = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed: " << GetCudaErrorMessage(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess(*(*all_pools_)[i], &map, 1)) {
        pool_ = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << i
            << " location id: " << map.location.id
            << " error: " << GetCudaErrorMessage(status);
      }
    }
  }
  all_pools_->push_back(&pool_);
  all_ids_->push_back(platform_device_id);

  VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator PoolSize " << pool_size;
#else   // TF_CUDA_MALLOC_ASYNC_SUPPORTED
  LOG(FATAL) << "GpuCudaMallocAsyncAllocator requires CUDA 11.2+";  // Crash OK.
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED
}

GpuCudaMallocAsyncAllocator::~GpuCudaMallocAsyncAllocator() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_2(mht_2_v, 444, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::~GpuCudaMallocAsyncAllocator");
}

void* GpuCudaMallocAsyncAllocator::AllocateRaw(size_t alignment,
                                               size_t num_bytes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_3(mht_3_v, 450, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::AllocateRaw");

#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  CHECK(cuda_stream_ != nullptr)
      << "A stream must be added to the GpuCudaMallocAsync allocator";
  if (pool_ == nullptr) {
    LOG(FATAL)  // Crash OK.
        << "The instantiation of GpuCudaMallocAsyncAllocator failed."
        << " See previous errors.";
  }
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  void* ptr = nullptr;
  if (auto result =
          cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&ptr),
                                  num_bytes, pool_, cuda_stream_)) {
    size_t free, total;
    cuMemGetInfo(&free, &total);
    LOG(ERROR) << Name() << " cuMemAllocAsync failed to allocate " << num_bytes
               << " bytes: " << GetCudaErrorMessage(result)
               << "\n Reported by CUDA: Free memory/Total memory: " << free
               << "/" << total;
    if (auto stats = GetStats())
      LOG(ERROR) << "Stats: " << stats->DebugString();

    PrintAllocatorStatistics();

    return nullptr;
  }

  // Update stats.
  if (stats_) {
    mutex_lock lock(lock_);
    ++(stats_->num_allocs);
    stats_->bytes_in_use += num_bytes;
    if (stats_->bytes_in_use > stats_->peak_bytes_in_use) {
      VLOG(9) << "New Peak memory usage of " << stats_->bytes_in_use
              << " bytes.";
    }
    stats_->peak_bytes_in_use =
        std::max(stats_->peak_bytes_in_use, stats_->bytes_in_use);
    stats_->largest_alloc_size =
        std::max<std::size_t>(stats_->largest_alloc_size, num_bytes);
    size_map_[ptr] = num_bytes;
  }
  VLOG(10) << Name() << " Allocated " << num_bytes << " at " << ptr;
  return ptr;
#else   // TF_CUDA_MALLOC_ASYNC_SUPPORTED
  return nullptr;
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED
}
void GpuCudaMallocAsyncAllocator::DeallocateRaw(void* ptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_4(mht_4_v, 502, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::DeallocateRaw");

#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  if (ptr == nullptr) return;
  if (auto result = cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(ptr),
                                   cuda_stream_)) {
    if (result == CUDA_ERROR_DEINITIALIZED) {
      // It happens with multi-GPU that TF free the GPU allocation after
      // the driver is unloaded. It is safe to ignore this error here.
      // TODO: Find how to fix the shutdown steps in TF.
      VLOG(1) << "Ignoring CUDA error: " << GetCudaErrorMessage(result);
    } else {
      size_t free, total;
      se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
      cuMemGetInfo(&free, &total);
      LOG(ERROR) << "cudaFreeAsync failed to free " << ptr << ": "
                 << GetCudaErrorMessage(result)
                 << "\n Free memory/Total memory: " << free << "/" << total;
      if (auto stats = GetStats())
        LOG(ERROR) << "Stats: " << stats->DebugString();
    }
  }

  // Updates the stats.
  if (stats_) {
    mutex_lock lock(lock_);
    DCHECK(size_map_.contains(ptr));
    size_t size = size_map_[ptr];
    stats_->bytes_in_use -= size;
    size_map_.erase(ptr);
  }

  VLOG(10) << Name() << " Freed ptr: " << ptr;
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED
}

bool GpuCudaMallocAsyncAllocator::TracksAllocationSizes() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_5(mht_5_v, 540, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::TracksAllocationSizes");

  return static_cast<bool>(stats_);
}

size_t GpuCudaMallocAsyncAllocator::RequestedSize(const void* ptr) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_6(mht_6_v, 547, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::RequestedSize");

  if (!stats_ || !ptr) return 0;
  mutex_lock l(lock_);
  return size_map_.at(ptr);
}

size_t GpuCudaMallocAsyncAllocator::AllocatedSize(const void* ptr) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_7(mht_7_v, 556, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::AllocatedSize");

  if (!stats_ || !ptr) return 0;
  mutex_lock l(lock_);
  return size_map_.at(ptr);
}

absl::optional<AllocatorStats> GpuCudaMallocAsyncAllocator::GetStats() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_8(mht_8_v, 565, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::GetStats");

  if (!stats_) return absl::nullopt;
  mutex_lock l(lock_);
  return *stats_;
}

bool GpuCudaMallocAsyncAllocator::ClearStats() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_9(mht_9_v, 574, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::ClearStats");

  if (!stats_) return false;
  mutex_lock l(lock_);
  stats_->num_allocs = 0;
  stats_->peak_bytes_in_use = stats_->bytes_in_use;
  stats_->largest_alloc_size = 0;
  return true;
}

void GpuCudaMallocAsyncAllocator::SetStreamAndPreallocateMemory(void* stream) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTcc mht_10(mht_10_v, 586, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.cc", "GpuCudaMallocAsyncAllocator::SetStreamAndPreallocateMemory");

#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  if (cuda_stream_ != nullptr) {
    LOG(FATAL) <<  // Crash OK.
        "Trying to set the stream twice. This isn't supported. ";
  }

  uint64_t pool_size_64 = 0;
  if (auto status = cuMemPoolGetAttribute(
          pool_, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &pool_size_64)) {
    LOG(FATAL) <<  // Crash OK.
        "Failed to get CUDA pool attribute: " << GetCudaErrorMessage(status);
  }
  cuda_stream_ = *(reinterpret_cast<CUstream*>(stream));
  int64 prealloc_size = 0;
  // TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=-1 is a special value that
  // preallocates the total pool size.
  TF_CHECK_OK(ReadInt64FromEnvVar("TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC", 0,
                                  &prealloc_size));
  if (prealloc_size == -1) {
    prealloc_size = pool_size_64;
  } else if (reserve_memory_) {
    prealloc_size = pool_size_64;
  }

  if (prealloc_size != 0) {
    void* ptr = AllocateRaw(0, prealloc_size);
    DeallocateRaw(ptr);
    VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator reserved the pool for "
            << prealloc_size << " bytes"
            << ". First ptr: " << ptr;
    ClearStats();
  }
#endif
}

}  // namespace tensorflow
