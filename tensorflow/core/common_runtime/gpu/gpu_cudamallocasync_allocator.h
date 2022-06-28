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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_CUDAMALLOCASYNC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_CUDAMALLOCASYNC_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTh() {
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

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

#if GOOGLE_CUDA
#define TF_CUDA_MALLOC_ASYNC_SUPPORTED CUDA_VERSION >= 11020
#endif

// An allocator that wraps cudaMallocAsync. It has fewer fragmentation
// issues then the BFC memory allocator.  The compute-sanitizer tool
// helps to detect OOB memory errors when using cudaMallocAsync. Use
// the environment variable `TF_GPU_ALLOCATOR=cuda_malloc_async` to
// enable it.
//
// It needs CUDA 11.2+. When using a container, this only needs the
// container driver to be 11.2. It has a WAR again a driver bug in
// multi-GPU setup with CUDA 11.2. The WAR creates an extra context on
// GPU 0.
//
// We configure cudaMallocAsync to grow when more memory is needed
// instead of preallocating everything up front and to keep a local
// pool up to pool_size bytes that is never released to other processes.
// So no other process will "steal" the GPU memory already used by the
// current process. This is to speed up execution and prevent crashes
// of long-running jobs. Use `reserve_memory=true` if you want to
// preallocate the full pool_size. You can also use the environment
// variable `TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=nb_bytes` to preallocate
// that amount of memory. `TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=-1` is a
// special value that preallocate all what the BFC memory allocator
// would have allocated. This is useful when benchmarking as it doesn't
// change when driver allocations are done.
//
// Here, the pool_size isn't the absolute max as for [Gpu]BFCAllocator.
// The pool can grow above that up to the total GPU memory.  But the
// driver can return the excess memory to other processes.
class GpuCudaMallocAsyncAllocator : public Allocator {
 public:
  explicit GpuCudaMallocAsyncAllocator(PlatformDeviceId platform_device_id,
                                       size_t pool_size,
                                       bool reserve_memory = false,
                                       bool compute_stats = true);
  ~GpuCudaMallocAsyncAllocator() override;
  string Name() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTh mht_0(mht_0_v, 241, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.h", "Name");
 return name_; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  bool TracksAllocationSizes() const override;

  size_t RequestedSize(const void* ptr) const override;

  size_t AllocatedSize(const void* ptr) const override;

  absl::optional<AllocatorStats> GetStats() override;

  bool ClearStats() override;

  void SetStreamAndPreallocateMemory(void* stream) override;

  // With the right VLOG set, it prints:
  // - the number of ptr currently allocated per size (histogram).
  // - each ptr value and its size.
  // - If CUDA_VERSION >= 11030, print cudaMallocAsync statistics.
  void PrintAllocatorStatistics();

  static int GetInstantiatedCountTestOnly() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTh mht_1(mht_1_v, 266, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.h", "GetInstantiatedCountTestOnly");
 return number_instantiated_; }

  AllocatorMemoryType GetMemoryType() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_cudamallocasync_allocatorDTh mht_2(mht_2_v, 271, "", "./tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.h", "GetMemoryType");

    return AllocatorMemoryType::kDevice;
  }

 private:
#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  se::StreamExecutor* stream_exec_;  // Not owned.

  // cudaMallocAsync is stream aware. But TF StreamExecutor use only 1
  // compute stream and already synchronize with the h2d, d2h and d2d
  // stream. So we do not need to ask cudaMallocAsync to add extra
  // synchronization.
  // Not owned.
  CUstream cuda_stream_;

  // Not owned. The default pool of the associated GPU.
  // If null, then the instanciation failed and the first allocation
  // will return an error.
  CUmemoryPool pool_;
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED

  // Just a counter for the number of time this class is instantiated.
  // Only useful for tests.
  static std::atomic<int> number_instantiated_;

  string name_;

  bool reserve_memory_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuCudaMallocAsyncAllocator);

  // Stats.
  // Structures mutable after construction
  mutable mutex lock_;
  std::unique_ptr<AllocatorStats> stats_ TF_PT_GUARDED_BY(lock_);
  absl::flat_hash_map<const void*, size_t> size_map_ TF_GUARDED_BY(lock_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_CUDAMALLOCASYNC_ALLOCATOR_H_
