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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc() {
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

#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"

#include <cstddef>
#include <vector>

#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"

#define MASK_WORDS 2
#define MASK_BYTES (MASK_WORDS * sizeof(int64_t))

namespace tensorflow {
namespace {

int64_t* NewMask(int64_t word) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "NewMask");

  int64_t* m = new int64_t[MASK_WORDS];
  for (int i = 0; i < MASK_WORDS; ++i) {
    m[i] = word;
  }
  return m;
}

int64_t* before_mask = NewMask(0xabababababababab);
int64_t* after_mask = NewMask(0xcdcdcdcdcdcdcdcd);

bool CheckMask(se::StreamExecutor* exec, void* ptr, int64_t* mask) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "CheckMask");

  se::DeviceMemory<int64_t> gpu_ptr{se::DeviceMemoryBase{ptr, MASK_BYTES}};
  int64_t tmp[MASK_WORDS];

  Status result = exec->SynchronousMemcpyD2H(gpu_ptr, MASK_BYTES, tmp);
  if (!result.ok()) {
    LOG(FATAL) << "Could not copy debug mask, " << result;
  }

  bool ok = true;
  for (int i = 0; i < MASK_WORDS; ++i) {
    ok &= (mask[i] == tmp[i]);
    if (!ok) {
      LOG(ERROR) << "i=" << i
                 << " mask=" << reinterpret_cast<const void*>(mask[i])
                 << " field=" << reinterpret_cast<const void*>(tmp[i]);
    }
  }

  return ok;
}

void InitMask(se::StreamExecutor* exec, void* ptr, int64_t* mask) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "InitMask");

  se::DeviceMemory<int64_t> gpu_ptr{se::DeviceMemoryBase{ptr, MASK_BYTES}};
  Status result = exec->SynchronousMemcpyH2D(mask, MASK_BYTES, &gpu_ptr);
  if (!result.ok()) {
    LOG(FATAL) << "Could not copy debug mask, " << result;
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// GPUDebugAllocator
// -----------------------------------------------------------------------------
GPUDebugAllocator::GPUDebugAllocator(Allocator* allocator,
                                     PlatformDeviceId platform_device_id)
    : base_allocator_(allocator) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::GPUDebugAllocator");

  stream_exec_ = DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                           platform_device_id)
                     .ValueOrDie();
}

GPUDebugAllocator::~GPUDebugAllocator() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_4(mht_4_v, 267, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::~GPUDebugAllocator");
 delete base_allocator_; }

void* GPUDebugAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_5(mht_5_v, 272, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::AllocateRaw");

  num_bytes += (2 * MASK_BYTES);
  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);
  if (allocated_ptr == nullptr) return allocated_ptr;

  // Return the pointer after the header
  void* rv = static_cast<char*>(allocated_ptr) + MASK_BYTES;

  // Write the header at allocated_ptr
  InitMask(stream_exec_, allocated_ptr, before_mask);

  // Write the footer at the end.
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  InitMask(stream_exec_,
           static_cast<char*>(allocated_ptr) + req_size - MASK_BYTES,
           after_mask);
  return rv;
}
void GPUDebugAllocator::DeallocateRaw(void* ptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_6(mht_6_v, 293, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::DeallocateRaw");

  if (ptr != nullptr) {
    CHECK(CheckHeader(ptr)) << "before_mask has been overwritten";
    CHECK(CheckFooter(ptr)) << "after_mask has been overwritten";

    // Backtrack to the beginning of the header.
    ptr = static_cast<void*>(static_cast<char*>(ptr) - MASK_BYTES);
  }
  // Deallocate the memory
  base_allocator_->DeallocateRaw(ptr);
}

bool GPUDebugAllocator::TracksAllocationSizes() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_7(mht_7_v, 308, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::TracksAllocationSizes");
 return true; }

size_t GPUDebugAllocator::RequestedSize(const void* ptr) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_8(mht_8_v, 313, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::RequestedSize");

  auto req_size = base_allocator_->RequestedSize(static_cast<const char*>(ptr) -
                                                 MASK_BYTES);
  return req_size - 2 * MASK_BYTES;
}

size_t GPUDebugAllocator::AllocatedSize(const void* ptr) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_9(mht_9_v, 322, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::AllocatedSize");

  return base_allocator_->AllocatedSize(static_cast<const char*>(ptr) -
                                        MASK_BYTES);
}

int64_t GPUDebugAllocator::AllocationId(const void* ptr) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_10(mht_10_v, 330, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::AllocationId");

  return base_allocator_->AllocationId(static_cast<const char*>(ptr) -
                                       MASK_BYTES);
}

absl::optional<AllocatorStats> GPUDebugAllocator::GetStats() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_11(mht_11_v, 338, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::GetStats");

  return base_allocator_->GetStats();
}

bool GPUDebugAllocator::ClearStats() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_12(mht_12_v, 345, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::ClearStats");
 return base_allocator_->ClearStats(); }

bool GPUDebugAllocator::CheckHeader(void* ptr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_13(mht_13_v, 350, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::CheckHeader");

  return CheckMask(stream_exec_, static_cast<char*>(ptr) - MASK_BYTES,
                   before_mask);
}

bool GPUDebugAllocator::CheckFooter(void* ptr) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_14(mht_14_v, 358, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUDebugAllocator::CheckFooter");

  char* original_ptr = static_cast<char*>(ptr) - MASK_BYTES;
  size_t req_size = base_allocator_->RequestedSize(original_ptr);
  return CheckMask(stream_exec_, original_ptr + req_size - MASK_BYTES,
                   after_mask);
}

// -----------------------------------------------------------------------------
// GPUNanResetAllocator
// -----------------------------------------------------------------------------
GPUNanResetAllocator::GPUNanResetAllocator(Allocator* allocator,
                                           PlatformDeviceId platform_device_id)
    : base_allocator_(allocator) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_15(mht_15_v, 373, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUNanResetAllocator::GPUNanResetAllocator");

  stream_exec_ = DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                           platform_device_id)
                     .ValueOrDie();
}

GPUNanResetAllocator::~GPUNanResetAllocator() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_16(mht_16_v, 382, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUNanResetAllocator::~GPUNanResetAllocator");
 delete base_allocator_; }

void* GPUNanResetAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_17(mht_17_v, 387, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUNanResetAllocator::AllocateRaw");

  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);
  if (allocated_ptr == nullptr) return allocated_ptr;

  // Initialize the buffer to Nans
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  std::vector<float> nans((req_size + sizeof(float) - 1) / sizeof(float),
                          std::nanf(""));
  se::DeviceMemory<float> nan_ptr{
      se::DeviceMemoryBase{static_cast<float*>(allocated_ptr), req_size}};

  Status result =
      stream_exec_->SynchronousMemcpyH2D(&nans[0], req_size, &nan_ptr);
  if (!result.ok()) {
    LOG(ERROR) << "Could not initialize to NaNs, " << result;
  }

  return allocated_ptr;
}
void GPUNanResetAllocator::DeallocateRaw(void* ptr) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_18(mht_18_v, 409, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUNanResetAllocator::DeallocateRaw");

  if (ptr != nullptr) {
    // Reset the buffer to Nans
    size_t req_size = base_allocator_->RequestedSize(ptr);
    std::vector<float> nans((req_size + sizeof(float) - 1) / sizeof(float),
                            std::nanf(""));
    se::DeviceMemory<float> nan_ptr{
        se::DeviceMemoryBase{static_cast<float*>(ptr), req_size}};
    Status result =
        stream_exec_->SynchronousMemcpyH2D(&nans[0], req_size, &nan_ptr);
    if (!result.ok()) {
      LOG(ERROR) << "Could not initialize to NaNs, " << result;
    }
  }

  // Deallocate the memory
  base_allocator_->DeallocateRaw(ptr);
}

size_t GPUNanResetAllocator::RequestedSize(const void* ptr) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_19(mht_19_v, 431, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUNanResetAllocator::RequestedSize");

  return base_allocator_->RequestedSize(ptr);
}

size_t GPUNanResetAllocator::AllocatedSize(const void* ptr) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_20(mht_20_v, 438, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUNanResetAllocator::AllocatedSize");

  return base_allocator_->AllocatedSize(ptr);
}

absl::optional<AllocatorStats> GPUNanResetAllocator::GetStats() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_21(mht_21_v, 445, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUNanResetAllocator::GetStats");

  return base_allocator_->GetStats();
}

bool GPUNanResetAllocator::ClearStats() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_debug_allocatorDTcc mht_22(mht_22_v, 452, "", "./tensorflow/core/common_runtime/gpu/gpu_debug_allocator.cc", "GPUNanResetAllocator::ClearStats");

  return base_allocator_->ClearStats();
}

}  // namespace tensorflow
