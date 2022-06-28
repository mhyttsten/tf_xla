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
class MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc() {
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

#include "tensorflow/stream_executor/cuda/cuda_driver.h"

#include <stdint.h>
#include <stdlib.h>

#include <map>
#include <set>
#include <utility>

#include "absl/base/casts.h"
#include "absl/base/const_init.h"
#include "absl/container/inlined_vector.h"
#include "absl/debugging/leak_check.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/human_readable.h"
#include "tensorflow/stream_executor/lib/stacktrace.h"
#include "tensorflow/stream_executor/lib/static_threadlocal.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

bool FLAGS_gpuexec_cuda_driver_inject_init_error = false;
bool FLAGS_gpuexec_cuda_sync_around_driver_calls = false;
bool FLAGS_gpuexec_cuda_device_0_only = false;

#define RETURN_IF_CUDA_RES_ERROR(expr, ...)                            \
  do {                                                                 \
    CUresult _res = (expr);                                            \
    if (TF_PREDICT_FALSE(_res != CUDA_SUCCESS)) {                      \
      return port::InternalError(absl::StrCat(                         \
          __VA_ARGS__, ": ", ::stream_executor::gpu::ToString(_res))); \
    }                                                                  \
  } while (0)

#define FAIL_IF_CUDA_RES_ERROR(expr, ...)                   \
  do {                                                      \
    CUresult _res = (expr);                                 \
    if (TF_PREDICT_FALSE(_res != CUDA_SUCCESS)) {           \
      LOG(FATAL) << absl::StrCat(__VA_ARGS__) << ": "       \
                 << ::stream_executor::gpu::ToString(_res); \
    }                                                       \
  } while (0)

// Debugging: on each push and pop of a cuda context, verify the current context
// matches the expected one.
constexpr bool kVerifyGpuContext = false;

namespace stream_executor {
namespace gpu {

/* static */ absl::Mutex CreatedContexts::mu_{absl::kConstInit};
/* static */ int64_t CreatedContexts::next_id_ = 1;  // 0 means "no context"

namespace {

bool UseCudaMallocAsyncAllocator() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_0(mht_0_v, 248, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "UseCudaMallocAsyncAllocator");

  static const char* debug_allocator_str = std::getenv("TF_GPU_ALLOCATOR");
  return debug_allocator_str != nullptr &&
         std::strcmp(debug_allocator_str, "cuda_malloc_async") == 0;
}

// Returns the current context and checks that it is in the set of CUDA contexts
// created by StreamExecutor (to ensure that the CUDA runtime didn't create a
// context behind our backs).
CUcontext CurrentContext() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_1(mht_1_v, 260, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "CurrentContext");

  CUcontext current = cuda::CurrentContextOrDie();
  if (current != nullptr && !CreatedContexts::Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "cuda_driver API: "
               << current
               << "; a CUDA runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
  return current;
}

// CUDA driver routines may require a large amount of stack (particularly
// cuModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
port::ThreadPool* GetDriverExecutor() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_2(mht_2_v, 280, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GetDriverExecutor");

  static port::ThreadPool* thread_pool = new port::ThreadPool(
      port::Env::Default(), port::ThreadOptions(), "cuda_driver", 1);
  return thread_pool;
}

}  // namespace

std::string MemorySpaceString(MemorySpace memory_space) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_3(mht_3_v, 291, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "MemorySpaceString");

  switch (memory_space) {
    case MemorySpace::kHost:
      return "host";
    case MemorySpace::kDevice:
      return "device";
    default:
      LOG(FATAL) << "impossible memory space";
  }
}

namespace {

// Call cuCtxtSynchronize and crash if it doesn't succeed.
void SynchronizeOrDie() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_4(mht_4_v, 308, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "SynchronizeOrDie");

  FAIL_IF_CUDA_RES_ERROR(cuCtxSynchronize(),
                         "Synchronize fail: ", port::CurrentStackTrace());
}

struct ThreadLocalData {
  int64_t id;
  GpuContext* context;  // Only valid if id == a known good context.
  int depth;
};

SE_STATIC_THREAD_LOCAL_POD(ThreadLocalData, tls_data);

}  // namespace

ScopedActivateContext::ScopedActivateContext(GpuContext* cuda_context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_5(mht_5_v, 326, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "ScopedActivateContext::ScopedActivateContext");

  if (FLAGS_gpuexec_cuda_sync_around_driver_calls) SynchronizeOrDie();

  auto* tls = &tls_data.get();

  // If this is an outermost scope, we must not assume that the CUDA context has
  // been left in the same state we left it. Other code may have run on this
  // thread and altered the context.
  if (tls->depth == 0) {
    VLOG(3) << "ScopedActivateContext switching to " << cuda_context->id();
    FAIL_IF_CUDA_RES_ERROR(cuCtxSetCurrent(cuda_context->context()),
                           "Failed setting context");
    tls->depth = 1;
    tls->id = cuda_context->id();
    tls->context = cuda_context;
    to_restore_ = nullptr;
    return;
  }

  tls->depth++;
  if (tls->id == cuda_context->id()) {
    if (kVerifyGpuContext) {
      CHECK_EQ(CurrentContext(), cuda_context->context());
    }
    DCHECK_EQ(CurrentContext(), cuda_context->context());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching context from " << tls->id
          << " to " << cuda_context->id();

  to_restore_ = tls->context;
  // Set the context and update thread local.
  FAIL_IF_CUDA_RES_ERROR(cuCtxSetCurrent(cuda_context->context()),
                         "Failed setting context");
  tls->id = cuda_context->id();
  tls->context = cuda_context;
}

ScopedActivateContext::~ScopedActivateContext() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_6(mht_6_v, 368, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "ScopedActivateContext::~ScopedActivateContext");

  if (FLAGS_gpuexec_cuda_sync_around_driver_calls) SynchronizeOrDie();

  auto* tls = &tls_data.get();

  if (kVerifyGpuContext) {
    // Note that if kVerifyGpuContext is used, and contexts are deleted, it's
    // possible this could fail in the CurrentContext() call.
    CHECK_EQ(CurrentContext(),
             tls->context == nullptr ? nullptr : tls->context->context());
  }

  tls->depth--;
  DCHECK_GE(tls->depth, 0);
  if (to_restore_ == nullptr) {
    // Leave context, tls->id, and tls->context set.
    return;
  }

  // Set context and update thread local.
  FAIL_IF_CUDA_RES_ERROR(cuCtxSetCurrent(to_restore_->context()),
                         "Failed setting context");
  tls->id = to_restore_->id();
  tls->context = to_restore_;
}

namespace {

// Returns a stringified device number associated with pointer, primarily for
// logging purposes. Returns "?" if the device could not be successfully
// queried.
std::string CUDAPointerToDeviceString(CUdeviceptr pointer) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_7(mht_7_v, 402, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "CUDAPointerToDeviceString");

  auto value = GpuDriver::GetPointerDevice(pointer);
  if (value.ok()) {
    return absl::StrCat(value.ValueOrDie());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified memory space associated with pointer, primarily for
// logging purposes. Returns "?" if the memory space could not be successfully
// queried.
std::string CUDAPointerToMemorySpaceString(CUdeviceptr pointer) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_8(mht_8_v, 417, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "CUDAPointerToMemorySpaceString");

  auto value = GpuDriver::GetPointerMemorySpace(pointer);
  if (value.ok()) {
    return MemorySpaceString(value.ValueOrDie());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified representation of whether or not peer access is
// permitted between the "from" and "to" pointers' associated contexts,
// primarily for logging purposes. Returns "error" if an error is encountered
// in the process of querying.
std::string CUDAPointersToCanAccessString(CUdeviceptr from, CUdeviceptr to) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_9(mht_9_v, 433, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "CUDAPointersToCanAccessString");

  auto from_context = GpuDriver::GetPointerContext(from);
  if (!from_context.ok()) {
    LOG(ERROR) << "could not retrieve source pointer's context: "
               << from_context.status();
    return "error";
  }
  auto to_context = GpuDriver::GetPointerContext(to);
  if (!to_context.ok()) {
    LOG(ERROR) << "could not retrieve destination pointer's context: "
               << to_context.status();
    return "error";
  }
  return GpuDriver::CanEnablePeerAccess(from_context.ValueOrDie(),
                                        to_context.ValueOrDie())
             ? "true"
             : "false";
}

// Actually performs the work of CUDA initialization. Wrapped up in one-time
// execution guard.
static port::Status InternalInit() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_10(mht_10_v, 457, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "InternalInit");

  CUresult res = CUDA_ERROR_NO_DEVICE;
  if (FLAGS_gpuexec_cuda_driver_inject_init_error) {
    LOG(ERROR) << "injecting CUDA init error; initialization will fail";
  } else {
    res = cuInit(0 /* = flags */);
  }

  if (res == CUDA_SUCCESS) {
    return port::Status::OK();
  } else if (res == CUDA_ERROR_SHARED_OBJECT_INIT_FAILED) {
    LOG(WARNING) << "failed call to cuInit: " << ToString(res);
  } else {
    LOG(ERROR) << "failed call to cuInit: " << ToString(res);
  }

  Diagnostician::LogDiagnosticInformation();
  return port::Status(port::error::ABORTED,
                      absl::StrCat("failed call to cuInit: ", ToString(res)));
}

}  // namespace

/* static */ port::Status GpuDriver::Init() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_11(mht_11_v, 483, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::Init");

  // Cached return value from calling InternalInit(), as cuInit need only be
  // called once, but GpuDriver::Init may be called many times.
  static port::Status* init_retval = [] {
    return new port::Status(InternalInit());
  }();
  return *init_retval;
}

/* static */ port::Status GpuDriver::GetDevice(int device_ordinal,
                                               CUdevice* device) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_12(mht_12_v, 496, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetDevice");

  RETURN_IF_CUDA_RES_ERROR(cuDeviceGet(device, device_ordinal),
                           "Failed call to cuDeviceGet");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::GetDeviceName(CUdevice device,
                                                   std::string* device_name) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_13(mht_13_v, 506, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetDeviceName");

  static const size_t kCharLimit = 64;
  absl::InlinedVector<char, 4> chars(kCharLimit);
  RETURN_IF_CUDA_RES_ERROR(
      cuDeviceGetName(chars.begin(), kCharLimit - 1, device),
      "Failed to get device name");
  chars[kCharLimit - 1] = '\0';
  *device_name = chars.begin();
  return port::Status::OK();
}

bool DeviceOptionsToContextFlags(const DeviceOptions& device_options,
                                 int* flags) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_14(mht_14_v, 521, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "DeviceOptionsToContextFlags");

  static_assert(DeviceOptions::kMask == 0xf,
                "needs update for new device options");

  if (device_options.flags() & DeviceOptions::kDoNotReclaimStackAllocation) {
    *flags |= CU_CTX_LMEM_RESIZE_TO_MAX;
  }

  // If no flags are set the default is CU_CTX_SCHED_AUTO, which
  // in Google environments is very likely to mean SPIN.
  if (device_options.flags() & DeviceOptions::kScheduleSpin) {
    *flags |= CU_CTX_SCHED_SPIN;
  }
  if (device_options.flags() & DeviceOptions::kScheduleYield) {
    *flags |= CU_CTX_SCHED_YIELD;
  }
  if (device_options.flags() & DeviceOptions::kScheduleBlockingSync) {
    *flags |= CU_CTX_SCHED_BLOCKING_SYNC;
  }

  return true;
}

/* static */ port::Status GpuDriver::CreateContext(
    int device_ordinal, CUdevice device, const DeviceOptions& device_options,
    GpuContext** context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_15(mht_15_v, 549, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::CreateContext");

  *context = nullptr;

  int flags = 0;
  if (!DeviceOptionsToContextFlags(device_options, &flags)) {
    LOG(WARNING) << "could not convert all device options into context flags";
  }

  CUresult res;
  CUcontext former_context;
  CUcontext new_context;

  unsigned int former_primary_context_flags;
  int former_primary_context_is_active;
  CHECK_EQ(CUDA_SUCCESS,
           cuDevicePrimaryCtxGetState(device, &former_primary_context_flags,
                                      &former_primary_context_is_active));
  if (former_primary_context_flags != flags) {
    if (former_primary_context_is_active) {
      LOG(ERROR)
          << "The primary context is active and has a different flag set ("
          << former_primary_context_flags << ") than the desired flag set ("
          << flags << ").";
    } else {
      CHECK_EQ(CUDA_SUCCESS, cuDevicePrimaryCtxSetFlags(device, flags));
    }
  }

  former_context = cuda::CurrentContextOrDie();
  res = cuDevicePrimaryCtxRetain(&new_context, device);
  if (former_context != nullptr) {
    CUdevice former_device;
    if (cuCtxGetDevice(&former_device) == CUDA_SUCCESS) {
      if (former_device == device) {
        if (former_context == new_context) {
          VLOG(2) << "The primary context " << former_context << " for device "
                  << device
                  << " exists before initializing the StreamExecutor.";
        } else {
          LOG(WARNING) << "A non-primary context " << former_context
                       << " for device " << device
                       << " exists before initializing the StreamExecutor. The "
                       << "primary context is now " << new_context << ". We "
                       << "haven't verified StreamExecutor works with that.";
        }
      }
    } else {
      LOG(ERROR) << "Failed to get the device of the current context "
                 << former_context;
    }
  }
  CHECK_EQ(CUDA_SUCCESS, cuCtxSetCurrent(former_context));

  if (res == CUDA_SUCCESS) {
    *context = CreatedContexts::Add(new_context, device_ordinal);
    CHECK(*context != nullptr)
        << "success in this call must entail non-null result";
    VLOG(2) << "created or reused context " << new_context
            << " for this thread";
    return port::Status::OK();
  }

  std::string message =
      "failed call to cuDevicePrimaryCtxRetain: " + ToString(res);
  if (res == CUDA_ERROR_OUT_OF_MEMORY) {
    uint64_t total_memory;
    if (GetDeviceTotalMemory(device, &total_memory)) {
      absl::StrAppend(&message, "; total memory reported: ", total_memory);
    } else {
      absl::StrAppend(&message, "; could not query total memory");
    }
  }

  return port::Status(port::error::INTERNAL, message);
}

/* static */ void GpuDriver::DestroyContext(GpuContext* context) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_16(mht_16_v, 628, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::DestroyContext");

  if (context == nullptr) {
    return;
  }
  CUcontext former_context = CurrentContext();
  CUresult res = cuCtxSetCurrent(context->context());
  CUdevice device;
  cuCtxGetDevice(&device);
  cuCtxSetCurrent(former_context);

  res = cuDevicePrimaryCtxRelease(device);

  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to release CUDA context; leaking: " << ToString(res);
  }

  CreatedContexts::Remove(context->context());
}

/* static */ CUcontext GpuDriver::GetContextHandle(GpuContext* context) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_17(mht_17_v, 650, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetContextHandle");

  return context->context();
}

/* static */ port::Status GpuDriver::FuncGetAttribute(
    CUfunction_attribute attribute, CUfunction func, int* attribute_value) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_18(mht_18_v, 658, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::FuncGetAttribute");

  RETURN_IF_CUDA_RES_ERROR(cuFuncGetAttribute(attribute_value, attribute, func),
                           "Failed to query kernel attribute: ", attribute);
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::FuncSetCacheConfig(
    CUfunction function, CUfunc_cache cache_config) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_19(mht_19_v, 668, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::FuncSetCacheConfig");

  RETURN_IF_CUDA_RES_ERROR(cuFuncSetCacheConfig(function, cache_config),
                           "Failed to set CUDA kernel cache config");
  return port::Status::OK();
}

/* static */ port::StatusOr<CUsharedconfig>
GpuDriver::ContextGetSharedMemConfig(GpuContext* context) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_20(mht_20_v, 678, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::ContextGetSharedMemConfig");

  CUsharedconfig shared_mem_config;
  ScopedActivateContext activation(context);
  RETURN_IF_CUDA_RES_ERROR(cuCtxGetSharedMemConfig(&shared_mem_config),
                           "Failed to get shared memory config");
  return shared_mem_config;
}

/* static */ port::Status GpuDriver::ContextSetSharedMemConfig(
    GpuContext* context, CUsharedconfig shared_mem_config) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_21(mht_21_v, 690, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::ContextSetSharedMemConfig");

  ScopedActivateContext activation(context);
  RETURN_IF_CUDA_RES_ERROR(cuCtxSetSharedMemConfig(shared_mem_config),
                           "Failed to set shared memory config");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::LaunchKernel(
    GpuContext* context, absl::string_view kernel_name, CUfunction function,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes, CUstream stream,
    void** kernel_params, void** extra) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_22(mht_22_v, 706, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::LaunchKernel");

  ScopedActivateContext activation(context);
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z;
  RETURN_IF_CUDA_RES_ERROR(
      cuLaunchKernel(function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                     block_dim_y, block_dim_z, shared_mem_bytes, stream,
                     kernel_params, extra),
      "Failed to launch CUDA kernel: ", kernel_name,
      " with block dimensions: ", block_dim_x, "x", block_dim_y, "x",
      block_dim_z, " and grid dimensions: ", grid_dim_x, "x", grid_dim_y, "x",
      grid_dim_z);
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::LoadCubin(GpuContext* context,
                                               const char* cubin_bytes,
                                               CUmodule* module) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("cubin_bytes: \"" + (cubin_bytes == nullptr ? std::string("nullptr") : std::string((char*)cubin_bytes)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_23(mht_23_v, 729, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::LoadCubin");

  ScopedActivateContext activation(context);
  RETURN_IF_CUDA_RES_ERROR(cuModuleLoadFatBinary(module, cubin_bytes),
                           "Failed to load in-memory CUBIN");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::LoadPtx(GpuContext* context,
                                             const char* ptx_contents,
                                             CUmodule* module) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("ptx_contents: \"" + (ptx_contents == nullptr ? std::string("nullptr") : std::string((char*)ptx_contents)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_24(mht_24_v, 742, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::LoadPtx");

  absl::Notification notification;
  port::Status ret = port::Status::OK();
  GetDriverExecutor()->Schedule([context, ptx_contents, module, &ret,
                                 &notification]() {
    ScopedActivateContext activation(context);
    void* ptx_data = const_cast<char*>(ptx_contents);
    static const unsigned int kLogBufferBytesLimit = 1024;
    unsigned int error_log_buffer_bytes = kLogBufferBytesLimit;
    unsigned int info_log_buffer_bytes = kLogBufferBytesLimit;
    absl::InlinedVector<char, 4> error_log_buffer(error_log_buffer_bytes);
    absl::InlinedVector<char, 4> info_log_buffer(info_log_buffer_bytes);
    bool log_verbose = true;
    CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_ERROR_LOG_BUFFER,
                              CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                              CU_JIT_INFO_LOG_BUFFER, CU_JIT_LOG_VERBOSE};
    // Note that the driver API wants the contents of this values to be stored
    // in an array of void*s, so we coerce them accordingly.
    void* option_values[] = {
        absl::bit_cast<void*>(uintptr_t(error_log_buffer_bytes)),
        absl::bit_cast<void*>(error_log_buffer.data()),
        absl::bit_cast<void*>(uintptr_t(info_log_buffer_bytes)),
        absl::bit_cast<void*>(info_log_buffer.data()),
        absl::bit_cast<void*>(uintptr_t(log_verbose))};
    CHECK(TF_ARRAYSIZE(options) == TF_ARRAYSIZE(option_values));

    CUresult res;
    {
      // TODO(leary) Need to see if NVIDIA can expunge the leakiness in their
      // module loading: see http://b/13248943
      absl::LeakCheckDisabler disabler;
      res = cuModuleLoadDataEx(module, ptx_data, TF_ARRAYSIZE(options), options,
                               option_values);
    }

    // The PTX JIT mutates the values in the option values array to reflect the
    // size of the logs it output; now that we've made the call, read the values
    // back out.
    error_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[0]);
    info_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[2]);
    CHECK_LE(error_log_buffer_bytes, kLogBufferBytesLimit);
    CHECK_LE(info_log_buffer_bytes, kLogBufferBytesLimit);

    if (res != CUDA_SUCCESS) {
      LOG(ERROR) << "failed to load PTX text as a module: " << ToString(res);
      // As a precaution for null termination of the API-provided value, ensure
      // that at least the last byte is null.
      error_log_buffer[error_log_buffer_bytes ? error_log_buffer_bytes - 1
                                              : 0] = '\0';
      LOG(ERROR) << "error log buffer (" << error_log_buffer_bytes
                 << " bytes): " << error_log_buffer.data();
      ret = port::InternalError(
          absl::StrCat("Failed to load PTX text as a module: ", ToString(res)));
      notification.Notify();
    }

    VLOG(3) << "PTX compilation info log (" << info_log_buffer_bytes
            << " bytes): " << info_log_buffer.data();
    VLOG(3) << "PTX compilation error log (" << error_log_buffer_bytes
            << " bytes): " << error_log_buffer.data();
    CHECK(module != nullptr);
    notification.Notify();
  });
  notification.WaitForNotification();

  return ret;
}

/* static */ port::Status GpuDriver::LoadHsaco(GpuContext* context,
                                               const char* hsaco_contents,
                                               CUmodule* module) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("hsaco_contents: \"" + (hsaco_contents == nullptr ? std::string("nullptr") : std::string((char*)hsaco_contents)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_25(mht_25_v, 817, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::LoadHsaco");

  return port::InternalError(
      "Feature not supported on CUDA platform (LoadHsaco)");
}

/* static */ port::Status GpuDriver::SynchronousMemsetUint8(
    GpuContext* context, CUdeviceptr location, uint8 value, size_t size) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_26(mht_26_v, 826, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::SynchronousMemsetUint8");

  ScopedActivateContext activation(context);
  RETURN_IF_CUDA_RES_ERROR(cuMemsetD8(location, value, size),
                           "Failed to memset memory");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::SynchronousMemsetUint32(
    GpuContext* context, CUdeviceptr location, uint32 value,
    size_t uint32_count) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_27(mht_27_v, 838, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::SynchronousMemsetUint32");

  ScopedActivateContext activation(context);
  RETURN_IF_CUDA_RES_ERROR(cuMemsetD32(location, value, uint32_count),
                           "Failed to memset memory");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::AsynchronousMemsetUint8(
    GpuContext* context, CUdeviceptr location, uint8 value, size_t uint32_count,
    CUstream stream) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_28(mht_28_v, 850, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::AsynchronousMemsetUint8");

  ScopedActivateContext activation(context);
  RETURN_IF_CUDA_RES_ERROR(
      cuMemsetD8Async(location, value, uint32_count, stream),
      "Failed to enqueue async memset operation");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::AsynchronousMemsetUint32(
    GpuContext* context, CUdeviceptr location, uint32 value,
    size_t uint32_count, CUstream stream) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_29(mht_29_v, 863, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::AsynchronousMemsetUint32");

  ScopedActivateContext activation(context);
  RETURN_IF_CUDA_RES_ERROR(
      cuMemsetD32Async(location, value, uint32_count, stream),
      "Failed to enqueue async memset operation");
  return port::Status::OK();
}

/* static */ bool GpuDriver::AddStreamCallback(GpuContext* context,
                                               CUstream stream,
                                               StreamCallback callback,
                                               void* data) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_30(mht_30_v, 877, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::AddStreamCallback");

  // Note: flags param is required to be zero according to CUDA 6.0.
  CUresult res = cuStreamAddCallback(stream, callback, data, 0 /* = flags */);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "unable to add host callback: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool GpuDriver::GetModuleFunction(GpuContext* context,
                                               CUmodule module,
                                               const char* kernel_name,
                                               CUfunction* function) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("kernel_name: \"" + (kernel_name == nullptr ? std::string("nullptr") : std::string((char*)kernel_name)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_31(mht_31_v, 894, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetModuleFunction");

  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  CUresult res = cuModuleGetFunction(function, module, kernel_name);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to get PTX kernel \"" << kernel_name
               << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::GetModuleSymbol(GpuContext* context,
                                             CUmodule module,
                                             const char* symbol_name,
                                             CUdeviceptr* dptr, size_t* bytes) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("symbol_name: \"" + (symbol_name == nullptr ? std::string("nullptr") : std::string((char*)symbol_name)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_32(mht_32_v, 914, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetModuleSymbol");

  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  CUresult res = cuModuleGetGlobal(dptr, bytes, module, symbol_name);
  if (res != CUDA_SUCCESS) {
    // symbol may not be found in the current module, but it may reside in
    // another module.
    VLOG(2) << "failed to get symbol \"" << symbol_name
            << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ void GpuDriver::UnloadModule(GpuContext* context,
                                          CUmodule module) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_33(mht_33_v, 934, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::UnloadModule");

  ScopedActivateContext activated{context};
  CUresult res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << ToString(res);
  }
}

/* static */ port::StatusOr<CUdevice> GpuDriver::DeviceFromContext(
    GpuContext* context) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_34(mht_34_v, 947, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::DeviceFromContext");

  ScopedActivateContext activated{context};
  CUdevice device = -1;
  CUresult result = cuCtxGetDevice(&device);
  if (result == CUDA_SUCCESS) {
    return device;
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed to get device for context: ", ToString(result)));
}

/* static */ bool GpuDriver::CreateStream(GpuContext* context, CUstream* stream,
                                          int priority) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_35(mht_35_v, 964, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::CreateStream");

  // TODO(leary) can we switch this to CU_STREAM_NON_BLOCKING or will that mess
  // up synchronization with respect to memsets and any other things that have
  // to occur on the default stream?
  ScopedActivateContext activated{context};
  CUresult res;
  // If the priority is 0, then use the previous api to create the stream with
  // the default priority for backward compatibility. Probably there is no
  // difference in using the new api call but leaving it as is for now.
  if (priority == 0) {
    res = cuStreamCreate(stream, 0);
  } else {
    res = cuStreamCreateWithPriority(stream, 0, priority);
  }
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not allocate CUDA stream for context "
               << context->context() << ": " << ToString(res);
    return false;
  }

  VLOG(2) << "successfully created stream " << *stream << " for context "
          << context->context() << " on thread";
  return true;
}

/* static */ void GpuDriver::DestroyStream(GpuContext* context,
                                           CUstream* stream) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_36(mht_36_v, 993, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::DestroyStream");

  if (*stream == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  CUresult res = cuStreamDestroy(*stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to destroy CUDA stream for context "
               << context->context() << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for context "
            << context->context();
    *stream = nullptr;
  }
}

/* static */ void* GpuDriver::DeviceAllocate(GpuContext* context,
                                             uint64_t bytes) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_37(mht_37_v, 1014, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::DeviceAllocate");

  if (bytes == 0) {
    return nullptr;
  }

  ScopedActivateContext activated{context};
  CUdeviceptr result = 0;
  CUresult res = cuMemAlloc(&result, bytes);
  if (res != CUDA_SUCCESS) {
    // LOG(INFO) because this isn't always important to users (e.g. BFCAllocator
    // implements a retry if the first allocation fails).
    LOG(INFO) << "failed to allocate "
              << port::HumanReadableNumBytes::ToString(bytes) << " (" << bytes
              << " bytes) from device: " << ToString(res);
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes";
  return ptr;
}

/* static */ void GpuDriver::DeviceDeallocate(GpuContext* context,
                                              void* location) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_38(mht_38_v, 1040, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::DeviceDeallocate");

  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  CUresult res = cuMemFree(pointer);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for context "
            << context->context();
  }
}

/* static */ void* GpuDriver::UnifiedMemoryAllocate(GpuContext* context,
                                                    uint64_t bytes) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_39(mht_39_v, 1057, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::UnifiedMemoryAllocate");

  ScopedActivateContext activation(context);
  CUdeviceptr result = 0;
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  CUresult res = cuMemAllocManaged(&result, bytes, CU_MEM_ATTACH_GLOBAL);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes unified memory; result: " << ToString(res);
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes in unified memory";
  return ptr;
}

/* static */ void GpuDriver::UnifiedMemoryDeallocate(GpuContext* context,
                                                     void* location) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_40(mht_40_v, 1077, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::UnifiedMemoryDeallocate");

  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  CUresult res = cuMemFree(pointer);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to free unified memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated unified memory at " << location << " for context "
            << context->context();
  }
}

/* static */ void* GpuDriver::HostAllocate(GpuContext* context,
                                           uint64_t bytes) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_41(mht_41_v, 1094, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::HostAllocate");

  ScopedActivateContext activation(context);
  void* host_mem = nullptr;
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  CUresult res = cuMemHostAlloc(&host_mem, bytes, CU_MEMHOSTALLOC_PORTABLE);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes on host: " << ToString(res);
  }
  return host_mem;
}

/* static */ void GpuDriver::HostDeallocate(GpuContext* context,
                                            void* location) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_42(mht_42_v, 1110, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::HostDeallocate");

  ScopedActivateContext activation(context);
  CUresult res = cuMemFreeHost(location);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << ToString(res);
  }
}

/* static */ bool GpuDriver::HostRegister(GpuContext* context, void* location,
                                          uint64_t bytes) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_43(mht_43_v, 1123, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::HostRegister");

  ScopedActivateContext activation(context);
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  CUresult res =
      cuMemHostRegister(location, bytes, CU_MEMHOSTREGISTER_PORTABLE);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool GpuDriver::HostUnregister(GpuContext* context,
                                            void* location) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_44(mht_44_v, 1140, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::HostUnregister");

  ScopedActivateContext activation(context);
  CUresult res = cuMemHostUnregister(location);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

#if CUDA_VERSION >= 10020
/* static */ port::StatusOr<GpuDriver::VmemSpan>
GpuDriver::ReserveVirtualMemory(GpuContext* context, uint64_t bytes) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_45(mht_45_v, 1156, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::ReserveVirtualMemory");

  ScopedActivateContext activation(context);
  CUdeviceptr base;
  CUresult res = cuMemAddressReserve(&base, bytes, /*alignment=*/0,
                                     /*addr=*/0, /*flags=*/0);
  if (res != CUDA_SUCCESS) {
    return port::InternalError(
        absl::StrFormat("error reserving %d bytes of virtual GPU memory: %s",
                        bytes, ToString(res)));
  }
  return {{base, bytes}};
}

/* static */ void GpuDriver::FreeVirtualMemory(
    GpuContext* context, GpuDriver::VmemSpan reservation) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_46(mht_46_v, 1173, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::FreeVirtualMemory");

  ScopedActivateContext activation(context);
  CUresult res = cuMemAddressFree(reservation.base, reservation.size_bytes);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "error freeing vmem reservation of size "
               << reservation.size_bytes << " at address " << reservation.base;
  }
}

/* static */ port::StatusOr<uint64_t> GpuDriver::GetMinAllocationGranularity(
    GpuDeviceHandle device) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_47(mht_47_v, 1186, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetMinAllocationGranularity");

  CUmemAllocationProp props = {};
  props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = device;

  size_t granularity;
  CUresult res = cuMemGetAllocationGranularity(
      &granularity, &props, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (res != CUDA_SUCCESS) {
    return port::InternalError(absl::StrCat(
        "failed to get min allocation granularity: ", ToString(res)));
  }
  return granularity;
}

/* static */ port::StatusOr<GpuDriver::GenericMemoryHandle>
GpuDriver::CreateMemoryHandle(GpuContext* context, uint64_t bytes) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_48(mht_48_v, 1206, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::CreateMemoryHandle");

  ScopedActivateContext activation(context);
  auto device = DeviceFromContext(context);
  if (!device.ok()) {
    LOG(ERROR) << "Failed to get device from context" << device.status();
    return device.status();
  }

  CUmemAllocationProp props = {};
  props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = device.ValueOrDie();

  CUmemGenericAllocationHandle mem_handle;
  CUresult res = cuMemCreate(&mem_handle, bytes, &props, 0);
  if (res != CUDA_SUCCESS) {
    return port::InternalError(
        absl::StrFormat("failed to create memory allocation of size %d: %s",
                        bytes, ToString(res)));
  }
  return GpuDriver::GenericMemoryHandle{mem_handle, bytes};
}

/* static */ void GpuDriver::ReleaseMemoryHandle(
    GpuContext* context, GpuDriver::GenericMemoryHandle handle) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_49(mht_49_v, 1233, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::ReleaseMemoryHandle");

  ScopedActivateContext activation(context);

  CUresult res = cuMemRelease(handle.handle);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to release memory handle " << handle.handle
               << " of size " << handle.bytes << ": " << ToString(res);
  }
}

/* static */ port::Status GpuDriver::MapMemory(
    GpuContext* context, CUdeviceptr va,
    const GpuDriver::GenericMemoryHandle& handle,
    const std::vector<GpuDeviceHandle>& device_handles) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_50(mht_50_v, 1249, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::MapMemory");

  ScopedActivateContext activation(context);

  auto device = DeviceFromContext(context);
  if (!device.ok()) {
    return device.status();
  }

  // NB: Zero is the only valid value for both flags and offset.
  CUresult res =
      cuMemMap(va, handle.bytes, /*offset=*/0, handle.handle, /*flags=*/0);
  if (res != CUDA_SUCCESS) {
    return port::InternalError(absl::StrFormat(
        "Failed to map %d bytes at %d: %s", handle.bytes, va, ToString(res)));
  }

  std::vector<CUmemAccessDesc> access_descriptors(device_handles.size());
  for (int i = 0; i < access_descriptors.size(); ++i) {
    access_descriptors[i].location.id = device_handles[i];
    access_descriptors[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_descriptors[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }

  res = cuMemSetAccess(va, handle.bytes, access_descriptors.data(),
                       access_descriptors.size());
  if (res != CUDA_SUCCESS) {
    // Unmap the memory that we failed to set access for.
    if (cuMemUnmap(va, handle.bytes) != CUDA_SUCCESS) {
      LOG(ERROR)
          << "Failed to unmap memory in GpuDriver::MapMemory error path.";
    }
    return port::InternalError(absl::StrFormat(
        "Failed to set read/write access on memory mapped at %d: %s", va,
        ToString(res)));
  }
  return port::Status::OK();
}

/* static */ void GpuDriver::UnmapMemory(GpuContext* context, CUdeviceptr va,
                                         uint64_t bytes) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_51(mht_51_v, 1291, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::UnmapMemory");

  ScopedActivateContext activation(context);

  CUresult res = cuMemUnmap(va, bytes);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "Failed to unmap memory at " << va << " of size " << bytes
               << ": " << ToString(res);
  }
}

#endif

/* static */ port::Status GpuDriver::DestroyEvent(GpuContext* context,
                                                  CUevent* event) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_52(mht_52_v, 1307, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::DestroyEvent");

  if (*event == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "input event cannot be null");
  }

  ScopedActivateContext activated{context};
  RETURN_IF_CUDA_RES_ERROR(cuEventDestroy(*event),
                           "Error destroying CUDA event");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::RecordEvent(GpuContext* context,
                                                 CUevent event,
                                                 CUstream stream) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_53(mht_53_v, 1324, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::RecordEvent");

  ScopedActivateContext activated{context};
  RETURN_IF_CUDA_RES_ERROR(cuEventRecord(event, stream),
                           "Error recording CUDA event");
  return port::Status::OK();
}

/* static */ port::StatusOr<CUresult> GpuDriver::QueryEvent(GpuContext* context,
                                                            CUevent event) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_54(mht_54_v, 1335, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::QueryEvent");

  ScopedActivateContext activated{context};
  CUresult res = cuEventQuery(event);
  if (res != CUDA_SUCCESS && res != CUDA_ERROR_NOT_READY) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat("failed to query event: %s", ToString(res)));
  }

  return res;
}

/* static */ bool GpuDriver::GetEventElapsedTime(GpuContext* context,
                                                 float* elapsed_milliseconds,
                                                 CUevent start, CUevent stop) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_55(mht_55_v, 1352, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetEventElapsedTime");

  ScopedActivateContext activated{context};
  // The stop event must have completed in order for cuEventElapsedTime to
  // work.
  CUresult res = cuEventSynchronize(stop);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to synchronize the stop event: " << ToString(res);
    return false;
  }
  res = cuEventElapsedTime(elapsed_milliseconds, start, stop);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to get elapsed time between events: "
               << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::WaitStreamOnEvent(GpuContext* context,
                                               CUstream stream, CUevent event) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_56(mht_56_v, 1375, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::WaitStreamOnEvent");

  ScopedActivateContext activation(context);
  CUresult res = cuStreamWaitEvent(stream, event, 0 /* = flags */);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not wait stream on event: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::SynchronizeContext(GpuContext* context) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_57(mht_57_v, 1389, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::SynchronizeContext");

  ScopedActivateContext activation(context);
  CUresult res = cuCtxSynchronize();
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not synchronize on CUDA context: " << ToString(res)
               << " :: " << port::CurrentStackTrace();
    return false;
  }

  return true;
}

/* static */ port::Status GpuDriver::SynchronizeStream(GpuContext* context,
                                                       CUstream stream) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_58(mht_58_v, 1405, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::SynchronizeStream");

  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  RETURN_IF_CUDA_RES_ERROR(cuStreamSynchronize(stream),
                           "Could not synchronize CUDA stream");
  return port::Status::OK();
}

/* static */ bool GpuDriver::IsStreamIdle(GpuContext* context,
                                          CUstream stream) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_59(mht_59_v, 1417, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::IsStreamIdle");

  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  CUresult res = cuStreamQuery(stream);
  if (res == CUDA_SUCCESS) {
    return true;
  }

  if (res != CUDA_ERROR_NOT_READY) {
    LOG(ERROR) << "stream in bad state on status query: " << ToString(res);
  }
  return false;
}

/* static */ port::Status GpuDriver::SynchronousMemcpyD2H(GpuContext* context,
                                                          void* host_dst,
                                                          CUdeviceptr gpu_src,
                                                          uint64_t size) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_60(mht_60_v, 1437, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::SynchronousMemcpyD2H");

  ScopedActivateContext activation(context);
  RETURN_IF_CUDA_RES_ERROR(
      cuMemcpyDtoH(host_dst, gpu_src, size),
      absl::StrFormat("failed to synchronous memcpy from device to host "
                      "host dst: %p; GPU src: %p; size: %u=0x%x",
                      host_dst, absl::bit_cast<void*>(gpu_src), size, size));
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::SynchronousMemcpyH2D(GpuContext* context,
                                                          CUdeviceptr gpu_dst,
                                                          const void* host_src,
                                                          uint64_t size) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_61(mht_61_v, 1455, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::SynchronousMemcpyH2D");

  ScopedActivateContext activation(context);
  RETURN_IF_CUDA_RES_ERROR(
      cuMemcpyHtoD(gpu_dst, host_src, size),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: GPU dst: %p;"
          " host src: %p; size: %u=0x%x",
          absl::bit_cast<void*>(gpu_dst), host_src, size, size));
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::SynchronousMemcpyD2D(GpuContext* context,
                                                          CUdeviceptr gpu_dst,
                                                          CUdeviceptr gpu_src,
                                                          uint64_t size) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_62(mht_62_v, 1473, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::SynchronousMemcpyD2D");

  ScopedActivateContext activation(context);

  CUresult result;
  // CreatedContexts::GetAnyContext() doesn't works when ptr == 0.
  // This happens when the size is 0.
  if (gpu_dst == 0 || gpu_src == 0 || !UseCudaMallocAsyncAllocator()) {
    result = cuMemcpyDtoD(gpu_dst, gpu_src, size);
  } else {
    // Any context work here.
    CUcontext dst_context =
        CreatedContexts::GetAnyContext(absl::bit_cast<void*>(gpu_dst));
    CUcontext src_context =
        CreatedContexts::GetAnyContext(absl::bit_cast<void*>(gpu_src));

    if (static_cast<void*>(dst_context) == nullptr) {
      port::StatusOr<GpuContext*> tmp_context = GetPointerContext(gpu_dst);
      if (tmp_context.ok()) {
        dst_context = tmp_context.ValueOrDie()->context();
      }
    }

    if (static_cast<void*>(src_context) == nullptr) {
      port::StatusOr<GpuContext*> tmp_context = GetPointerContext(gpu_src);
      if (tmp_context.ok()) {
        src_context = tmp_context.ValueOrDie()->context();
      }
    }

    result = cuMemcpyPeer(gpu_dst, dst_context, gpu_src, src_context, size);
  }

  RETURN_IF_CUDA_RES_ERROR(
      result,
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: GPU dst: %p; "
          "GPU src: %p; size: %u=0x%x",
          absl::bit_cast<void*>(gpu_dst), absl::bit_cast<void*>(gpu_src), size,
          size));
  VLOG(2) << "successfully sync memcpy'd d2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ bool GpuDriver::AsynchronousMemcpyD2H(GpuContext* context,
                                                   void* host_dst,
                                                   CUdeviceptr gpu_src,
                                                   uint64_t size,
                                                   CUstream stream) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_63(mht_63_v, 1523, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::AsynchronousMemcpyD2H");

  ScopedActivateContext activation(context);
  CUresult res = cuMemcpyDtoHAsync(host_dst, gpu_src, size, stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to host: %s; host dst: %p; "
        "GPU src: %p; size: %u=0x%x",
        ToString(res), host_dst, absl::bit_cast<void*>(gpu_src), size, size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return true;
}

/* static */ bool GpuDriver::AsynchronousMemcpyH2D(GpuContext* context,
                                                   CUdeviceptr gpu_dst,
                                                   const void* host_src,
                                                   uint64_t size,
                                                   CUstream stream) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_64(mht_64_v, 1546, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::AsynchronousMemcpyH2D");

  ScopedActivateContext activation(context);
  CUresult res = cuMemcpyHtoDAsync(gpu_dst, host_src, size, stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from host to device: %s; GPU dst: %p; "
        "host src: %p; size: %u=0x%x",
        ToString(res), absl::bit_cast<void*>(gpu_dst), host_src, size, size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " on stream " << stream;
  return true;
}

/* static */ bool GpuDriver::AsynchronousMemcpyD2D(GpuContext* context,
                                                   CUdeviceptr gpu_dst,
                                                   CUdeviceptr gpu_src,
                                                   uint64_t size,
                                                   CUstream stream) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_65(mht_65_v, 1568, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::AsynchronousMemcpyD2D");

  ScopedActivateContext activation(context);
  CUresult result;
  // CreatedContexts::GetAnyContext() doesn't works when ptr == 0.
  // This happens when the size is 0.
  if (gpu_dst == 0 || gpu_src == 0 || !UseCudaMallocAsyncAllocator()) {
    result = cuMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream);
  } else {
    // Any context work here.
    CUcontext dst_context =
        CreatedContexts::GetAnyContext(absl::bit_cast<void*>(gpu_dst));
    CUcontext src_context =
        CreatedContexts::GetAnyContext(absl::bit_cast<void*>(gpu_src));

    if (static_cast<void*>(dst_context) == nullptr) {
      port::StatusOr<GpuContext*> tmp_context = GetPointerContext(gpu_dst);
      if (tmp_context.ok()) {
        dst_context = tmp_context.ValueOrDie()->context();
      }
    }

    if (static_cast<void*>(src_context) == nullptr) {
      port::StatusOr<GpuContext*> tmp_context = GetPointerContext(gpu_src);
      if (tmp_context.ok()) {
        src_context = tmp_context.ValueOrDie()->context();
      }
    }

    result = cuMemcpyPeerAsync(gpu_dst, dst_context, gpu_src, src_context, size,
                               stream);
  }
  if (result != CUDA_SUCCESS) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to device: %s"
        "; GPU dst: %p on %s %s"
        "; GPU src: %p on %s %s"
        "; can access? %s; size: %u=0x%x",
        ToString(result), absl::bit_cast<void*>(gpu_dst),
        CUDAPointerToMemorySpaceString(gpu_dst),
        CUDAPointerToDeviceString(gpu_dst), absl::bit_cast<void*>(gpu_src),
        CUDAPointerToMemorySpaceString(gpu_src),
        CUDAPointerToDeviceString(gpu_src),
        CUDAPointersToCanAccessString(gpu_src, gpu_dst), size, size);

    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes";
  return true;
}

/* static */ port::Status GpuDriver::InitEvent(GpuContext* context,
                                               CUevent* result,
                                               EventFlags flags) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_66(mht_66_v, 1623, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::InitEvent");

  int cuflags;
  switch (flags) {
    case EventFlags::kDefault:
      cuflags = CU_EVENT_DEFAULT;
      break;
    case EventFlags::kDisableTiming:
      cuflags = CU_EVENT_DISABLE_TIMING;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(flags);
  }

  ScopedActivateContext activated{context};
  CUresult res = cuEventCreate(result, cuflags);

  if (res == CUDA_SUCCESS) {
    return port::Status::OK();
  } else if (res == CUDA_ERROR_OUT_OF_MEMORY) {
    return port::Status(port::error::RESOURCE_EXHAUSTED,
                        "could not create CUDA event: out of device memory");
  } else {
    return port::Status(
        port::error::FAILED_PRECONDITION,
        absl::StrCat("could not create CUDA event: ", ToString(res)));
  }
}

/* static */ int GpuDriver::GetDeviceCount() {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_67(mht_67_v, 1654, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetDeviceCount");

  int device_count = 0;
  CUresult res = cuDeviceGetCount(&device_count);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "could not retrieve CUDA device count: " << ToString(res);
    return 0;
  }

  if (FLAGS_gpuexec_cuda_device_0_only && device_count > 1) {
    device_count = 1;
  }
  return device_count;
}

/* static */ port::StatusOr<GpuContext*> GpuDriver::GetPointerContext(
    CUdeviceptr pointer) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_68(mht_68_v, 1672, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetPointerContext");

  GpuContext* context = nullptr;
  CUresult result =
      cuPointerGetAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT, pointer);
  if (result == CUDA_SUCCESS) {
    // For cudaMallocAsync, the context returned is null.  For now
    // return not-available. But how to manage that correctly
    // everywhere in TF?  Currently this is only used during error
    // handling.  So all is working fine, but TF have a different
    // error then the original one.
    if (context == nullptr) {
      return port::Status(
          port::error::UNAVAILABLE,
          absl::StrCat("failed to query context for device pointer: ",
                       ToString(result)));
    }
    return context;
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed to query context for device pointer: ",
                   ToString(result)));
}

/* static */ port::StatusOr<MemorySpace> GpuDriver::GetPointerMemorySpace(
    CUdeviceptr pointer) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_69(mht_69_v, 1701, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetPointerMemorySpace");

  unsigned int value;
  CUresult result =
      cuPointerGetAttribute(&value, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer);
  if (result == CUDA_SUCCESS) {
    switch (value) {
      case CU_MEMORYTYPE_DEVICE:
        return MemorySpace::kDevice;
      case CU_MEMORYTYPE_HOST:
        return MemorySpace::kHost;
      default:
        return port::Status(
            port::error::INTERNAL,
            absl::StrCat("unknown memory space provided by CUDA API: ", value));
    }
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed to query device pointer for memory space: ",
                   ToString(result)));
}

/* static */ port::Status GpuDriver::GetPointerAddressRange(CUdeviceptr dptr,
                                                            CUdeviceptr* base,
                                                            size_t* size) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_70(mht_70_v, 1729, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetPointerAddressRange");

  CUresult result = cuMemGetAddressRange(base, size, dptr);
  if (result == CUDA_SUCCESS) {
    return port::Status::OK();
  } else if (result == CUDA_ERROR_NOT_FOUND) {
    // We differentiate between "this pointer is unknown" (return here) and
    // "there was an internal error while performing this operation" (return
    // below).
    return port::Status(
        port::error::NOT_FOUND,
        absl::StrFormat("not a device pointer %p; %s",
                        reinterpret_cast<void*>(dptr), ToString(result)));
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrFormat("failed to get pointer into for device pointer %p; %s",
                      reinterpret_cast<void*>(dptr), ToString(result)));
}

/* static */ port::StatusOr<CUdevice> GpuDriver::GetPointerDevice(
    CUdeviceptr pointer) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_71(mht_71_v, 1753, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetPointerDevice");

  auto result = GetPointerContext(pointer);
  if (!result.ok()) {
    return result.status();
  }

  return DeviceFromContext(result.ValueOrDie());
}

/* static */ port::Status GpuDriver::GetComputeCapability(int* cc_major,
                                                          int* cc_minor,
                                                          CUdevice device) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_72(mht_72_v, 1767, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetComputeCapability");

  *cc_major = 0;
  *cc_minor = 0;

  CUresult res = cuDeviceGetAttribute(
      cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (res != CUDA_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat(
            "failed to get compute capability major for device: %s; %d",
            ToString(res), device));
  }

  res = cuDeviceGetAttribute(
      cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  if (res != CUDA_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat(
            "failed to get compute capability minor for device: %s; %d",
            ToString(res), device));
  }

  return port::Status::OK();
}

/* static */ port::Status GpuDriver::GetGpuISAVersion(int* version,
                                                      CUdevice device) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_73(mht_73_v, 1798, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetGpuISAVersion");

  return port::Status{
      port::error::INTERNAL,
      "Feature not supported on CUDA platform (GetGpuISAVersion)"};
}

/* static */ port::Status GpuDriver::GetGpuGCNArchName(CUdevice, std::string*) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_74(mht_74_v, 1807, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetGpuGCNArchName");

  return port::Status{
      port::error::INTERNAL,
      "Feature not supported on CUDA platform (GetGpuGCNArchName)"};
}

// Helper function that turns the integer output of cuDeviceGetAttribute to type
// T and wraps it in a StatusOr.
template <typename T>
static port::StatusOr<T> GetSimpleAttribute(CUdevice device,
                                            CUdevice_attribute attribute) {
  int value = -1;
  RETURN_IF_CUDA_RES_ERROR(cuDeviceGetAttribute(&value, attribute, device),
                           "Could not retrieve CUDA device attribute (",
                           attribute);
  T converted = value;
  return converted;
}

/* static */ port::StatusOr<int> GpuDriver::GetMultiprocessorCount(
    CUdevice device) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_75(mht_75_v, 1830, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetMultiprocessorCount");

  return GetSimpleAttribute<int>(device,
                                 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerCore(
    CUdevice device) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_76(mht_76_v, 1839, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetMaxSharedMemoryPerCore");

  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerBlock(
    CUdevice device) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_77(mht_77_v, 1848, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetMaxSharedMemoryPerBlock");

  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxThreadsPerMultiprocessor(
    CUdevice device) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_78(mht_78_v, 1857, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetMaxThreadsPerMultiprocessor");

  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxThreadsPerBlock(
    CUdevice device) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_79(mht_79_v, 1866, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetMaxThreadsPerBlock");

  return GetSimpleAttribute<int64_t>(device,
                                     CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxRegistersPerBlock(
    CUdevice device) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_80(mht_80_v, 1875, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetMaxRegistersPerBlock");

  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetThreadsPerWarp(
    CUdevice device) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_81(mht_81_v, 1884, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetThreadsPerWarp");

  return GetSimpleAttribute<int64_t>(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
}

/* static */ bool GpuDriver::GetGridLimits(int* x, int* y, int* z,
                                           CUdevice device) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_82(mht_82_v, 1892, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetGridLimits");

  int value;
  CUresult res =
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim x: " << ToString(res);
    return false;
  }
  *x = value;

  res =
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim y: " << ToString(res);
    return false;
  }
  *y = value;

  res =
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim z: " << ToString(res);
    return false;
  }
  *z = value;
  return true;
}

/* static */ bool GpuDriver::GetDriverVersion(int* driver_version) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_83(mht_83_v, 1923, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetDriverVersion");

  CUresult res = cuDriverGetVersion(driver_version);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query driver version: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::GetDeviceProperties(CUdevprop* device_properties,
                                                 int device_ordinal) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_84(mht_84_v, 1937, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetDeviceProperties");

  CUresult res = cuDeviceGetProperties(device_properties, device_ordinal);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ port::StatusOr<int> GpuDriver::GetDeviceAttribute(
    CUdevice_attribute attribute, CUdevice device) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_85(mht_85_v, 1951, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetDeviceAttribute");

  int val;
  CUresult res = cuDeviceGetAttribute(&val, attribute, device);
  if (res != CUDA_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat("failed to get device attribute %d for device %d: %s",
                        attribute, device, ToString(res)));
  }
  return val;
}

/* static */ bool GpuDriver::IsEccEnabled(CUdevice device, bool* result) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_86(mht_86_v, 1966, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::IsEccEnabled");

  int value = -1;
  CUresult res =
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query ECC status: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ bool GpuDriver::GetDeviceMemoryInfo(GpuContext* context,
                                                 int64_t* free_out,
                                                 int64_t* total_out) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_87(mht_87_v, 1984, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetDeviceMemoryInfo");

  ScopedActivateContext activation(context);
  size_t free = 0;
  size_t total = 0;
  CUresult res = cuMemGetInfo(&free, &total);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query device memory info: " << ToString(res);
    return false;
  }

  *free_out = free;
  *total_out = total;
  return true;
}

/* static */ bool GpuDriver::GetDeviceTotalMemory(CUdevice device,
                                                  uint64_t* result) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_88(mht_88_v, 2003, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetDeviceTotalMemory");

  size_t value = -1;
  CUresult res = cuDeviceTotalMem(&value, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query total available memory: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ std::string GpuDriver::GetPCIBusID(CUdevice device) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_89(mht_89_v, 2018, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetPCIBusID");

  std::string pci_bus_id;
  static const int kBufferSize = 64;
  absl::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  CUresult res = cuDeviceGetPCIBusId(chars.begin(), kBufferSize - 1, device);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << ToString(res);
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

/* static */ bool GpuDriver::CanEnablePeerAccess(GpuContext* from,
                                                 GpuContext* to) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_90(mht_90_v, 2036, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::CanEnablePeerAccess");

  if (from == to) {
    return true;  // A context can always access its own memory.
  }

  auto from_device = DeviceFromContext(from);
  if (!from_device.ok()) {
    LOG(ERROR) << "failed to resolve 'from' peer access context to a device: "
               << from_device.status();
    return false;
  }
  auto to_device = DeviceFromContext(to);
  if (!to_device.ok()) {
    LOG(ERROR) << "failed to resolve 'to' peer access context to a device: "
               << to_device.status();
    return false;
  }
  return CanEnablePeerAccess(from_device.ValueOrDie(), to_device.ValueOrDie());
}

/* static */ bool GpuDriver::CanEnablePeerAccess(GpuDeviceHandle from,
                                                 GpuDeviceHandle to) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_91(mht_91_v, 2060, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::CanEnablePeerAccess");

  int can_access_peer = -1;
  CUresult result = cuDeviceCanAccessPeer(&can_access_peer, from, to);
  if (result != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to detect peer access capability: "
               << ToString(result);
    return false;
  }
  return can_access_peer;
}

/* static */ port::Status GpuDriver::EnablePeerAccess(GpuContext* from,
                                                      GpuContext* to) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_92(mht_92_v, 2075, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::EnablePeerAccess");

  if (from == to) {
    return port::Status::OK();  // A context can always access its own memory.
  }

  ScopedActivateContext activated{from};
  CUresult result = cuCtxEnablePeerAccess(to->context(), 0 /* = flags */);
  if (result != CUDA_SUCCESS &&
      result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrFormat("failed to enable peer access from %p to %p: %s", from,
                        to, ToString(result)));
  }

  return port::Status::OK();
}

/* static */ port::StatusOr<int> GpuDriver::GetMaxOccupiedBlocksPerCore(
    GpuContext* context, CUfunction kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_93(mht_93_v, 2098, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "GpuDriver::GetMaxOccupiedBlocksPerCore");

  ScopedActivateContext activation(context);

  int max_blocks;
  RETURN_IF_CUDA_RES_ERROR(
      cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &max_blocks, kernel, threads_per_block, dynamic_shared_memory_bytes),
      absl::StrFormat("Failed to calculate occupancy of kernel %p", kernel));
  return max_blocks;
}

}  // namespace gpu

namespace cuda {

CUcontext CurrentContextOrDie() {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_driverDTcc mht_94(mht_94_v, 2116, "", "./tensorflow/stream_executor/cuda/cuda_driver.cc", "CurrentContextOrDie");

  CUcontext current = nullptr;
  FAIL_IF_CUDA_RES_ERROR(cuCtxGetCurrent(&current),
                         "Failed to query current context");
  return current;
}

}  // namespace cuda
}  // namespace stream_executor
