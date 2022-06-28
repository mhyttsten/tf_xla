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
class MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc() {
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

#include <stdint.h>
#include <stdlib.h>

#include <map>
#include <set>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/stream_executor/gpu/gpu_diagnostics.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/human_readable.h"
#include "tensorflow/stream_executor/lib/stacktrace.h"
#include "tensorflow/stream_executor/lib/static_threadlocal.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/rocm/rocm_driver_wrapper.h"

bool FLAGS_gpuexec_rocm_driver_inject_init_error = false;
bool FLAGS_gpuexec_rocm_sync_around_driver_calls = false;
bool FLAGS_gpuexec_rocm_device_0_only = false;

#define RETURN_IF_ROCM_ERROR(expr, ...)                                \
  do {                                                                 \
    hipError_t _res = (expr);                                          \
    if (TF_PREDICT_FALSE(_res != hipSuccess)) {                        \
      return port::InternalError(absl::StrCat(                         \
          __VA_ARGS__, ": ", ::stream_executor::gpu::ToString(_res))); \
    }                                                                  \
  } while (0)

// Debugging: on each push and pop of a rocm context, verify the current device
// matches the expected one.
constexpr bool kVerifyGpuContext = false;

namespace stream_executor {
namespace gpu {

// GpuContext wraps the device_ordinal.
// Only reason we need this wrapper class is to make the GpuDriver* API
class GpuContext {
 public:
  GpuContext(const int v) : device_ordinal_(v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_0(mht_0_v, 234, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuContext");
}

  int device_ordinal() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_1(mht_1_v, 239, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "device_ordinal");
 return device_ordinal_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  const int device_ordinal_;
};

namespace {

// Formats hipError_t to output prettified values into a log stream.
// Error summaries taken from:
string ToString(hipError_t result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_2(mht_2_v, 258, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "ToString");

#define OSTREAM_ROCM_ERROR(__name) \
  case hipError##__name:           \
    return "HIP_ERROR_" #__name;

  switch (result) {
    OSTREAM_ROCM_ERROR(InvalidValue)
    OSTREAM_ROCM_ERROR(OutOfMemory)
    OSTREAM_ROCM_ERROR(NotInitialized)
    OSTREAM_ROCM_ERROR(Deinitialized)
    OSTREAM_ROCM_ERROR(NoDevice)
    OSTREAM_ROCM_ERROR(InvalidDevice)
    OSTREAM_ROCM_ERROR(InvalidImage)
    OSTREAM_ROCM_ERROR(InvalidContext)
    OSTREAM_ROCM_ERROR(InvalidHandle)
    OSTREAM_ROCM_ERROR(NotFound)
    OSTREAM_ROCM_ERROR(NotReady)
    OSTREAM_ROCM_ERROR(NoBinaryForGpu)

    // Encountered an uncorrectable ECC error during execution.
    OSTREAM_ROCM_ERROR(ECCNotCorrectable)

    // Load/store on an invalid address. Must reboot all context.
    case 700:
      return "ROCM_ERROR_ILLEGAL_ADDRESS";
    // Passed too many / wrong arguments, too many threads for register count.
    case 701:
      return "ROCM_ERROR_LAUNCH_OUT_OF_RESOURCES";

      OSTREAM_ROCM_ERROR(ContextAlreadyInUse)
      OSTREAM_ROCM_ERROR(PeerAccessUnsupported)
      OSTREAM_ROCM_ERROR(Unknown)  // Unknown internal error to ROCM.
    default:
      return absl::StrCat("hipError_t(", static_cast<int>(result), ")");
  }
}

// ROCM driver routines may require a large amount of stack (particularly
// hipModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
port::ThreadPool* GetDriverExecutor() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_3(mht_3_v, 303, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GetDriverExecutor");

  static port::ThreadPool* thread_pool = new port::ThreadPool(
      port::Env::Default(), port::ThreadOptions(), "rocm_driver", 1);
  return thread_pool;
}

}  // namespace

string MemorySpaceString(MemorySpace memory_space) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_4(mht_4_v, 314, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "MemorySpaceString");

  switch (memory_space) {
    case MemorySpace::kHost:
      return "host";
    case MemorySpace::kDevice:
      return "device";
    default:
      LOG(FATAL) << "impossible memory space";
  }
}

// Returns the current device set in HIP. This is done by calling the
// HIP driver (e.g., this value is not our cached view of the current device).
static int CurrentDeviceOrDie() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_5(mht_5_v, 330, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "CurrentDeviceOrDie");

  int current = -1;
  hipError_t result = tensorflow::wrap::hipGetDevice(&current);
  if (result != hipSuccess) {
    LOG(FATAL) << "failed to query current device: " << ToString(result);
  }
  return current;
}

namespace {

// Call hipDeviceSynchronize and crash if it doesn't succeed.
void SynchronizeOrDie() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_6(mht_6_v, 345, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "SynchronizeOrDie");

  auto res = tensorflow::wrap::hipDeviceSynchronize();
  if (res != hipSuccess) {
    LOG(FATAL) << "Synchronize found " << ToString(res)
               << " :: " << port::CurrentStackTrace();
  }
}

struct ThreadLocalData {
  int current_device_ordinal;
  int depth;
};

SE_STATIC_THREAD_LOCAL_POD(ThreadLocalData, tls_data);

}  // namespace

ScopedActivateContext::ScopedActivateContext(GpuContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_7(mht_7_v, 365, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "ScopedActivateContext::ScopedActivateContext");

  if (FLAGS_gpuexec_rocm_sync_around_driver_calls) {
    SynchronizeOrDie();
  }

  auto* tls = &tls_data.get();
  if (tls->depth == 0) {
    tls->current_device_ordinal = CurrentDeviceOrDie();
  }

  if (kVerifyGpuContext) {
    CHECK_EQ(CurrentDeviceOrDie(), tls->current_device_ordinal);
  }

  tls->depth++;

  to_restore_ = context;

  if (context->device_ordinal() == tls->current_device_ordinal) {
    DCHECK_EQ(CurrentDeviceOrDie(), context->device_ordinal());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching device from "
          << tls->current_device_ordinal << " to " << context->device_ordinal();

  // Set the device and update thread local.
  CHECK_EQ(hipSuccess,
           tensorflow::wrap::hipSetDevice(context->device_ordinal()));
  tls->current_device_ordinal = context->device_ordinal();
}

ScopedActivateContext::~ScopedActivateContext() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_8(mht_8_v, 400, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "ScopedActivateContext::~ScopedActivateContext");

  if (FLAGS_gpuexec_rocm_sync_around_driver_calls) {
    SynchronizeOrDie();
  }

  auto* tls = &tls_data.get();

  if (kVerifyGpuContext) {
    CHECK_EQ(CurrentDeviceOrDie(), tls->current_device_ordinal);
  }

  tls->depth--;
  DCHECK_GE(tls->depth, 0);

  if (to_restore_->device_ordinal() == tls->current_device_ordinal) {
    DCHECK_EQ(CurrentDeviceOrDie(), to_restore_->device_ordinal());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching device from "
          << tls->current_device_ordinal << " to "
          << to_restore_->device_ordinal();

  // Set context and update thread local.
  CHECK_EQ(hipSuccess,
           tensorflow::wrap::hipSetDevice(to_restore_->device_ordinal()));
  tls->current_device_ordinal = to_restore_->device_ordinal();
}

namespace {

// Returns a stringified device number associated with pointer, primarily for
// logging purposes. Returns "?" if the device could not be successfully
// queried.
string ROCMPointerToDeviceString(hipDeviceptr_t pointer) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_9(mht_9_v, 437, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "ROCMPointerToDeviceString");

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
string ROCMPointerToMemorySpaceString(hipDeviceptr_t pointer) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_10(mht_10_v, 452, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "ROCMPointerToMemorySpaceString");

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
string ROCMPointersToCanAccessString(hipDeviceptr_t from, hipDeviceptr_t to) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_11(mht_11_v, 468, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "ROCMPointersToCanAccessString");

  hipPointerAttribute_t from_pointerAttributes;
  hipError_t result =
      tensorflow::wrap::hipPointerGetAttributes(&from_pointerAttributes, from);
  if (result != hipSuccess) {
    LOG(ERROR) << "could not retrieve source pointer's device: "
               << ToString(result);
    return "error";
  }

  hipPointerAttribute_t to_pointerAttributes;
  result = tensorflow::wrap::hipPointerGetAttributes(&to_pointerAttributes, to);
  if (result != hipSuccess) {
    LOG(ERROR) << "could not retrieve destination pointer's device: "
               << ToString(result);
    return "error";
  }

  GpuContext fromCtx(from_pointerAttributes.device);
  GpuContext toCtx(to_pointerAttributes.device);

  return GpuDriver::CanEnablePeerAccess(&fromCtx, &toCtx) ? "true" : "false";
}

// Actually performs the work of ROCM initialization. Wrapped up in one-time
// execution guard.
static port::Status InternalInit() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_12(mht_12_v, 497, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "InternalInit");

  hipError_t res = hipErrorNoDevice;
  if (FLAGS_gpuexec_rocm_driver_inject_init_error) {
    LOG(ERROR) << "injecting ROCM init error; initialization will fail";
  } else {
    res = tensorflow::wrap::hipInit(0 /* = flags */);
  }

  if (res == hipSuccess) {
    return port::Status::OK();
  }

  LOG(ERROR) << "failed call to hipInit: " << ToString(res);
  Diagnostician::LogDiagnosticInformation();
  return port::Status{port::error::ABORTED,
                      absl::StrCat("failed call to hipInit: ", ToString(res))};
}

}  // namespace

/* static */ port::Status GpuDriver::Init() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_13(mht_13_v, 520, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::Init");

  // Cached return value from calling InternalInit(), as hipInit need only be
  // called once, but GpuDriver::Init may be called many times.
  static port::Status* init_retval = [] {
    return new port::Status(InternalInit());
  }();
  return *init_retval;
}

/* static */ port::Status GpuDriver::GetDevice(int device_ordinal,
                                               hipDevice_t* device) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_14(mht_14_v, 533, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetDevice");

  hipError_t res = tensorflow::wrap::hipDeviceGet(device, device_ordinal);
  if (res == hipSuccess) {
    return port::Status::OK();
  }

  return port::Status{
      port::error::INTERNAL,
      absl::StrCat("failed call to hipDeviceGet: ", ToString(res))};
}

/* static */ port::Status GpuDriver::GetDeviceName(hipDevice_t device,
                                                   string* device_name) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_15(mht_15_v, 548, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetDeviceName");

  static const size_t kCharLimit = 64;
  absl::InlinedVector<char, 4> chars(kCharLimit);
  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipDeviceGetName(chars.begin(), kCharLimit - 1, device),
      "Failed to get device name");
  chars[kCharLimit - 1] = '\0';
  *device_name = chars.begin();
  return port::Status::OK();
}

bool DeviceOptionsToContextFlags(const DeviceOptions& device_options,
                                 int* flags) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_16(mht_16_v, 563, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "DeviceOptionsToContextFlags");

  static_assert(DeviceOptions::kMask == 0xf,
                "needs update for new device options");
  return true;
}

/* static */ port::Status GpuDriver::CreateContext(
    int device_ordinal, hipDevice_t device, const DeviceOptions& device_options,
    GpuContext** context) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_17(mht_17_v, 574, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::CreateContext");

  // TODO(hanbinyoon): Create a real context, i.e., by calling hipCtxCreate().
  *context = new GpuContext(device_ordinal);
  return port::Status::OK();
}
/* static */ void GpuDriver::DestroyContext(GpuContext* context) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_18(mht_18_v, 582, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::DestroyContext");

  if (context == nullptr) {
    return;
  }
  delete context;
}

/* static */ hipCtx_t GpuDriver::GetContextHandle(GpuContext* context) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_19(mht_19_v, 592, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetContextHandle");

  // TODO(hanbinyoon): Return a real context.
  return nullptr;
}

/* static */ port::Status GpuDriver::FuncGetAttribute(
    hipDeviceAttribute_t attribute, hipFunction_t func, int* attribute_value) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_20(mht_20_v, 601, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::FuncGetAttribute");

  // TODO(ROCm) properly implement this feature in HIP
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::FuncSetCacheConfig(
    hipFunction_t function, hipFuncCache_t cache_config) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_21(mht_21_v, 610, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::FuncSetCacheConfig");

  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipFuncSetCacheConfig(function, cache_config),
      "Failed to set ROCM kernel cache config.");
  return port::Status::OK();
}

/* static */ port::StatusOr<hipSharedMemConfig>
GpuDriver::ContextGetSharedMemConfig(GpuContext* context) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_22(mht_22_v, 621, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::ContextGetSharedMemConfig");

  hipSharedMemConfig shared_mem_config;
  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipDeviceGetSharedMemConfig(&shared_mem_config),
      "Failed to get shared memory config");
  return shared_mem_config;
}

/* static */ port::Status GpuDriver::ContextSetSharedMemConfig(
    GpuContext* context, hipSharedMemConfig shared_mem_config) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_23(mht_23_v, 634, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::ContextSetSharedMemConfig");

  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipDeviceSetSharedMemConfig(shared_mem_config),
      "Failed to set ROCM device shared memory config");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::LaunchKernel(
    GpuContext* context, absl::string_view kernel_name, hipFunction_t function,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    GpuStreamHandle stream, void** kernel_params, void** extra) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_24(mht_24_v, 651, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::LaunchKernel");

  ScopedActivateContext activation{context};
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << " smem: " << shared_mem_bytes;
  RETURN_IF_ROCM_ERROR(tensorflow::wrap::hipModuleLaunchKernel(
                           function, grid_dim_x, grid_dim_y, grid_dim_z,
                           block_dim_x, block_dim_y, block_dim_z,
                           shared_mem_bytes, stream, kernel_params, extra),
                       "Failed to launch ROCm kernel: ", kernel_name,
                       " with block dimensions: ", block_dim_x, "x",
                       block_dim_y, "x", block_dim_z);
  VLOG(2) << "successfully launched kernel";
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::LoadPtx(GpuContext* context,
                                             const char* ptx_contents,
                                             hipModule_t* module) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("ptx_contents: \"" + (ptx_contents == nullptr ? std::string("nullptr") : std::string((char*)ptx_contents)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_25(mht_25_v, 674, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::LoadPtx");

  LOG(ERROR) << "Feature not supported on ROCm platform (LoadPtx)";
  return port::InternalError("Not Implemented");
}

/* static */ port::Status GpuDriver::LoadCubin(GpuContext* context,
                                               const char* cubin_bytes,
                                               hipModule_t* module) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("cubin_bytes: \"" + (cubin_bytes == nullptr ? std::string("nullptr") : std::string((char*)cubin_bytes)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_26(mht_26_v, 685, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::LoadCubin");

  return port::Status{port::error::INTERNAL,
                      "Feature not supported on ROCm platform (LoadCubin)"};
}

/* static */ port::Status GpuDriver::LoadHsaco(GpuContext* context,
                                               const char* hsaco_contents,
                                               hipModule_t* module) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("hsaco_contents: \"" + (hsaco_contents == nullptr ? std::string("nullptr") : std::string((char*)hsaco_contents)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_27(mht_27_v, 696, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::LoadHsaco");

  absl::Notification notification;
  port::Status ret = port::Status::OK();
  GetDriverExecutor()->Schedule([context, hsaco_contents, module, &ret,
                                 &notification]() {
    ScopedActivateContext activation{context};
    void* hsaco_data = const_cast<char*>(hsaco_contents);

    hipError_t res = tensorflow::wrap::hipModuleLoadData(module, hsaco_data);

    if (res != hipSuccess) {
      ret = port::InternalError(
          absl::StrCat("Failed to load HSACO: ", ToString(res)));
      notification.Notify();
    }

    CHECK(module != nullptr);
    notification.Notify();
  });
  notification.WaitForNotification();

  return ret;
}

/* static */ port::Status GpuDriver::SynchronousMemsetUint8(
    GpuContext* context, hipDeviceptr_t location, uint8 value, size_t size) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_28(mht_28_v, 724, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::SynchronousMemsetUint8");

  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(tensorflow::wrap::hipMemsetD8(location, value, size),
                       "Failed to memset memory");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::SynchronousMemsetUint32(
    GpuContext* context, hipDeviceptr_t location, uint32 value,
    size_t uint32_count) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_29(mht_29_v, 736, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::SynchronousMemsetUint32");

  ScopedActivateContext activation{context};
  void* pointer = absl::bit_cast<void*>(location);
  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipMemsetD32(pointer, value, uint32_count),
      "Failed to memset memory");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::AsynchronousMemsetUint8(
    GpuContext* context, hipDeviceptr_t location, uint8 value,
    size_t uint32_count, GpuStreamHandle stream) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_30(mht_30_v, 750, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::AsynchronousMemsetUint8");

  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipMemsetAsync(location, value, uint32_count, stream),
      "Failed to enqueue async memset operation");
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::AsynchronousMemsetUint32(
    GpuContext* context, hipDeviceptr_t location, uint32 value,
    size_t uint32_count, GpuStreamHandle stream) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_31(mht_31_v, 763, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::AsynchronousMemsetUint32");

  ScopedActivateContext activation{context};
  void* pointer = absl::bit_cast<void*>(location);
  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipMemsetD32Async(pointer, value, uint32_count, stream),
      "Failed to enqueue async memset operation");
  VLOG(2) << "successfully enqueued async memset operation";
  return port::Status::OK();
}

/* static */ bool GpuDriver::AddStreamCallback(GpuContext* context,
                                               GpuStreamHandle stream,
                                               StreamCallback callback,
                                               void* data) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_32(mht_32_v, 779, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::AddStreamCallback");

  hipError_t res = tensorflow::wrap::hipStreamAddCallback(
      stream, (hipStreamCallback_t)callback, data, 0 /* = flags */);
  if (res != hipSuccess) {
    LOG(ERROR) << "unable to add host callback: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool GpuDriver::GetModuleFunction(GpuContext* context,
                                               hipModule_t module,
                                               const char* kernel_name,
                                               hipFunction_t* function) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("kernel_name: \"" + (kernel_name == nullptr ? std::string("nullptr") : std::string((char*)kernel_name)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_33(mht_33_v, 796, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetModuleFunction");

  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  hipError_t res =
      tensorflow::wrap::hipModuleGetFunction(function, module, kernel_name);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to get kernel \"" << kernel_name
               << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::GetModuleSymbol(GpuContext* context,
                                             hipModule_t module,
                                             const char* symbol_name,
                                             hipDeviceptr_t* dptr,
                                             size_t* bytes) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("symbol_name: \"" + (symbol_name == nullptr ? std::string("nullptr") : std::string((char*)symbol_name)) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_34(mht_34_v, 818, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetModuleSymbol");

  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  hipError_t res =
      tensorflow::wrap::hipModuleGetGlobal(dptr, bytes, module, symbol_name);
  if (res != hipSuccess) {
    // symbol may not be found in the current module, but it may reside in
    // another module.
    VLOG(2) << "failed to get symbol \"" << symbol_name
            << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ void GpuDriver::UnloadModule(GpuContext* context,
                                          hipModule_t module) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_35(mht_35_v, 839, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::UnloadModule");

  ScopedActivateContext activated{context};
  hipError_t res = tensorflow::wrap::hipModuleUnload(module);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << ToString(res);
  }
}

/* static */ bool GpuDriver::CreateStream(GpuContext* context,
                                          GpuStreamHandle* stream,
                                          int priority) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_36(mht_36_v, 853, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::CreateStream");

  ScopedActivateContext activated{context};
  hipError_t res;
  if (priority == 0) {
    res = tensorflow::wrap::hipStreamCreateWithFlags(
        stream, hipStreamDefault);  // switch to hipStreamNonBlocking?
  } else {
    res = tensorflow::wrap::hipStreamCreateWithPriority(
        stream, hipStreamDefault, priority);  // switch to hipStreamNonBlocking?
  }
  if (res != hipSuccess) {
    LOG(ERROR) << "could not allocate ROCM stream for device "
               << context->device_ordinal() << ": " << ToString(res);
    return false;
  }

  VLOG(2) << "successfully created stream " << *stream << " for device "
          << context->device_ordinal() << " on thread";
  return true;
}

/* static */ void GpuDriver::DestroyStream(GpuContext* context,
                                           GpuStreamHandle* stream) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_37(mht_37_v, 878, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::DestroyStream");

  if (*stream == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  hipError_t res = tensorflow::wrap::hipStreamDestroy(*stream);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to destroy ROCM stream for device "
               << context->device_ordinal() << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for device "
            << context->device_ordinal();
    *stream = nullptr;
  }
}

/* static */ void* GpuDriver::DeviceAllocate(GpuContext* context,
                                             uint64_t bytes) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_38(mht_38_v, 899, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::DeviceAllocate");

  ScopedActivateContext activated{context};
  hipDeviceptr_t result = 0;
  hipError_t res = tensorflow::wrap::hipMalloc(&result, bytes);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to allocate "
               << port::HumanReadableNumBytes::ToString(bytes) << " (" << bytes
               << " bytes) from device: " << ToString(res);
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for device " << context->device_ordinal()
          << " of " << bytes << " bytes";
  return ptr;
}

/* static */ void GpuDriver::DeviceDeallocate(GpuContext* context,
                                              void* location) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_39(mht_39_v, 919, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::DeviceDeallocate");

  ScopedActivateContext activation{context};
  hipDeviceptr_t pointer = absl::bit_cast<hipDeviceptr_t>(location);
  hipError_t res = tensorflow::wrap::hipFree(pointer);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for device "
            << context->device_ordinal();
  }
}

/* static */ void* GpuDriver::UnifiedMemoryAllocate(GpuContext* context,
                                                    uint64_t bytes) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_40(mht_40_v, 936, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::UnifiedMemoryAllocate");

  ScopedActivateContext activated{context};

  LOG(ERROR)
      << "Feature not supported on ROCm platform (UnifiedMemoryAllocate)";
  return nullptr;
}

/* static */ void GpuDriver::UnifiedMemoryDeallocate(GpuContext* context,
                                                     void* location) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_41(mht_41_v, 948, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::UnifiedMemoryDeallocate");

  LOG(ERROR)
      << "Feature not supported on ROCm platform (UnifiedMemoryDeallocate)";
}

/* static */ void* GpuDriver::HostAllocate(GpuContext* context,
                                           uint64_t bytes) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_42(mht_42_v, 957, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::HostAllocate");

  ScopedActivateContext activation{context};
  void* host_mem = nullptr;
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  hipError_t res =
      tensorflow::wrap::hipHostMalloc(&host_mem, bytes, hipHostMallocPortable);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes on host: " << ToString(res);
  }
  return host_mem;
}

/* static */ void GpuDriver::HostDeallocate(GpuContext* context,
                                            void* location) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_43(mht_43_v, 974, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::HostDeallocate");

  ScopedActivateContext activation{context};
  hipError_t res = tensorflow::wrap::hipHostFree(location);
  if (res != hipSuccess) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << ToString(res);
  }
}

/* static */ bool GpuDriver::HostRegister(GpuContext* context, void* location,
                                          uint64_t bytes) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_44(mht_44_v, 987, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::HostRegister");

  ScopedActivateContext activation{context};
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  hipError_t res = tensorflow::wrap::hipHostRegister(location, bytes,
                                                     hipHostRegisterPortable);
  if (res != hipSuccess) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool GpuDriver::HostUnregister(GpuContext* context,
                                            void* location) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_45(mht_45_v, 1004, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::HostUnregister");

  ScopedActivateContext activation{context};
  hipError_t res = tensorflow::wrap::hipHostUnregister(location);
  if (res != hipSuccess) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ port::Status GpuDriver::DestroyEvent(GpuContext* context,
                                                  GpuEventHandle* event) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_46(mht_46_v, 1019, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::DestroyEvent");

  if (*event == nullptr) {
    return port::Status{port::error::INVALID_ARGUMENT,
                        "input event cannot be null"};
  }

  ScopedActivateContext activated{context};
  hipError_t res = tensorflow::wrap::hipEventDestroy(*event);
  *event = nullptr;

  switch (res) {
    case hipSuccess:
      return port::Status::OK();
    case hipErrorDeinitialized:
    case hipErrorNotInitialized:
      return port::Status{
          port::error::FAILED_PRECONDITION,
          absl::StrFormat("error destroying ROCM event in device %d: %s",
                          context->device_ordinal(), ToString(res).c_str())};
    default:
      return port::Status{
          port::error::INTERNAL,
          absl::StrFormat("error destroying ROCM event in device %d: %s",
                          context->device_ordinal(), ToString(res).c_str())};
  }
}

/* static */ port::Status GpuDriver::RecordEvent(GpuContext* context,
                                                 GpuEventHandle event,
                                                 GpuStreamHandle stream) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_47(mht_47_v, 1051, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::RecordEvent");

  ScopedActivateContext activated{context};
  hipError_t res = tensorflow::wrap::hipEventRecord(event, stream);
  switch (res) {
    case hipSuccess:
      return port::Status::OK();
    case hipErrorDeinitialized:
    case hipErrorNotInitialized:
      return port::Status{
          port::error::FAILED_PRECONDITION,
          absl::StrFormat("error recording ROCM event on stream %p: %s", stream,
                          ToString(res).c_str())};
    default:
      return port::Status{
          port::error::INVALID_ARGUMENT,
          absl::StrFormat("error recording ROCM event on stream %p: %s", stream,
                          ToString(res).c_str())};
  }
}

/* static */ port::StatusOr<hipError_t> GpuDriver::QueryEvent(
    GpuContext* context, GpuEventHandle event) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_48(mht_48_v, 1075, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::QueryEvent");

  ScopedActivateContext activated{context};
  hipError_t res = tensorflow::wrap::hipEventQuery(event);
  if (res != hipSuccess && res != hipErrorNotReady) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat("failed to query event: %s", ToString(res).c_str())};
  }

  return res;
}

/* static */ bool GpuDriver::GetEventElapsedTime(GpuContext* context,
                                                 float* elapsed_milliseconds,
                                                 GpuEventHandle start,
                                                 GpuEventHandle stop) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_49(mht_49_v, 1093, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetEventElapsedTime");

  ScopedActivateContext activated{context};
  // The stop event must have completed in order for hipEventElapsedTime to
  // work.
  hipError_t res = tensorflow::wrap::hipEventSynchronize(stop);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to synchronize the stop event: " << ToString(res);
    return false;
  }
  res =
      tensorflow::wrap::hipEventElapsedTime(elapsed_milliseconds, start, stop);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to get elapsed time between events: "
               << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::WaitStreamOnEvent(GpuContext* context,
                                               GpuStreamHandle stream,
                                               GpuEventHandle event) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_50(mht_50_v, 1118, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::WaitStreamOnEvent");

  ScopedActivateContext activation{context};
  hipError_t res =
      tensorflow::wrap::hipStreamWaitEvent(stream, event, 0 /* = flags */);
  if (res != hipSuccess) {
    LOG(ERROR) << "could not wait stream on event: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::SynchronizeContext(GpuContext* context) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_51(mht_51_v, 1133, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::SynchronizeContext");

  ScopedActivateContext activation{context};
  hipError_t res = tensorflow::wrap::hipDeviceSynchronize();
  if (res != hipSuccess) {
    LOG(ERROR) << "could not synchronize on ROCM device: " << ToString(res)
               << " :: " << port::CurrentStackTrace();
    return false;
  }

  return true;
}

/* static */ port::Status GpuDriver::SynchronizeStream(GpuContext* context,
                                                       GpuStreamHandle stream) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_52(mht_52_v, 1149, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::SynchronizeStream");

  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  RETURN_IF_ROCM_ERROR(tensorflow::wrap::hipStreamSynchronize(stream),
                       "Could not synchronize on ROCM stream");
  VLOG(2) << "successfully synchronized stream " << stream << " on device "
          << context->device_ordinal();
  return port::Status::OK();
}

/* static */ bool GpuDriver::IsStreamIdle(GpuContext* context,
                                          GpuStreamHandle stream) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_53(mht_53_v, 1163, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::IsStreamIdle");

  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  hipError_t res = tensorflow::wrap::hipStreamQuery(stream);
  if (res == hipSuccess) {
    return true;
  }

  if (res != hipErrorNotReady) {
    LOG(ERROR) << "stream in bad state on status query: " << ToString(res);
  }
  return false;
}

/* static */ port::Status GpuDriver::SynchronousMemcpyD2H(
    GpuContext* context, void* host_dst, hipDeviceptr_t gpu_src,
    uint64_t size) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_54(mht_54_v, 1182, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::SynchronousMemcpyD2H");

  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipMemcpyDtoH(host_dst, gpu_src, size),
      absl::StrFormat("failed to synchronous memcpy from device to host: "
                      "host dst: %p; Gpu src: %p; size: %llu=0x%llx",
                      host_dst, absl::bit_cast<void*>(gpu_src), size, size));
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::SynchronousMemcpyH2D(
    GpuContext* context, hipDeviceptr_t gpu_dst, const void* host_src,
    uint64_t size) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_55(mht_55_v, 1199, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::SynchronousMemcpyH2D");

  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipMemcpyHtoD(gpu_dst, const_cast<void*>(host_src),
                                      size),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: Gpu dst: %p;"
          " host src: %p; size: %llu=0x%llx",
          absl::bit_cast<void*>(gpu_dst), host_src, size, size));
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ port::Status GpuDriver::SynchronousMemcpyD2D(
    GpuContext* context, hipDeviceptr_t gpu_dst, hipDeviceptr_t gpu_src,
    uint64_t size) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_56(mht_56_v, 1217, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::SynchronousMemcpyD2D");

  ScopedActivateContext activation{context};
  RETURN_IF_ROCM_ERROR(
      tensorflow::wrap::hipMemcpyDtoD(gpu_dst, gpu_src, size),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device:Gpu dst: %p; "
          "Gpu src: %p; size: %llu=0x%llx",
          absl::bit_cast<void*>(gpu_dst), absl::bit_cast<void*>(gpu_src), size,
          size));
  VLOG(2) << "successfully sync memcpy'd d2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ bool GpuDriver::AsynchronousMemcpyD2H(GpuContext* context,
                                                   void* host_dst,
                                                   hipDeviceptr_t gpu_src,
                                                   uint64_t size,
                                                   GpuStreamHandle stream) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_57(mht_57_v, 1237, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::AsynchronousMemcpyD2H");

  ScopedActivateContext activation{context};
  hipError_t res =
      tensorflow::wrap::hipMemcpyDtoHAsync(host_dst, gpu_src, size, stream);
  if (res != hipSuccess) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to host: %s; host dst: %p; "
        "Gpu src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), host_dst, absl::bit_cast<void*>(gpu_src), size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return true;
}

/* static */ bool GpuDriver::AsynchronousMemcpyH2D(GpuContext* context,
                                                   hipDeviceptr_t gpu_dst,
                                                   const void* host_src,
                                                   uint64_t size,
                                                   GpuStreamHandle stream) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_58(mht_58_v, 1262, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::AsynchronousMemcpyH2D");

  ScopedActivateContext activation{context};
  hipError_t res = tensorflow::wrap::hipMemcpyHtoDAsync(
      gpu_dst, const_cast<void*>(host_src), size, stream);
  if (res != hipSuccess) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from host to device: %s; Gpu dst: %p; "
        "host src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), absl::bit_cast<void*>(gpu_dst), host_src, size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " on stream " << stream;
  return true;
}

/* static */ bool GpuDriver::AsynchronousMemcpyD2D(GpuContext* context,
                                                   hipDeviceptr_t gpu_dst,
                                                   hipDeviceptr_t gpu_src,
                                                   uint64_t size,
                                                   GpuStreamHandle stream) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_59(mht_59_v, 1286, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::AsynchronousMemcpyD2D");

  ScopedActivateContext activation{context};
  hipError_t result =
      tensorflow::wrap::hipMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream);
  if (result != hipSuccess) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to device: %s"
        "; Gpu dst: %p on %s %s"
        "; Gpu src: %p on %s %s"
        "; can access? %s; size: %llu=0x%llx",
        ToString(result).c_str(), absl::bit_cast<void*>(gpu_dst),
        ROCMPointerToMemorySpaceString(gpu_dst).c_str(),
        ROCMPointerToDeviceString(gpu_dst).c_str(),
        absl::bit_cast<void*>(gpu_src),
        ROCMPointerToMemorySpaceString(gpu_src).c_str(),
        ROCMPointerToDeviceString(gpu_src).c_str(),
        ROCMPointersToCanAccessString(gpu_src, gpu_dst).c_str(), size, size);

    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes";
  return true;
}

/* static */ port::Status GpuDriver::InitEvent(GpuContext* context,
                                               GpuEventHandle* event,
                                               EventFlags flags) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_60(mht_60_v, 1315, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::InitEvent");

  int hipflags;
  switch (flags) {
    case EventFlags::kDefault:
      hipflags = hipEventDefault;
      break;
    case EventFlags::kDisableTiming:
      hipflags = hipEventDisableTiming | hipEventReleaseToSystem;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(hipflags);
  }

  ScopedActivateContext activated{context};
  hipError_t res = tensorflow::wrap::hipEventCreateWithFlags(event, hipflags);

  if (res == hipSuccess) {
    return port::Status::OK();
  } else if (res == hipErrorMemoryAllocation) {
    return port::Status{port::error::RESOURCE_EXHAUSTED,
                        "could not create ROCM event: out of device memory"};
  } else {
    return port::Status{
        port::error::FAILED_PRECONDITION,
        absl::StrCat("could not create ROCM event: ", ToString(res))};
  }
}

/* static */ int GpuDriver::GetDeviceCount() {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_61(mht_61_v, 1346, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetDeviceCount");

  int device_count = 0;
  hipError_t res = tensorflow::wrap::hipGetDeviceCount(&device_count);
  if (res != hipSuccess) {
    LOG(ERROR) << "could not retrieve ROCM device count: " << ToString(res);
    return 0;
  }

  if (FLAGS_gpuexec_rocm_device_0_only && device_count > 1) {
    device_count = 1;
  }
  return device_count;
}

/* static */ port::Status GpuDriver::GetComputeCapability(int* cc_major,
                                                          int* cc_minor,
                                                          hipDevice_t device) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_62(mht_62_v, 1365, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetComputeCapability");

  return port::Status(
      port::error::INTERNAL,
      absl::StrFormat("failed to get compute capability for device: %d "
                      "(unsupported API on AMD Gpus)",
                      device));
}

/* static */ port::Status GpuDriver::GetPointerAddressRange(
    hipDeviceptr_t dptr, hipDeviceptr_t* base, size_t* size) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_63(mht_63_v, 1377, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetPointerAddressRange");

  hipError_t result = tensorflow::wrap::hipMemGetAddressRange(base, size, dptr);
  if (result == hipSuccess) {
    return port::Status::OK();
  } else if (result == hipErrorNotFound) {
    // We differentiate between "this pointer is unknown" (return here) and
    // "there was an internal error while performing this operation" (return
    // below).
    return port::Status{port::error::NOT_FOUND,
                        absl::StrFormat("not a device pointer %p; %s",
                                        reinterpret_cast<void*>(dptr),
                                        ToString(result).c_str())};
  }

  return port::Status{
      port::error::INTERNAL,
      absl::StrFormat("failed to get pointer into for device pointer %p; %s",
                      reinterpret_cast<void*>(dptr), ToString(result).c_str())};
}

/* static */ port::StatusOr<MemorySpace> GpuDriver::GetPointerMemorySpace(
    hipDeviceptr_t pointer) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_64(mht_64_v, 1401, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetPointerMemorySpace");

  unsigned int value;
  hipError_t result = hipSuccess;
  if (result == hipSuccess) {
    switch (value) {
      case hipMemoryTypeDevice:
        return MemorySpace::kDevice;
      case hipMemoryTypeHost:
        return MemorySpace::kHost;
      default:
        return port::Status{
            port::error::INTERNAL,
            absl::StrCat("unknown memory space provided by ROCM API: ", value)};
    }
  }

  return port::Status{
      port::error::INTERNAL,
      absl::StrCat("failed to query device pointer for memory space: ",
                   ToString(result))};
}

/* static */ port::StatusOr<hipDevice_t> GpuDriver::GetPointerDevice(
    hipDeviceptr_t pointer) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_65(mht_65_v, 1427, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetPointerDevice");

  hipPointerAttribute_t pointerAttributes;
  hipError_t result =
      tensorflow::wrap::hipPointerGetAttributes(&pointerAttributes, pointer);
  if (result != hipSuccess) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrCat("failed to get device for pointer: ", ToString(result))};
  }

  hipDevice_t device;
  result = tensorflow::wrap::hipDeviceGet(&device, pointerAttributes.device);
  if (result != hipSuccess) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrCat("failed to get device for pointer: ", ToString(result))};
  }

  return device;
}

/* static */ port::Status GpuDriver::GetGpuISAVersion(int* version,
                                                      hipDevice_t device) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_66(mht_66_v, 1452, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetGpuISAVersion");

  hipDeviceProp_t props;
  hipError_t result = tensorflow::wrap::hipGetDeviceProperties(&props, device);
  if (result == hipSuccess) {
    *version = props.gcnArch;
    return port::Status::OK();
  }
  *version = 0;
  return port::Status{
      port::error::INTERNAL,
      absl::StrFormat("failed to determine AMDGpu ISA version for device %d",
                      device)};
}

/* static */ port::Status GpuDriver::GetGpuGCNArchName(
    hipDevice_t device, std::string* gcnArchName) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_67(mht_67_v, 1470, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetGpuGCNArchName");

  hipDeviceProp_t props;
  hipError_t result = tensorflow::wrap::hipGetDeviceProperties(&props, device);
  if (result == hipSuccess) {
    *gcnArchName = props.gcnArchName;
    return port::Status::OK();
  }
  *gcnArchName = "";
  return port::Status{
      port::error::INTERNAL,
      absl::StrFormat("failed to determine AMDGpu GCN Arch Name for device %d",
                      device)};
}

/* static */ port::StatusOr<bool> GpuDriver::GetMFMASupport() {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_68(mht_68_v, 1487, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetMFMASupport");

  hipDeviceProp_t props;
  int dev = 0;
  hipError_t result = hipGetDevice(&dev);
  result = tensorflow::wrap::hipGetDeviceProperties(&props, dev);
  if (result == hipSuccess) {
    std::string gcnArchName = props.gcnArchName;
    VLOG(1) << "GCN arch name " << gcnArchName;
    auto pos = gcnArchName.find(":");
    if (pos != string::npos) gcnArchName = gcnArchName.substr(0, pos);
    pos = gcnArchName.find("gfx");
    if (pos != string::npos) gcnArchName = gcnArchName.substr(pos + 3);
    VLOG(1) << "GCN arch name (stripped) " << gcnArchName;
    return ((gcnArchName == "908") || (gcnArchName == "909"));
  }
  return port::Status{
      port::error::INTERNAL,
      absl::StrFormat("failed to determine AMDGpu GCN Arch Name for device %d",
                      dev)};
}

// Helper function that turns the integer output of hipDeviceGetAttribute to
// type T and wraps it in a StatusOr.
template <typename T>
static port::StatusOr<T> GetSimpleAttribute(hipDevice_t device,
                                            hipDeviceAttribute_t attribute) {
  int value = -1;
  hipError_t result =
      tensorflow::wrap::hipDeviceGetAttribute(&value, attribute, device);
  if (result != hipSuccess) {
    return port::Status{
        port::error::NOT_FOUND,
        absl::StrCat("could not retrieve ROCM device attribute (", attribute,
                     "): ", ToString(result))};
  }
  T converted = value;
  return converted;
}

/* static */ port::StatusOr<int> GpuDriver::GetMultiprocessorCount(
    hipDevice_t device) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_69(mht_69_v, 1530, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetMultiprocessorCount");

  return GetSimpleAttribute<int>(device, hipDeviceAttributeMultiprocessorCount);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerCore(
    hipDevice_t device) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_70(mht_70_v, 1538, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetMaxSharedMemoryPerCore");

  return GetSimpleAttribute<int64_t>(
      device, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerBlock(
    hipDevice_t device) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_71(mht_71_v, 1547, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetMaxSharedMemoryPerBlock");

  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxSharedMemoryPerBlock);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxThreadsPerMultiprocessor(
    hipDevice_t device) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_72(mht_72_v, 1556, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetMaxThreadsPerMultiprocessor");

  return GetSimpleAttribute<int64_t>(
      device, hipDeviceAttributeMaxThreadsPerMultiProcessor);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxThreadsPerBlock(
    hipDevice_t device) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_73(mht_73_v, 1565, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetMaxThreadsPerBlock");

  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxThreadsPerBlock);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetMaxRegistersPerBlock(
    hipDevice_t device) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_74(mht_74_v, 1574, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetMaxRegistersPerBlock");

  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxRegistersPerBlock);
}

/* static */ port::StatusOr<int64_t> GpuDriver::GetThreadsPerWarp(
    hipDevice_t device) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_75(mht_75_v, 1583, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetThreadsPerWarp");

  return GetSimpleAttribute<int64_t>(device, hipDeviceAttributeWarpSize);
}

/* static */ bool GpuDriver::GetGridLimits(int* x, int* y, int* z,
                                           hipDevice_t device) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_76(mht_76_v, 1591, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetGridLimits");

  int value;
  hipError_t res = tensorflow::wrap::hipDeviceGetAttribute(
      &value, hipDeviceAttributeMaxGridDimX, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query max grid dim x: " << ToString(res);
    return false;
  }
  *x = value;

  res = tensorflow::wrap::hipDeviceGetAttribute(
      &value, hipDeviceAttributeMaxGridDimY, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query max grid dim y: " << ToString(res);
    return false;
  }
  *y = value;

  res = tensorflow::wrap::hipDeviceGetAttribute(
      &value, hipDeviceAttributeMaxGridDimZ, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query max grid dim z: " << ToString(res);
    return false;
  }
  *z = value;
  return true;
}

/* static */ bool GpuDriver::GetDriverVersion(int* driver_version) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_77(mht_77_v, 1622, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetDriverVersion");

  hipError_t res = tensorflow::wrap::hipDriverGetVersion(driver_version);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query driver version: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool GpuDriver::GetDeviceProperties(
    hipDeviceProp_t* device_properties, int device_ordinal) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_78(mht_78_v, 1636, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetDeviceProperties");

  hipError_t res = tensorflow::wrap::hipGetDeviceProperties(device_properties,
                                                            device_ordinal);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ port::StatusOr<int> GpuDriver::GetDeviceAttribute(
    hipDeviceAttribute_t attribute, hipDevice_t device) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_79(mht_79_v, 1651, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetDeviceAttribute");

  return GetSimpleAttribute<int>(device, attribute);
}

/* static */ bool GpuDriver::IsEccEnabled(hipDevice_t device, bool* result) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_80(mht_80_v, 1658, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::IsEccEnabled");

  int value = -1;
  hipError_t res = hipSuccess;
  // TODO(ROCm) implement this feature in HIP
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query ECC status: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ bool GpuDriver::GetDeviceMemoryInfo(GpuContext* context,
                                                 int64_t* free_out,
                                                 int64_t* total_out) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_81(mht_81_v, 1676, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetDeviceMemoryInfo");

  ScopedActivateContext activation{context};
  size_t free = 0;
  size_t total = 0;
  hipError_t res = tensorflow::wrap::hipMemGetInfo(&free, &total);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device memory info: " << ToString(res);
    return false;
  }

  *free_out = free;
  *total_out = total;
  return true;
}

/* static */ bool GpuDriver::GetDeviceTotalMemory(hipDevice_t device,
                                                  uint64_t* result) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_82(mht_82_v, 1695, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetDeviceTotalMemory");

  size_t value = -1;
  hipError_t res = tensorflow::wrap::hipDeviceTotalMem(&value, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query total available memory: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ string GpuDriver::GetPCIBusID(hipDevice_t device) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_83(mht_83_v, 1710, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetPCIBusID");

  string pci_bus_id;
  static const int kBufferSize = 64;
  absl::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  hipError_t res = tensorflow::wrap::hipDeviceGetPCIBusId(
      chars.begin(), kBufferSize - 1, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << ToString(res);
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

/* static */ bool GpuDriver::CanEnablePeerAccess(GpuContext* from,
                                                 GpuContext* to) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_84(mht_84_v, 1729, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::CanEnablePeerAccess");

  if (from->device_ordinal() == to->device_ordinal()) {
    return true;  // A device can always access its own memory.
  }

  int can_access_peer = -1;
  hipError_t res = tensorflow::wrap::hipDeviceCanAccessPeer(
      &can_access_peer, from->device_ordinal(), to->device_ordinal());
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to detect peer access capability: " << ToString(res);
    return false;
  }

  return can_access_peer;
}

/* static */ port::Status GpuDriver::EnablePeerAccess(GpuContext* from,
                                                      GpuContext* to) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_85(mht_85_v, 1749, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::EnablePeerAccess");

  if (from->device_ordinal() == to->device_ordinal()) {
    return port::Status::OK();  // A device can always access its own memory.
  }

  ScopedActivateContext activated{from};
  hipError_t result = tensorflow::wrap::hipDeviceEnablePeerAccess(
      to->device_ordinal(), 0 /* = flags */);
  if (result != hipSuccess && result != hipErrorPeerAccessAlreadyEnabled) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat("failed to enable peer access from %d to %d: %s",
                        from->device_ordinal(), to->device_ordinal(),
                        ToString(result).c_str())};
  }

  return port::Status::OK();
}

/* static */ port::StatusOr<int> GpuDriver::GetMaxOccupiedBlocksPerCore(
    GpuContext* context, hipFunction_t kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_driverDTcc mht_86(mht_86_v, 1773, "", "./tensorflow/stream_executor/rocm/rocm_driver.cc", "GpuDriver::GetMaxOccupiedBlocksPerCore");

  ScopedActivateContext activation{context};

  int max_blocks = 0;
  hipError_t result = hipSuccess;
  // TODO(ROCm) implement this feature in HIP
  if (result != hipSuccess) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat("failed to calculate occupancy of kernel %p: %s",
                        kernel, ToString(result).c_str())};
  }

  return max_blocks;
}

}  // namespace gpu
}  // namespace stream_executor
