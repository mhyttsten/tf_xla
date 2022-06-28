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
class MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc() {
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
// This file extends/implements core stream executor base classes in terms of
// the C API defined in stream_executor.h. A class "CSomething" represents a
// "Something" that can be manipulated via calls in the C interface and a C
// struct called "SP_Something".
//
// This file also contains stream_executor::Platform registration for pluggable
// device.
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

#include <string>

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/c_api_macros_internal.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/device/device_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/stream_executor/timer.h"

using tensorflow::StatusFromTF_Status;

namespace stream_executor {
using tensorflow::StringPiece;

// TODO(penporn): Remove OwnedTFStatus.
using OwnedTFStatus = tensorflow::TF_StatusPtr;

namespace {
port::Status ValidateSPPlatform(const SP_Platform& platform) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_0(mht_0_v, 223, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "ValidateSPPlatform");

  TF_VALIDATE_STRUCT_SIZE(SP_Platform, platform, SP_PLATFORM_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(SP_Platform, platform, name);
  TF_VALIDATE_NOT_NULL(SP_Platform, platform, type);
  TF_RETURN_IF_ERROR(
      tensorflow::device_utils::ValidateDeviceType(platform.name));
  TF_RETURN_IF_ERROR(
      tensorflow::device_utils::ValidateDeviceType(platform.type));
  // `visible_device_count` could be 0 at initialization time.
  return port::Status::OK();
}

port::Status ValidateSPPlatformFns(const SP_PlatformFns& platform_fns) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_1(mht_1_v, 238, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "ValidateSPPlatformFns");

  TF_VALIDATE_STRUCT_SIZE(SP_PlatformFns, platform_fns,
                          SP_PLATFORM_FNS_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, create_device);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, destroy_device);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, create_stream_executor);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, destroy_stream_executor);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, create_timer_fns);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, destroy_timer_fns);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, create_device_fns);
  TF_VALIDATE_NOT_NULL(SP_PlatformFns, platform_fns, destroy_device_fns);
  return port::Status::OK();
}

port::Status ValidateSPTimerFns(const SP_TimerFns& timer_fns) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_2(mht_2_v, 255, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "ValidateSPTimerFns");

  TF_VALIDATE_STRUCT_SIZE(SP_TimerFns, timer_fns, SP_TIMER_FNS_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(SP_TimerFns, timer_fns, nanoseconds);
  return port::Status::OK();
}

port::Status ValidateSPAllocatorStats(const SP_AllocatorStats& stats) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_3(mht_3_v, 264, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "ValidateSPAllocatorStats");

  TF_VALIDATE_STRUCT_SIZE(SP_AllocatorStats, stats,
                          SP_ALLOCATORSTATS_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return port::Status::OK();
}

port::Status ValidateSPDeviceMemoryBase(const SP_DeviceMemoryBase& mem) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_4(mht_4_v, 274, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "ValidateSPDeviceMemoryBase");

  TF_VALIDATE_STRUCT_SIZE(SP_DeviceMemoryBase, mem,
                          SP_DEVICE_MEMORY_BASE_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return port::Status::OK();
}

port::Status ValidateSPDevice(const SP_Device& device) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_5(mht_5_v, 284, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "ValidateSPDevice");

  TF_VALIDATE_STRUCT_SIZE(SP_Device, device, SP_DEVICE_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return port::Status::OK();
}

port::Status ValidateSPDeviceFns(const SP_DeviceFns& device_fns) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_6(mht_6_v, 293, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "ValidateSPDeviceFns");

  TF_VALIDATE_STRUCT_SIZE(SP_DeviceFns, device_fns, SP_DEVICE_FNS_STRUCT_SIZE);
  // All other fields could theoretically be zero/null.
  return port::Status::OK();
}

port::Status ValidateSPStreamExecutor(const SP_StreamExecutor& se,
                                      const SP_Platform& platform) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_7(mht_7_v, 303, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "ValidateSPStreamExecutor");

  TF_VALIDATE_STRUCT_SIZE(SP_StreamExecutor, se,
                          SP_STREAM_EXECUTOR_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, allocate);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, deallocate);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, get_allocator_stats);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, host_memory_allocate);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, host_memory_deallocate);
  if (platform.supports_unified_memory) {
    TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, unified_memory_allocate);
    TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, unified_memory_deallocate);
  }
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, device_memory_usage);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, create_stream);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, destroy_stream);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, create_stream_dependency);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, get_stream_status);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, create_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, destroy_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, get_event_status);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, record_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, wait_for_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, create_timer);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, destroy_timer);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, start_timer);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, stop_timer);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, memcpy_dtoh);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, memcpy_htod);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, sync_memcpy_dtoh);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, sync_memcpy_htod);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, block_host_for_event);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, synchronize_all_activity);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, host_callback);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, mem_zero);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, memset);
  TF_VALIDATE_NOT_NULL(SP_StreamExecutor, se, memset32);
  return port::Status::OK();
}

port::Status ValidateSEPlatformRegistrationParams(
    const SE_PlatformRegistrationParams& params) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_8(mht_8_v, 346, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "ValidateSEPlatformRegistrationParams");

  TF_VALIDATE_STRUCT_SIZE(SE_PlatformRegistrationParams, params,
                          SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(SE_PlatformRegistrationParams, params, destroy_platform);
  TF_VALIDATE_NOT_NULL(SE_PlatformRegistrationParams, params,
                       destroy_platform_fns);
  return port::Status::OK();
}
#undef TF_VALIDATE_NOT_NULL

// Converts SE_EventStatus to Event::Status.
Event::Status SEEventStatusToEventStatus(SE_EventStatus s) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_9(mht_9_v, 360, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "SEEventStatusToEventStatus");

  switch (s) {
    case SE_EVENT_ERROR:
      return Event::Status::kError;
    case SE_EVENT_PENDING:
      return Event::Status::kPending;
    case SE_EVENT_COMPLETE:
      return Event::Status::kComplete;
    default:
      return Event::Status::kUnknown;
  }
}

// Converts DeviceMemoryBase to a C struct.
SP_DeviceMemoryBase DeviceMemoryBaseToC(const DeviceMemoryBase* mem) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_10(mht_10_v, 377, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "DeviceMemoryBaseToC");

  SP_DeviceMemoryBase device_memory_base{SP_DEVICE_MEMORY_BASE_STRUCT_SIZE};
  // `opaque` field inside SP_DeviceMemoryBase is not const.
  // Therefore, we need to cast away the constness before setting it.
  device_memory_base.opaque = const_cast<void*>(mem->opaque());
  device_memory_base.size = mem->size();
  device_memory_base.payload = mem->payload();
  return device_memory_base;
}

DeviceMemoryBase DeviceMemoryBaseFromC(const SP_DeviceMemoryBase& mem) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_11(mht_11_v, 390, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "DeviceMemoryBaseFromC");

  DeviceMemoryBase base(mem.opaque, mem.size);
  base.SetPayload(mem.payload);
  return base;
}

// Wrapper that allows passing std::function across C API.
struct HostCallbackContext {
  std::function<port::Status()> callback;
};

// This wrapper allows calling `HostCallbackContext::callback` across C API.
// This function matches `SE_StatusCallbackFn` signature and will be passed as
// `callback_fn` to `host_callback` in `SP_StreamExecutor`.
void HostCallbackTrampoline(void* ctx, TF_Status* status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_12(mht_12_v, 407, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "HostCallbackTrampoline");

  HostCallbackContext* host_ctx = static_cast<HostCallbackContext*>(ctx);
  port::Status s = host_ctx->callback();
  Set_TF_Status_from_Status(status, s);
  delete host_ctx;
}

class CStreamExecutor : public internal::StreamExecutorInterface {
 public:
  explicit CStreamExecutor(SP_Device device, SP_DeviceFns* device_fns,
                           SP_StreamExecutor* stream_executor,
                           SP_Platform* platform, SP_PlatformFns* platform_fns,
                           SP_TimerFns* timer_fns, const std::string& name,
                           int visible_device_count)
      : device_(std::move(device)),
        device_fns_(device_fns),
        stream_executor_(stream_executor),
        platform_(platform),
        platform_fns_(platform_fns),
        timer_fns_(timer_fns),
        platform_name_(name),
        visible_device_count_(visible_device_count) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_13(mht_13_v, 432, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CStreamExecutor");
}

  ~CStreamExecutor() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_14(mht_14_v, 437, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "~CStreamExecutor");

    platform_fns_->destroy_device(platform_, &device_);
  }

  port::Status Init(int device_ordinal, DeviceOptions device_options) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_15(mht_15_v, 444, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "Init");

    return port::Status::OK();
  }

  DeviceMemoryBase Allocate(uint64 size, int64_t memory_space) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_16(mht_16_v, 451, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "Allocate");

    SP_DeviceMemoryBase mem = {SP_DEVICE_MEMORY_BASE_STRUCT_SIZE};
    stream_executor_->allocate(&device_, size, memory_space, &mem);
    port::Status status = ValidateSPDeviceMemoryBase(mem);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
    return DeviceMemoryBaseFromC(mem);
  }
  DeviceMemoryBase Allocate(uint64 size) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_17(mht_17_v, 463, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "Allocate");

    return Allocate(size, /*memory_space=*/0);
  }
  void* GetSubBuffer(DeviceMemoryBase* parent, uint64 offset,
                     uint64 size) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_18(mht_18_v, 470, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "GetSubBuffer");

    LOG(FATAL) << "GetSubBuffer is not supported by pluggable device.";
  }

  void Deallocate(DeviceMemoryBase* mem) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_19(mht_19_v, 477, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "Deallocate");

    SP_DeviceMemoryBase device_memory_base = DeviceMemoryBaseToC(mem);
    stream_executor_->deallocate(&device_, &device_memory_base);
  }

  void* HostMemoryAllocate(uint64 size) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_20(mht_20_v, 485, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "HostMemoryAllocate");

    return stream_executor_->host_memory_allocate(&device_, size);
  }

  void HostMemoryDeallocate(void* mem) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_21(mht_21_v, 492, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "HostMemoryDeallocate");

    stream_executor_->host_memory_deallocate(&device_, mem);
  }

  bool HostMemoryRegister(void* mem, uint64 size) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_22(mht_22_v, 499, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "HostMemoryRegister");
 return false; }
  bool HostMemoryUnregister(void* mem) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_23(mht_23_v, 503, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "HostMemoryUnregister");
 return false; }

  void* UnifiedMemoryAllocate(uint64 size) override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_24(mht_24_v, 508, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "UnifiedMemoryAllocate");

    CHECK(stream_executor_->unified_memory_allocate);
    return stream_executor_->unified_memory_allocate(&device_, size);
  }

  void UnifiedMemoryDeallocate(void* mem) override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_25(mht_25_v, 516, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "UnifiedMemoryDeallocate");

    CHECK(stream_executor_->unified_memory_deallocate);
    stream_executor_->unified_memory_deallocate(&device_, mem);
  }

  absl::optional<AllocatorStats> GetAllocatorStats() override {
    SP_AllocatorStats c_stats{SP_ALLOCATORSTATS_STRUCT_SIZE};
    TF_Bool has_stats =
        stream_executor_->get_allocator_stats(&device_, &c_stats);
    if (!has_stats) {
      return absl::nullopt;
    }
    port::Status status = ValidateSPAllocatorStats(c_stats);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
      return absl::nullopt;
    }
    ::stream_executor::AllocatorStats stats;
    stats.num_allocs = c_stats.num_allocs;
    stats.bytes_in_use = c_stats.bytes_in_use;
    stats.peak_bytes_in_use = c_stats.peak_bytes_in_use;
    stats.largest_alloc_size = c_stats.largest_alloc_size;
    if (c_stats.has_bytes_limit) {
      stats.bytes_limit = c_stats.bytes_limit;
    }
    stats.bytes_reserved = c_stats.bytes_reserved;
    stats.peak_bytes_reserved = c_stats.peak_bytes_reserved;
    if (c_stats.has_bytes_reservable_limit) {
      stats.bytes_reservable_limit = c_stats.bytes_reservable_limit;
    }
    stats.largest_free_block_bytes = c_stats.largest_free_block_bytes;
    return stats;
  }
  bool SynchronizeAllActivity() override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_26(mht_26_v, 552, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "SynchronizeAllActivity");

    OwnedTFStatus c_status(TF_NewStatus());
    stream_executor_->synchronize_all_activity(&device_, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  port::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64 size) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_27(mht_27_v, 565, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "SynchronousMemZero");

    // TODO(annarev): figure out if we should support memzero/memset
    // functionality by allocating on host and then copying to device.
    return port::UnimplementedError(
        "SynchronousMemZero is not supported by pluggable device.");
  }
  port::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                 uint64 size) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_28(mht_28_v, 575, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "SynchronousMemSet");

    return port::UnimplementedError(
        "SynchronousMemSet is not supported by pluggable device.");
  }
  port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64 size) override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_29(mht_29_v, 583, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "SynchronousMemcpy");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_memory_base = DeviceMemoryBaseToC(gpu_dst);
    stream_executor_->sync_memcpy_htod(&device_, &device_memory_base, host_src,
                                       size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  port::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64 size) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_30(mht_30_v, 595, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "SynchronousMemcpy");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_memory_base = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->sync_memcpy_dtoh(&device_, host_dst, &device_memory_base,
                                       size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                               const DeviceMemoryBase& gpu_src,
                                               uint64 size) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_31(mht_31_v, 607, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "SynchronousMemcpyDeviceToDevice");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_DeviceMemoryBase device_mem_dst = DeviceMemoryBaseToC(gpu_dst);
    SP_DeviceMemoryBase device_mem_src = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->sync_memcpy_dtod(&device_, &device_mem_dst,
                                       &device_mem_src, size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  port::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64 size) override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_32(mht_32_v, 619, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "MemZero");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_DeviceMemoryBase device_mem = DeviceMemoryBaseToC(location);
    stream_executor_->mem_zero(&device_, stream_handle, &device_mem, size,
                               c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  port::Status Memset(Stream* stream, DeviceMemoryBase* location, uint8 pattern,
                      uint64 size) override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_33(mht_33_v, 632, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "Memset");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_DeviceMemoryBase device_mem = DeviceMemoryBaseToC(location);
    stream_executor_->memset(&device_, stream_handle, &device_mem, pattern,
                             size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  port::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32 pattern, uint64 size) override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_34(mht_34_v, 645, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "Memset32");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_DeviceMemoryBase device_mem = DeviceMemoryBaseToC(location);
    stream_executor_->memset32(&device_, stream_handle, &device_mem, pattern,
                               size, c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64 size) override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_35(mht_35_v, 658, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "Memcpy");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_DeviceMemoryBase device_mem_src = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->memcpy_dtoh(&device_, stream_handle, host_dst,
                                  &device_mem_src, size, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64 size) override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_36(mht_36_v, 675, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "Memcpy");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_DeviceMemoryBase device_mem_dst = DeviceMemoryBaseToC(gpu_dst);
    stream_executor_->memcpy_htod(&device_, stream_handle, &device_mem_dst,
                                  host_src, size, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64 size) override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_37(mht_37_v, 693, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "MemcpyDeviceToDevice");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_DeviceMemoryBase device_mem_dst = DeviceMemoryBaseToC(gpu_dst);
    SP_DeviceMemoryBase device_mem_src = DeviceMemoryBaseToC(&gpu_src);
    stream_executor_->memcpy_dtod(&device_, stream_handle, &device_mem_dst,
                                  &device_mem_src, size, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool HostCallback(Stream* stream,
                    std::function<port::Status()> callback) override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_38(mht_38_v, 711, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "HostCallback");

    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    HostCallbackContext* ctx = new HostCallbackContext{callback};
    return stream_executor_->host_callback(&device_, stream_handle,
                                           &HostCallbackTrampoline, ctx);
  }
  port::Status AllocateEvent(Event* event) override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_39(mht_39_v, 721, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "AllocateEvent");

    DCHECK(event != nullptr);
    return static_cast<CEvent*>(event->implementation())->Create();
  }
  port::Status DeallocateEvent(Event* event) override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_40(mht_40_v, 728, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "DeallocateEvent");

    static_cast<CEvent*>(event->implementation())->Destroy();
    return port::Status::OK();
  }
  port::Status RecordEvent(Stream* stream, Event* event) override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_41(mht_41_v, 735, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "RecordEvent");

    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    return static_cast<CEvent*>(event->implementation())->Record(stream_handle);
  }
  port::Status WaitForEvent(Stream* stream, Event* event) override {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_42(mht_42_v, 743, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "WaitForEvent");

    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_Event event_handle =
        static_cast<CEvent*>(event->implementation())->Handle();
    OwnedTFStatus c_status(TF_NewStatus());
    stream_executor_->wait_for_event(&device_, stream_handle, event_handle,
                                     c_status.get());
    port::Status s = StatusFromTF_Status(c_status.get());
    return s;
  }
  Event::Status PollForEventStatus(Event* event) override {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_43(mht_43_v, 757, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "PollForEventStatus");

    SP_Event event_handle =
        static_cast<CEvent*>(event->implementation())->Handle();
    SE_EventStatus event_status =
        stream_executor_->get_event_status(&device_, event_handle);
    return SEEventStatusToEventStatus(event_status);
  }
  bool AllocateStream(Stream* stream) override {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_44(mht_44_v, 767, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "AllocateStream");

    DCHECK(stream != nullptr);
    port::Status status =
        static_cast<CStream*>(stream->implementation())->Create();
    // TODO(annarev): update AllocateStream to return status instead
    // (similar to AllocateEvent).
    return status.ok();
  }
  void DeallocateStream(Stream* stream) override {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_45(mht_45_v, 778, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "DeallocateStream");

    static_cast<CStream*>(stream->implementation())->Destroy();
  }
  bool CreateStreamDependency(Stream* dependent, Stream* other) override {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_46(mht_46_v, 784, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CreateStreamDependency");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream dependent_handle =
        static_cast<CStream*>(dependent->implementation())->Handle();
    SP_Stream other_handle =
        static_cast<CStream*>(other->implementation())->Handle();
    stream_executor_->create_stream_dependency(&device_, dependent_handle,
                                               other_handle, c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool AllocateTimer(Timer* timer) override {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_47(mht_47_v, 801, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "AllocateTimer");

    port::Status status =
        static_cast<CTimer*>(timer->implementation())->Create();
    // TODO(annarev): change return value of AllocateTimer
    // to status (similar to AllocateEvent).
    return status.ok();
  }
  void DeallocateTimer(Timer* timer) override {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_48(mht_48_v, 811, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "DeallocateTimer");

    static_cast<CTimer*>(timer->implementation())->Destroy();
  }
  bool StartTimer(Stream* stream, Timer* timer) override {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_49(mht_49_v, 817, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "StartTimer");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_Timer timer_handle =
        static_cast<CTimer*>(timer->implementation())->Handle();
    stream_executor_->start_timer(&device_, stream_handle, timer_handle,
                                  c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  bool StopTimer(Stream* stream, Timer* timer) override {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_50(mht_50_v, 834, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "StopTimer");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    SP_Timer timer_handle =
        static_cast<CTimer*>(timer->implementation())->Handle();
    stream_executor_->stop_timer(&device_, stream_handle, timer_handle,
                                 c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return false;
    }
    return true;
  }
  port::Status BlockHostForEvent(Stream* stream, Event* event) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_51(mht_51_v, 851, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "BlockHostForEvent");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Event event_handle =
        static_cast<CEvent*>(event->implementation())->Handle();
    stream_executor_->block_host_for_event(&device_, event_handle,
                                           c_status.get());
    return StatusFromTF_Status(c_status.get());
  }

  port::Status BlockHostUntilDone(Stream* stream) override {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_52(mht_52_v, 863, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "BlockHostUntilDone");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();

    // If `block_host_until_done` is set, use it.
    if (stream_executor_->block_host_until_done != nullptr) {
      stream_executor_->block_host_until_done(&device_, stream_handle,
                                              c_status.get());
      return StatusFromTF_Status(c_status.get());
    }
    // Create and record an event and then wait for it.
    SP_Event event_handle;
    stream_executor_->create_event(&device_, &event_handle, c_status.get());
    TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status.get()));
    stream_executor_->record_event(&device_, stream_handle, event_handle,
                                   c_status.get());
    port::Status s = StatusFromTF_Status(c_status.get());
    if (!s.ok()) {
      stream_executor_->destroy_event(&device_, event_handle);
      return s;
    }
    stream_executor_->block_host_for_event(&device_, event_handle,
                                           c_status.get());
    stream_executor_->destroy_event(&device_, event_handle);
    return StatusFromTF_Status(c_status.get());
  }

  port::Status GetStatus(Stream* stream) override {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_53(mht_53_v, 894, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "GetStatus");

    OwnedTFStatus c_status(TF_NewStatus());
    SP_Stream stream_handle =
        static_cast<CStream*>(stream->implementation())->Handle();
    stream_executor_->get_stream_status(&device_, stream_handle,
                                        c_status.get());
    return StatusFromTF_Status(c_status.get());
  }
  int PlatformDeviceCount() override {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_54(mht_54_v, 905, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "PlatformDeviceCount");
 return visible_device_count_; }
  port::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_55(mht_55_v, 909, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "EnablePeerAccessTo");

    return port::UnimplementedError(
        "EnablePeerAccessTo is not supported by pluggable device.");
  }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_56(mht_56_v, 916, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CanEnablePeerAccessTo");

    return false;
  }

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_57(mht_57_v, 923, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "DeviceMemoryUsage");

    return stream_executor_->device_memory_usage(
        &device_, reinterpret_cast<int64_t*>(free),
        reinterpret_cast<int64_t*>(total));
  }

  // Creates a new DeviceDescription object.
  // Ownership is transferred to the caller.
  port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    OwnedTFStatus c_status(TF_NewStatus());

    internal::DeviceDescriptionBuilder builder;
    if (device_.hardware_name != nullptr) {
      builder.set_name(device_.hardware_name);
    }
    if (device_.device_vendor != nullptr) {
      builder.set_device_vendor(device_.device_vendor);
    }
    if (device_.pci_bus_id != nullptr) {
      builder.set_pci_bus_id(device_.pci_bus_id);
    }

    if (device_fns_->get_numa_node != nullptr) {
      int32_t numa_node = device_fns_->get_numa_node(&device_);
      if (numa_node >= 0) {
        builder.set_numa_node(numa_node);
      }
    }

    if (device_fns_->get_memory_bandwidth != nullptr) {
      int64_t memory_bandwidth = device_fns_->get_memory_bandwidth(&device_);
      if (memory_bandwidth >= 0) {
        builder.set_memory_bandwidth(memory_bandwidth);
      }
    }
    // TODO(annarev): Add gflops field in DeviceDescription and set it here.
    // TODO(annarev): Perhaps add `supports_unified_memory` in
    // DeviceDescription.
    return builder.Build();
  }

  // Each call creates a new instance of the platform-specific implementation of
  // the corresponding interface type.
  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override {
    return std::unique_ptr<internal::EventInterface>(
        new CEvent(&device_, stream_executor_));
  }
  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    LOG(FATAL)
        << "CreateKernelImplementation is not supported by pluggable device.";
  }
  std::unique_ptr<internal::StreamInterface> GetStreamImplementation()
      override {
    return std::unique_ptr<internal::StreamInterface>(
        new CStream(&device_, stream_executor_));
  }
  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
    return std::unique_ptr<internal::TimerInterface>(
        new CTimer(&device_, stream_executor_, timer_fns_));
  }

 private:
  SP_Device device_;
  SP_DeviceFns* device_fns_;
  SP_StreamExecutor* stream_executor_;
  SP_Platform* platform_;
  SP_PlatformFns* platform_fns_;
  SP_TimerFns* timer_fns_;
  std::string platform_name_;
  int visible_device_count_;
};
}  // namespace

CPlatform::CPlatform(SP_Platform platform,
                     void (*destroy_platform)(SP_Platform*),
                     SP_PlatformFns platform_fns,
                     void (*destroy_platform_fns)(SP_PlatformFns*),
                     SP_DeviceFns device_fns, SP_StreamExecutor stream_executor,
                     SP_TimerFns timer_fns)
    : platform_(std::move(platform)),
      destroy_platform_(destroy_platform),
      platform_fns_(std::move(platform_fns)),
      destroy_platform_fns_(destroy_platform_fns),
      device_fns_(std::move(device_fns)),
      stream_executor_(std::move(stream_executor)),
      timer_fns_(std::move(timer_fns)),
      name_(platform.name) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_58(mht_58_v, 1015, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CPlatform::CPlatform");
}

CPlatform::~CPlatform() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_59(mht_59_v, 1020, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CPlatform::~CPlatform");

  executor_cache_.DestroyAllExecutors();
  platform_fns_.destroy_device_fns(&platform_, &device_fns_);
  platform_fns_.destroy_stream_executor(&platform_, &stream_executor_);
  platform_fns_.destroy_timer_fns(&platform_, &timer_fns_);
  destroy_platform_(&platform_);
  destroy_platform_fns_(&platform_fns_);
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
CPlatform::DescriptionForDevice(int ordinal) const {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_60(mht_60_v, 1033, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CPlatform::DescriptionForDevice");

  // TODO(annarev): see if we can get StreamExecutor instance
  // and call GetDeviceDescription. executor_cache_.Get would need
  // to be made const for it to work.
  internal::DeviceDescriptionBuilder builder;
  builder.set_name(name_);
  return builder.Build();
}
port::StatusOr<StreamExecutor*> CPlatform::ExecutorForDevice(int ordinal) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_61(mht_61_v, 1044, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CPlatform::ExecutorForDevice");

  stream_executor::StreamExecutorConfig config;
  config.ordinal = ordinal;
  return GetExecutor(config);
}
port::StatusOr<StreamExecutor*> CPlatform::ExecutorForDeviceWithPluginConfig(
    int ordinal, const PluginConfig& plugin_config) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_62(mht_62_v, 1053, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CPlatform::ExecutorForDeviceWithPluginConfig");

  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = plugin_config;
  return GetExecutor(config);
}
port::StatusOr<StreamExecutor*> CPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_63(mht_63_v, 1063, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CPlatform::GetExecutor");

  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}
port::StatusOr<std::unique_ptr<StreamExecutor>> CPlatform::GetUncachedExecutor(
    const StreamExecutorConfig& config) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_64(mht_64_v, 1071, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "CPlatform::GetUncachedExecutor");

  // Fill device creation params
  SE_CreateDeviceParams device_params{SE_CREATE_DEVICE_PARAMS_STRUCT_SIZE};
  SP_Device device{SP_DEVICE_STRUCT_SIZE};
  device_params.device = &device;
  device_params.ext = nullptr;
  device_params.ordinal = config.ordinal;
  OwnedTFStatus c_status(TF_NewStatus());

  // Create Device
  platform_fns_.create_device(&platform_, &device_params, c_status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPDevice(device));

  // Get Device Count
  int visible_device_count = 0;
  platform_fns_.get_device_count(&platform_, &visible_device_count,
                                 c_status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(c_status.get()));

  auto executor = absl::make_unique<CStreamExecutor>(
      std::move(device), &device_fns_, &stream_executor_, &platform_,
      &platform_fns_, &timer_fns_, name_, visible_device_count);
  auto result = absl::make_unique<StreamExecutor>(this, std::move(executor),
                                                  config.ordinal);
  return result;
}

port::Status InitStreamExecutorPlugin(void* dso_handle,
                                      std::string* device_type,
                                      std::string* platform_name) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_65(mht_65_v, 1104, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "InitStreamExecutorPlugin");

  tensorflow::Env* env = tensorflow::Env::Default();

  // Step 1: Load symbol for `TF_InitPlugin`
  void* dso_symbol;
  TF_RETURN_IF_ERROR(
      env->GetSymbolFromLibrary(dso_handle, "SE_InitPlugin", &dso_symbol));

  // Step 2: Call `TF_InitPlugin`
  auto init_fn = reinterpret_cast<SEInitPluginFn>(dso_symbol);
  return InitStreamExecutorPlugin(init_fn, device_type, platform_name);
}

port::Status InitStreamExecutorPlugin(SEInitPluginFn init_fn,
                                      std::string* device_type,
                                      std::string* platform_name) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executorDTcc mht_66(mht_66_v, 1122, "", "./tensorflow/c/experimental/stream_executor/stream_executor.cc", "InitStreamExecutorPlugin");

  SE_PlatformRegistrationParams params{
      SE_PLATFORM_REGISTRATION_PARAMS_STRUCT_SIZE};
  SP_Platform platform{SP_PLATFORM_STRUCT_SIZE};
  SP_PlatformFns platform_fns{SP_PLATFORM_FNS_STRUCT_SIZE};
  params.major_version = SE_MAJOR;
  params.minor_version = SE_MINOR;
  params.patch_version = SE_PATCH;
  params.platform = &platform;
  params.platform_fns = &platform_fns;

  OwnedTFStatus c_status(TF_NewStatus());
  init_fn(&params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSEPlatformRegistrationParams(params));
  TF_RETURN_IF_ERROR(ValidateSPPlatform(platform));
  TF_RETURN_IF_ERROR(ValidateSPPlatformFns(platform_fns));

  // Fill SP_DeviceFns creation params
  SE_CreateDeviceFnsParams device_fns_params{
      SE_CREATE_DEVICE_FNS_PARAMS_STRUCT_SIZE};
  SP_DeviceFns device_fns{SP_DEVICE_FNS_STRUCT_SIZE};
  device_fns_params.device_fns = &device_fns;

  // Create StreamExecutor
  platform_fns.create_device_fns(&platform, &device_fns_params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPDeviceFns(device_fns));

  // Fill stream executor creation params
  SE_CreateStreamExecutorParams se_params{
      SE_CREATE_STREAM_EXECUTOR_PARAMS_STRUCT_SIZE};
  SP_StreamExecutor se{SP_STREAMEXECUTOR_STRUCT_SIZE};
  se_params.stream_executor = &se;

  // Create StreamExecutor
  platform_fns.create_stream_executor(&platform, &se_params, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPStreamExecutor(se, platform));

  SP_TimerFns timer_fns{SP_TIMER_FNS_STRUCT_SIZE};
  platform_fns.create_timer_fns(&platform, &timer_fns, c_status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(c_status.get()));
  TF_RETURN_IF_ERROR(ValidateSPTimerFns(timer_fns));

  // Register new platform
  *device_type = std::string(platform.type);
  *platform_name = std::string(platform.name);
  std::unique_ptr<stream_executor::CPlatform> cplatform(
      new stream_executor::CPlatform(
          std::move(platform), params.destroy_platform, std::move(platform_fns),
          params.destroy_platform_fns, std::move(device_fns), std::move(se),
          std::move(timer_fns)));
  SE_CHECK_OK(stream_executor::MultiPlatformManager::RegisterPlatform(
      std::move(cplatform)));
  // TODO(annarev): Return `use_bfc_allocator` value in some way so that it is
  // available in `PluggableDeviceProcessState` once the latter is checked in.
  return port::Status::OK();
}
}  // namespace stream_executor
