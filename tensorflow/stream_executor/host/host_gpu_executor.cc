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
class MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Implementation of HostExecutor class [of those methods not defined in the
// class declaration].
#include "tensorflow/stream_executor/host/host_gpu_executor.h"

#include <stdint.h>
#include <string.h>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace host {

HostStream* AsHostStream(Stream* stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_0(mht_0_v, 207, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "AsHostStream");

  DCHECK(stream != nullptr);
  return dynamic_cast<HostStream*>(stream->implementation());
}

HostExecutor::HostExecutor(const PluginConfig& plugin_config)
    : plugin_config_(plugin_config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_1(mht_1_v, 216, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::HostExecutor");
}

HostExecutor::~HostExecutor() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_2(mht_2_v, 221, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::~HostExecutor");
}

port::Status HostExecutor::Init(int device_ordinal,
                                DeviceOptions device_options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_3(mht_3_v, 227, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::Init");

  auto it =
      device_options.non_portable_tags.find("host_thread_stack_size_in_bytes");
  if (it != device_options.non_portable_tags.end()) {
    if (!absl::SimpleAtoi(it->second, &thread_stack_size_in_bytes_)) {
      return port::InvalidArgumentError(absl::StrCat(
          "Unable to parse host_thread_stack_size_in_bytes as an integer: ",
          it->second));
    }
  }
  return port::Status::OK();
}

bool HostExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_4(mht_4_v, 243, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::DeviceMemoryUsage");

  tensorflow::port::MemoryInfo mem_info = tensorflow::port::GetMemoryInfo();
  *free = (mem_info.free != INT64_MAX) ? mem_info.free : -1;
  *total = (mem_info.total != INT64_MAX) ? mem_info.total : -1;
  return true;
}

DeviceMemoryBase HostExecutor::Allocate(uint64_t size, int64_t memory_space) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_5(mht_5_v, 253, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::Allocate");

  CHECK_EQ(memory_space, 0);
  // Use a minimum alignment of 64 bytes to be friendly to AVX512 code.
  // This should probably be kept in sync with
  // tensorflow::Allocator::kAllocatorAlignment.
  return DeviceMemoryBase(
      tensorflow::port::AlignedMalloc(size, /*minimum_alignment=*/64), size);
}

void* HostExecutor::GetSubBuffer(DeviceMemoryBase* parent,
                                 uint64_t offset_bytes, uint64_t size_bytes) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_6(mht_6_v, 266, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::GetSubBuffer");

  return reinterpret_cast<char*>(parent->opaque()) + offset_bytes;
}

void HostExecutor::Deallocate(DeviceMemoryBase* mem) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_7(mht_7_v, 273, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::Deallocate");

  tensorflow::port::AlignedFree(mem->opaque());
}

port::Status HostExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64_t size) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_8(mht_8_v, 281, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::SynchronousMemZero");

  memset(location->opaque(), 0, size);
  return port::Status::OK();
}

port::Status HostExecutor::SynchronousMemSet(DeviceMemoryBase* location,
                                             int value, uint64_t size) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_9(mht_9_v, 290, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::SynchronousMemSet");

  memset(location->opaque(), value, size);
  return port::Status::OK();
}

bool HostExecutor::Memcpy(Stream* stream, void* host_dst,
                          const DeviceMemoryBase& gpu_src, uint64_t size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_10(mht_10_v, 299, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::Memcpy");

  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  AsHostStream(stream)->EnqueueTask(
      [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return true;
}

bool HostExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                          const void* host_src, uint64_t size) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_11(mht_11_v, 312, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::Memcpy");

  void* dst_mem = gpu_dst->opaque();
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return true;
}

bool HostExecutor::MemcpyDeviceToDevice(Stream* stream,
                                        DeviceMemoryBase* gpu_dst,
                                        const DeviceMemoryBase& gpu_src,
                                        uint64_t size) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_12(mht_12_v, 327, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::MemcpyDeviceToDevice");

  void* dst_mem = gpu_dst->opaque();
  void* src_mem = const_cast<void*>(gpu_src.opaque());
  // Enqueue this [asynchronous] "device-to-device" (i.e., host-to-host, given
  // the nature of the HostExecutor) memcpy  on the stream (HostStream)
  // associated with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [src_mem, dst_mem, size]() { memcpy(dst_mem, src_mem, size); });
  return true;
}

port::Status HostExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                   uint64_t size) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_13(mht_13_v, 342, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::MemZero");

  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size]() { memset(gpu_mem, 0, size); });
  return port::Status::OK();
}

port::Status HostExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                  uint8 pattern, uint64_t size) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_14(mht_14_v, 355, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::Memset");

  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return port::Status::OK();
}

port::Status HostExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                    uint32 pattern, uint64_t size) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_15(mht_15_v, 368, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::Memset32");

  void* gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return port::Status::OK();
}

port::Status HostExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_16(mht_16_v, 382, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::SynchronousMemcpy");

  memcpy(gpu_dst->opaque(), host_src, size);
  return port::Status::OK();
}

port::Status HostExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_17(mht_17_v, 392, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::SynchronousMemcpy");

  memcpy(host_dst, gpu_src.opaque(), size);
  return port::Status::OK();
}

port::Status HostExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src, uint64_t size) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_18(mht_18_v, 401, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::SynchronousMemcpyDeviceToDevice");

  memcpy(gpu_dst->opaque(), gpu_src.opaque(), size);
  return port::Status::OK();
}

bool HostExecutor::HostCallback(Stream* stream,
                                std::function<port::Status()> callback) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_19(mht_19_v, 410, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::HostCallback");

  AsHostStream(stream)->EnqueueTaskWithStatus(callback);
  return true;
}

bool HostExecutor::AllocateStream(Stream* stream) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_20(mht_20_v, 418, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::AllocateStream");
 return true; }

void HostExecutor::DeallocateStream(Stream* stream) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_21(mht_21_v, 423, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::DeallocateStream");
}

bool HostExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_22(mht_22_v, 428, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::CreateStreamDependency");

  auto event = std::make_shared<absl::Notification>();
  AsHostStream(other)->EnqueueTask([event]() { event->Notify(); });
  AsHostStream(dependent)->EnqueueTask(
      [event]() { event->WaitForNotification(); });
  return true;
}

class HostEvent : public internal::EventInterface {
 public:
  HostEvent() : notification_(std::make_shared<absl::Notification>()) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_23(mht_23_v, 441, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostEvent");
}

  std::shared_ptr<absl::Notification>& notification() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_24(mht_24_v, 446, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "notification");
 return notification_; }

 private:
  // We use a std::shared_ptr here because the client may delete the HostEvent
  // object while there are still RecordEvent and WaitForEvent callbacks pending
  // on a stream.
  std::shared_ptr<absl::Notification> notification_;
};

std::unique_ptr<internal::EventInterface>
HostExecutor::CreateEventImplementation() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_25(mht_25_v, 459, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::CreateEventImplementation");

  return std::unique_ptr<internal::EventInterface>(new HostEvent());
}

static HostEvent* AsHostEvent(Event* event) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_26(mht_26_v, 466, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "AsHostEvent");

  DCHECK(event != nullptr);
  return static_cast<HostEvent*>(event->implementation());
}

port::Status HostExecutor::AllocateEvent(Event* /*event*/) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_27(mht_27_v, 474, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::AllocateEvent");

  return port::Status::OK();
}

port::Status HostExecutor::DeallocateEvent(Event* /*event*/) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_28(mht_28_v, 481, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::DeallocateEvent");

  return port::Status::OK();
}

port::Status HostExecutor::RecordEvent(Stream* stream, Event* event) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_29(mht_29_v, 488, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::RecordEvent");

  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->notification();
  AsHostStream(stream)->EnqueueTask([notification]() {
    CHECK(!notification->HasBeenNotified());
    notification->Notify();
  });
  return port::Status::OK();
}

port::Status HostExecutor::WaitForEvent(Stream* stream, Event* event) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_30(mht_30_v, 501, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::WaitForEvent");

  std::shared_ptr<absl::Notification> notification =
      AsHostEvent(event)->notification();
  AsHostStream(stream)->EnqueueTask(
      [notification]() { notification->WaitForNotification(); });
  return port::Status::OK();
}

Event::Status HostExecutor::PollForEventStatus(Event* event) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_31(mht_31_v, 512, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::PollForEventStatus");

  absl::Notification& notification = *AsHostEvent(event)->notification();
  return notification.HasBeenNotified() ? Event::Status::kComplete
                                        : Event::Status::kPending;
}

bool HostExecutor::StartTimer(Stream* stream, Timer* timer) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_32(mht_32_v, 521, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::StartTimer");

  dynamic_cast<HostTimer*>(timer->implementation())->Start(stream);
  return true;
}

bool HostExecutor::StopTimer(Stream* stream, Timer* timer) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_33(mht_33_v, 529, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::StopTimer");

  dynamic_cast<HostTimer*>(timer->implementation())->Stop(stream);
  return true;
}

port::Status HostExecutor::BlockHostUntilDone(Stream* stream) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_34(mht_34_v, 537, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::BlockHostUntilDone");

  return AsHostStream(stream)->BlockUntilDone();
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
HostExecutor::CreateDeviceDescription(int device_ordinal) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_35(mht_35_v, 545, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::CreateDeviceDescription");

  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  // TODO(rspringer): How to report a value that's based in reality but that
  // doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
  builder.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);

  float cycle_counter_frequency = static_cast<float>(
      tensorflow::profile_utils::CpuUtils::GetCycleCounterFrequency());
  builder.set_clock_rate_ghz(cycle_counter_frequency / 1e9);

  builder.set_name("Host");
  builder.set_platform_version("Default Version");

  return builder.Build();
}

bool HostExecutor::SupportsBlas() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_36(mht_36_v, 567, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::SupportsBlas");

  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::BlasFactory>(kHostPlatformId,
                                                plugin_config_.blas())
      .ok();
}

blas::BlasSupport* HostExecutor::CreateBlas() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_37(mht_37_v, 577, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::CreateBlas");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(kHostPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

bool HostExecutor::SupportsFft() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_38(mht_38_v, 594, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::SupportsFft");

  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::FftFactory>(kHostPlatformId,
                                               plugin_config_.fft())
      .ok();
}

fft::FftSupport* HostExecutor::CreateFft() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_39(mht_39_v, 604, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::CreateFft");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(kHostPlatformId,
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

bool HostExecutor::SupportsRng() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_40(mht_40_v, 621, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::SupportsRng");

  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::RngFactory>(kHostPlatformId,
                                               plugin_config_.rng())
      .ok();
}

rng::RngSupport* HostExecutor::CreateRng() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_41(mht_41_v, 631, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::CreateRng");

  PluginRegistry* registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(kHostPlatformId,
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

std::unique_ptr<internal::StreamInterface>
HostExecutor::GetStreamImplementation() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTcc mht_42(mht_42_v, 649, "", "./tensorflow/stream_executor/host/host_gpu_executor.cc", "HostExecutor::GetStreamImplementation");

  return std::unique_ptr<internal::StreamInterface>(
      new HostStream(thread_stack_size_in_bytes_));
}

}  // namespace host
}  // namespace stream_executor
