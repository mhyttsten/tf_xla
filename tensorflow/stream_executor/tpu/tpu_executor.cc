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
class MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc() {
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

#include "tensorflow/stream_executor/tpu/tpu_executor.h"

#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_event.h"
#include "tensorflow/stream_executor/tpu/tpu_stream.h"
#include "tensorflow/stream_executor/tpu/tpu_timer.h"

using stream_executor::DeviceMemoryBase;

namespace tensorflow {
namespace tpu {

namespace {
using ::stream_executor::port::Status;
}  // namespace

TpuExecutor::~TpuExecutor() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_0(mht_0_v, 204, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::~TpuExecutor");

  tpu::ExecutorApiFn()->TpuExecutor_FreeFn(executor_);
}

Status TpuExecutor::Init(int device_ordinal,
                         ::stream_executor::DeviceOptions device_options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_1(mht_1_v, 212, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::Init");

  StatusHelper status;
  SE_DeviceOptions* options =
      tpu::ExecutorApiFn()->TpuExecutor_NewDeviceOptionsFn(
          device_options.flags());
  tpu::ExecutorApiFn()->TpuExecutor_InitFn(executor_, device_ordinal, options,
                                           status.c_status);
  tpu::ExecutorApiFn()->TpuExecutor_FreeDeviceOptionsFn(options);
  return status.status();
}

int TpuExecutor::PlatformDeviceCount() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_2(mht_2_v, 226, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::PlatformDeviceCount");

  return tpu::ExecutorApiFn()->TpuExecutor_PlatformDeviceCountFn(executor_);
}

void TpuExecutor::SyncAndForgetFailedStreams() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_3(mht_3_v, 233, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::SyncAndForgetFailedStreams");

  tpu::ExecutorApiFn()->TpuExecutor_SyncAndForgetFailedStreamsFn(executor_);
}

bool TpuExecutor::SynchronizeAllActivity() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_4(mht_4_v, 240, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::SynchronizeAllActivity");

  return tpu::ExecutorApiFn()->TpuExecutor_SynchronizeAllActivityFn(executor_);
}

Status TpuExecutor::BlockHostUntilDone(Stream* stream) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_5(mht_5_v, 247, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::BlockHostUntilDone");

  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_BlockHostUntilDoneFn(
      executor_, get_stream(stream->implementation()), status.c_status);
  return status.status();
}

Status TpuExecutor::BlockUntilDoneOrFailed() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_6(mht_6_v, 257, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::BlockUntilDoneOrFailed");

  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_BlockUntilDoneOrFailedFn(executor_,
                                                             status.c_status);
  return status.status();
}

Status TpuExecutor::GetStatus(Stream* stream) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_7(mht_7_v, 267, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::GetStatus");

  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_GetStatusFn(
      executor_, get_stream(stream->implementation()), status.c_status);
  return status.status();
}

tpu::TpuCoreLocationExternal TpuExecutor::GetCoreLocationExternal() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_8(mht_8_v, 277, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::GetCoreLocationExternal");

  return tpu::TpuCoreLocationExternal(
      tpu::ExecutorApiFn()->TpuExecutor_GetCoreLocationFn(executor_));
}

bool TpuExecutor::AllocateStream(Stream* stream) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_9(mht_9_v, 285, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::AllocateStream");

  return tpu::ExecutorApiFn()->TpuExecutor_AllocateStreamFn(
      executor_, get_stream(stream->implementation()));
}

void TpuExecutor::DeallocateStream(Stream* stream) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_10(mht_10_v, 293, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::DeallocateStream");

  tpu::ExecutorApiFn()->TpuExecutor_DeallocateStreamFn(
      executor_, get_stream(stream->implementation()));
  tpu_platform().mutex().lock();
  stream_map().erase(stream->implementation());
  tpu_platform().mutex().unlock();
}

bool TpuExecutor::CreateStreamDependency(Stream* dependent, Stream* other) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_11(mht_11_v, 304, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::CreateStreamDependency");

  return tpu::ExecutorApiFn()->TpuExecutor_CreateStreamDependencyFn(
      executor_, get_stream(dependent->implementation()),
      get_stream(other->implementation()));
}

Status TpuExecutor::AllocateEvent(Event* event) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_12(mht_12_v, 313, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::AllocateEvent");
 return Status::OK(); }

Status TpuExecutor::DeallocateEvent(Event* event) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_13(mht_13_v, 318, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::DeallocateEvent");

  tpu_platform().EraseEvent(event->implementation());
  return Status::OK();
}

// AllocateTimer/DeallocateTimer have no specialization.
bool TpuExecutor::AllocateTimer(Timer* timer) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_14(mht_14_v, 327, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::AllocateTimer");
 return true; }

void TpuExecutor::DeallocateTimer(Timer* timer) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_15(mht_15_v, 332, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::DeallocateTimer");
}

bool TpuExecutor::StartTimer(Stream* stream, ::stream_executor::Timer* timer) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_16(mht_16_v, 337, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::StartTimer");

  return tpu::ExecutorApiFn()->TpuExecutor_StartTimerFn(
      executor_, get_stream(stream->implementation()),
      timer_map_.at(timer->implementation()));
}

bool TpuExecutor::StopTimer(Stream* stream, ::stream_executor::Timer* timer) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_17(mht_17_v, 346, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::StopTimer");

  return tpu::ExecutorApiFn()->TpuExecutor_StopTimerFn(
      executor_, get_stream(stream->implementation()),
      timer_map_.at(timer->implementation()));
}

stream_executor::Event::Status TpuExecutor::PollForEventStatus(
    stream_executor::Event* event) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_18(mht_18_v, 356, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::PollForEventStatus");

  auto se_event = tpu_platform().LookupEvent(event->implementation());
  return stream_executor::Event::Status(
      tpu::ExecutorApiFn()->TpuExecutor_PollForEventStatusFn(executor_,
                                                             se_event));
}

Status TpuExecutor::RecordEvent(Stream* stream,
                                ::stream_executor::Event* event) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_19(mht_19_v, 367, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::RecordEvent");

  StatusHelper status;
  auto se_event = tpu_platform().LookupEvent(event->implementation());
  tpu::ExecutorApiFn()->TpuExecutor_RecordEventFn(
      executor_, get_stream(stream->implementation()), se_event,
      status.c_status);
  return status.status();
}

Status TpuExecutor::WaitForEvent(Stream* stream,
                                 ::stream_executor::Event* event) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_20(mht_20_v, 380, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::WaitForEvent");

  StatusHelper status;
  auto se_event = tpu_platform().LookupEvent(event->implementation());
  tpu::ExecutorApiFn()->TpuExecutor_WaitForEventFn(
      executor_, get_stream(stream->implementation()), se_event,
      status.c_status);
  return status.status();
}

// Implementations for Timer, Stream, Event
// We need to map these implementations to internal equivalents -- thus we
// allocate the internal Timer, Stream and Event operations here, and map
// the implementations to the internal values. The "wrapper" interfaces are
// responsible for deallocating the internal value when they are destroyed.

// Called by Timer::Timer
std::unique_ptr<::stream_executor::internal::TimerInterface>
TpuExecutor::GetTimerImplementation() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_21(mht_21_v, 400, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::GetTimerImplementation");

  SE_Timer* tpu_timer = tpu::ExecutorApiFn()->TpuTimer_NewFn(executor_);
  auto ptr = absl::make_unique<TpuTimer>(tpu_timer);
  timer_map_[ptr.get()] = tpu_timer;
  return ptr;
}

// Called by Stream::Stream
std::unique_ptr<::stream_executor::internal::StreamInterface>
TpuExecutor::GetStreamImplementation() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_22(mht_22_v, 412, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::GetStreamImplementation");

  SE_Stream* tpu_stream = tpu::ExecutorApiFn()->TpuStream_NewFn(executor_);
  auto ptr = absl::make_unique<tpu::TpuStream>(tpu_stream);
  tpu_platform().mutex().lock();
  stream_map()[ptr.get()] = tpu_stream;
  tpu_platform().mutex().unlock();
  return ptr;
}

// Called by Event::Event
std::unique_ptr<::stream_executor::internal::EventInterface>
TpuExecutor::CreateEventImplementation() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_23(mht_23_v, 426, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::CreateEventImplementation");

  SE_Event* tpu_event = tpu::ExecutorApiFn()->TpuEvent_NewFn(executor_);
  auto ptr = absl::make_unique<TpuEvent>(tpu_event);
  tpu_platform().InsertEvent(ptr.get(), tpu_event);
  return ptr;
}

DeviceMemoryBase TpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_24(mht_24_v, 436, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::Allocate");

  SE_DeviceMemoryBase se_base = tpu::ExecutorApiFn()->TpuExecutor_AllocateFn(
      executor_, size, memory_space);
  return ApiConverter::FromC(se_base);
}

void TpuExecutor::Deallocate(const DeviceMemoryBase& memory) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_25(mht_25_v, 445, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::Deallocate");

  SE_DeviceMemoryBase se_base = ApiConverter::ToC(memory);
  tpu::ExecutorApiFn()->TpuExecutor_DeallocateFn(executor_, &se_base);
}

void TpuExecutor::Deallocate(DeviceMemoryBase* memory) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_26(mht_26_v, 453, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::Deallocate");

  SE_DeviceMemoryBase se_base = ApiConverter::ToC(*memory);
  tpu::ExecutorApiFn()->TpuExecutor_DeallocateFn(executor_, &se_base);
}

bool TpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_27(mht_27_v, 461, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::DeviceMemoryUsage");

  int64_t _free;
  int64_t _total;
  if (tpu::ExecutorApiFn()->TpuExecutor_DeviceMemoryUsageFn(executor_, &_free,
                                                            &_total)) {
    *free = _free;
    *total = _total;
    return true;
  }
  return false;
}

absl::optional<stream_executor::AllocatorStats>
TpuExecutor::GetAllocatorStats() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_28(mht_28_v, 477, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::GetAllocatorStats");

  SE_AllocatorStats c_stats;
  if (tpu::ExecutorApiFn()->TpuExecutor_GetAllocatorStatsFn(executor_,
                                                            &c_stats)) {
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
  return {};
}

Status TpuExecutor::WaitForInfeedReady(int32_t infeed_queue_index) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_29(mht_29_v, 503, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::WaitForInfeedReady");

  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_WaitForInfeedReadyFn(
      executor_, infeed_queue_index, status.c_status);
  return status.status();
}

Status TpuExecutor::WaitForOutfeedReady(int32_t outfeed_queue_index) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_30(mht_30_v, 513, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::WaitForOutfeedReady");

  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_WaitForOutfeedReadyFn(
      executor_, outfeed_queue_index, status.c_status);
  return status.status();
}

void TpuExecutor::DequeueOutfeed(int32_t outfeed_queue_index,
                                 absl::Span<uint8> bytes, StatusCallback done) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_31(mht_31_v, 524, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::DequeueOutfeed");

  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_DequeueOutfeedFn(
      executor_, outfeed_queue_index, bytes.data(), bytes.size(),
      status.c_status);
  done(status.status());
}

Status TpuExecutor::EnqueueInfeed(int32_t infeed_queue_index,
                                  absl::Span<const uint8> bytes) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_32(mht_32_v, 536, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::EnqueueInfeed");

  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_EnqueueInfeedFn(
      executor_, infeed_queue_index, bytes.data(), bytes.size(),
      status.c_status);
  return status.status();
}

bool TpuExecutor::Memcpy(Stream* stream, void* host_dst,
                         const ::stream_executor::DeviceMemoryBase& device_src,
                         uint64_t size) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_33(mht_33_v, 549, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::Memcpy");

  SE_DeviceMemoryBase se_base = ApiConverter::ToC(device_src);
  return tpu::ExecutorApiFn()->TpuExecutor_MemcpyToHostFn(
      executor_, get_stream(stream->implementation()), host_dst, &se_base,
      size);
}

bool TpuExecutor::Memcpy(Stream* stream,
                         ::stream_executor::DeviceMemoryBase* device_dst,
                         const void* host_src, uint64_t size) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_34(mht_34_v, 561, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::Memcpy");

  SE_DeviceMemoryBase se_base = ApiConverter::ToC(*device_dst);
  return tpu::ExecutorApiFn()->TpuExecutor_MemcpyFromHostFn(
      executor_, get_stream(stream->implementation()), &se_base, host_src,
      size);
}

Status TpuExecutor::SynchronousMemcpy(
    ::stream_executor::DeviceMemoryBase* device_dst, const void* host_src,
    uint64_t size) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_35(mht_35_v, 573, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::SynchronousMemcpy");

  StatusHelper status;
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(*device_dst);
  tpu::ExecutorApiFn()->TpuExecutor_SynchronousMemcpyFromHostFn(
      executor_, &se_base, host_src, size, status.c_status);
  return status.status();
}

Status TpuExecutor::SynchronousMemcpy(
    void* host_dst, const ::stream_executor::DeviceMemoryBase& device_src,
    uint64_t size) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_36(mht_36_v, 586, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::SynchronousMemcpy");

  StatusHelper status;
  SE_DeviceMemoryBase se_base = ApiConverter::ToC(device_src);
  tpu::ExecutorApiFn()->TpuExecutor_SynchronousMemcpyToHostFn(
      executor_, host_dst, &se_base, size, status.c_status);
  return status.status();
}

Status TpuExecutor::SynchronousMemcpyDeviceToDevice(
    ::stream_executor::DeviceMemoryBase* device_dst,
    const ::stream_executor::DeviceMemoryBase& device_src, uint64_t size) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_37(mht_37_v, 599, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::SynchronousMemcpyDeviceToDevice");

  return ::stream_executor::port::UnimplementedError(
      "This operation not supported on TPU");
}

bool TpuExecutor::MemcpyDeviceToDevice(
    Stream* stream, ::stream_executor::DeviceMemoryBase* gpu_dst,
    const ::stream_executor::DeviceMemoryBase& host_src, uint64_t size) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_38(mht_38_v, 609, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::MemcpyDeviceToDevice");

  LOG(FATAL) << __func__ << " not supported on TpuExecutor";
}

Status TpuExecutor::UnloadAllPrograms() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_39(mht_39_v, 616, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::UnloadAllPrograms");

  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_UnloadAllProgramsFn(executor_,
                                                        status.c_status);
  return status.status();
}

Status TpuExecutor::EnqueueCompactionOnStreamForHbm(Stream* compaction_stream) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_40(mht_40_v, 626, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::EnqueueCompactionOnStreamForHbm");

  StatusHelper status;
  tpu::ExecutorApiFn()->TpuExecutor_EnqueueCompactionOnStreamForHbmFn(
      executor_, get_stream(compaction_stream->implementation()),
      status.c_status);
  return status.status();
}

struct HostCallbackContext {
  std::function<Status()> callback;
};

TF_Status* HostCallbackTrampoline(void* ctx) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_41(mht_41_v, 641, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "HostCallbackTrampoline");

  HostCallbackContext* host_ctx = reinterpret_cast<HostCallbackContext*>(ctx);
  Status status = host_ctx->callback();
  TF_Status* c_status = tpu::ExecutorApiFn()->TpuStatus_CreateFn(
      status.code(), status.error_message().c_str());
  delete host_ctx;
  return c_status;
}

bool TpuExecutor::HostCallback(Stream* stream,
                               std::function<Status()> callback) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_42(mht_42_v, 654, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::HostCallback");

  HostCallbackContext* ctx = new HostCallbackContext{callback};
  return tpu::ExecutorApiFn()->TpuExecutor_HostCallbackFn(
      executor_, get_stream(stream->implementation()), &HostCallbackTrampoline,
      ctx);
}

TpuExecutor::StatusOr<std::unique_ptr<::stream_executor::DeviceDescription>>
TpuExecutor::CreateDeviceDescription() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTcc mht_43(mht_43_v, 665, "", "./tensorflow/stream_executor/tpu/tpu_executor.cc", "TpuExecutor::CreateDeviceDescription");

  StatusHelper status;
  SE_DeviceDescription* description =
      tpu::ExecutorApiFn()->TpuDeviceDescription_NewFn();
  auto cleanup = tensorflow::gtl::MakeCleanup([description]() {
    tpu::ExecutorApiFn()->TpuDeviceDescription_FreeFn(description);
  });
  tpu::ExecutorApiFn()->TpuExecutor_CreateDeviceDescriptionFn(
      executor_, description, status.c_status);
  if (status.status().ok()) {
    stream_executor::internal::DeviceDescriptionBuilder builder;
    CHECK_NE(description->device_vendor, nullptr);
    builder.set_device_vendor(description->device_vendor);
    builder.set_name(description->name);
    builder.set_clock_rate_ghz(description->clock_rate_ghz);
    builder.set_core_count(description->core_count);
    builder.set_ecc_enabled(description->ecc_enabled);
    builder.set_device_memory_size(description->device_memory_size);
    builder.set_platform_version(description->platform_version);
    return builder.Build();
  }
  return status.status();
}

}  // namespace tpu
}  // namespace tensorflow
