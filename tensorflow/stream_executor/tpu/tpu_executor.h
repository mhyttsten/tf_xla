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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_H_
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
class MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"
#include "tensorflow/stream_executor/timer.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_stream.h"

namespace tensorflow {
namespace tpu {

class TpuExecutor : public tensorflow::tpu::TpuExecutorInterface {
 public:
  using Status = ::stream_executor::port::Status;
  template <typename T>
  using StatusOr = ::stream_executor::port::StatusOr<T>;
  using StatusCallback = std::function<void(const Status&)>;
  using Stream = ::stream_executor::Stream;
  using Event = ::stream_executor::Event;
  using Timer = ::stream_executor::Timer;
  using DeviceMemoryBase = ::stream_executor::DeviceMemoryBase;
  using StreamInterface = ::stream_executor::internal::StreamInterface;
  using StreamExecutorInterface =
      ::stream_executor::internal::StreamExecutorInterface;

  using TimerMap =
      absl::flat_hash_map<stream_executor::internal::TimerInterface*,
                          SE_Timer*>;

  explicit TpuExecutor(::tensorflow::tpu::TpuPlatformInterface* platform,
                       SE_StreamExecutor* executor)
      : platform_(platform), executor_(executor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_0(mht_0_v, 231, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "TpuExecutor");
}

  ~TpuExecutor() override;

  Status Init(int device_ordinal,
              ::stream_executor::DeviceOptions device_options) override;

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;

  Status AllocateEvent(Event* event) override;

  bool AllocateStream(Stream* stream) override;

  bool AllocateTimer(Timer* timer) override;

  Status BlockHostUntilDone(::stream_executor::Stream* stream) override;

  Status BlockUntilDoneOrFailed();

  StatusOr<std::unique_ptr<::stream_executor::DeviceDescription>>
  CreateDeviceDescription() const override;

  bool CreateStreamDependency(Stream* dependent, Stream* other) override;

  void DeallocateStream(Stream* stream) override;

  void Deallocate(const DeviceMemoryBase& memory);

  void Deallocate(DeviceMemoryBase* memory) override;

  Status DeallocateEvent(Event* event) override;

  void DeallocateTimer(Timer* timer) override;

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  void DequeueOutfeed(int32_t outfeed_queue_index, absl::Span<uint8> bytes,
                      StatusCallback done);

  Status EnqueueInfeed(int32_t infeed_queue_index,
                       absl::Span<const uint8> bytes);

  absl::optional<stream_executor::AllocatorStats> GetAllocatorStats() override;

  tpu::TpuCoreLocationExternal GetCoreLocationExternal() const override;

  Status GetStatus(Stream* stream) override;

  std::unique_ptr<::stream_executor::internal::StreamInterface>
  GetStreamImplementation() override;

  std::unique_ptr<::stream_executor::internal::TimerInterface>
  GetTimerImplementation() override;

  std::unique_ptr<::stream_executor::internal::EventInterface>
  CreateEventImplementation() override;

  bool HostCallback(Stream* stream, std::function<Status()> callback) override;

  bool Memcpy(Stream* stream, void* host_dst,
              const ::stream_executor::DeviceMemoryBase& device_src,
              uint64_t size) override;

  bool Memcpy(Stream* stream, ::stream_executor::DeviceMemoryBase* device_dst,
              const void* host_src, uint64_t size) override;

  bool MemcpyDeviceToDevice(Stream* stream,
                            ::stream_executor::DeviceMemoryBase* gpu_dst,
                            const ::stream_executor::DeviceMemoryBase& host_src,
                            uint64_t size) override;

  void SyncAndForgetFailedStreams();
  bool SynchronizeAllActivity() override;

  Status SynchronousMemcpy(::stream_executor::DeviceMemoryBase* device_dst,
                           const void* host_src, uint64_t size) override;
  Status SynchronousMemcpy(
      void* host_dst, const ::stream_executor::DeviceMemoryBase& device_src,
      uint64_t size) override;
  Status SynchronousMemcpyDeviceToDevice(
      ::stream_executor::DeviceMemoryBase* device_dst,
      const ::stream_executor::DeviceMemoryBase& device_src,
      uint64_t size) override;

  int PlatformDeviceCount() override;

  Event::Status PollForEventStatus(Event* event) override;
  Status RecordEvent(Stream* stream, ::stream_executor::Event* event) override;
  Status WaitForEvent(Stream* stream, ::stream_executor::Event* event) override;

  bool StartTimer(Stream* stream, ::stream_executor::Timer* timer) override;
  bool StopTimer(Stream* stream, ::stream_executor::Timer* timer) override;

  Status WaitForInfeedReady(int32_t infeed_queue_index);

  Status WaitForOutfeedReady(int32_t outfeed_queue_index);

  Status UnloadAllPrograms() override;

  Status EnqueueCompactionOnStreamForHbm(Stream* compaction_stream) override;

  const ::tensorflow::tpu::TpuPlatformInterface& platform() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_1(mht_1_v, 335, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "platform");

    return *platform_;
  }

  ::tensorflow::tpu::TpuPlatformInterface& platform() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_2(mht_2_v, 342, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "platform");

    return *platform_;
  }

  // TODO(henrytan): convert this to override once the base interface is changed
  // to TpuExecutorInterface.
  StatusOr<std::unique_ptr<
      tensorflow::tpu::TpuExecutorInterface::TemporaryDeviceMemory>>
  CreateTemporaryDeviceMemory(int64_t memory_space, int64_t byte_offset,
                              int64_t size) override {
    LOG(FATAL) << "Unimplemented.";
  }

  // -- Unimplemented (stubbed out) methods.
  std::unique_ptr<stream_executor::internal::KernelInterface>
  CreateKernelImplementation() override {
    LOG(FATAL) << "Not yet implemented";
  }

  void* GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset,
                     uint64_t size) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_3(mht_3_v, 365, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "GetSubBuffer");

    LOG(FATAL) << "not yet implemented";
  }
  Status MemZero(Stream* stream, DeviceMemoryBase* location,
                 uint64_t size) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_4(mht_4_v, 372, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "MemZero");

    LOG(FATAL) << "not yet implemented";
  }
  Status Memset32(Stream* stream, DeviceMemoryBase* location, uint32 pattern,
                  uint64_t size) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_5(mht_5_v, 379, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "Memset32");

    LOG(FATAL) << "not yet implemented";
  }
  Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_6(mht_6_v, 385, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "EnablePeerAccessTo");

    LOG(FATAL) << "not yet implemented";
  }
  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_7(mht_7_v, 391, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "CanEnablePeerAccessTo");

    LOG(FATAL) << "not yet implemented";
  }

  void* HostMemoryAllocate(uint64_t size) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_8(mht_8_v, 398, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "HostMemoryAllocate");

    LOG(FATAL) << "not yet implemented";
  }
  void HostMemoryDeallocate(void* mem) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_9(mht_9_v, 404, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "HostMemoryDeallocate");

    LOG(FATAL) << "not yet implemented";
  }
  bool HostMemoryRegister(void* mem, uint64_t size) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_10(mht_10_v, 410, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "HostMemoryRegister");

    LOG(FATAL) << "not yet implemented";
  }
  bool HostMemoryUnregister(void* mem) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_11(mht_11_v, 416, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "HostMemoryUnregister");

    LOG(FATAL) << "not yet implemented";
  }
  Status SynchronousMemZero(DeviceMemoryBase* location,
                            uint64_t size) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_12(mht_12_v, 423, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "SynchronousMemZero");

    LOG(FATAL) << "not yet implemented";
  }
  Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                           uint64_t size) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_13(mht_13_v, 430, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "SynchronousMemSet");

    LOG(FATAL) << "not yet implemented";
  }

  SE_StreamExecutor* se_executor() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_14(mht_14_v, 437, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "se_executor");
 return executor_; }

 private:
  TpuPlatform& tpu_platform() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_15(mht_15_v, 443, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "tpu_platform");

    return *(tensorflow::down_cast<TpuPlatform*>(platform_));
  }

  TpuPlatform::StreamMap& stream_map() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_16(mht_16_v, 450, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "stream_map");

    return *(tpu_platform().stream_map());
  }

  SE_Stream* get_stream(StreamInterface* ptr) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executorDTh mht_17(mht_17_v, 457, "", "./tensorflow/stream_executor/tpu/tpu_executor.h", "get_stream");

    tensorflow::mutex_lock m(tpu_platform().mutex());
    return stream_map()[ptr];
  }

  TimerMap timer_map_;
  tensorflow::tpu::TpuPlatformInterface* platform_;
  SE_StreamExecutor* executor_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_H_
