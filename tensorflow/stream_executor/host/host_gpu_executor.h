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

// Declares the HostExecutor class, which is a CPU-only implementation of
// the StreamExecutor interface. For now, this is used for testing and to
// examine the performance of host-based StreamExecutor code.
#ifndef TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_GPU_EXECUTOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_GPU_EXECUTOR_H_
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
class MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh() {
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


#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace host {

// An implementation of StreamExecutor that does no communication or interaction
// with a device, but DOES perform memory operations backed by the host.
// Plugin routines (RNG, BLAS) are also supported and functional.
// Kernel invocations will fail, but host callbacks may be enqueued on this
// executor and its associated stream, and should follow standard ordering
// semantics.
//
// This is useful for evaluating the performance of host-based or fallback
// routines executed under the context of a GPU executor.
// See stream_executor.h for description of the below operations.
class HostExecutor : public internal::StreamExecutorInterface {
 public:
  explicit HostExecutor(const PluginConfig& plugin_config);
  ~HostExecutor() override;

  // The stack size used for host streams can be set via
  // device_options.non_portable_tags["host_stack_size"].
  port::Status Init(int device_ordinal, DeviceOptions device_options) override;

  port::Status GetKernel(const MultiKernelLoaderSpec& spec,
                         KernelBase* kernel) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_0(mht_0_v, 223, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "GetKernel");

    return port::UnimplementedError("Not Implemented");
  }
  port::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                      const BlockDim& block_dims, const KernelBase& kernel,
                      const KernelArgsArrayBase& args) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_1(mht_1_v, 231, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "Launch");

    return port::UnimplementedError("Not Implemented");
  }

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;
  void* GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset_bytes,
                     uint64_t size_bytes) override;
  void Deallocate(DeviceMemoryBase* mem) override;

  void* HostMemoryAllocate(uint64_t size) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_2(mht_2_v, 243, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "HostMemoryAllocate");
 return new char[size]; }
  void HostMemoryDeallocate(void* mem) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_3(mht_3_v, 247, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "HostMemoryDeallocate");

    delete[] static_cast<char*>(mem);
  }
  bool HostMemoryRegister(void* mem, uint64_t size) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_4(mht_4_v, 253, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "HostMemoryRegister");
 return true; }
  bool HostMemoryUnregister(void* mem) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_5(mht_5_v, 257, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "HostMemoryUnregister");
 return true; }

  bool Memcpy(Stream* stream, void* host_dst, const DeviceMemoryBase& gpu_src,
              uint64_t size) override;
  bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst, const void* host_src,
              uint64_t size) override;
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                            const DeviceMemoryBase& gpu_src,
                            uint64_t size) override;

  port::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64_t size) override;
  port::Status Memset(Stream* stream, DeviceMemoryBase* location, uint8 pattern,
                      uint64_t size) override;
  port::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32 pattern, uint64_t size) override;

  // No "synchronize all activity" implemented for this platform at the moment.
  bool SynchronizeAllActivity() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_6(mht_6_v, 278, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "SynchronizeAllActivity");
 return true; }
  port::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override;

  port::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                 uint64_t size) override;

  port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) override;
  port::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override;
  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase* gpu_dst,
                                               const DeviceMemoryBase& gpu_src,
                                               uint64_t size) override;

  bool HostCallback(Stream* stream,
                    std::function<port::Status()> callback) override;

  port::Status AllocateEvent(Event* event) override;
  port::Status DeallocateEvent(Event* event) override;
  port::Status RecordEvent(Stream* stream, Event* event) override;
  port::Status WaitForEvent(Stream* stream, Event* event) override;
  Event::Status PollForEventStatus(Event* event) override;

  bool AllocateStream(Stream* stream) override;
  void DeallocateStream(Stream* stream) override;
  bool CreateStreamDependency(Stream* dependent, Stream* other) override;

  // No special initialization is necessary for host timers.
  bool AllocateTimer(Timer* timer) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_7(mht_7_v, 311, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "AllocateTimer");
 return true; }

  void DeallocateTimer(Timer* timer) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_8(mht_8_v, 316, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "DeallocateTimer");
}

  bool StartTimer(Stream* stream, Timer* timer) override;

  bool StopTimer(Stream* stream, Timer* timer) override;

  port::Status BlockHostUntilDone(Stream* stream) override;

  int PlatformDeviceCount() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_9(mht_9_v, 327, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "PlatformDeviceCount");
 return 1; }

  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const override;

  port::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return CreateDeviceDescription(0);
  }

  static port::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  port::Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_10(mht_10_v, 342, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "EnablePeerAccessTo");

    return port::Status::OK();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_11(mht_11_v, 349, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "CanEnablePeerAccessTo");

    return true;
  }

  bool SupportsBlas() const override;
  blas::BlasSupport* CreateBlas() override;

  bool SupportsDnn() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_12(mht_12_v, 359, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "SupportsDnn");
 return false; }
  dnn::DnnSupport* CreateDnn() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_13(mht_13_v, 363, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "CreateDnn");
 return nullptr; }

  bool SupportsFft() const override;
  fft::FftSupport* CreateFft() override;

  bool SupportsRng() const override;
  rng::RngSupport* CreateRng() override;

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override;

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override;

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override {
    return std::unique_ptr<internal::TimerInterface>(new HostTimer());
  }

  void* GpuContextHack() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_gpu_executorDTh mht_14(mht_14_v, 388, "", "./tensorflow/stream_executor/host/host_gpu_executor.h", "GpuContextHack");
 return nullptr; }

 private:
  const PluginConfig plugin_config_;
  // Size of thread stacks for streams in bytes. '0' means "the default size".
  size_t thread_stack_size_in_bytes_ = 0;
};

}  // namespace host
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_HOST_HOST_GPU_EXECUTOR_H_
