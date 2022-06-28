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

// Interfaces for platform-dependent implementations to satisfy. This are
// delegated to from the StreamExecutor in pointer-to-implementation style; i.e.
// the StreamExecutor is just a husk that delegates calls to the
// platform-specific objects which implement the interfaces defined here.

#ifndef TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
#define TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh() {
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


#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/stream_executor/allocator_stats.h"
#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/kernel_spec.h"
#include "tensorflow/stream_executor/launch_dim.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/module_spec.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/trace_listener.h"

namespace stream_executor {

class Stream;
class Timer;

// An opaque handle to a loaded module.
//
// An instance of this is returned from StreamExecutor::GetModule.
class ModuleHandle {
 public:
  /*implicit*/ ModuleHandle(void* id = nullptr) : id_(id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_0(mht_0_v, 228, "", "./tensorflow/stream_executor/stream_executor_internal.h", "ModuleHandle");
}

  // A ModuleHandle with id() == nullptr is an invalid module handle, akin to a
  // null pointer.
  void* id() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_1(mht_1_v, 235, "", "./tensorflow/stream_executor/stream_executor_internal.h", "id");
 return id_; }

  explicit operator bool() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_2(mht_2_v, 240, "", "./tensorflow/stream_executor/stream_executor_internal.h", "bool");
 return id() != nullptr; }

 private:
  void* id_;
};

namespace internal {

// Platform-dependent interface class for the generic Events interface, in
// the PIMPL style.
class EventInterface {
 public:
  EventInterface() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_3(mht_3_v, 255, "", "./tensorflow/stream_executor/stream_executor_internal.h", "EventInterface");
}
  virtual ~EventInterface() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_4(mht_4_v, 259, "", "./tensorflow/stream_executor/stream_executor_internal.h", "~EventInterface");
}

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(EventInterface);
};

// Pointer-to-implementation object type (i.e. the KernelBase class delegates to
// this interface) with virtual destruction. This class exists for the
// platform-dependent code to hang any kernel data/resource info/functionality
// off of.
class KernelInterface {
 public:
  // Default constructor for the abstract interface.
  KernelInterface() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_5(mht_5_v, 275, "", "./tensorflow/stream_executor/stream_executor_internal.h", "KernelInterface");
}

  // Default destructor for the abstract interface.
  virtual ~KernelInterface() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_6(mht_6_v, 281, "", "./tensorflow/stream_executor/stream_executor_internal.h", "~KernelInterface");
}

  // Returns the number of formal parameters that this kernel accepts.
  virtual unsigned Arity() const = 0;

  // Sets the preferred cache configuration.
  virtual void SetPreferredCacheConfig(KernelCacheConfig config) = 0;

  // Gets the preferred cache configuration.
  virtual KernelCacheConfig GetPreferredCacheConfig() const = 0;

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(KernelInterface);
};

// Pointer-to-implementation object type (i.e. the Stream class delegates to
// this interface) with virtual destruction. This class exists for the
// platform-dependent code to hang any kernel data/resource info/functionality
// off of.
class StreamInterface {
 public:
  // Default constructor for the abstract interface.
  StreamInterface() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_7(mht_7_v, 306, "", "./tensorflow/stream_executor/stream_executor_internal.h", "StreamInterface");
}

  // Default destructor for the abstract interface.
  virtual ~StreamInterface() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_8(mht_8_v, 312, "", "./tensorflow/stream_executor/stream_executor_internal.h", "~StreamInterface");
}

  // Returns the GPU stream associated with this platform's stream
  // implementation.
  //
  // WARNING: checks that the underlying platform is, in fact, CUDA or ROCm,
  // causing a fatal error if it is not. This hack is made available solely for
  // use from distbelief code, which temporarily has strong ties to CUDA or
  // ROCm as a platform.
  virtual void* GpuStreamHack() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_9(mht_9_v, 324, "", "./tensorflow/stream_executor/stream_executor_internal.h", "GpuStreamHack");
 return nullptr; }

  // See the above comment on GpuStreamHack -- this further breaks abstraction
  // for Eigen within distbelief, which has strong ties to CUDA or ROCm as a
  // platform, and a historical attachment to a programming model which takes a
  // stream-slot rather than a stream-value.
  virtual void** GpuStreamMemberHack() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_10(mht_10_v, 333, "", "./tensorflow/stream_executor/stream_executor_internal.h", "GpuStreamMemberHack");
 return nullptr; }

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(StreamInterface);
};

// Pointer-to-implementation object type (i.e. the Timer class delegates to
// this interface) with virtual destruction. This class exists for the
// platform-dependent code to hang any timer data/resource info/functionality
// off of.
class TimerInterface {
 public:
  // Default constructor for the abstract interface.
  TimerInterface() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_11(mht_11_v, 349, "", "./tensorflow/stream_executor/stream_executor_internal.h", "TimerInterface");
}

  // Default destructor for the abstract interface.
  virtual ~TimerInterface() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_12(mht_12_v, 355, "", "./tensorflow/stream_executor/stream_executor_internal.h", "~TimerInterface");
}

  // Returns the number of microseconds elapsed in a completed timer.
  virtual uint64_t Microseconds() const = 0;

  // Returns the number of nanoseconds elapsed in a completed timer.
  virtual uint64_t Nanoseconds() const = 0;

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(TimerInterface);
};

// Interface for the different StreamExecutor platforms (i.e. CUDA, OpenCL).
//
// Various platforms will provide an implementation that satisfy this interface.
class StreamExecutorInterface {
 public:
  // Default constructor for the abstract interface.
  StreamExecutorInterface() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_13(mht_13_v, 376, "", "./tensorflow/stream_executor/stream_executor_internal.h", "StreamExecutorInterface");
}

  // Default destructor for the abstract interface.
  virtual ~StreamExecutorInterface() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_14(mht_14_v, 382, "", "./tensorflow/stream_executor/stream_executor_internal.h", "~StreamExecutorInterface");
}

  // Returns the (transitively) wrapped executor if this executor is
  // wrapping another executor; otherwise, returns this.
  virtual StreamExecutorInterface* GetUnderlyingExecutor() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_15(mht_15_v, 389, "", "./tensorflow/stream_executor/stream_executor_internal.h", "GetUnderlyingExecutor");
 return this; }

  // See the StreamExecutor interface for comments on the same-named methods.
  virtual port::Status Init(int device_ordinal,
                            DeviceOptions device_options) = 0;

  virtual port::Status GetKernel(const MultiKernelLoaderSpec& spec,
                                 KernelBase* kernel) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_16(mht_16_v, 399, "", "./tensorflow/stream_executor/stream_executor_internal.h", "GetKernel");

    return port::UnimplementedError("Not Implemented");
  }
  virtual bool UnloadModule(ModuleHandle module_handle) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_17(mht_17_v, 405, "", "./tensorflow/stream_executor/stream_executor_internal.h", "UnloadModule");
 return false; }
  virtual port::Status LoadModule(const MultiModuleLoaderSpec& spec,
                                  ModuleHandle* module_handle) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_18(mht_18_v, 410, "", "./tensorflow/stream_executor/stream_executor_internal.h", "LoadModule");

    return port::UnimplementedError("Not Implemented");
  }
  virtual port::StatusOr<std::shared_ptr<DeviceMemoryBase>>
  CreateOrShareConstant(Stream* stream, const std::vector<uint8_t>& content) {
    return port::UnimplementedError("Not Implemented");
  }
  virtual port::Status Launch(Stream* stream, const ThreadDim& thread_dims,
                              const BlockDim& block_dims, const KernelBase& k,
                              const KernelArgsArrayBase& args) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_19(mht_19_v, 422, "", "./tensorflow/stream_executor/stream_executor_internal.h", "Launch");

    return port::UnimplementedError("Not Implemented");
  }

  // Releases any state associated with the kernel.
  virtual void UnloadKernel(const KernelBase* kernel) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_20(mht_20_v, 430, "", "./tensorflow/stream_executor/stream_executor_internal.h", "UnloadKernel");
}
  virtual DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) = 0;
  DeviceMemoryBase Allocate(uint64_t size) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_21(mht_21_v, 435, "", "./tensorflow/stream_executor/stream_executor_internal.h", "Allocate");

    return Allocate(size, /*memory_space=*/0);
  }
  virtual void* GetSubBuffer(DeviceMemoryBase* parent, uint64_t offset,
                             uint64_t size) = 0;
  virtual void Deallocate(DeviceMemoryBase* mem) = 0;
  // Allocates unified memory space of the given size, if supported.
  // See
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
  // for more details on unified memory.
  virtual void* UnifiedMemoryAllocate(uint64_t size) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_22(mht_22_v, 448, "", "./tensorflow/stream_executor/stream_executor_internal.h", "UnifiedMemoryAllocate");
 return nullptr; }

  // Deallocates unified memory space previously allocated with
  // UnifiedMemoryAllocate.
  virtual void UnifiedMemoryDeallocate(void* mem) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_23(mht_23_v, 455, "", "./tensorflow/stream_executor/stream_executor_internal.h", "UnifiedMemoryDeallocate");
}
  virtual void* HostMemoryAllocate(uint64_t size) = 0;
  virtual void HostMemoryDeallocate(void* mem) = 0;
  virtual bool HostMemoryRegister(void* mem, uint64_t size) = 0;
  virtual bool HostMemoryUnregister(void* mem) = 0;
  virtual bool SynchronizeAllActivity() = 0;
  virtual port::Status SynchronousMemZero(DeviceMemoryBase* location,
                                          uint64_t size) = 0;
  virtual port::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                         uint64_t size) = 0;
  virtual port::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                         const void* host_src,
                                         uint64_t size) = 0;
  virtual port::Status SynchronousMemcpy(void* host_dst,
                                         const DeviceMemoryBase& gpu_src,
                                         uint64_t size) = 0;
  virtual port::Status SynchronousMemcpyDeviceToDevice(
      DeviceMemoryBase* gpu_dst, const DeviceMemoryBase& gpu_src,
      uint64_t size) = 0;
  virtual port::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                               uint64_t size) = 0;
  virtual port::Status Memset(Stream* stream, DeviceMemoryBase* location,
                              uint8 pattern, uint64_t size) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_24(mht_24_v, 480, "", "./tensorflow/stream_executor/stream_executor_internal.h", "Memset");

    return port::InternalError("Not implemented");
  }
  virtual port::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                                uint32 pattern, uint64_t size) = 0;
  virtual bool Memcpy(Stream* stream, void* host_dst,
                      const DeviceMemoryBase& gpu_src, uint64_t size) = 0;
  virtual bool Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                      const void* host_src, uint64_t size) = 0;
  virtual bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* gpu_dst,
                                    const DeviceMemoryBase& gpu_src,
                                    uint64_t size) = 0;
  virtual bool HostCallback(Stream* stream, std::function<void()> callback);
  virtual bool HostCallback(Stream* stream,
                            std::function<port::Status()> callback) = 0;
  virtual port::Status AllocateEvent(Event* event) = 0;
  virtual port::Status DeallocateEvent(Event* event) = 0;
  virtual port::Status RecordEvent(Stream* stream, Event* event) = 0;
  virtual port::Status WaitForEvent(Stream* stream, Event* event) = 0;
  virtual Event::Status PollForEventStatus(Event* event) = 0;
  virtual bool AllocateStream(Stream* stream) = 0;
  virtual void DeallocateStream(Stream* stream) = 0;
  virtual bool CreateStreamDependency(Stream* dependent, Stream* other) = 0;
  virtual bool AllocateTimer(Timer* timer) = 0;
  virtual void DeallocateTimer(Timer* timer) = 0;
  virtual bool StartTimer(Stream* stream, Timer* timer) = 0;
  virtual bool StopTimer(Stream* stream, Timer* timer) = 0;
  virtual port::Status BlockHostUntilDone(Stream* stream) = 0;
  virtual port::Status GetStatus(Stream* stream) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_25(mht_25_v, 511, "", "./tensorflow/stream_executor/stream_executor_internal.h", "GetStatus");

    return port::Status(port::error::UNIMPLEMENTED,
                        "GetStatus is not supported on this executor.");
  }
  virtual int PlatformDeviceCount() = 0;
  virtual port::Status EnablePeerAccessTo(StreamExecutorInterface* other) = 0;
  virtual bool CanEnablePeerAccessTo(StreamExecutorInterface* other) = 0;

  virtual int64_t GetDeviceLoad() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_26(mht_26_v, 522, "", "./tensorflow/stream_executor/stream_executor_internal.h", "GetDeviceLoad");
 return -1; }

  virtual bool DeviceMemoryUsage(int64_t* free, int64_t* total) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_27(mht_27_v, 527, "", "./tensorflow/stream_executor/stream_executor_internal.h", "DeviceMemoryUsage");

    return false;
  }

  // Retrieves device pointer and size for a symbol. The device pointer is
  // stored at mem, and the size is stored at size. Either mem or bytes can be
  // null, however, both of them cannot be null at the same time. To use
  // constant memory in CUDA, GetSymbol has to be used. Returns true if symbol
  // is found.
  //
  // If ModuleHandle is set then we search for `symbol_name` only within the
  // module corresponding to `module_handle`.  Otherwise all loaded modules are
  // searched.
  virtual bool GetSymbol(const std::string& symbol_name,
                         ModuleHandle module_handle, void** mem,
                         size_t* bytes) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("symbol_name: \"" + symbol_name + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_28(mht_28_v, 546, "", "./tensorflow/stream_executor/stream_executor_internal.h", "GetSymbol");

    return false;
  }

  // Creates a new DeviceDescription object. Ownership is transferred to the
  // caller.
  virtual port::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription() const = 0;

  // Attempts to register the provided TraceListener with the device-specific
  // Executor implementation. When this is called, the PIMPL interface has
  // already taken ownership of the object and is managing the generic tracing
  // events. The device-specific implementation must determine if the passed
  // listener is of a type appropriate for it to trace during registration (and
  // before dispatching events to it).
  // Returns true if the listener was successfully registered, false otherwise.
  // Does not take ownership of listener.
  virtual bool RegisterTraceListener(TraceListener* listener) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_29(mht_29_v, 566, "", "./tensorflow/stream_executor/stream_executor_internal.h", "RegisterTraceListener");
 return false; }

  // Unregisters the specified listener from the device-specific Executor.
  // Returns true if the listener was successfully registered, false otherwise.
  virtual bool UnregisterTraceListener(TraceListener* listener) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_30(mht_30_v, 573, "", "./tensorflow/stream_executor/stream_executor_internal.h", "UnregisterTraceListener");

    return false;
  }

  // Returns whether this StreamExecutor has BLAS support for its underlying
  // platform.
  virtual bool SupportsBlas() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_31(mht_31_v, 582, "", "./tensorflow/stream_executor/stream_executor_internal.h", "SupportsBlas");
 return false; }

  // Creates a new BlasSupport object, ownership is transferred to the caller.
  // If SupportsBlas() is false, this will always return null.
  //
  // If SupportsBlas() is true, this may return null, for example, if the BLAS
  // initialization fails.
  virtual blas::BlasSupport* CreateBlas() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_32(mht_32_v, 592, "", "./tensorflow/stream_executor/stream_executor_internal.h", "CreateBlas");
 return nullptr; }

  // Returns whether this StreamExecutor has FFT support for its underlying
  // platform.
  virtual bool SupportsFft() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_33(mht_33_v, 599, "", "./tensorflow/stream_executor/stream_executor_internal.h", "SupportsFft");
 return false; }

  // Creates a new fft::FftSupport object, ownership is transferred to the
  // caller.
  // If SupportsFft() is false, this will always return null.
  //
  // If SupportsFft() is true, this may return null, for example, if the FFT
  // initialization fails.
  virtual fft::FftSupport* CreateFft() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_34(mht_34_v, 610, "", "./tensorflow/stream_executor/stream_executor_internal.h", "CreateFft");
 return nullptr; }

  // Returns whether this StreamExecutor has Random Number Generation support
  // for
  // its underlying platform.
  virtual bool SupportsRng() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_35(mht_35_v, 618, "", "./tensorflow/stream_executor/stream_executor_internal.h", "SupportsRng");
 return false; }

  // Returns whether this StreamExecutor has neural net support for its
  // underlying
  // platform.
  virtual bool SupportsDnn() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_36(mht_36_v, 626, "", "./tensorflow/stream_executor/stream_executor_internal.h", "SupportsDnn");
 return false; }

  // Creates a new RngSupport object, ownership is transferred to the caller.
  // If SupportsRng() is false, this will always return null.
  //
  // If SupportsRng() is true, this may return null, for example, if the RNG
  // initialization fails.
  virtual rng::RngSupport* CreateRng() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_37(mht_37_v, 636, "", "./tensorflow/stream_executor/stream_executor_internal.h", "CreateRng");
 return nullptr; }

  // Creates a new DnnSupport object, ownership is transferred to the caller.
  // If SupportsDnn() is false, this will always return null.
  //
  // If SupportsDnn() is true, this may return null, for example, if the DNN
  // initialization fails.
  virtual dnn::DnnSupport* CreateDnn() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_38(mht_38_v, 646, "", "./tensorflow/stream_executor/stream_executor_internal.h", "CreateDnn");
 return nullptr; }

  // Each call creates a new instance of the platform-specific implementation of
  // the corresponding interface type.
  virtual std::unique_ptr<EventInterface> CreateEventImplementation() = 0;
  virtual std::unique_ptr<KernelInterface> CreateKernelImplementation() = 0;
  virtual std::unique_ptr<StreamInterface> GetStreamImplementation() = 0;
  virtual std::unique_ptr<TimerInterface> GetTimerImplementation() = 0;

  // Returns the CUDA or ROCm context associated with this StreamExecutor
  // platform implementation.
  //
  // WARNING: checks that the underlying platform is, in fact, CUDA or ROCm,
  // causing a fatal error if it is not. This hack is made available solely for
  // use from distbelief code, which temporarily has strong ties to CUDA or ROCm
  // as a platform.
  virtual void* GpuContextHack() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_39(mht_39_v, 665, "", "./tensorflow/stream_executor/stream_executor_internal.h", "GpuContextHack");
 return nullptr; }

  // Return allocator statistics.
  virtual absl::optional<AllocatorStats> GetAllocatorStats() {
    return absl::nullopt;
  }

  // If implemented, clears the internal stats except for the `in_use` fields
  // and sets the `peak_bytes_in_use` to be equal to the `bytes_in_use`. Returns
  // true if implemented.
  //
  // REQUIRES: GetAllocatorStats is overridden.
  virtual bool ClearAllocatorStats() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_40(mht_40_v, 680, "", "./tensorflow/stream_executor/stream_executor_internal.h", "ClearAllocatorStats");
 return false; }

  // Clears the compilation cache from volatile memory. Returns OK if no
  // compilation cache exists or if clearing the compilation cache is
  // unsupported. Caches in non-volatile storage are unaffected.
  virtual port::Status FlushCompilationCache() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSstream_executorPSstream_executor_internalDTh mht_41(mht_41_v, 688, "", "./tensorflow/stream_executor/stream_executor_internal.h", "FlushCompilationCache");
 return port::Status::OK(); }

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(StreamExecutorInterface);
};

}  // namespace internal
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
