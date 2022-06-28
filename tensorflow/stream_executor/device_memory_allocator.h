/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DEVICE_MEMORY_ALLOCATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DEVICE_MEMORY_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh() {
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


#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {

class DeviceMemoryAllocator;

// Owning pointer for memory on a device.
//
// ScopedDeviceMemory is an owning pointer like std::unique_ptr, but it can
// point to memory that resides on a "device" (e.g. a GPU).  When a
// ScopedDeviceMemory goes out of scope, it frees the memory it owns.
//
// We say that an instance of ScopedDeviceMemory is "active" if it currently
// owns a (possibly empty) slice of memory on the device.  Moving,
// Release()'ing, Free()'ing, and other actions can deactive an active object.
template <typename ElemT>
class ScopedDeviceMemory {
 public:
  // Default construction initializes the internal state to nullptr.  This
  // mirrors the std::unique_ptr<> functionality, where default construction
  // produces a nullptr unique_ptr, which can be assigned later.
  ScopedDeviceMemory() : device_ordinal_(-1), allocator_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_0(mht_0_v, 219, "", "./tensorflow/stream_executor/device_memory_allocator.h", "ScopedDeviceMemory");
}

  // Construct a ScopedDeviceMemory from a custom allocator.
  //
  // Parameters:
  //  mem: Already-allocated device memory value for this scoped mechanism to
  //       deallocate. This memory must have been allocated by parent.
  //  device_ordinal: Device on which the memory was allocated.
  //  allocator: Allocator used to deallocate memory when this instance goes
  //             out of scope.
  ScopedDeviceMemory(DeviceMemoryBase mem, int device_ordinal,
                     DeviceMemoryAllocator *allocator)
      : wrapped_(mem), device_ordinal_(device_ordinal), allocator_(allocator) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_1(mht_1_v, 234, "", "./tensorflow/stream_executor/device_memory_allocator.h", "ScopedDeviceMemory");

    DCHECK_GE(device_ordinal_, 0);
  }

  // A helper constructor to generate a scoped device memory given an already
  // allocated memory and a stream executor.
  //
  // Precondition: memory was allocated by the stream executor `parent`.
  ScopedDeviceMemory(StreamExecutor *parent, DeviceMemoryBase value);

  // Constructor overload that places a literal array into device memory.
  //
  // Relies on the allocation function exposed by the stream executor `parent`,
  // which will be also used for deallocating the memory
  ScopedDeviceMemory(StreamExecutor *parent,
                     std::initializer_list<ElemT> values);

  // Moves ownership of the memory from other to the constructed
  // object.
  //
  // Postcondition: other == nullptr.
  ScopedDeviceMemory(ScopedDeviceMemory &&other)
      : wrapped_(other.Release()),
        device_ordinal_(other.device_ordinal_),
        allocator_(other.allocator_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_2(mht_2_v, 261, "", "./tensorflow/stream_executor/device_memory_allocator.h", "ScopedDeviceMemory");
}

  // Releases the memory that was provided in the constructor, through the
  // "parent" StreamExecutor.
  ~ScopedDeviceMemory() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_3(mht_3_v, 268, "", "./tensorflow/stream_executor/device_memory_allocator.h", "~ScopedDeviceMemory");
 TF_CHECK_OK(Free()); }

  // Moves ownership of the memory from other to this object.
  //
  // Postcondition: other == nullptr.
  ScopedDeviceMemory &operator=(ScopedDeviceMemory &&other) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_4(mht_4_v, 276, "", "./tensorflow/stream_executor/device_memory_allocator.h", "=");

    TF_CHECK_OK(Free());
    wrapped_ = other.Release();
    allocator_ = other.allocator_;
    device_ordinal_ = other.device_ordinal_;
    return *this;
  }

  // Returns the memory that backs this scoped allocation converted to
  // DeviceMemory<T> apparent type. This is useful for cases where the
  // DeviceMemory must be passed by const-ref, as the ScopedDeviceMemory doesn't
  // allow copying, for scoped-object-lifetime reasons.
  const DeviceMemory<ElemT> &cref() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_5(mht_5_v, 291, "", "./tensorflow/stream_executor/device_memory_allocator.h", "cref");
 return wrapped_; }

  // Returns a pointer to the DeviceMemory<T> apparent type for use in mutable
  // operations. The value returned should not be used outside the scope of this
  // ScopedDeviceMemory object's lifetime.
  DeviceMemory<ElemT> *ptr() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_6(mht_6_v, 299, "", "./tensorflow/stream_executor/device_memory_allocator.h", "ptr");
 return &wrapped_; }
  const DeviceMemory<ElemT> *ptr() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_7(mht_7_v, 303, "", "./tensorflow/stream_executor/device_memory_allocator.h", "ptr");
 return &wrapped_; }

  // Smart-pointer-like operators for the wrapped DeviceMemory.
  // This reference must not be used outside the lifetime of this
  // ScopedDeviceMemory.
  const DeviceMemory<ElemT> &operator*() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_8(mht_8_v, 311, "", "./tensorflow/stream_executor/device_memory_allocator.h", "*");
 return cref(); }
  DeviceMemory<ElemT> *operator->() { return ptr(); }
  const DeviceMemory<ElemT> *operator->() const { return ptr(); }

  bool is_null() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_9(mht_9_v, 318, "", "./tensorflow/stream_executor/device_memory_allocator.h", "is_null");
 return wrapped_.is_null(); }
  bool operator==(std::nullptr_t other) const { return is_null(); }
  bool operator!=(std::nullptr_t other) const { return !is_null(); }

  // Analogous to std::unique_ptr::release, releases ownership of the held
  // memory and transfers it to the caller.
  //
  // Postcondition: *this == nullptr
  DeviceMemory<ElemT> Release() {
    DeviceMemory<ElemT> tmp = wrapped_;
    wrapped_ = DeviceMemory<ElemT>{};
    return tmp;
  }

  // The returned allocator is nonnull iff this object is active.
  DeviceMemoryAllocator *allocator() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_10(mht_10_v, 336, "", "./tensorflow/stream_executor/device_memory_allocator.h", "allocator");
 return allocator_; }

  int device_ordinal() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_11(mht_11_v, 341, "", "./tensorflow/stream_executor/device_memory_allocator.h", "device_ordinal");
 return device_ordinal_; }

  // Frees the existing memory, resets the wrapped memory to null.
  port::Status Free();

 private:
  DeviceMemory<ElemT> wrapped_;       // Value we wrap with scoped-release.
  int device_ordinal_;                // Negative one for inactive object.
  DeviceMemoryAllocator *allocator_;  // Null if this object is inactive.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedDeviceMemory);
};

// Type alias for compatibility with the previous managed memory implementation.
using OwningDeviceMemory = ScopedDeviceMemory<uint8>;

// Memory allocator interface for the device.
//
// Intended usage is through Allocate() functions which return an owning smart
// pointer.
class DeviceMemoryAllocator {
 public:
  // Parameter platform indicates which platform the allocator allocates memory
  // on. Must be non-null.
  explicit DeviceMemoryAllocator(const Platform* platform)
      : platform_(platform) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_12(mht_12_v, 369, "", "./tensorflow/stream_executor/device_memory_allocator.h", "DeviceMemoryAllocator");
}
  virtual ~DeviceMemoryAllocator() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_13(mht_13_v, 373, "", "./tensorflow/stream_executor/device_memory_allocator.h", "~DeviceMemoryAllocator");
}

  // Allocates memory on the device.
  //
  // If size > 0 and the returned StatusOr is OK, the wrapped OwningDeviceMemory
  // must not be null.  If size == 0, must return a null OwningDeviceMemory.
  //
  // 'retry_on_failure': If false, and the first attempt to allocate the memory
  // fails, the allocation should return immediately without retrying.  An
  // example use case is optional scratch spaces where a failure has only
  // performance impact.
  virtual port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal,
                                                      uint64_t size,
                                                      bool retry_on_failure,
                                                      int64_t memory_space) = 0;

  // Two-arg version of Allocate(), which sets retry-on-failure to true and
  // memory_space to default (0).
  //
  // (We don't simply use a default argument on the virtual Allocate function
  // because default args on virtual functions are disallowed by the Google
  // style guide.)
  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal,
                                              uint64_t size) {
    return Allocate(device_ordinal, size, /*retry_on_failure=*/true,
                    /*memory_space=*/0);
  }

  // Three-arg version of Allocate(), which sets memory_space to default (0).
  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                              bool retry_on_failure) {
    return Allocate(device_ordinal, size, retry_on_failure,
                    /*memory_space=*/0);
  }

  // Typed version of the allocation, returning typed memory.
  template <typename ElemT>
  port::StatusOr<ScopedDeviceMemory<ElemT>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure = true,
      int64_t memory_space = 0) {
    return Allocate(device_ordinal, size, retry_on_failure, memory_space);
  }

  // Must be a nop for null pointers. Should not be used.
  //
  // TODO(cheshire): Add deprecation notice.
  virtual port::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) = 0;

  // Return the platform that the allocator allocates memory on.
  const Platform* platform() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_14(mht_14_v, 425, "", "./tensorflow/stream_executor/device_memory_allocator.h", "platform");
 return platform_; }

  // Can we call Deallocate() as soon as a computation has been scheduled on
  // a stream, or do we have to wait for the computation to complete first?
  virtual bool AllowsAsynchronousDeallocation() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_15(mht_15_v, 432, "", "./tensorflow/stream_executor/device_memory_allocator.h", "AllowsAsynchronousDeallocation");
 return false; }

  // Returns a stream pointer on which it is always safe to access memory
  // allocated by this allocator. It is not necessary to use the returned stream
  // though, as clients may have additional information letting them safely use
  // a different stream.
  virtual port::StatusOr<Stream *> GetStream(int device_ordinal) = 0;

 protected:
  const Platform* platform_;
};

// Default memory allocator for a platform which uses
// StreamExecutor::Allocate/Deallocate.
class StreamExecutorMemoryAllocator : public DeviceMemoryAllocator {
 public:
  // Create an allocator supporting a single device, corresponding to the passed
  // executor.
  explicit StreamExecutorMemoryAllocator(StreamExecutor *executor);

  // Create an allocator supporting multiple stream executors.
  //
  // Precondition: all stream_executors have different device ordinals.
  StreamExecutorMemoryAllocator(
      const Platform *platform,
      absl::Span<StreamExecutor *const> stream_executors);

  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                              bool retry_on_failure,
                                              int64_t memory_space) override;

  // Pull in two-arg overload that sets retry_on_failure to true.
  using DeviceMemoryAllocator::Allocate;

  port::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override;

  bool AllowsAsynchronousDeallocation() const override;

  // Gets-or-creates a stream for a given `device_ordinal` from an appropriate
  // stream executor.
  port::StatusOr<Stream *> GetStream(int device_ordinal) override;

  // Gets the stream executor for given device ordinal.
  port::StatusOr<StreamExecutor *> GetStreamExecutor(int device_ordinal) const;

 private:
  // Available stream executors. Each stream executor has a different device
  // ordinal.
  std::vector<StreamExecutor *> stream_executors_;

  absl::Mutex mutex_;

  // Cache of streams for GetStream.
  std::map<int, Stream> streams_ TF_GUARDED_BY(mutex_);
};

template <typename ElemT>
port::Status ScopedDeviceMemory<ElemT>::Free() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memory_allocatorDTh mht_16(mht_16_v, 492, "", "./tensorflow/stream_executor/device_memory_allocator.h", "ScopedDeviceMemory<ElemT>::Free");

  if (!wrapped_.is_null()) {
    CHECK(allocator_ != nullptr) << "Owning pointer in inconsistent state";
    TF_RETURN_IF_ERROR(allocator_->Deallocate(device_ordinal_, wrapped_));
  }
  wrapped_ = DeviceMemory<ElemT>{};
  return port::Status::OK();
}

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DEVICE_MEMORY_ALLOCATOR_H_
