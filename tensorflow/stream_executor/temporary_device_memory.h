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

// Temporary memories are used to allocate scratch space required by an
// operation about to be enqueued onto a stream.
//
//    std::unique_ptr<TemporaryDeviceMemory<float>> temporary_memory =
//        stream.AllocateTemporaryArray<float>(1024).ConsumeValueOrDie();
//    // ... enqueue stuff onto the stream using the temporary memory ...
//    // Note that the memory is accessible via
//    // temporary_memory->device_memory() and similar.
//
//    // Finalize the temporary memory. The underlying device memory may
//    // be released any time after this program point, as another thread may
//    // call Stream::BlockHostUntilDone, causing synchronization. This
//    // finalization also happens automatically for the user if the unique_ptr
//    // goes out of scope.
//    temporary_memory.Finalize();
//
// WARNING: do NOT hold onto the device memory associated with temporary_memory
// after finalization. If temporary_memory->device_memory() is used after the
// temporary memory is finalized, it will cause a DCHECK failure.
//
// Note that standard usage takes advantage of the type-safe wrapper,
// TemporaryDeviceMemory<T>, defined below.
//
// Also see tests for executable sample usage.

#ifndef TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_
#define TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_
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
class MHTracer_DTPStensorflowPSstream_executorPStemporary_device_memoryDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPStemporary_device_memoryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStemporary_device_memoryDTh() {
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


#include "tensorflow/stream_executor/device_memory.h"

namespace stream_executor {

class Stream;
namespace internal {
class TemporaryMemoryManager;
}

// Untyped base class (analogous to a void*) for temporary device memory
// allocations associated with a stream.
class TemporaryDeviceMemoryBase {
 public:
  // Marks the temporary memory as finalized if it is not already marked as
  // such.
  ~TemporaryDeviceMemoryBase();

  // Precondition: !IsFinalized()
  DeviceMemoryBase* mutable_device_memory();

  // Precondition: !IsFinalized()
  const DeviceMemoryBase& device_memory() const;

  // "Finalizes" this temporary memory, making it acceptable to release at the
  // next stream synchronization point -- the device memory can be reclaimed at
  // any time after the temporary memory is marked as finalized (e.g. if a
  // separate thread is calls Stream::BlockHostUntilDone). This may only be
  // called once -- see the precondition below.
  //
  // Precondition: !IsFinalized()
  void Finalize();

  // Returns true iff the temporary memory is finalized (that is, the user is
  // done referring to the temporary device memory, and thus it can be released
  // at the next stream synchronization point).
  bool IsFinalized() const;

  // Returns true iff the temporary memory is still allocated.
  //
  // Note: this is a polling call, no guarantee is made that the temporary
  // memory is still allocated after the call has completed.
  bool IsAllocated() const;

 private:
  friend class internal::TemporaryMemoryManager;
  friend class TemporaryDeviceMemoryTest;

  // Note: construction DCHECKs that the memory is known-allocated in the
  // stream's temporary-allocation-manager.
  TemporaryDeviceMemoryBase(Stream* parent, DeviceMemoryBase device_memory,
                            uint64_t allocation_generation);

  // The device memory region that has allocated.
  DeviceMemoryBase device_memory_;

  // The generation counter value for the temporary memory record in the
  // temporary memory manager.
  uint64_t allocation_generation_;

  // The stream that this temporary memory was allocated for.
  Stream* parent_;
};

// Type-safe wrapper around the base type (which is analogous to a void*).
template <typename T>
class TemporaryDeviceMemory : public TemporaryDeviceMemoryBase {
 public:
  // Type-safe wrapper around TemporaryDeviceMemoryBase::mutable_device_memory.
  DeviceMemory<T>* mutable_device_memory() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_device_memoryDTh mht_0(mht_0_v, 281, "", "./tensorflow/stream_executor/temporary_device_memory.h", "mutable_device_memory");

    StaticSlicingAssertionDummy();
    return reinterpret_cast<DeviceMemory<T>*>(
        TemporaryDeviceMemoryBase::mutable_device_memory());
  }

  // Type-safe wrapper around TemporaryDeviceMemoryBase::device_memory.
  const DeviceMemory<T>& device_memory() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_device_memoryDTh mht_1(mht_1_v, 291, "", "./tensorflow/stream_executor/temporary_device_memory.h", "device_memory");

    StaticSlicingAssertionDummy();
    return reinterpret_cast<const DeviceMemory<T>&>(
        TemporaryDeviceMemoryBase::device_memory());
  }

 private:
  static void StaticSlicingAssertionDummy() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_device_memoryDTh mht_2(mht_2_v, 301, "", "./tensorflow/stream_executor/temporary_device_memory.h", "StaticSlicingAssertionDummy");

    static_assert(
        sizeof(TemporaryDeviceMemory) == sizeof(TemporaryDeviceMemoryBase),
        "derived class is simply a wrapper, no members may be added due to "
        "slicing");
  }
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_
