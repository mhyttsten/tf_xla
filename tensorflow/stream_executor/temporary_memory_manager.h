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

// The temporary-memory-manager is a helper class for a Stream to keep track of
// temporary allocations. These allocations defer their deallocation to the next
// Stream::BlockHostUntilDone call for efficiency purposes (as deallocation
// itself generally forces synchronization to occur).

#ifndef TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_MEMORY_MANAGER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_MEMORY_MANAGER_H_
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
class MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTh() {
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


#include <map>
#include <memory>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"

namespace stream_executor {
namespace internal {

// Record used inside the TemporaryMemoryManager as metadata for a given device
// memory region.
struct TemporaryMemoryRecord {
  // What "generation" this record was allocated in.
  //
  // Currently the generation counter is bumped for every allocation, but this
  // could be made coarser if necessary.
  uint64_t allocation_generation;

  // Notes whether the temporary memory has been marked as finalized, such that
  // we can release the DeviceMemory associated with this record at
  // synchronization time.
  bool finalized;
};

// Manages temporary memories associated with a stream -- keeps records of
// outstanding temporaries and their state, and can deallocate them
// appropriately at points in the Stream lifecycle (e.g. BlockHostUntilDone,
// destruction).
class TemporaryMemoryManager {
 public:
  explicit TemporaryMemoryManager(Stream* stream) : stream_(stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTh mht_0(mht_0_v, 227, "", "./tensorflow/stream_executor/temporary_memory_manager.h", "TemporaryMemoryManager");
}

  // Allocates a temporary array that is then managed by this object.
  template <typename T>
  port::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>> AllocateArray(
      uint64_t element_count);

  // Forces deallocation of all managed temporary memory regions.
  //
  // Called, for example, when the Stream owning this temporary memory manager
  // is destroyed.
  //
  // Note: These calls to Deallocate will likely force synchronization.
  void ForceDeallocateAll();

  // Marks the given memory region as finalized.
  //
  // If must_exist is set, this will check-fail if the temporary memory record
  // is not found.
  void MarkFinalized(const DeviceMemoryBase& device_memory, uint64_t generation,
                     bool must_exist);

  // Deallocates temporary memories that have been finalized.
  //
  // Note: These calls to Deallocate will likely force synchronization, so it is
  // meant to be called before a "BlockHostUntilDone" is about to be performed.
  void DeallocateFinalizedTemporaries();

  // Returns whether the provided device_memory is finalized.
  //
  // In the vacuous case where the device memory doesn't appear in the temporary
  // memory records, it is either not a temporary at all, or has already been
  // deallocated, and thus returns true.
  bool IsFinalized(const DeviceMemoryBase& device_memory,
                   uint64_t allocation_generation) const;

  // Returns whether the manager has a live allocation record for the given
  // device memory pointer with the given generation counter.
  //
  // Note: this is a polling call -- there is no guarantee that the region is
  // still allocated once the call has completed.
  bool HasAllocated(const DeviceMemoryBase& device_memory,
                    uint64_t generation) const;

 private:
  // Allocates an array without type parameterization, so that the
  // implementation can live in the source file. Without this base allocation
  // method, we incur a circular dependency between the StreamExecutor
  // definition and this class' definition.
  port::StatusOr<std::unique_ptr<TemporaryDeviceMemoryBase>> AllocateArrayBase(
      uint64_t element_count, uint64 element_size);

  // Mutex to guard temporary record state.
  mutable absl::Mutex mutex_;

  // Mapping from device memory to the current (live) temporary memory record.
  //
  // If a device memory is not in this mapping, it is not a temporary currently
  // allocated and owned by this temporary memory manager.
  std::map<DeviceMemoryBase, TemporaryMemoryRecord> records_
      TF_GUARDED_BY(mutex_);

  // Allocation generation -- we bump this counter to distinguish temporary
  // memory handles that have been deallocated and later reallocated at the same
  // device memory address.
  uint64_t generation_ TF_GUARDED_BY(mutex_);

  // The stream (parent object) for this temporary memory manager -- allocations
  // are performed through this stream handle.
  Stream* stream_;

  SE_DISALLOW_COPY_AND_ASSIGN(TemporaryMemoryManager);
};

////////////
// Inlines

template <typename T>
port::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>>
TemporaryMemoryManager::AllocateArray(uint64_t element_count) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTh mht_1(mht_1_v, 309, "", "./tensorflow/stream_executor/temporary_memory_manager.h", "TemporaryMemoryManager::AllocateArray");

  port::StatusOr<std::unique_ptr<TemporaryDeviceMemoryBase>> temporary_memory =
      AllocateArrayBase(element_count, sizeof(T));
  if (!temporary_memory.ok()) {
    return temporary_memory.status();
  }

  return std::unique_ptr<TemporaryDeviceMemory<T>>(
      reinterpret_cast<TemporaryDeviceMemory<T>*>(
          temporary_memory.ConsumeValueOrDie().release()));
}

}  // namespace internal
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_MEMORY_MANAGER_H_
