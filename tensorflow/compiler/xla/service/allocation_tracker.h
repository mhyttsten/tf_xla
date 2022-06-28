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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALLOCATION_TRACKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALLOCATION_TRACKER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTh() {
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
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Tracks allocations for the XLA service; allocations can be registered
// with shape/device/tag and resolved from a handle for later use.
class AllocationTracker {
 public:
  // The allocator is used for deallocating memory when allocations are
  // deregistered. All registered allocations must have the same platform as the
  // allocator.
  AllocationTracker(Backend* backend) : backend_(backend), next_handle_(1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTh mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/service/allocation_tracker.h", "AllocationTracker");
}

  // Registers a shaped buffer of device memory, and returns a corresponding
  // handle that can be used for talking to XLA clients. The given shaped buffer
  // will be treated as the buffer corresponding to the only replica.
  StatusOr<GlobalDataHandle> Register(ScopedShapedBuffer shaped_buffer,
                                      const std::string& tag);

  // Registers a vector of shaped buffers of device memory, one per replica, and
  // returns a corresponding handle that can be used for talking to XLA clients.
  StatusOr<GlobalDataHandle> RegisterReplicatedBuffers(
      std::vector<ScopedShapedBuffer> replicated_buffers,
      const std::string& tag);

  // Unregister the allocation for the given data handle.
  Status Unregister(const GlobalDataHandle& data);

  // Returns a vector of global data handles that point to the tuple elements.
  StatusOr<std::vector<GlobalDataHandle>> DeconstructTuple(
      const GlobalDataHandle& Data);

  // Resolve a handle from an XLA client to a vector of shaped buffers, one per
  // replica, or provide an error status to say whether any of those buffers
  // were not found (or found, but found deallocated).
  StatusOr<std::vector<const ShapedBuffer*>> Resolve(
      const GlobalDataHandle& data) const;

  // Resolves a handle from an XLA client and replica id to a shaped buffer, or
  // provide an error status to say whether it was not found (or found, but
  // found deallocated).
  StatusOr<const ShapedBuffer*> ResolveForReplica(const GlobalDataHandle& data,
                                                  int replica_id) const;

 private:
  // Data structure encapsulating single memory allocation on the device.
  struct Allocation {
    // The pointer to this allocation.
    se::OwningDeviceMemory device_memory;

    // This is the number of times this memory allocation is referred to by
    // registered data handles.
    int ref_count;
  };

  // Internal helper which resolves the given GlobalDataHandle to a
  // list of ScopedShapedBuffers.
  StatusOr<std::vector<const ShapedBuffer*>> ResolveInternal(
      const GlobalDataHandle& data) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Internal helper which registers a vector of shaped buffers, one per
  // replica.  ShapedBufferTy is either ScopedShapedBuffer or ShapedBuffer.  If
  // it's ShapedBuffer, all of the given buffers must already be tracked by this
  // object -- presumably this is a call from DeconstructTuple.
  template <typename ShapedBufferTy>
  StatusOr<GlobalDataHandle> RegisterInternal(
      std::vector<ShapedBufferTy> replicated_buffers, const std::string& tag)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Adds the given device address to the allocation tracker, or if it already
  // exists, then increment its reference count.
  void AddAllocationOrIncrementRefCount(se::DeviceMemoryBase device_memory,
                                        int device_ordinal)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Decrements the reference count of the given device memory. Then, if it is
  // zero, deallocate the memory.
  Status DecrementRefCount(se::DeviceMemoryBase device_memory,
                           int device_ordinal)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // A map from device memory opaque value to allocation. One such map is
  // maintained per device ordinal.
  using AllocationMap = absl::flat_hash_map<const void*, Allocation>;

  mutable absl::Mutex mutex_;

  // Backend to use with this tracker. The backend supplies the memory allocator
  // to use when deallocating memory.
  Backend* backend_;

  // The next handle to assign to an allocation, guarded by the same mutex as
  // the mapping as they'll be mutated at the same time.
  int64_t next_handle_ ABSL_GUARDED_BY(mutex_);

  // A map from device ordinal to AllocationMap.
  absl::flat_hash_map<int, AllocationMap> opaque_to_allocation_map_
      ABSL_GUARDED_BY(mutex_);

  // A map from data handle to a vector of shaped buffers that represent the
  // buffers for different replicas.
  //
  // The ShapedBuffers in this map's vectors need to be unique_ptrs, because our
  // public API returns pointers to them.  We expect the concrete class to be
  // ShapedBuffer and never ScopedShapedBuffer; deallocation of buffers is
  // handled by opaque_to_allocation_map_.
  //
  // The elements of the vectors need to be unique_ptrs because we return
  // pointers to them.  (In theory we could use std::list or something instead,
  // but we also want to be able to null out these elements.)
  //
  // The reason that the elements can't be unique_ptr<ScopedShapedBuffer>s is
  // the existence of DeconstructTuple().  This function allows us to create a
  // non-owning "view" into a tuple's sub-buffers.  The sub-buffers are then
  // free'd when both the view *and* the original tuple are Unregistered.  This
  // refcounting is managed in opaque_to_allocation_map_.
  absl::flat_hash_map<int64_t, std::vector<std::unique_ptr<ShapedBuffer>>>
      handle_to_shaped_buffers_ ABSL_GUARDED_BY(mutex_);

  AllocationTracker(const AllocationTracker&) = delete;
  AllocationTracker& operator=(const AllocationTracker&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALLOCATION_TRACKER_H_
