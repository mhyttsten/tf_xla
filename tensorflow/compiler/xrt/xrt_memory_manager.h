/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XRT_XRT_MEMORY_MANAGER_H_
#define TENSORFLOW_COMPILER_XRT_XRT_MEMORY_MANAGER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTh() {
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


#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt_refptr.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

// The XRTMemoryManager manages all the XRT allocations. It is a ResourceBase
// object which leaves within the ResourceMgr. This is only one XRT memory
// manager object within the ResourceMgr container.
class XRTMemoryManager : public ResourceBase {
  // The DeviceContext class, defined and implemented locally inside the
  // xrt_memory_manager.cc file, holds, for each device, all the information
  // related to the XRT memory management for such device.
  class DeviceContext;

 public:
  // A working set is a set of tuple allocations which are the input of a given
  // operation, and as such they must be pinned on the device memory. The tuple
  // allocations added to the WorkingSet will be unpinned at object destruction.
  class WorkingSet {
   public:
    explicit WorkingSet(RefPtr<XRTMemoryManager> memory_manager);

    ~WorkingSet();

    // Looks up the tuple handle within the memory manager, and pins it to the
    // device (if not already pinned).
    Status LookupAndPin(xla::Backend* backend, int64_t handle,
                        se::DeviceMemoryAllocator* allocator);

    const std::vector<RefPtr<XRTTupleAllocation>>& PinnedTuples() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTh mht_0(mht_0_v, 231, "", "./tensorflow/compiler/xrt/xrt_memory_manager.h", "PinnedTuples");

      return pinned_tuples_;
    }

    const RefPtr<XRTMemoryManager>& MemoryManager() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTh mht_1(mht_1_v, 238, "", "./tensorflow/compiler/xrt/xrt_memory_manager.h", "MemoryManager");

      return memory_manager_;
    }

   private:
    RefPtr<XRTMemoryManager> memory_manager_;
    std::vector<RefPtr<XRTTupleAllocation>> pinned_tuples_;
  };

  // Retrieves the XRTMemoryManager singleton stored within the ResourceMgr.
  static RefPtr<XRTMemoryManager> Get(ResourceMgr* rm);

  // Registers an XRTTupleAllocation and returns the unique handle identifying
  // it.
  int64_t Register(RefPtr<XRTTupleAllocation> tuple);

  // Looks up an handle returned by the Register() API and returns the
  // XRTTupleAllocation behind it.
  xla::StatusOr<RefPtr<XRTTupleAllocation>> Lookup(int64_t handle);

  Status Lookup(int64_t handle, RefPtr<XRTTupleAllocation>* tuple) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTh mht_2(mht_2_v, 261, "", "./tensorflow/compiler/xrt/xrt_memory_manager.h", "Lookup");

    TF_ASSIGN_OR_RETURN(*tuple, Lookup(handle));
    return Status::OK();
  }

  // Releases an handle by dropping the references count held on the
  // XRTTupleAllocation by the XRTMemoryManager. Existing XRTTupleAllocation
  // references will continue to be valid.
  Status Release(int64_t handle);

  // Tries to compact all the memory allocations on a given device. This is
  // currently done by swapping-out all the existing allocation, and swapping
  // them back in.
  Status CompactAllocations(xla::Backend* backend, int device_ordinal,
                            se::DeviceMemoryAllocator* allocator);

  // Releases all the device memory allocated by XRT within the resource
  // manager.
  void ReleaseAllAllocations();

  // Tries to allocate size bytes of device memory from the device_ordinal
  // device. Might attempt to free some unpinned device memory, if the underline
  // allocator call fails, and try the allocation again.
  xla::StatusOr<se::OwningDeviceMemory> Allocate(
      xla::Backend* backend, int device_ordinal, size_t size,
      se::DeviceMemoryAllocator* allocator);

  // Runs the specified function and handling the error::RESOURCE_EXHAUSTED
  // status code coming out of it. In such cases, we run different memory
  // freeing operations trying to make runfn succeed. The requested_free_size
  // argument represents an hint of the requested memory size which would make
  // runfn succeed.
  template <typename T>
  xla::StatusOr<T> Run(const std::function<xla::StatusOr<T>()>& runfn,
                       xla::Backend* backend, int device_ordinal,
                       size_t requested_free_size,
                       se::DeviceMemoryAllocator* allocator);

  string DebugString() const override;

  // Returns the invalid key value, which will be never generated by the
  // Intern() API.
  static int64_t InvalidKey() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTh mht_3(mht_3_v, 306, "", "./tensorflow/compiler/xrt/xrt_memory_manager.h", "InvalidKey");
 return 0; }

 private:
  // Structure used to track the progress of a try-to-free operation. It is
  // initialized and the passed to the TryFreeMemoryStep() API.
  struct MemoryReclaimContext {
    MemoryReclaimContext(xla::Backend* backend, int device_ordinal,
                         size_t requested_free_size,
                         se::DeviceMemoryAllocator* specific_allocator)
        : backend(backend),
          device_ordinal(device_ordinal),
          requested_free_size(requested_free_size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTh mht_4(mht_4_v, 320, "", "./tensorflow/compiler/xrt/xrt_memory_manager.h", "MemoryReclaimContext");

      allocator = specific_allocator;
    }

    xla::Backend* const backend = nullptr;
    se::DeviceMemoryAllocator* allocator = nullptr;
    const int device_ordinal = 0;
    const size_t requested_free_size = 0;
    size_t free_size = 0;
    bool done_freeing = false;
    bool done_compacting = false;
  };

  DeviceContext* GetDeviceContext(int device_ordinal, bool create_if_missing);

  // Called multiple times while trying to make a memory consuming function call
  // to fit. Performs progressively more expensive memory reduction operations,
  // until returning error::RESOURCE_EXHAUSTED when no further reductions are
  // possible.
  Status TryFreeMemoryStep(MemoryReclaimContext* mrctx, const Status& status);

  mutex lock_;
  std::vector<std::unique_ptr<DeviceContext>> device_contexts_;
};

template <typename T>
xla::StatusOr<T> XRTMemoryManager::Run(
    const std::function<xla::StatusOr<T>()>& runfn, xla::Backend* backend,
    int device_ordinal, size_t requested_free_size,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_memory_managerDTh mht_5(mht_5_v, 352, "", "./tensorflow/compiler/xrt/xrt_memory_manager.h", "XRTMemoryManager::Run");

  MemoryReclaimContext mrctx(backend, device_ordinal, requested_free_size,
                             allocator);
  while (true) {
    // We assume that runfn is a relatively fast-fail function compared to the
    // operations required to free up the required memory. Here we call into the
    // TryFreeMemoryStep() API multiple times, which will run progressively more
    // expensive operations.
    auto result_or = runfn();
    if (result_or.status().code() != error::RESOURCE_EXHAUSTED) {
      return result_or;
    }
    TF_RETURN_IF_ERROR(TryFreeMemoryStep(&mrctx, result_or.status()));
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_MEMORY_MANAGER_H_
