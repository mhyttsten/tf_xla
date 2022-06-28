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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TRACKING_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TRACKING_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTh() {
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


#include <unordered_map>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// TrackingAllocator is a wrapper for an Allocator. It keeps a running
// count of the number of bytes allocated through the wrapper. It is
// used by the Executor to "charge" allocations to particular Op
// executions. Each Op gets a separate TrackingAllocator wrapper
// around the underlying allocator.
//
// The implementation assumes the invariant that all calls to
// AllocateRaw by an Op (or work items spawned by the Op) will occur
// before the Op's Compute method returns. Thus the high watermark is
// established once Compute returns.
//
// DeallocateRaw can be called long after the Op has finished,
// e.g. when an output tensor is deallocated, and the wrapper cannot
// be deleted until the last of these calls has occurred.  The
// TrackingAllocator keeps track of outstanding calls using a
// reference count, and deletes itself once the last call has been
// received and the high watermark has been retrieved.
struct AllocRecord {
  AllocRecord(int64_t a_btyes, int64_t a_micros)
      : alloc_bytes(a_btyes), alloc_micros(a_micros) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTh mht_0(mht_0_v, 216, "", "./tensorflow/core/framework/tracking_allocator.h", "AllocRecord");
}
  AllocRecord() : AllocRecord(0, 0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTh mht_1(mht_1_v, 220, "", "./tensorflow/core/framework/tracking_allocator.h", "AllocRecord");
}

  int64_t alloc_bytes;
  int64_t alloc_micros;
};

class TrackingAllocator : public Allocator {
 public:
  explicit TrackingAllocator(Allocator* allocator, bool track_ids);
  std::string Name() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTh mht_2(mht_2_v, 232, "", "./tensorflow/core/framework/tracking_allocator.h", "Name");
 return allocator_->Name(); }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTh mht_3(mht_3_v, 236, "", "./tensorflow/core/framework/tracking_allocator.h", "AllocateRaw");

    return AllocateRaw(alignment, num_bytes, AllocationAttributes());
  }
  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;
  void DeallocateRaw(void* ptr) override;
  bool TracksAllocationSizes() const override;
  size_t RequestedSize(const void* ptr) const override;
  size_t AllocatedSize(const void* ptr) const override;
  int64_t AllocationId(const void* ptr) const override;
  absl::optional<AllocatorStats> GetStats() override;
  bool ClearStats() override;

  AllocatorMemoryType GetMemoryType() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTh mht_4(mht_4_v, 252, "", "./tensorflow/core/framework/tracking_allocator.h", "GetMemoryType");

    return allocator_->GetMemoryType();
  }

  // If the underlying allocator tracks allocation sizes, this returns
  // a tuple where the first value is the total number of bytes
  // allocated through this wrapper, the second value is the high
  // watermark of bytes allocated through this wrapper and the third value is
  // the allocated bytes through this wrapper that are still alive. If the
  // underlying allocator does not track allocation sizes the first
  // value is the total number of bytes requested through this wrapper
  // and the second and the third are 0.
  //
  std::tuple<size_t, size_t, size_t> GetSizes();
  // After GetRecordsAndUnRef is called, the only further calls allowed
  // on this wrapper are calls to DeallocateRaw with pointers that
  // were allocated by this wrapper and have not yet been
  // deallocated. After this call completes and all allocated pointers
  // have been deallocated the wrapper will delete itself.
  gtl::InlinedVector<AllocRecord, 4> GetRecordsAndUnRef();
  // Returns a copy of allocation records collected so far.
  gtl::InlinedVector<AllocRecord, 4> GetCurrentRecords();

 protected:
  ~TrackingAllocator() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTh mht_5(mht_5_v, 279, "", "./tensorflow/core/framework/tracking_allocator.h", "~TrackingAllocator");
}

 private:
  bool UnRef() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Allocator* allocator_;  // not owned.
  mutable mutex mu_;
  // the number of calls to AllocateRaw that have not yet been matched
  // by a corresponding call to DeAllocateRaw, plus 1 if the Executor
  // has not yet read out the high watermark.
  int ref_ TF_GUARDED_BY(mu_);
  // the current number of outstanding bytes that have been allocated
  // by this wrapper, or 0 if the underlying allocator does not track
  // allocation sizes.
  size_t allocated_ TF_GUARDED_BY(mu_);
  // the maximum number of outstanding bytes that have been allocated
  // by this wrapper, or 0 if the underlying allocator does not track
  // allocation sizes.
  size_t high_watermark_ TF_GUARDED_BY(mu_);
  // the total number of bytes that have been allocated by this
  // wrapper if the underlying allocator tracks allocation sizes,
  // otherwise the total number of bytes that have been requested by
  // this allocator.
  size_t total_bytes_ TF_GUARDED_BY(mu_);

  gtl::InlinedVector<AllocRecord, 4> allocations_ TF_GUARDED_BY(mu_);

  // Track allocations locally if requested in the constructor and the
  // underlying allocator doesn't already do it for us.
  const bool track_sizes_locally_;
  struct Chunk {
    size_t requested_size;
    size_t allocated_size;
    int64_t allocation_id;
  };
  std::unordered_map<const void*, Chunk> in_use_ TF_GUARDED_BY(mu_);
  int64_t next_allocation_id_ TF_GUARDED_BY(mu_);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TRACKING_ALLOCATOR_H_
