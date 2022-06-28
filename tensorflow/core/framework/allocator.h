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

#ifndef TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh() {
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


#include <stdlib.h>

#include <functional>
#include <limits>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Attributes for a single allocation call. Different calls to the same
// allocator could potentially have different allocation attributes.
struct AllocationAttributes {
  AllocationAttributes() = default;

  AllocationAttributes(bool retry_on_failure, bool allocation_will_be_logged,
                       std::function<uint64()>* freed_by_func)
      : retry_on_failure(retry_on_failure),
        allocation_will_be_logged(allocation_will_be_logged),
        freed_by_func(freed_by_func) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_0(mht_0_v, 213, "", "./tensorflow/core/framework/allocator.h", "AllocationAttributes");
}

  // If the first attempt to allocate the memory fails, the allocation should
  // wait and retry (with a timeout).
  //
  // This is usually set to true, but we may set it to false in cases where a
  // failure has only performance impact (e.g. optional scratch space
  // allocation).
  bool retry_on_failure = true;
  // If a Tensor is allocated without the following set to true, then
  // it is logged as an unknown allocation. During execution Tensors
  // should be allocated through the OpKernelContext which records
  // which Op is performing the allocation, and sets this flag to
  // true.
  bool allocation_will_be_logged = false;
  // EXPERIMENTAL: If provided, then evaluates to a timing count such that only
  // a memory chunk whose freed_at_count is at this value or earlier may be
  // returned.
  std::function<uint64()>* freed_by_func = nullptr;  // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(AllocationAttributes);
};

// Runtime statistics collected by an allocator. Exactly the same as
// stream_executor::AllocatorStats, but independently defined to preserve the
// mutual independence of StreamExecutor and TensorFlow.
struct AllocatorStats {
  int64_t num_allocs;          // Number of allocations.
  int64_t bytes_in_use;        // Number of bytes in use.
  int64_t peak_bytes_in_use;   // The peak bytes in use.
  int64_t largest_alloc_size;  // The largest single allocation seen.

  // The upper limit of bytes of user allocatable device memory, if such a limit
  // is known.
  absl::optional<int64_t> bytes_limit;

  // Stats for reserved memory usage.
  int64_t bytes_reserved;       // Number of bytes reserved.
  int64_t peak_bytes_reserved;  // The peak number of bytes reserved.
  // The upper limit on the number bytes of reservable memory,
  // if such a limit is known.
  absl::optional<int64_t> bytes_reservable_limit;

  int64_t largest_free_block_bytes;  // Largest free block's size in heap.

  AllocatorStats()
      : num_allocs(0),
        bytes_in_use(0),
        peak_bytes_in_use(0),
        largest_alloc_size(0),
        bytes_reserved(0),
        peak_bytes_reserved(0),
        largest_free_block_bytes(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_1(mht_1_v, 268, "", "./tensorflow/core/framework/allocator.h", "AllocatorStats");
}

  std::string DebugString() const;
};

// The type of the allocated memory.
enum class AllocatorMemoryType {
  kUnknown = 0,       // Memory type unknown.
  kDevice = 1,        // Memory on device.
  kHostPageable = 2,  // Memory on host and it is pagable.
  kHostPinned = 3,    // Memory on host and it is pinned.
};

// Allocator is an abstract interface for allocating and deallocating
// device memory.
class Allocator {
 public:
  // Align to 64 byte boundary.
  static constexpr size_t kAllocatorAlignment = 64;

  virtual ~Allocator();

  // Return a string identifying this allocator
  virtual std::string Name() = 0;

  // Return an uninitialized block of memory that is "num_bytes" bytes
  // in size.  The returned pointer is guaranteed to be aligned to a
  // multiple of "alignment" bytes.
  // REQUIRES: "alignment" is a power of 2.
  virtual void* AllocateRaw(size_t alignment, size_t num_bytes) = 0;

  // Return an uninitialized block of memory that is "num_bytes" bytes
  // in size with specified allocation attributes.  The returned pointer is
  // guaranteed to be aligned to a multiple of "alignment" bytes.
  // REQUIRES: "alignment" is a power of 2.
  virtual void* AllocateRaw(size_t alignment, size_t num_bytes,
                            const AllocationAttributes& allocation_attr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_2(mht_2_v, 307, "", "./tensorflow/core/framework/allocator.h", "AllocateRaw");

    // The default behavior is to use the implementation without any allocation
    // attributes.
    return AllocateRaw(alignment, num_bytes);
  }

  // Deallocate a block of memory pointer to by "ptr"
  // REQUIRES: "ptr" was previously returned by a call to AllocateRaw
  virtual void DeallocateRaw(void* ptr) = 0;

  // Returns true if this allocator tracks the sizes of allocations.
  // RequestedSize and AllocatedSize must be overridden if
  // TracksAllocationSizes is overridden to return true.
  virtual bool TracksAllocationSizes() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_3(mht_3_v, 323, "", "./tensorflow/core/framework/allocator.h", "TracksAllocationSizes");
 return false; }

  // Returns true if this allocator allocates an opaque handle rather than the
  // requested number of bytes.
  //
  // This method returns false for most allocators, but may be used by
  // special-case allocators that track tensor usage. If this method returns
  // true, AllocateRaw() should be invoked for all values of `num_bytes`,
  // including 0.
  //
  // NOTE: It is the caller's responsibility to track whether an allocated
  // object is a buffer or an opaque handle. In particular, when this method
  // returns `true`, users of this allocator must not run any constructors or
  // destructors for complex objects, since there is no backing store for the
  // tensor in which to place their outputs.
  virtual bool AllocatesOpaqueHandle() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_4(mht_4_v, 341, "", "./tensorflow/core/framework/allocator.h", "AllocatesOpaqueHandle");
 return false; }

  // Returns the user-requested size of the data allocated at
  // 'ptr'.  Note that the actual buffer allocated might be larger
  // than requested, but this function returns the size requested by
  // the user.
  //
  // REQUIRES: TracksAllocationSizes() is true.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  virtual size_t RequestedSize(const void* ptr) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_5(mht_5_v, 355, "", "./tensorflow/core/framework/allocator.h", "RequestedSize");

    CHECK(false) << "allocator doesn't track sizes";
    return size_t(0);
  }

  // Returns the allocated size of the buffer at 'ptr' if known,
  // otherwise returns RequestedSize(ptr). AllocatedSize(ptr) is
  // guaranteed to be >= RequestedSize(ptr).
  //
  // REQUIRES: TracksAllocationSizes() is true.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  virtual size_t AllocatedSize(const void* ptr) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_6(mht_6_v, 371, "", "./tensorflow/core/framework/allocator.h", "AllocatedSize");

    return RequestedSize(ptr);
  }

  // Returns either 0 or an identifier assigned to the buffer at 'ptr'
  // when the buffer was returned by AllocateRaw. If non-zero, the
  // identifier differs from every other ID assigned by this
  // allocator.
  //
  // REQUIRES: TracksAllocationSizes() is true.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  virtual int64_t AllocationId(const void* ptr) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_7(mht_7_v, 387, "", "./tensorflow/core/framework/allocator.h", "AllocationId");
 return 0; }

  // Returns the allocated size of the buffer at 'ptr' if known,
  // otherwise returns 0. This method can be called when
  // TracksAllocationSizes() is false, but can be extremely slow.
  //
  // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
  // allocated by this allocator.
  virtual size_t AllocatedSizeSlow(const void* ptr) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_8(mht_8_v, 398, "", "./tensorflow/core/framework/allocator.h", "AllocatedSizeSlow");

    if (TracksAllocationSizes()) {
      return AllocatedSize(ptr);
    }
    return 0;
  }

  // Fills in 'stats' with statistics collected by this allocator.
  virtual absl::optional<AllocatorStats> GetStats() { return absl::nullopt; }

  // If implemented, clears the internal stats except for the `in_use` fields
  // and sets the `peak_bytes_in_use` to be equal to the `bytes_in_use`. Returns
  //  true if implemented.
  //
  // REQUIRES: GetStats is overridden.
  virtual bool ClearStats() TF_MUST_USE_RESULT { return false; }

  virtual void SetSafeFrontier(uint64 count) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_9(mht_9_v, 418, "", "./tensorflow/core/framework/allocator.h", "SetSafeFrontier");
}

  // For allocator that are stream aware, allow to specify the compute
  // stream this allocator is used for. This can also trigger memory
  // preallocation.
  virtual void SetStreamAndPreallocateMemory(void* stream) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_10(mht_10_v, 426, "", "./tensorflow/core/framework/allocator.h", "SetStreamAndPreallocateMemory");
}

  // Returns the type of the memory allocated by this allocator.
  virtual AllocatorMemoryType GetMemoryType() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_11(mht_11_v, 432, "", "./tensorflow/core/framework/allocator.h", "GetMemoryType");

    return AllocatorMemoryType::kUnknown;
  }
};

// An implementation of Allocator that delegates all calls to another Allocator.
//
// Useful to clients who want to override part of the functionality of another
// allocator.
class AllocatorWrapper : public Allocator {
 public:
  explicit AllocatorWrapper(Allocator* wrapped) : wrapped_(wrapped) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_12(mht_12_v, 446, "", "./tensorflow/core/framework/allocator.h", "AllocatorWrapper");
}

  ~AllocatorWrapper() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_13(mht_13_v, 451, "", "./tensorflow/core/framework/allocator.h", "~AllocatorWrapper");
}

  // Returns the wrapped allocator to which all calls are delegated.
  Allocator* wrapped() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_14(mht_14_v, 457, "", "./tensorflow/core/framework/allocator.h", "wrapped");
 return wrapped_; }

  std::string Name() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_15(mht_15_v, 462, "", "./tensorflow/core/framework/allocator.h", "Name");
 return wrapped_->Name(); }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_16(mht_16_v, 467, "", "./tensorflow/core/framework/allocator.h", "AllocateRaw");

    return wrapped_->AllocateRaw(alignment, num_bytes);
  }

  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_17(mht_17_v, 475, "", "./tensorflow/core/framework/allocator.h", "AllocateRaw");

    return wrapped_->AllocateRaw(alignment, num_bytes, allocation_attr);
  }

  void DeallocateRaw(void* ptr) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_18(mht_18_v, 482, "", "./tensorflow/core/framework/allocator.h", "DeallocateRaw");
 wrapped_->DeallocateRaw(ptr); }

  bool TracksAllocationSizes() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_19(mht_19_v, 487, "", "./tensorflow/core/framework/allocator.h", "TracksAllocationSizes");

    return wrapped_->TracksAllocationSizes();
  }

  bool AllocatesOpaqueHandle() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_20(mht_20_v, 494, "", "./tensorflow/core/framework/allocator.h", "AllocatesOpaqueHandle");

    return wrapped_->AllocatesOpaqueHandle();
  }

  size_t RequestedSize(const void* ptr) const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_21(mht_21_v, 501, "", "./tensorflow/core/framework/allocator.h", "RequestedSize");

    return wrapped_->RequestedSize(ptr);
  }

  size_t AllocatedSize(const void* ptr) const override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_22(mht_22_v, 508, "", "./tensorflow/core/framework/allocator.h", "AllocatedSize");

    return wrapped_->AllocatedSize(ptr);
  }

  int64_t AllocationId(const void* ptr) const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_23(mht_23_v, 515, "", "./tensorflow/core/framework/allocator.h", "AllocationId");

    return wrapped_->AllocationId(ptr);
  }

  size_t AllocatedSizeSlow(const void* ptr) const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_24(mht_24_v, 522, "", "./tensorflow/core/framework/allocator.h", "AllocatedSizeSlow");

    return wrapped_->AllocatedSizeSlow(ptr);
  }

  AllocatorMemoryType GetMemoryType() const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_25(mht_25_v, 529, "", "./tensorflow/core/framework/allocator.h", "GetMemoryType");

    return wrapped_->GetMemoryType();
  }

 private:
  Allocator* const wrapped_;
};

// A tensorflow Op may need access to different kinds of memory that
// are not simply a function of the device to which the Op has been
// assigned.  For example, an Op executing on a GPU may still need
// to allocate CPU RAM for some purpose.  Internal to the tensorflow
// runtime we may choose to allocate CPU ram from special regions
// that have been prepared for higher performance in some use
// contexts, e.g. doing DMA with particular devices.  For these
// reasons, the Device interface does not expose just one memory
// Allocator, but instead provides an accessor that takes a
// specification of the desired memory attributes in order to select
// an Allocator.
//
// Example use:
//  // Allocator for ordinary device memory:
//  Allocator* a = allocator(AllocatorAttributes());
// ...
//  // Allocator for CPU RAM, regardless of where Op is executing:
//  AllocatorAttributes attr;
//  attr.set_on_host(true);
//  Allocator* a = allocator(attr);
struct AllocatorAttributes {
  void set_on_host(bool v) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_26(mht_26_v, 561, "", "./tensorflow/core/framework/allocator.h", "set_on_host");
 value |= (static_cast<int>(v)); }
  bool on_host() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_27(mht_27_v, 565, "", "./tensorflow/core/framework/allocator.h", "on_host");
 return value & 0x1; }
  void set_nic_compatible(bool v) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_28(mht_28_v, 569, "", "./tensorflow/core/framework/allocator.h", "set_nic_compatible");
 value |= (static_cast<int>(v) << 1); }
  bool nic_compatible() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_29(mht_29_v, 573, "", "./tensorflow/core/framework/allocator.h", "nic_compatible");
 return value & (0x1 << 1); }
  void set_gpu_compatible(bool v) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_30(mht_30_v, 577, "", "./tensorflow/core/framework/allocator.h", "set_gpu_compatible");
 value |= (static_cast<int>(v) << 2); }
  bool gpu_compatible() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_31(mht_31_v, 581, "", "./tensorflow/core/framework/allocator.h", "gpu_compatible");
 return value & (0x1 << 2); }
  void Merge(AllocatorAttributes other) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_32(mht_32_v, 585, "", "./tensorflow/core/framework/allocator.h", "Merge");

    value |= other.value;
    if (scope_id != other.scope_id) {
      CHECK(scope_id == 0 || other.scope_id == 0)
          << "At least one scope_id should be zero to merge "
             "AllocatorAttributes but found this.scope_id="
          << scope_id << " and other.scope_id=" << other.scope_id;
      scope_id = scope_id == 0 ? other.scope_id : scope_id;
    }
  }
  // Returns true if the fields set in *this is a subset of or equal to
  // those set in other.
  bool IsEqualOrLessRestrictiveThan(const AllocatorAttributes& other) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_33(mht_33_v, 600, "", "./tensorflow/core/framework/allocator.h", "IsEqualOrLessRestrictiveThan");

    return (value | other.value) == other.value;
  }

  // NOTE: The upper 8 bits of the value are reserved for
  // device-specific uses.  Implementors of a device can interpret these
  // upper 8 bits in device-specific ways, and ops implemented for those
  // devices are responsible for setting those 8 bits appropriately.
  uint32 value = 0;
  // EXPERIMENTAL: If this is greater than zero, then allocation is delegated to
  // a named special-purpose allocator on the same device.
  int32 scope_id = 0;

  // Returns a human readable representation of this.
  std::string DebugString() const;
};

// Returns a trivial implementation of Allocator, which is a process singleton.
// Access through this function is only intended for use by restricted parts
// of the infrastructure.
Allocator* cpu_allocator_base();

// If available, calls ProcessState::GetCPUAllocator(numa_node).
// If not, falls back to cpu_allocator_base().
// Intended for use in contexts where ProcessState is not visible at
// compile time. Where ProcessState is visible, it's preferable to
// call it directly.
Allocator* cpu_allocator(int numa_node = port::kNUMANoAffinity);

// Enables AllocatorStats in the default CPU allocator implementation.  By
// default, it's disabled.
void EnableCPUAllocatorStats();
// Disables AllocatorStats in the default CPU allocator implementation.  By
// default, it's disabled.
void DisableCPUAllocatorStats();
bool CPUAllocatorStatsEnabled();

// Enables full statistics collection in the default CPU allocator
// implementation.  By default, it's disabled.
void EnableCPUAllocatorFullStats();
bool CPUAllocatorFullStatsEnabled();

// An object that does the underlying suballoc/free of memory for a higher-level
// allocator.  The expectation is that the higher-level allocator is doing some
// kind of cache or pool management so that it will call SubAllocator::Alloc and
// Free relatively infrequently, compared to the number of times its own
// AllocateRaw and Free methods are called.
class SubAllocator {
 public:
  // Visitor gets called with a pointer to a memory area and its
  // size in bytes.  The index value will be numa_node for a CPU
  // allocator and GPU id for a GPU allocator.
  typedef std::function<void(void*, int index, size_t)> Visitor;

  SubAllocator(const std::vector<Visitor>& alloc_visitors,
               const std::vector<Visitor>& free_visitors);

  virtual ~SubAllocator() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_34(mht_34_v, 660, "", "./tensorflow/core/framework/allocator.h", "~SubAllocator");
}
  // Allocates at least num_bytes. Returns actual number of bytes allocated in
  // bytes_received. The caller can safely use the full bytes_received sized
  // buffer following the returend pointer.
  virtual void* Alloc(size_t alignment, size_t num_bytes,
                      size_t* bytes_received) = 0;
  virtual void Free(void* ptr, size_t num_bytes) = 0;

  // Returns true if the BFC allocator can safely coalesce adjacent regions
  // returned by this allocator.
  virtual bool SupportsCoalescing() const = 0;

  // Returns the type of the memory allocated by this SubAllocator.
  virtual AllocatorMemoryType GetMemoryType() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTh mht_35(mht_35_v, 676, "", "./tensorflow/core/framework/allocator.h", "GetMemoryType");

    return AllocatorMemoryType::kUnknown;
  }

 protected:
  // Implementation of Alloc() method must call this on newly allocated
  // value.
  void VisitAlloc(void* ptr, int index, size_t num_bytes);

  // Implementation of Free() method must call this on value to be
  // freed immediately before deallocation.
  void VisitFree(void* ptr, int index, size_t num_bytes);

  const std::vector<Visitor> alloc_visitors_;
  const std::vector<Visitor> free_visitors_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_
