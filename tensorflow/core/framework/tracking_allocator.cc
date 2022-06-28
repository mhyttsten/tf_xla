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
class MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc() {
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

#include "tensorflow/core/framework/tracking_allocator.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

TrackingAllocator::TrackingAllocator(Allocator* allocator, bool track_sizes)
    : allocator_(allocator),
      ref_(1),
      allocated_(0),
      high_watermark_(0),
      total_bytes_(0),
      track_sizes_locally_(track_sizes && !allocator_->TracksAllocationSizes()),
      next_allocation_id_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::TrackingAllocator");
}

void* TrackingAllocator::AllocateRaw(
    size_t alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::AllocateRaw");

  void* ptr = allocator_->AllocateRaw(alignment, num_bytes, allocation_attr);
  // If memory is exhausted AllocateRaw returns nullptr, and we should
  // pass this through to the caller
  if (nullptr == ptr) {
    return ptr;
  }
  if (allocator_->TracksAllocationSizes()) {
    size_t allocated_bytes = allocator_->AllocatedSize(ptr);
    {
      mutex_lock lock(mu_);
      allocated_ += allocated_bytes;
      high_watermark_ = std::max(high_watermark_, allocated_);
      total_bytes_ += allocated_bytes;
      allocations_.emplace_back(allocated_bytes, Env::Default()->NowMicros());
      ++ref_;
    }
  } else if (track_sizes_locally_) {
    // Call the underlying allocator to try to get the allocated size
    // whenever possible, even when it might be slow. If this fails,
    // use the requested size as an approximation.
    size_t allocated_bytes = allocator_->AllocatedSizeSlow(ptr);
    allocated_bytes = std::max(num_bytes, allocated_bytes);
    mutex_lock lock(mu_);
    next_allocation_id_ += 1;
    Chunk chunk = {num_bytes, allocated_bytes, next_allocation_id_};
    in_use_.emplace(std::make_pair(ptr, chunk));
    allocated_ += allocated_bytes;
    high_watermark_ = std::max(high_watermark_, allocated_);
    total_bytes_ += allocated_bytes;
    allocations_.emplace_back(allocated_bytes, Env::Default()->NowMicros());
    ++ref_;
  } else {
    mutex_lock lock(mu_);
    total_bytes_ += num_bytes;
    allocations_.emplace_back(num_bytes, Env::Default()->NowMicros());
    ++ref_;
  }
  return ptr;
}

void TrackingAllocator::DeallocateRaw(void* ptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::DeallocateRaw");

  // freeing a null ptr is a no-op
  if (nullptr == ptr) {
    return;
  }
  bool should_delete;
  // fetch the following outside the lock in case the call to
  // AllocatedSize is slow
  bool tracks_allocation_sizes = allocator_->TracksAllocationSizes();
  size_t allocated_bytes = 0;
  if (tracks_allocation_sizes) {
    allocated_bytes = allocator_->AllocatedSize(ptr);
  } else if (track_sizes_locally_) {
    mutex_lock lock(mu_);
    auto itr = in_use_.find(ptr);
    if (itr != in_use_.end()) {
      tracks_allocation_sizes = true;
      allocated_bytes = (*itr).second.allocated_size;
      in_use_.erase(itr);
    }
  }
  Allocator* allocator = allocator_;
  {
    mutex_lock lock(mu_);
    if (tracks_allocation_sizes) {
      CHECK_GE(allocated_, allocated_bytes);
      allocated_ -= allocated_bytes;
      allocations_.emplace_back(-allocated_bytes, Env::Default()->NowMicros());
    }
    should_delete = UnRef();
  }
  allocator->DeallocateRaw(ptr);
  if (should_delete) {
    delete this;
  }
}

bool TrackingAllocator::TracksAllocationSizes() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_3(mht_3_v, 290, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::TracksAllocationSizes");

  return track_sizes_locally_ || allocator_->TracksAllocationSizes();
}

size_t TrackingAllocator::RequestedSize(const void* ptr) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_4(mht_4_v, 297, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::RequestedSize");

  if (track_sizes_locally_) {
    mutex_lock lock(mu_);
    auto it = in_use_.find(ptr);
    if (it != in_use_.end()) {
      return (*it).second.requested_size;
    }
    return 0;
  } else {
    return allocator_->RequestedSize(ptr);
  }
}

size_t TrackingAllocator::AllocatedSize(const void* ptr) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_5(mht_5_v, 313, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::AllocatedSize");

  if (track_sizes_locally_) {
    mutex_lock lock(mu_);
    auto it = in_use_.find(ptr);
    if (it != in_use_.end()) {
      return (*it).second.allocated_size;
    }
    return 0;
  } else {
    return allocator_->AllocatedSize(ptr);
  }
}

int64_t TrackingAllocator::AllocationId(const void* ptr) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_6(mht_6_v, 329, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::AllocationId");

  if (track_sizes_locally_) {
    mutex_lock lock(mu_);
    auto it = in_use_.find(ptr);
    if (it != in_use_.end()) {
      return (*it).second.allocation_id;
    }
    return 0;
  } else {
    return allocator_->AllocationId(ptr);
  }
}

absl::optional<AllocatorStats> TrackingAllocator::GetStats() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_7(mht_7_v, 345, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::GetStats");

  return allocator_->GetStats();
}

bool TrackingAllocator::ClearStats() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_8(mht_8_v, 352, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::ClearStats");
 return allocator_->ClearStats(); }

std::tuple<size_t, size_t, size_t> TrackingAllocator::GetSizes() {
  size_t high_watermark;
  size_t total_bytes;
  size_t still_live_bytes;
  {
    mutex_lock lock(mu_);
    high_watermark = high_watermark_;
    total_bytes = total_bytes_;
    still_live_bytes = allocated_;
  }
  return std::make_tuple(total_bytes, high_watermark, still_live_bytes);
}

gtl::InlinedVector<AllocRecord, 4> TrackingAllocator::GetRecordsAndUnRef() {
  bool should_delete;
  gtl::InlinedVector<AllocRecord, 4> allocations;
  {
    mutex_lock lock(mu_);
    allocations.swap(allocations_);
    should_delete = UnRef();
  }
  if (should_delete) {
    delete this;
  }
  return allocations;
}

gtl::InlinedVector<AllocRecord, 4> TrackingAllocator::GetCurrentRecords() {
  gtl::InlinedVector<AllocRecord, 4> allocations;
  {
    mutex_lock lock(mu_);
    for (const AllocRecord& alloc : allocations_) {
      allocations.push_back(alloc);
    }
  }
  return allocations;
}

bool TrackingAllocator::UnRef() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStracking_allocatorDTcc mht_9(mht_9_v, 395, "", "./tensorflow/core/framework/tracking_allocator.cc", "TrackingAllocator::UnRef");

  CHECK_GE(ref_, 1);
  --ref_;
  return (ref_ == 0);
}

}  // end namespace tensorflow
