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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/scoped_allocator.h"

#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/platform/dynamic_annotations.h"

namespace tensorflow {

ScopedAllocator::ScopedAllocator(const Tensor& backing_tensor, int32_t scope_id,
                                 const string& name,
                                 const gtl::ArraySlice<Field> fields,
                                 int32_t expected_call_count,
                                 ScopedAllocatorContainer* container)
    : backing_tensor_(backing_tensor),
      tbuf_(backing_tensor_.buf_),
      id_(scope_id),
      name_(name),
      container_(container),
      fields_(fields.begin(), fields.end()),
      expected_call_count_(expected_call_count),
      live_alloc_count_(0) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocator::ScopedAllocator");

  // Hold this until all aliases have been deallocated.
  tbuf_->Ref();
  // Hold this until all expected_calls have been made.
  container->Ref();
  CHECK_GE(tbuf_->size(), fields.back().offset + fields.back().bytes_requested);
}

ScopedAllocator::~ScopedAllocator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocator::~ScopedAllocator");

  mutex_lock l(mu_);
  VLOG(1) << "~ScopedAllocator " << this << " tbuf_ " << tbuf_ << " data "
          << static_cast<void*>(tbuf_->data());
  // In the absence of incomplete graph execution situations
  // (interruption by error status or control flow branch crossing
  // ScopedAllocation region) we expect expected_call_count_ == 0 at
  // exit.
  if (VLOG_IS_ON(1)) {
    if (expected_call_count_ > 0)
      VLOG(1) << "expected_call_count_ = " << expected_call_count_
              << " at deallocation";
  }
  if (tbuf_) tbuf_->Unref();
}

void* ScopedAllocator::AllocateRaw(int32_t field_index, size_t num_bytes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocator::AllocateRaw");

  VLOG(1) << "ScopedAllocator index " << id_ << " AllocateRaw "
          << "field " << field_index << " num_bytes " << num_bytes;
  void* ptr = nullptr;
  const Field* field = nullptr;
  {
    mutex_lock l(mu_);
    if (expected_call_count_ <= 0) {
      LOG(ERROR) << "Scoped allocator " << name_
                 << " could not satisfy request for " << num_bytes
                 << " bytes, expected uses exhausted. ";
      return nullptr;
    }

    int32_t num_fields = static_cast<int32>(fields_.size());
    if (field_index >= num_fields) {
      LOG(ERROR) << "ScopedAllocator " << name_
                 << " received unexpected field number " << field_index;
      return nullptr;
    }

    field = &fields_[field_index];
    if (num_bytes != field->bytes_requested) {
      LOG(ERROR) << "ScopedAllocator " << name_ << " got request for "
                 << num_bytes << " bytes from field " << field_index
                 << " which has precalculated size " << field->bytes_requested
                 << " and offset " << field->offset;
      return nullptr;
    }

    ptr = static_cast<void*>((tbuf_->template base<char>() + field->offset));

    ++live_alloc_count_;
    --expected_call_count_;
    if (0 == expected_call_count_) {
      for (auto& f : fields_) {
        container_->Drop(f.scope_id, this);
      }
      container_->Drop(id_, this);
      container_->Unref();
      container_ = nullptr;
    }
  }
  VLOG(2) << "AllocateRaw returning " << ptr << " bytes_requested "
          << field->bytes_requested << " bytes_allocated "
          << field->bytes_allocated;

  // If there is overshoot due to alignment, let MSAN believe that the padding
  // is initialized.  This is okay because we do not use this memory region for
  // anything meaningful.
  if (field->bytes_allocated > field->bytes_requested) {
    size_t extra_bytes = field->bytes_allocated - field->bytes_requested;
    void* extra_buf = static_cast<void*>(static_cast<char*>(ptr) +
                                         field->bytes_allocated - extra_bytes);
    VLOG(2) << "AllocateRaw requested " << num_bytes
            << " bytes which is not divisible by kAllocatorAlignment="
            << Allocator::kAllocatorAlignment << " and hence we allocated "
            << field->bytes_allocated << ". Annotating " << extra_bytes
            << " bytes starting at " << extra_buf
            << " with TF_ANNOTATE_MEMORY_IS_INITIALIZED";
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(extra_buf, extra_bytes);
  }

  return ptr;
}

void ScopedAllocator::DeallocateRaw(void* p) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_3(mht_3_v, 303, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocator::DeallocateRaw");

  CHECK(VerifyPointer(p));

  bool dead = false;
  {
    mutex_lock l(mu_);
    CHECK_GT(live_alloc_count_, 0);
    if (0 == --live_alloc_count_) {
      if (0 == expected_call_count_) {
        dead = true;
      }
    }
  }
  if (dead) {
    delete this;
  }
}

bool ScopedAllocator::VerifyPointer(const void* p) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_4(mht_4_v, 324, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocator::VerifyPointer");

  void* base = tbuf_->data();
  CHECK_GE(p, base);
  for (auto& f : fields_) {
    void* f_ptr = static_cast<void*>(static_cast<char*>(base) + f.offset);
    if (f_ptr == p) {
      return true;
      break;
    }
  }
  VLOG(1) << "ScopedAllocator index " << id_ << " VerifyPointer for p=" << p
          << " failed.";
  return false;
}

bool ScopedAllocator::VerifyTensor(const Tensor* t) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_5(mht_5_v, 342, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocator::VerifyTensor");

  return VerifyPointer(t->buf_->data());
}

ScopedAllocatorInstance::ScopedAllocatorInstance(ScopedAllocator* sa,
                                                 int32_t field_index)
    : scoped_allocator_(sa),
      field_index_(field_index),
      allocated_(false),
      deallocated_(false),
      in_table_(true) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_6(mht_6_v, 355, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocatorInstance::ScopedAllocatorInstance");

  VLOG(1) << "new ScopedAllocatorInstance " << this << " on SA " << sa
          << " field_index " << field_index;
}

void ScopedAllocatorInstance::DropFromTable() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_7(mht_7_v, 363, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocatorInstance::DropFromTable");

  bool del = false;
  {
    mutex_lock l(mu_);
    CHECK(in_table_);
    in_table_ = false;
    VLOG(2) << "ScopedAllocatorInstance::DropFromTable " << this
            << " allocated_ " << allocated_ << " deallocated_ " << deallocated_
            << " in_table_ " << in_table_;
    // Single use is complete when it is allocated and deallocated.
    // This check prevents a race between Allocating the tensor slice and
    // Dropping it from the parent container's table.
    if (allocated_ && deallocated_) {
      del = true;
    }
  }
  if (del) delete this;
}

void* ScopedAllocatorInstance::AllocateRaw(size_t alignment, size_t num_bytes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_8(mht_8_v, 385, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocatorInstance::AllocateRaw");

  void* ptr = scoped_allocator_->AllocateRaw(field_index_, num_bytes);
  {
    mutex_lock l(mu_);
    if (nullptr == ptr) {
      VLOG(2) << "ScopedAllocatorInstance::AllocateRaw " << this
              << " call to underlying ScopedAllocator unsuccessful,"
              << " allocated_ " << allocated_ << " deallocated_ "
              << deallocated_ << " in_table_ " << in_table_
              << " returning nullptr.";
    } else {
      allocated_ = true;
      VLOG(2) << "ScopedAllocatorInstance::AllocateRaw " << this
              << " allocated_ " << allocated_ << " deallocated_ "
              << deallocated_ << " in_table_ " << in_table_
              << " returning ptr = " << ptr;
    }
  }
  return ptr;
}

void ScopedAllocatorInstance::DeallocateRaw(void* p) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_9(mht_9_v, 409, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocatorInstance::DeallocateRaw");

  scoped_allocator_->DeallocateRaw(p);
  bool del = false;
  {
    mutex_lock l(mu_);
    CHECK(allocated_);
    deallocated_ = true;
    VLOG(2) << "ScopedAllocatorInstance::DeallocateRaw " << this
            << " allocated_ " << allocated_ << " deallocated_ " << deallocated_
            << " in_table_ " << in_table_;
    // Single use is now complete, but only delete this instance when it is
    // no longer in a ScopedAllocatorContainer's table.
    if (!in_table_) {
      del = true;
    }
  }
  if (del) delete this;
}

string ScopedAllocatorInstance::Name() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTcc mht_10(mht_10_v, 431, "", "./tensorflow/core/common_runtime/scoped_allocator.cc", "ScopedAllocatorInstance::Name");

  return strings::StrCat(scoped_allocator_->name(), "_field_", field_index_);
}

}  // namespace tensorflow
