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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SCOPED_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SCOPED_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh() {
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


#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
class ScopedAllocatorContainer;
class ScopedAllocatorInstance;

// Manages a single backing tensor and a collection of aliases.
class ScopedAllocator {
 public:
  static constexpr int32_t kInvalidId = 0;
  static constexpr size_t kMaxAlignment = 64;

  // A subrange of the TensorBuffer associated with this object that
  // will be the backing memory for one aliased tensor.
  struct Field {
    int32 scope_id;
    size_t offset;
    size_t bytes_requested;
    size_t bytes_allocated;
  };
  // Field index that refers to backing tensor, not any aliased field.
  static constexpr int32_t kBackingIndex = -1;

  // backing_tensor is expected to be newly allocated by a ScopedAllocatorOp
  // instance.  It must be large enough to back all of the specified
  // (offset, byte) ranges of the fields.
  ScopedAllocator(const Tensor& backing_tensor, int32_t scope_id,
                  const std::string& name, const gtl::ArraySlice<Field> fields,
                  int32_t expected_call_count,
                  ScopedAllocatorContainer* container);

  // Automatically deletes when last use expires, or when
  // ScopedAllocatorContainer decides to delete.
  ~ScopedAllocator() TF_LOCKS_EXCLUDED(mu_);

  // For debugging: returns true iff p is a pointer that could have
  // been returned by AllocateRaw.
  bool VerifyPointer(const void* p);
  bool VerifyTensor(const Tensor* t);

  const Tensor& tensor() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh mht_0(mht_0_v, 230, "", "./tensorflow/core/common_runtime/scoped_allocator.h", "tensor");
 return backing_tensor_; }

  const std::string& name() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh mht_1(mht_1_v, 235, "", "./tensorflow/core/common_runtime/scoped_allocator.h", "name");
 return name_; }

 private:
  friend class ScopedAllocatorInstance;
  // Only ScopedAllocatorInstances can call AllocateRaw and DeallocateRaw on a
  // ScopedAllocator
  void* AllocateRaw(int32_t field_index, size_t num_bytes)
      TF_LOCKS_EXCLUDED(mu_);
  void DeallocateRaw(void* p) TF_LOCKS_EXCLUDED(mu_);
  Tensor backing_tensor_;
  TensorBuffer* tbuf_;
  int32 id_;
  std::string name_;
  ScopedAllocatorContainer* container_;
  std::vector<Field> fields_;
  mutex mu_;
  int32 expected_call_count_ TF_GUARDED_BY(mu_);
  int32 live_alloc_count_ TF_GUARDED_BY(mu_);
};

// An Allocator that will return a pointer into the backing buffer of
// a previously allocated tensor, allowing creation of an alias
// tensor.  There is a one-to-one mapping between the fields of a
// ScopedAllocator and ScopedAllocatorInstances.  There is also a one-to-one
// mapping between scope_ids and ScopedAllocatorInstances.  It should be
// discarded immediately after a single use.
class ScopedAllocatorInstance : public Allocator {
 public:
  explicit ScopedAllocatorInstance(ScopedAllocator* sa, int32_t field_index);

 private:
  ~ScopedAllocatorInstance() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh mht_2(mht_2_v, 269, "", "./tensorflow/core/common_runtime/scoped_allocator.h", "~ScopedAllocatorInstance");

    VLOG(1) << "~ScopedAllocatorInstance " << this;
  }

 public:
  // When a ScopedAllocatorContainer "Drops" a scope_id, it calls DropFromTable
  // on the underlying ScopedAllocatorInstance.  If this instance has already
  // deallocated the tensor slice, we can safely delete this.
  void DropFromTable() TF_LOCKS_EXCLUDED(mu_);
  void* AllocateRaw(size_t alignment, size_t num_bytes)
      TF_LOCKS_EXCLUDED(mu_) override;
  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocator_attr) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh mht_3(mht_3_v, 284, "", "./tensorflow/core/common_runtime/scoped_allocator.h", "AllocateRaw");

    return AllocateRaw(alignment, num_bytes);
  }
  void DeallocateRaw(void* p) TF_LOCKS_EXCLUDED(mu_) override;
  bool TracksAllocationSizes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh mht_4(mht_4_v, 291, "", "./tensorflow/core/common_runtime/scoped_allocator.h", "TracksAllocationSizes");
 return false; }
  size_t RequestedSize(const void* ptr) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh mht_5(mht_5_v, 295, "", "./tensorflow/core/common_runtime/scoped_allocator.h", "RequestedSize");
 return 0; }
  size_t AllocatedSize(const void* ptr) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh mht_6(mht_6_v, 299, "", "./tensorflow/core/common_runtime/scoped_allocator.h", "AllocatedSize");
 return 0; }
  int64_t AllocationId(const void* ptr) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh mht_7(mht_7_v, 303, "", "./tensorflow/core/common_runtime/scoped_allocator.h", "AllocationId");
 return 0; }
  size_t AllocatedSizeSlow(const void* ptr) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocatorDTh mht_8(mht_8_v, 307, "", "./tensorflow/core/common_runtime/scoped_allocator.h", "AllocatedSizeSlow");
 return 0; }
  std::string Name() override;

 private:
  mutex mu_;
  ScopedAllocator* scoped_allocator_;
  int32 field_index_;
  bool allocated_ TF_GUARDED_BY(mu_);
  bool deallocated_ TF_GUARDED_BY(mu_);
  bool in_table_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SCOPED_ALLOCATOR_H_
