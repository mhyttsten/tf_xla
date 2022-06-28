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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc() {
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
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"

#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/framework/allocator.h"

namespace tensorflow {

Status ScopedAllocatorContainer::AddScopedAllocator(
    const Tensor& backing_tensor, int32_t scope_id, const string& scope_name,
    const gtl::ArraySlice<ScopedAllocator::Field>& fields,
    int32_t expected_call_count) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("scope_name: \"" + scope_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorContainer::AddScopedAllocator");

  VLOG(1) << "AddScopedAllocator " << mgr_->device_name()
          << " step_id_=" << step_id_ << " scope_id=" << scope_id;
  mutex_lock l(mu_);
  // Ensure none of the new scope_ids are in use.
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    return errors::Internal("Cannot create ScopedAllocator because scope_id ",
                            scope_id, " for name ", scope_name,
                            " already exists");
  }
  for (auto& f : fields) {
    if (allocators_.find(f.scope_id) != allocators_.end()) {
      return errors::Internal(
          "Cannot create ScopedAllocator because field scope_id ", f.scope_id,
          " for name ", scope_name, " already exists");
    }
  }
  VLOG(2) << " container " << this << " step_id " << step_id_;
  ScopedAllocator* sa = new ScopedAllocator(
      backing_tensor, scope_id, scope_name, fields, expected_call_count, this);
  allocators_[scope_id] =
      ScopedAllocatorContainer::SAField(ScopedAllocator::kBackingIndex, sa);
  VLOG(2) << "#fields " << fields.size();
  for (int i = 0; i < fields.size(); ++i) {
    const ScopedAllocator::Field& f = fields[i];
    VLOG(2) << "Adding instance with for " << mgr_->device_name()
            << " scope_id=" << f.scope_id;
    allocators_[f.scope_id] = ScopedAllocatorContainer::SAField(
        i, new ScopedAllocatorInstance(sa, i));
  }
  return Status::OK();
}

ScopedAllocator* ScopedAllocatorContainer::GetAllocator(int32_t scope_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorContainer::GetAllocator");

  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    CHECK_EQ(ScopedAllocator::kBackingIndex, it->second.field_index);
    return it->second.scoped_allocator;
  } else {
    LOG(ERROR) << "Failed to find ScopedAllocator for " << scope_id
               << " in container for step " << step_id_ << " on "
               << mgr_->device_name();
    return nullptr;
  }
}

ScopedAllocatorInstance* ScopedAllocatorContainer::GetInstance(
    int32_t scope_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorContainer::GetInstance");

  VLOG(2) << "GetInstance " << scope_id << " step " << step_id_ << " on "
          << mgr_->device_name();
  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    return it->second.instance;
  }
  LOG(FATAL) << "Failed to find instance " << scope_id << " in container "
             << step_id_ << " on " << mgr_->device_name();
  return nullptr;
}

void ScopedAllocatorContainer::Drop(int32_t scope_id, ScopedAllocator* sa) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorContainer::Drop");

  VLOG(2) << "Drop " << scope_id << " from container " << this << " step "
          << step_id_ << " on " << mgr_->device_name();
  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    if (it->second.field_index != ScopedAllocator::kBackingIndex) {
      it->second.instance->DropFromTable();
    }
    allocators_.erase(it);
  }
}

ScopedAllocatorContainer::~ScopedAllocatorContainer() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_4(mht_4_v, 282, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorContainer::~ScopedAllocatorContainer");

  VLOG(2) << "~ScopedAllocatorContainer " << this << " step " << step_id_
          << " on " << mgr_->device_name();
  mutex_lock l(mu_);
  // In normal execution the table should be empty and all of its contents
  // deleted via Drop.  When a step ends early (e.g. through abnormal
  // termination) we need to clean up explicitly.  So long as graph execution
  // of the associated step has completely terminated this should be safe.
  for (auto& it : allocators_) {
    if (it.second.field_index == ScopedAllocator::kBackingIndex) {
      delete it.second.scoped_allocator;
    } else {
      it.second.instance->DropFromTable();
    }
  }
}

ScopedAllocatorMgr::~ScopedAllocatorMgr() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_5(mht_5_v, 302, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorMgr::~ScopedAllocatorMgr");

  mutex_lock l(mu_);
  for (auto it : per_step_map_) {
    // In normal execution the associated ScopedAllocatorContainer is
    // empty and gone by the end of the step.  But in abnormal termination,
    // such as when an error has interrupted execution or in a unittest,
    // we need to remove all of its Refs here to avoid memory leaks.
    // This is safe so long as graph execution has ceased.
    while (!it.second->Unref()) {
    }
  }
}

void ScopedAllocatorMgr::Cleanup(int64_t step_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_6(mht_6_v, 318, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorMgr::Cleanup");

  mutex_lock l(mu_);
  auto it = per_step_map_.find(step_id);
  if (it != per_step_map_.end()) {
    it->second->Unref();
    per_step_map_.erase(it);
  }
}

ScopedAllocatorContainer* ScopedAllocatorMgr::GetContainer(int64_t step_id) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_7(mht_7_v, 330, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorMgr::GetContainer");

  VLOG(2) << "GetContainer " << step_id << " on " << device_name();
  ScopedAllocatorContainer* sac = nullptr;
  mutex_lock l(mu_);
  auto it = per_step_map_.find(step_id);
  if (it == per_step_map_.end()) {
    sac = new ScopedAllocatorContainer(this, step_id);
    per_step_map_[step_id] = sac;
  } else {
    sac = it->second;
  }
  return sac;
}

Status ScopedAllocatorMgr::AddScopedAllocator(
    const Tensor& backing_tensor, int64_t step_id, int32_t scope_id,
    const string& scope_name,
    const gtl::ArraySlice<ScopedAllocator::Field>& fields,
    int32_t expected_call_count) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("scope_name: \"" + scope_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_8(mht_8_v, 352, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorMgr::AddScopedAllocator");

  ScopedAllocatorContainer* sac = GetContainer(step_id);
  return sac->AddScopedAllocator(backing_tensor, scope_id, scope_name, fields,
                                 expected_call_count);
}

/*static*/
size_t ScopedAllocatorMgr::PopulateFields(
    int32_t scope_id, const gtl::ArraySlice<TensorShape>& shapes,
    const DataType dtype, std::vector<ScopedAllocator::Field>* fields) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSscoped_allocator_mgrDTcc mht_9(mht_9_v, 364, "", "./tensorflow/core/common_runtime/scoped_allocator_mgr.cc", "ScopedAllocatorMgr::PopulateFields");

  const int32_t num_fields = static_cast<int32>(shapes.size());
  fields->resize(num_fields);
  // At the end of iteration `i`, `offset` points to the offset from the start
  // of the backing buffer until the end of `field[i].bytes_allocated`.  This
  // is aligned to `kAllocatorAlignment`.
  size_t offset = 0;
  for (int32_t i = 0; i < num_fields; ++i) {
    size_t bytes_requested = shapes[i].num_elements() * DataTypeSize(dtype);
    auto* field = &((*fields)[i]);
    field->scope_id = scope_id + 1 + i;
    field->bytes_requested = bytes_requested;
    field->offset = offset;
    offset += bytes_requested;

    // Compute actual #bytes allocated, which may include padding due to
    // alignment.
    size_t bytes_allocated = bytes_requested;
    size_t overshoot = offset % Allocator::kAllocatorAlignment;
    if (overshoot > 0) {
      size_t alignment_bytes = Allocator::kAllocatorAlignment - overshoot;
      bytes_allocated += alignment_bytes;
      offset += alignment_bytes;
    }
    field->bytes_allocated = bytes_allocated;

    VLOG(1) << "field=" << i << " scope_id=" << field->scope_id
            << " bytes_requested=" << field->bytes_requested
            << " offset=" << field->offset
            << " bytes_allocated=" << field->bytes_allocated;
  }

  return offset;
}

}  // namespace tensorflow
