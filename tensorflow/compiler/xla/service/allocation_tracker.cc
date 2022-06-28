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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc() {
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

#include "tensorflow/compiler/xla/service/allocation_tracker.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

StatusOr<GlobalDataHandle> AllocationTracker::Register(
    ScopedShapedBuffer shaped_buffer, const std::string& tag) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::Register");

  absl::MutexLock lock(&mutex_);
  VLOG(2) << "Register";
  std::vector<ScopedShapedBuffer> replicated_buffers;
  replicated_buffers.emplace_back(std::move(shaped_buffer));
  return RegisterInternal(std::move(replicated_buffers), tag);
}

StatusOr<GlobalDataHandle> AllocationTracker::RegisterReplicatedBuffers(
    std::vector<ScopedShapedBuffer> replicated_buffers,
    const std::string& tag) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::RegisterReplicatedBuffers");

  absl::MutexLock lock(&mutex_);
  VLOG(2) << "RegisterReplicatedBuffers";
  return RegisterInternal(std::move(replicated_buffers), tag);
}

// ReleaseIfScopedShapedBuffer lets RegisterInternal<ShapedBufferTy>(b) call
// b.release() if b is a ScopedShapedBuffer, or otherwise pass b through
// unmodified.
static ShapedBuffer ReleaseIfScopedShapedBuffer(ShapedBuffer b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_2(mht_2_v, 231, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "ReleaseIfScopedShapedBuffer");
 return b; }
static ShapedBuffer ReleaseIfScopedShapedBuffer(ScopedShapedBuffer b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_3(mht_3_v, 235, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "ReleaseIfScopedShapedBuffer");

  return b.release();
}

template <typename ShapedBufferTy>
StatusOr<GlobalDataHandle> AllocationTracker::RegisterInternal(
    std::vector<ShapedBufferTy> replicated_buffers, const std::string& tag) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_4(mht_4_v, 245, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::RegisterInternal");

  static_assert(std::is_same<ShapedBufferTy, ShapedBuffer>::value ||
                    std::is_same<ShapedBufferTy, ScopedShapedBuffer>::value,
                "ShapedBufferTy must be ShapedBuffer or ScopedShapedBuffer.");
  VLOG(2) << "RegisterInternal("
          << "tag: \"" << tag << "\" with " << replicated_buffers.size()
          << " shaped_buffers.";

  int64_t handle = next_handle_++;
  for (auto& shaped_buffer : replicated_buffers) {
    std::vector<ShapeIndex> shape_indices;
    ShapeUtil::ForEachSubshape(
        shaped_buffer.on_device_shape(),
        [&](const Shape& /*subshape*/, const ShapeIndex& index) {
          shape_indices.push_back(index);
        });
    // Add shaped_buffer's buffers to opaque_to_allocation_map_, which owns
    // them.
    for (const ShapeIndex& index : shape_indices) {
      AddAllocationOrIncrementRefCount(shaped_buffer.buffer(index),
                                       shaped_buffer.device_ordinal());
    }
    // If ShapedBufferTy is ScopedShapedBuffer, release the ScopedShapedBuffer
    // into a regular ShapedBuffer, which is stored in
    // handle_to_shaped_buffers_.
    handle_to_shaped_buffers_[handle].emplace_back(
        absl::make_unique<ShapedBuffer>(
            ReleaseIfScopedShapedBuffer(std::move(shaped_buffer))));
  }

  GlobalDataHandle result;
  result.set_handle(handle);
  VLOG(2) << "handle: " << handle;
  return result;
}

Status AllocationTracker::Unregister(const GlobalDataHandle& data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_5(mht_5_v, 284, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::Unregister");

  absl::MutexLock lock(&mutex_);
  VLOG(2) << "Unregister("
          << "handle: " << data.handle() << ")";
  TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                      ResolveInternal(data));
  for (const auto& shaped_buffer : replicated_buffers) {
    std::vector<ShapeIndex> shape_indices;
    ShapeUtil::ForEachSubshape(
        shaped_buffer->on_device_shape(),
        [&shape_indices](const Shape& /*subshape*/, const ShapeIndex& index) {
          shape_indices.push_back(index);
        });
    for (const ShapeIndex& index : shape_indices) {
      TF_RETURN_IF_ERROR(DecrementRefCount(shaped_buffer->buffer(index),
                                           shaped_buffer->device_ordinal()));
    }
  }
  // Keep a nullptr as a tombstone for unregistered handles. This enables
  // better error messages. That is, "handle has been deallocated" versus
  // "handle does not exist".
  auto it = handle_to_shaped_buffers_.find(data.handle());
  if (it == handle_to_shaped_buffers_.end()) {
    return NotFound("no allocation record for global data handle: %d",
                    data.handle());
  }
  for (auto& shaped_buffer : it->second) {
    shaped_buffer.reset();
  }
  return Status::OK();
}

StatusOr<std::vector<GlobalDataHandle>> AllocationTracker::DeconstructTuple(
    const GlobalDataHandle& data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_6(mht_6_v, 320, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::DeconstructTuple");

  absl::MutexLock lock(&mutex_);

  TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                      ResolveInternal(data));
  // We only need to care about replica id 0 here, since the GlobalDataHandle is
  // the same for all buffers across replicas.
  const ShapedBuffer* shaped_buffer = replicated_buffers[0];
  if (!shaped_buffer->on_device_shape().IsTuple()) {
    return InvalidArgument("global data handle %d is not a tuple",
                           data.handle());
  }

  if (ShapeUtil::IsNestedTuple(shaped_buffer->on_device_shape())) {
    return Unimplemented("Deconstructing nested tuples is not implemented.");
  }

  std::vector<GlobalDataHandle> element_handles;
  const auto n = ShapeUtil::TupleElementCount(shaped_buffer->on_device_shape());
  element_handles.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto element_buffer = ShapedBuffer(
        ShapeUtil::GetTupleElementShape(shaped_buffer->on_device_shape(), i),
        shaped_buffer->device_ordinal());
    element_buffer.set_buffer(shaped_buffer->buffer(/*index=*/{i}),
                              /*index=*/{});
    std::vector<ShapedBuffer> replicated_buffers;
    replicated_buffers.push_back(std::move(element_buffer));
    TF_ASSIGN_OR_RETURN(
        GlobalDataHandle element_handle,
        RegisterInternal(std::move(replicated_buffers), "deconstructed tuple"));

    element_handles.push_back(element_handle);
  }
  return std::move(element_handles);
}

StatusOr<std::vector<const ShapedBuffer*>> AllocationTracker::Resolve(
    const GlobalDataHandle& data) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_7(mht_7_v, 361, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::Resolve");

  absl::MutexLock lock(&mutex_);
  return AllocationTracker::ResolveInternal(data);
}

StatusOr<const ShapedBuffer*> AllocationTracker::ResolveForReplica(
    const GlobalDataHandle& data, int replica_id) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_8(mht_8_v, 370, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::ResolveForReplica");

  absl::MutexLock lock(&mutex_);
  TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                      ResolveInternal(data));
  if (replica_id >= replicated_buffers.size()) {
    return InvalidArgument(
        "Requesting buffer for replica %d, but found buffers only for %lu "
        "replicas.",
        replica_id, replicated_buffers.size());
  }
  return replicated_buffers[replica_id];
}

StatusOr<std::vector<const ShapedBuffer*>> AllocationTracker::ResolveInternal(
    const GlobalDataHandle& data) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_9(mht_9_v, 387, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::ResolveInternal");

  VLOG(2) << "resolve:" << data.handle();
  auto it = handle_to_shaped_buffers_.find(data.handle());
  if (it == handle_to_shaped_buffers_.end()) {
    return NotFound("no allocation record for global data handle: %d",
                    data.handle());
  }
  std::vector<const ShapedBuffer*> replicated_buffers;
  for (const auto& shaped_buffer : it->second) {
    if (shaped_buffer == nullptr) {
      return InvalidArgument("global data handle %d was previously deallocated",
                             data.handle());
    }
    replicated_buffers.push_back(shaped_buffer.get());
  }

  return replicated_buffers;
}

void AllocationTracker::AddAllocationOrIncrementRefCount(
    se::DeviceMemoryBase device_memory, int device_ordinal) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_10(mht_10_v, 410, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::AddAllocationOrIncrementRefCount");

  AllocationMap& allocation_map = opaque_to_allocation_map_[device_ordinal];
  auto it = allocation_map.find(device_memory.opaque());
  if (it == allocation_map.end()) {
    allocation_map[device_memory.opaque()] = {
        se::OwningDeviceMemory(device_memory, device_ordinal,
                               backend_->memory_allocator()),
        /*ref_count=*/1};
  } else {
    it->second.ref_count++;
  }
}

Status AllocationTracker::DecrementRefCount(se::DeviceMemoryBase device_memory,
                                            int device_ordinal) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSallocation_trackerDTcc mht_11(mht_11_v, 427, "", "./tensorflow/compiler/xla/service/allocation_tracker.cc", "AllocationTracker::DecrementRefCount");

  AllocationMap& allocation_map = opaque_to_allocation_map_[device_ordinal];
  auto it = allocation_map.find(device_memory.opaque());
  TF_RET_CHECK(it != allocation_map.end());
  Allocation& allocation = it->second;
  TF_RET_CHECK(allocation.ref_count >= 1);
  if (allocation.ref_count == 1) {
    TF_RETURN_IF_ERROR(allocation.device_memory.Free());
    allocation_map.erase(it);
  } else {
    allocation.ref_count--;
  }
  return Status::OK();
}

}  // namespace xla
