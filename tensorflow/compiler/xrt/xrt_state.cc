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
class MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc() {
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

// Classes for allocating XLA literals in device memory and managing handles
// that refer to them.

#include "tensorflow/compiler/xrt/xrt_state.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xrt/xrt_memory_manager.h"

namespace tensorflow {
namespace {

// Helper typedef to make ShapeTree ForEach helper lambda signatures more
// readable. They need a type of const T& where in this case T is the
// following pointer.
typedef XRTBufferAllocation* XRTBufferAllocationPtr;

class BufferAllocStats {
 public:
  struct Stats {
    int64_t count = 0;
    int64_t size = 0;
  };

  Stats ReportAlloc(int64_t device, int64_t msize) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xrt/xrt_state.cc", "ReportAlloc");

    mutex_lock lock(lock_);
    Stats* device_stats = &stats_[device];
    device_stats->count += 1;
    device_stats->size += msize;
    return *device_stats;
  }

  Stats ReportFree(int64_t device, int64_t msize) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/xrt/xrt_state.cc", "ReportFree");

    mutex_lock lock(lock_);
    Stats* device_stats = &stats_[device];
    device_stats->count -= 1;
    device_stats->size -= msize;
    return *device_stats;
  }

 private:
  mutable mutex lock_;
  std::map<int64_t, Stats> stats_;
};

BufferAllocStats* GetAllocStats() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_2(mht_2_v, 242, "", "./tensorflow/compiler/xrt/xrt_state.cc", "GetAllocStats");

  static BufferAllocStats* stats = new BufferAllocStats();
  return stats;
}

Status AllocateScopedShapedBuffer(
    XRTMemoryManager* memory_manager, xla::Backend* backend, int device_ordinal,
    const xla::Shape& shape, std::unique_ptr<xla::ScopedShapedBuffer>* buffer,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_3(mht_3_v, 253, "", "./tensorflow/compiler/xrt/xrt_state.cc", "AllocateScopedShapedBuffer");

  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal));

  // XLA may use a different representation on device than the representation on
  // the host. XLA does not document any contract for the relationship between
  // these representations :/ Right now, the device shape is always a superset
  // of the host shape, meaning that for any valid ShapeIndex in the host shape
  // that ShapeIndex is also valid in the device shape, but not vice versa. In
  // particular, some host-side types are rewritten to be tuples. We rely on
  // this property when making sub-buffers, because we assume that if the client
  // requests the host-shape sub-buffer at index i, that will correspond to the
  // right device-shape sub-buffer at the same index.
  xla::Shape on_device_shape = transfer_manager->HostShapeToDeviceShape(shape);
  VLOG(3) << "Allocating literal buffer: host_shape="
          << xla::ShapeUtil::HumanStringWithLayout(shape) << " device_shape="
          << xla::ShapeUtil::HumanStringWithLayout(on_device_shape);

  // The ScopedShapedBuffer frees the buffers that have so far been allocated if
  // it goes out of scope. That's useful if we return early as the result of an
  // error allocating one of the later buffers.
  *buffer = absl::make_unique<xla::ScopedShapedBuffer>(
      shape, on_device_shape, allocator, device_ordinal);
  for (auto& index_to_buffer : (*buffer)->buffers()) {
    const xla::Shape& subshape =
        xla::ShapeUtil::GetSubshape(on_device_shape, index_to_buffer.first);
    uint64 size = transfer_manager->GetByteSizeRequirement(subshape);
    TF_ASSIGN_OR_RETURN(
        se::OwningDeviceMemory buffer,
        memory_manager->Allocate(backend, device_ordinal, size, allocator));
    // Move our buffer into shaped_buffer, which takes ownership of it.
    index_to_buffer.second = buffer.Release();
    VLOG(2) << "Allocated buffer at " << index_to_buffer.second.opaque()
            << " index " << index_to_buffer.first.ToString() << " (" << size
            << " bytes)";
  }

  TF_RETURN_IF_ERROR(
      transfer_manager->WriteTupleIndexTables(stream.get(), *(buffer->get())));

  return Status::OK();
}

}  // namespace

XRTBufferAllocation::XRTBufferAllocation(const se::DeviceMemoryBase& allocation,
                                         int device_ordinal,
                                         se::DeviceMemoryAllocator* allocator)
    : allocation_(allocation),
      device_ordinal_(device_ordinal),
      allocator_(allocator) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_4(mht_4_v, 306, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTBufferAllocation::XRTBufferAllocation");

  if (VLOG_IS_ON(2)) {
    auto stats =
        GetAllocStats()->ReportAlloc(device_ordinal_, allocation_.size());
    LOG(INFO) << "XRT Allocation Stats: device=" << device_ordinal_
              << " count=" << stats.count << " size=" << stats.size;
  }
}

XRTBufferAllocation::~XRTBufferAllocation() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_5(mht_5_v, 318, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTBufferAllocation::~XRTBufferAllocation");

  if (VLOG_IS_ON(2)) {
    GetAllocStats()->ReportFree(device_ordinal_, allocation_.size());
  }
  // Deallocate explicitly allows allocation_ to be null.
  TF_CHECK_OK(allocator_->Deallocate(device_ordinal_, allocation_));
  VLOG(2) << "Freed buffer at " << allocation_.opaque() << " ("
          << allocation_.size() << " bytes)";
}

const se::DeviceMemoryBase& XRTBufferAllocation::allocation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_6(mht_6_v, 331, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTBufferAllocation::allocation");

  return allocation_;
}

XRTTupleAllocation::XRTTupleAllocation(int device_ordinal,
                                       se::DeviceMemoryAllocator* allocator,
                                       const xla::Shape& on_host_shape,
                                       const xla::Shape& on_device_shape)
    : device_ordinal_(device_ordinal),
      allocator_(allocator),
      on_host_shape_(on_host_shape),
      on_device_shape_(on_device_shape),
      buffers_(&on_device_shape_),
      pin_count_(0) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_7(mht_7_v, 347, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::XRTTupleAllocation");
}

XRTTupleAllocation::~XRTTupleAllocation() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_8(mht_8_v, 352, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::~XRTTupleAllocation");
 ReleaseBuffers(); }

void XRTTupleAllocation::ReleaseBuffers() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_9(mht_9_v, 357, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::ReleaseBuffers");

  for (auto& index_buffer : buffers_) {
    if (index_buffer.second != nullptr) {
      index_buffer.second->Unref();
      index_buffer.second = nullptr;
    }
  }
}

/*static*/ Status XRTTupleAllocation::CreateAndTransfer(
    const xla::LiteralBase& literal, XRTMemoryManager* memory_manager,
    xla::Backend* backend, int device_ordinal, XRTTupleAllocation** allocation,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_10(mht_10_v, 372, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::CreateAndTransfer");

  auto transfer_manager = backend->transfer_manager();
  std::unique_ptr<xla::ScopedShapedBuffer> scoped_buffer;
  TF_RETURN_IF_ERROR(AllocateScopedShapedBuffer(memory_manager, backend,
                                                device_ordinal, literal.shape(),
                                                &scoped_buffer, allocator));
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal));
  TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDevice(
      stream.get(), literal, *scoped_buffer));

  // By releasing the ScopedShapedBuffer we ensure that the underlying storage
  // won't be freed when the buffer goes out of scope at the end of this
  // call. To avoid a leak, there must be no error-case returns from here until
  // the end of the method.
  auto shaped_buffer = scoped_buffer->release();
  *allocation = new XRTTupleAllocation(device_ordinal, allocator,
                                       shaped_buffer.on_host_shape(),
                                       shaped_buffer.on_device_shape());
  (*allocation)
      ->InitializeFromShapedBuffer(shaped_buffer, allocator, device_ordinal);
  (*allocation)->SetDeviceMemorySize();
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::CreateUninitialized(
    const xla::Shape& shape, XRTMemoryManager* memory_manager,
    xla::Backend* backend, int device_ordinal, XRTTupleAllocation** allocation,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_11(mht_11_v, 402, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::CreateUninitialized");

  std::unique_ptr<xla::ScopedShapedBuffer> scoped_buffer;
  TF_RETURN_IF_ERROR(AllocateScopedShapedBuffer(memory_manager, backend,
                                                device_ordinal, shape,
                                                &scoped_buffer, allocator));

  // By releasing the ScopedShapedBuffer we ensure that the underlying storage
  // won't be freed when the buffer goes out of scope at the end of this
  // call. To avoid a leak, there must be no error-case returns from here until
  // the end of the method.
  auto shaped_buffer = scoped_buffer->release();
  *allocation = new XRTTupleAllocation(device_ordinal, allocator,
                                       shaped_buffer.on_host_shape(),
                                       shaped_buffer.on_device_shape());
  (*allocation)
      ->InitializeFromShapedBuffer(shaped_buffer, allocator, device_ordinal);
  (*allocation)->SetDeviceMemorySize();
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::CreateFromBuffer(
    const xla::ShapedBuffer& shaped_buffer, const xla::Shape& on_host_shape,
    const xla::Shape& on_device_shape, xla::Backend* backend,
    int device_ordinal, XRTTupleAllocation** allocation,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_12(mht_12_v, 429, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::CreateFromBuffer");

  *allocation = new XRTTupleAllocation(device_ordinal, allocator, on_host_shape,
                                       on_device_shape);
  (*allocation)
      ->InitializeFromShapedBuffer(shaped_buffer, allocator, device_ordinal);
  (*allocation)->SetDeviceMemorySize();
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::CreateFromBuffer(
    const xla::ShapedBuffer& shaped_buffer, xla::Backend* backend,
    int device_ordinal, XRTTupleAllocation** allocation,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_13(mht_13_v, 444, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::CreateFromBuffer");

  return CreateFromBuffer(shaped_buffer, shaped_buffer.on_host_shape(),
                          shaped_buffer.on_device_shape(), backend,
                          device_ordinal, allocation, allocator);
}

Status XRTTupleAllocation::ToLiteral(xla::Backend* backend,
                                     xla::MutableLiteralBase* literal) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_14(mht_14_v, 454, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::ToLiteral");

  mutex_lock lock(lock_);
  return literal_ == nullptr ? StoreToLiteral(backend, literal)
                             : literal->CopyFrom(*literal_);
}

Status XRTTupleAllocation::StoreToLiteral(xla::Backend* backend,
                                          xla::MutableLiteralBase* literal) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_15(mht_15_v, 464, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::StoreToLiteral");

  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal()));
  TF_ASSIGN_OR_RETURN(xla::ShapedBuffer shaped_buffer, ToShapedBuffer());
  return transfer_manager->TransferLiteralFromDevice(stream.get(),
                                                     shaped_buffer, literal);
}

Status XRTTupleAllocation::WriteLiteral(xla::Backend* backend,
                                        const xla::Literal& literal) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_16(mht_16_v, 476, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::WriteLiteral");

  if (!xla::ShapeUtil::Equal(literal.shape(), on_host_shape())) {
    return errors::InvalidArgument(
        "New literal shape not matching the existing one: literal=",
        xla::ShapeUtil::HumanStringWithLayout(literal.shape()),
        " device=", xla::ShapeUtil::HumanStringWithLayout(on_host_shape()));
  }
  mutex_lock lock(lock_);
  if (literal_ != nullptr) {
    // The allocation is currently swapped out, and we have a host literal for
    // its content. Just update the host literal with the new value.
    return literal_->CopyFrom(literal);
  }
  TF_ASSIGN_OR_RETURN(xla::ShapedBuffer shaped_buffer, ToShapedBuffer());
  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal()));
  return transfer_manager->TransferLiteralToDevice(stream.get(), literal,
                                                   shaped_buffer);
}

xla::StatusOr<bool> XRTTupleAllocation::SwapOut(xla::Backend* backend,
                                                bool swap_pinned) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_17(mht_17_v, 500, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::SwapOut");

  mutex_lock lock(lock_);
  if (literal_ == nullptr && (!IsPinned() || swap_pinned)) {
    xla::Literal literal(on_host_shape());
    TF_RETURN_IF_ERROR(StoreToLiteral(backend, &literal));
    ReleaseBuffers();
    literal_ = absl::make_unique<xla::Literal>(std::move(literal));
    return true;
  }
  return false;
}

xla::StatusOr<bool> XRTTupleAllocation::SwapIn(
    XRTMemoryManager* memory_manager, xla::Backend* backend,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_18(mht_18_v, 517, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::SwapIn");

  // We need to call AllocateScopedShapedBuffer() outside the locks, since the
  // XRTMemoryManager might end up calling back into the SwapOut() API.
  // So we do a quick check before using the IsSwapped() API, and it can happen
  // that the allocation becomes swapped in after the check. This means which we
  // will end up doing an allocation, and then releasing it soon after (via its
  // scoped variables). This is an unlikely scenario (two threads calling
  // SwapIn() on the same allocation) though.
  if (!IsSwapped()) {
    return false;
  }

  auto transfer_manager = backend->transfer_manager();
  std::unique_ptr<xla::ScopedShapedBuffer> scoped_buffer;
  TF_RETURN_IF_ERROR(
      AllocateScopedShapedBuffer(memory_manager, backend, device_ordinal(),
                                 on_host_shape(), &scoped_buffer, allocator));
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal()));

  mutex_lock lock(lock_);
  if (literal_ != nullptr) {
    TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDevice(
        stream.get(), *literal_, *scoped_buffer));

    auto shaped_buffer = scoped_buffer->release();
    InitializeFromShapedBuffer(shaped_buffer, allocator, device_ordinal());
    literal_ = nullptr;
    return true;
  }
  return false;
}

xla::StatusOr<bool> XRTTupleAllocation::PinAndSwapIn(
    XRTMemoryManager* memory_manager, xla::Backend* backend,
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_19(mht_19_v, 554, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::PinAndSwapIn");

  Pin();
  return SwapIn(memory_manager, backend, allocator);
}

bool XRTTupleAllocation::IsSwapped() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_20(mht_20_v, 562, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::IsSwapped");

  mutex_lock lock(lock_);
  return literal_ != nullptr;
}

int64_t XRTTupleAllocation::Pin() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_21(mht_21_v, 570, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::Pin");
 return pin_count_.fetch_add(1); }

int64_t XRTTupleAllocation::Unpin() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_22(mht_22_v, 575, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::Unpin");
 return pin_count_.fetch_sub(1); }

bool XRTTupleAllocation::IsPinned() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_23(mht_23_v, 580, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::IsPinned");
 return pin_count_ != 0; }

void XRTTupleAllocation::DiscardAllocation(
    const xla::ShapeIndex& buffer_index) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_24(mht_24_v, 586, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::DiscardAllocation");

  buffers_.element(buffer_index)->DiscardAllocation();
}

const xla::Shape& XRTTupleAllocation::on_host_shape() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_25(mht_25_v, 593, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::on_host_shape");

  return on_host_shape_;
}

const xla::Shape& XRTTupleAllocation::on_device_shape() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_26(mht_26_v, 600, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::on_device_shape");

  return on_device_shape_;
}

int XRTTupleAllocation::device_ordinal() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_27(mht_27_v, 607, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::device_ordinal");
 return device_ordinal_; }

const se::DeviceMemoryBase& XRTTupleAllocation::root_allocation() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_28(mht_28_v, 612, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::root_allocation");

  return buffers_.element({})->allocation();
}

/*static*/ Status XRTTupleAllocation::MakeSubBuffer(
    XRTTupleAllocation* parent, const xla::ShapeIndex& subshape,
    XRTTupleAllocation** allocation, bool alias_parent_allocation) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_29(mht_29_v, 621, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::MakeSubBuffer");

  TF_ASSIGN_OR_RETURN(
      const xla::Shape* host_sub_shape,
      xla::ShapeUtil::TryGetSubshape(parent->on_host_shape(), subshape));
  TF_ASSIGN_OR_RETURN(
      const xla::Shape* device_sub_shape,
      xla::ShapeUtil::TryGetSubshape(parent->on_device_shape(), subshape));

  *allocation =
      new XRTTupleAllocation(parent->device_ordinal(), parent->allocator_,
                             *host_sub_shape, *device_sub_shape);
  if (alias_parent_allocation) {
    // Copy the subtree of allocations from the parent allocation.
    (*allocation)->buffers_.CopySubtreeFrom(parent->buffers_, subshape, {});
    // Increment the refcount on each aliased buffer.
    (*allocation)
        ->buffers_.ForEachElement(
            [](const xla::ShapeIndex& index,
               const XRTBufferAllocationPtr& buffer) { buffer->Ref(); });
  } else {
    // Find the buffers in the parent allocation that match the subtree, and
    // move the parent allocation's buffer over to the new allocation.
    (*allocation)
        ->buffers_.ForEachMutableElement(
            [&](const xla::ShapeIndex& index, XRTBufferAllocationPtr* buffer) {
              // Extend the allocation's index to the parent's frame by adding
              // subshape as a prefix.
              xla::ShapeIndex parent_index = subshape;
              for (int i = 0; i < index.size(); ++i) {
                parent_index.push_back(index[i]);
              }
              *buffer = parent->buffers_.element(parent_index);
              *parent->buffers_.mutable_element(parent_index) = nullptr;
            });
  }
  (*allocation)->SetDeviceMemorySize();
  return Status::OK();
}

void XRTTupleAllocation::SetDeviceMemorySize() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_30(mht_30_v, 663, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::SetDeviceMemorySize");

  size_t size = 0;
  for (auto& index_buffer : buffers_) {
    if (index_buffer.second != nullptr) {
      size += index_buffer.second->allocation().size();
    }
  }
  device_memory_size_ = size;
}

/* static */ Status XRTTupleAllocation::ExpandTreeOfTuples(
    const xla::ShapeTree<ExpandedTupleInput>& elements, int device_ordinal,
    se::DeviceMemoryAllocator* allocator, xla::Shape* host_shape,
    xla::Shape* device_shape) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_31(mht_31_v, 679, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::ExpandTreeOfTuples");

  // Initialize both host and device shape to be the 'spine' of the new tuple
  // shape, given by the shape of the tree of tuples.
  *host_shape = elements.shape();
  *device_shape = elements.shape();
  // Now go over the leaves of the tree of tuples, and 'graft' the host/device
  // shapes of the allocation at that leaf onto the expanded host/device shapes
  // at the leaf position.
  TF_RETURN_IF_ERROR(elements.ForEachElementWithStatus(
      [&](const xla::ShapeIndex& index, const ExpandedTupleInput& element) {
        if (elements.IsLeaf(index)) {
          if (element.allocation == nullptr) {
            return errors::InvalidArgument(
                "MakeTuple elements has a null internal node at index ",
                index.ToString());
          }
          if (device_ordinal != element.allocation->device_ordinal() ||
              allocator != element.allocation->allocator_) {
            return errors::InvalidArgument(
                "MakeTuple elements must all be allocated on the same device "
                "as the destination.");
          }
          *xla::ShapeUtil::GetMutableSubshape(host_shape, index) =
              element.allocation->on_host_shape();
          *xla::ShapeUtil::GetMutableSubshape(device_shape, index) =
              element.allocation->on_device_shape();
        } else {
          if (element.allocation != nullptr) {
            return errors::InvalidArgument(
                "MakeTuple elements has a non-null internal node at index ",
                index.ToString());
          }
        }
        return Status::OK();
      }));
  return Status::OK();
}

/*static*/ Status XRTTupleAllocation::MakeTuple(
    XRTMemoryManager* memory_manager, xla::Backend* backend, int device_ordinal,
    const xla::ShapeTree<ExpandedTupleInput>& elements,
    XRTTupleAllocation** allocation, se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_32(mht_32_v, 723, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::MakeTuple");

  auto transfer_manager = backend->transfer_manager();
  TF_ASSIGN_OR_RETURN(auto stream, backend->BorrowStream(device_ordinal));

  xla::Shape host_shape;
  xla::Shape device_shape;
  TF_RETURN_IF_ERROR(ExpandTreeOfTuples(elements, device_ordinal, allocator,
                                        &host_shape, &device_shape));

  // The aliasing is determined below based on whether or not all the inputs are
  // released while being transferred. allocation_tmp is a local pointer that is
  // copied to *allocation at the end only if the method succeeds.
  XRTTupleAllocation* allocation_tmp = new XRTTupleAllocation(
      device_ordinal, allocator, host_shape, device_shape);
  core::ScopedUnref allocation_unref(allocation_tmp);
  // First allocate device memory for the new tuple index tables, one at each
  // internal node of the elements tree. Do this in a separate pass into a
  // ScopedShapedBuffer so that it's easy to free the newly-allocated memory if
  // an allocation fails. Make sure the shape has layout so that the code that
  // writes index tables will be happy lower down.
  xla::Shape spine_shape = elements.shape();
  xla::LayoutUtil::SetToDefaultLayout(&spine_shape);
  auto new_tuple_buffers = absl::make_unique<xla::ScopedShapedBuffer>(
      spine_shape, spine_shape, allocator, device_ordinal);
  TF_RETURN_IF_ERROR(elements.ForEachElementWithStatus(
      [&](const xla::ShapeIndex& index, const ExpandedTupleInput& element) {
        if (!elements.IsLeaf(index)) {
          const xla::Shape& subshape =
              xla::ShapeUtil::GetSubshape(device_shape, index);
          uint64 size = transfer_manager->GetByteSizeRequirement(subshape);
          TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory buffer,
                              memory_manager->Allocate(backend, device_ordinal,
                                                       size, allocator));
          VLOG(2) << "Allocated buffer at " << buffer->opaque() << " index "
                  << index.ToString();
          // Move the new buffer into new_tuple_buffers, which takes ownership
          // of it.
          new_tuple_buffers->set_buffer(std::move(buffer), index);
        }
        return Status::OK();
      }));
  // Transfer from the ScopedShapedBuffer to a ShapedBuffer, which does not own
  // the newly-allocated index tables. Right now there's no owner for the new
  // index tables, so next we will transfer ownership to the new allocation,
  // taking care not to return early on any errors in the meantime.
  xla::ShapedBuffer tuple_buffers = new_tuple_buffers->release();
  // Now fill in the remaining datastructures. After this ForEachElement
  // completes:
  //   1) Every leaf element of tuple_buffers will be the root buffer of
  //      an existing allocation, and every internal element of tuple_buffers
  //      will be a newly-allocated index table. tuple_buffers does not own any
  //      of these.
  //   2) Every element of allocation_tmp->buffers_ will be a correctly
  //   constructed
  //      XRTBufferAllocation wrapping the necessary allocations. For buffers in
  //      existing allocations there will be a new reference owned by the new
  //      allocation, and for newly-allocated index tables there will be a
  //      single reference owned by the new allocation.
  elements.ForEachElement([&](const xla::ShapeIndex& index,
                              const ExpandedTupleInput& element) {
    if (elements.IsLeaf(index)) {
      allocation_tmp->buffers_.CopySubtreeFrom(element.allocation->buffers_, {},
                                               index);
      tuple_buffers.set_buffer(element.allocation->root_allocation(), index);
      if (element.release_allocation_after_use) {
        // Transfer the references from element's buffers to the new allocation
        // rather than incrementing the refcount. The caller should have
        // validated that release_allocation_after_use is false if
        // element.allocation appears in more than one leaf.
        element.allocation->buffers_.ForEachMutableElement(
            [&](const xla::ShapeIndex&, XRTBufferAllocationPtr* buffer) {
              *buffer = nullptr;
            });
      } else {
        // Increment the refcount on each newly-aliased buffer.
        element.allocation->buffers_.ForEachElement(
            [](const xla::ShapeIndex& index,
               const XRTBufferAllocationPtr& buffer) { buffer->Ref(); });
      }
    } else {
      // This is an internal node of the tuple tree so take ownership of the
      // newly-created index table.
      *allocation_tmp->buffers_.mutable_element(index) =
          new XRTBufferAllocation(tuple_buffers.buffer(index), device_ordinal,
                                  allocator);
    }
  });
  allocation_tmp->SetDeviceMemorySize();
  // Because the internal nodes of tuple_buffers are exactly the new index
  // tables, WriteTupleIndexTables will write only the new index tables and not
  // rewrite the index tables for the existing allocations.
  TF_RETURN_IF_ERROR(
      transfer_manager->WriteTupleIndexTables(stream.get(), tuple_buffers));

  *allocation = allocation_tmp;
  // Get another reference since allocation_tmp will be Unrefed automatically on
  // exit.
  (*allocation)->Ref();
  return Status::OK();
}

bool XRTTupleAllocation::IsExclusiveOwner() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_33(mht_33_v, 827, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::IsExclusiveOwner");

  for (const auto& index_buffer : buffers_) {
    if (index_buffer.second != nullptr &&
        !index_buffer.second->RefCountIsOne()) {
      return false;
    }
  }
  return true;
}

size_t XRTTupleAllocation::GetDeviceMemorySize() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_34(mht_34_v, 840, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::GetDeviceMemorySize");

  return device_memory_size_;
}

void XRTTupleAllocation::InitializeFromShapedBuffer(
    const xla::ShapedBuffer& shaped_buffer,
    se::DeviceMemoryAllocator* allocator, int device_ordinal) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_35(mht_35_v, 849, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::InitializeFromShapedBuffer");

  for (auto& index_buffer : buffers_) {
    if (index_buffer.second != nullptr) {
      index_buffer.second->Unref();
    }
    // Make a reference-counted version of the allocated buffer.
    index_buffer.second = new XRTBufferAllocation(
        shaped_buffer.buffer(index_buffer.first), device_ordinal, allocator);
  }
}

xla::StatusOr<xla::ShapedBuffer> XRTTupleAllocation::ToShapedBuffer() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_36(mht_36_v, 863, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::ToShapedBuffer");

  xla::ShapedBuffer shaped_buffer(on_host_shape(), on_device_shape(),
                                  device_ordinal_);
  for (const auto& index_buffer : buffers_) {
    if (index_buffer.second == nullptr ||
        (index_buffer.second->allocation().is_null() &&
         index_buffer.second->allocation().size() > 0)) {
      return errors::InvalidArgument("Literal buffer at index ",
                                     index_buffer.first.ToString(),
                                     " has been released");
    }
    shaped_buffer.set_buffer(index_buffer.second->allocation(),
                             index_buffer.first);
  }
  return std::move(shaped_buffer);
}

Status XRTTupleAllocation::AliasBufferFrom(const XRTTupleAllocation& source,
                                           const xla::ShapeIndex& source_index,
                                           const xla::ShapeIndex& dest_index) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_37(mht_37_v, 885, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::AliasBufferFrom");

  XRTBufferAllocation* source_buffer = source.buffers_.element(source_index);
  XRTBufferAllocation* dest_buffer = buffers_.element(dest_index);
  if (dest_buffer != nullptr) {
    // We allow the destination size being zero, because there are cases where
    // we are coming in later filling in null/uninitialized device buffers. In
    // all other cases, the size of the new buffer must match.
    if (source_buffer->allocation().size() !=
            dest_buffer->allocation().size() &&
        dest_buffer->allocation().size() != 0) {
      return errors::InvalidArgument(
          "Source buffer at index ", source_index.ToString(),
          " does not match the size of destination buffer at index ",
          dest_index.ToString(), ": ", source_buffer->allocation().size(),
          " vs ", dest_buffer->allocation().size());
    }
  } else {
    const xla::Shape& source_subshape =
        xla::ShapeUtil::GetSubshape(source.on_device_shape(), source_index);
    const xla::Shape& dest_subshape =
        xla::ShapeUtil::GetSubshape(on_device_shape(), dest_index);
    if (!xla::ShapeUtil::Equal(source_subshape, dest_subshape)) {
      return errors::InvalidArgument(
          "Source and destination subshapes do not match: source=",
          xla::ShapeUtil::HumanStringWithLayout(source_subshape),
          " dest=", xla::ShapeUtil::HumanStringWithLayout(dest_subshape));
    }
  }
  *buffers_.mutable_element(dest_index) = source_buffer;
  source_buffer->Ref();
  if (dest_buffer != nullptr) {
    // If we handed over the ownership of a buffer in ToExecutionInput(), we
    // will be called here on the way back from execution, to alias back the
    // buffer at that index. In that case the buffers will be the same. So we
    // need to discard the memory at the destination buffer, before releasing
    // the reference.
    if (dest_buffer->allocation().IsSameAs(source_buffer->allocation()) &&
        dest_buffer != source_buffer) {
      dest_buffer->DiscardAllocation();
    }
    dest_buffer->Unref();
  }
  return Status::OK();
}

xla::StatusOr<xla::ExecutionInput> XRTTupleAllocation::ToExecutionInput(
    const std::function<xla::StatusOr<bool>(const xla::ShapeIndex&)>&
        alias_checker) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_stateDTcc mht_38(mht_38_v, 935, "", "./tensorflow/compiler/xrt/xrt_state.cc", "XRTTupleAllocation::ToExecutionInput");

  xla::ExecutionInput result(on_device_shape(), on_host_shape());
  for (const auto& index_buffer : buffers_) {
    if (index_buffer.second == nullptr ||
        (index_buffer.second->allocation().is_null() &&
         index_buffer.second->allocation().size() > 0)) {
      return errors::InvalidArgument("Literal buffer at index ",
                                     index_buffer.first.ToString(),
                                     " has been released");
    }
    TF_ASSIGN_OR_RETURN(bool should_alias, alias_checker(index_buffer.first));
    if (!should_alias) {
      result.SetBuffer(
          index_buffer.first,
          xla::MaybeOwningDeviceMemory(index_buffer.second->allocation()));
    } else {
      // We keep the ownership of the device memory here.
      result.SetUnownedBuffer(
          index_buffer.first,
          xla::MaybeOwningDeviceMemory(se::OwningDeviceMemory(
              index_buffer.second->allocation(), device_ordinal_, allocator_)));
    }
  }
  return std::move(result);
}

}  // namespace tensorflow
