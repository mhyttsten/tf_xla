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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_transfer_managerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_transfer_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_transfer_managerDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/gpu_transfer_manager.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "llvm/IR/DataLayout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace xla {
namespace gpu {

// TODO(b/30467474) Once GPU infeed implementation settles, consider
// folding back the cpu and gpu infeed implementations into a generic
// one if possible.
GpuTransferManager::GpuTransferManager(se::Platform::Id id,
                                       unsigned pointer_size)
    : GenericTransferManager(id, pointer_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_transfer_managerDTcc mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/service/gpu/gpu_transfer_manager.cc", "GpuTransferManager::GpuTransferManager");

  StatusOr<se::Platform*> platform =
      se::MultiPlatformManager::PlatformWithId(id);
  if (!platform.ok()) {
    LOG(WARNING)
        << "GpuTransferManager couldn't get StreamExecutor platform for " << id
        << ".  This may lead to failures later.";
    return;
  }

  int64_t num_devices = (*platform)->VisibleDeviceCount();
  pinned_buffer_mutexes_.resize(num_devices);
  pinned_buffers_.resize(num_devices);

  for (int64_t device_id = 0; device_id < num_devices; ++device_id) {
    pinned_buffer_mutexes_[device_id] = absl::make_unique<absl::Mutex>();

    StatusOr<se::StreamExecutor*> executor =
        (*platform)->ExecutorForDevice(device_id);
    if (!executor.ok()) {
      LOG(WARNING)
          << "GpuTransferManager couldn't get StreamExecutor for device "
          << device_id
          << ".  As a result, all dynamic shape copies from that device will "
             "be unpinned (and therefore potentially slow).";
      continue;
    }
    char* pinned_chunk = reinterpret_cast<char*>(
        (*executor)->HostMemoryAllocate(kPinnedChunkBytes));
    static_assert(kPinnedChunkBytes % kPinnedBufferBytes == 0,
                  "assumption of loop below");
    for (char* buf = pinned_chunk; buf < pinned_chunk + kPinnedChunkBytes;
         buf += kPinnedBufferBytes) {
      pinned_buffers_[device_id].push_back(buf);
    }
  }
}

Status GpuTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_transfer_managerDTcc mht_1(mht_1_v, 259, "", "./tensorflow/compiler/xla/service/gpu/gpu_transfer_manager.cc", "GpuTransferManager::TransferLiteralToInfeed");

  return gpu::GetOrCreateInfeedManager(executor)->TransferLiteralToInfeed(
      executor, literal);
}

Status GpuTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, MutableBorrowingLiteral literal) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_transfer_managerDTcc mht_2(mht_2_v, 268, "", "./tensorflow/compiler/xla/service/gpu/gpu_transfer_manager.cc", "GpuTransferManager::TransferLiteralFromOutfeed");

  return gpu::GetOrCreateOutfeedManager(executor)->TransferLiteralFromOutfeed(
      executor, literal);
}

Status GpuTransferManager::ReadDynamicShapes(se::Stream* stream,
                                             ShapedBuffer* device_buffer,
                                             Shape* device_shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_transfer_managerDTcc mht_3(mht_3_v, 278, "", "./tensorflow/compiler/xla/service/gpu/gpu_transfer_manager.cc", "GpuTransferManager::ReadDynamicShapes");

  DCHECK(device_shape->is_dynamic());
  Shape original_device_shape = *device_shape;

  TF_ASSIGN_OR_RETURN(auto compiler,
                      Compiler::GetForPlatform(stream->parent()->platform()));
  auto shape_size_fn = compiler->ShapeSizeBytesFunction();

  // First, figure out which parts of `device_shape` are dynamic and where the
  // dynamic shapes live in GPU memory.  We'll copy the bytes at the
  // DeviceMemoryBase into the Shape*'s dimensions.
  std::vector<std::pair<se::DeviceMemoryBase, Shape*>> copies;

  TF_RETURN_IF_ERROR(device_buffer->buffers().ForEachMutableElementWithStatus(
      [&](const ShapeIndex& index, se::DeviceMemoryBase* buffer) {
        const Shape& buffer_shape =
            ShapeUtil::GetSubshape(*device_shape, index);
        if (buffer_shape.IsTuple()) {
          return Status::OK();
        }
        Shape& device_sub_shape =
            *ShapeUtil::GetMutableSubshape(device_shape, index);
        if (device_sub_shape.is_static()) {
          return Status::OK();
        }

        // Read the dynamic shape metadata from the device stream.  The dynamic
        // shape itself is stored at the end of the buffer.
        Shape buffer_shape_static = ShapeUtil::MakeStaticShape(buffer_shape);
        const int64_t offset = shape_size_fn(buffer_shape_static);
        int64_t metadata_size = shape_size_fn(buffer_shape) - offset;
        if (metadata_size == 0) {
          return InvalidArgument("Dynamic shape metadata size should not be 0");
        }

        auto buffer_8 = se::DeviceMemory<uint8_t>(*buffer);
        auto metadata_buffer =
            stream->parent()->GetSubBuffer(&buffer_8, offset, metadata_size);
        copies.push_back(std::make_pair(metadata_buffer, &device_sub_shape));

        return Status::OK();
      }));

  // Check out pinned memory for each buffer we want to copy.  If there aren't
  // enough pinned buffers available, or if one of our buffers is so big it
  // doesn't fit, allocate an entry for it in fallback_buffers.
  const int device_id = stream->parent()->device_ordinal();
  std::vector<int32_t*> h2d_memcpy_dsts;
  std::vector<void*> checked_out_buffers;
  std::vector<std::unique_ptr<char[]>> fallback_buffers;

  // Return checked-out buffers at the end of this function.
  auto cleanup = tensorflow::gtl::MakeCleanup([&] {
    absl::MutexLock lock(pinned_buffer_mutexes_[device_id].get());
    std::vector<void*>& available_buffers =
        pinned_buffers_[device_id];  // guarded by lock
    available_buffers.insert(available_buffers.end(),
                             checked_out_buffers.begin(),
                             checked_out_buffers.end());
  });

  CHECK_LT(device_id, pinned_buffers_.size());
  {
    absl::MutexLock lock(pinned_buffer_mutexes_[device_id].get());
    std::vector<void*>& available_buffers =
        pinned_buffers_[device_id];  // guarded by lock

    for (const auto& src_dst : copies) {
      se::DeviceMemoryBase src = src_dst.first;
      if (!available_buffers.empty() && src.size() <= kPinnedBufferBytes) {
        void* buf = available_buffers.back();
        available_buffers.pop_back();
        checked_out_buffers.push_back(buf);
        h2d_memcpy_dsts.push_back(reinterpret_cast<int32_t*>(buf));
      } else {
        LOG_FIRST_N(WARNING, 10)
            << "Unable to copy dynamic shape buffer of size " << src.size()
            << " to host using pinned memory.  Falling back to unpinned "
               "memory, which may be slow.";
        fallback_buffers.push_back(absl::make_unique<char[]>(src.size()));
        h2d_memcpy_dsts.push_back(
            reinterpret_cast<int32_t*>(fallback_buffers.back().get()));
      }
    }
  }

  // Copy into the h2d_memcpy_dsts.
  for (int i = 0; i < copies.size(); i++) {
    se::DeviceMemoryBase src = copies[i].first;
    void* dst = h2d_memcpy_dsts[i];
    stream->ThenMemcpy(dst, src, src.size());
  }

  // Wait for all the async copies to complete, then write into device_shape.
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  for (int i = 0; i < copies.size(); i++) {
    Shape* dst_shape = copies[i].second;
    int32_t* dst = h2d_memcpy_dsts[i];
    for (int64_t j = 0; j < dst_shape->rank(); j++) {
      dst_shape->mutable_dimensions()[j] = dst[j];
    }
  }

  device_shape->clear_dynamic_dimensions();
  TF_RET_CHECK(ShapeUtil::DynamicShapeIsCompatible(*device_shape,
                                                   original_device_shape));
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreateNVPTXTransferManager() {
  return absl::make_unique<xla::gpu::GpuTransferManager>(
      /*id=*/stream_executor::cuda::kCudaPlatformId,
      /*pointer_size=*/llvm::DataLayout(xla::gpu::nvptx::DataLayout())
          .getPointerSize(0 /* default address space */));
}

static std::unique_ptr<xla::TransferManager> CreateAMDGPUTransferManager() {
  return absl::make_unique<xla::gpu::GpuTransferManager>(
      /*id=*/stream_executor::rocm::kROCmPlatformId,
      /*pointer_size=*/llvm::DataLayout(xla::gpu::amdgpu::DataLayout())
          .getPointerSize(0 /* default address space */));
}

static bool InitModule() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_transfer_managerDTcc mht_4(mht_4_v, 407, "", "./tensorflow/compiler/xla/service/gpu/gpu_transfer_manager.cc", "InitModule");

  xla::TransferManager::RegisterTransferManager(
      stream_executor::cuda::kCudaPlatformId, &CreateNVPTXTransferManager);
  xla::TransferManager::RegisterTransferManager(
      stream_executor::rocm::kROCmPlatformId, &CreateAMDGPUTransferManager);
  return true;
}
static bool module_initialized = InitModule();
