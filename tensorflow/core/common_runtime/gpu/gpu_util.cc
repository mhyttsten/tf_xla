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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc() {
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

#include "tensorflow/core/common_runtime/gpu/gpu_util.h"

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/util.h"

// IMPLEMENTATION NOTE:
//
// 1. Within this module, we intentionally LOG(FATAL) if any stream
//    involved in memcpy becomes !stream->ok(), because TF process
//    today (1/2016) can not properly recover from such an error.
//
// 2. When 0-size tensor is being copied, we should not schedule a
//    copy ThenMemcpy since there is no byte to move. However, we must
//    ensure the causal ordering by arranging the copy done callback
//    happens-after all activities scheduled on the given stream being
//    finished.

// If this need to be runtime configurable, consider adding options to
// ConfigProto.
const int64_t FLAGS_brain_gpu_util_debug_string_maxlen = 128;
extern bool FLAGS_brain_gpu_record_mem_types;

namespace tensorflow {

using se::DeviceMemoryBase;
using se::Stream;

Status PrepareCopy(Device* device, const DeviceContext* ctx, const Tensor& src,
                   const Tensor* dst,
                   const DeviceBase::AcceleratorDeviceInfo** dev_info,
                   se::Stream** stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_0(mht_0_v, 233, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "PrepareCopy");

  if (device == nullptr) {
    return errors::Internal("Unexpected null device.");
  }
  auto di = device->tensorflow_accelerator_device_info();
  if (di == nullptr) {
    return errors::Internal("Unexpected null device info.");
  }
  *dev_info = di;
  if (ctx == nullptr) {
    return errors::Internal("Unexpected null device context.");
  }
  auto gs = static_cast<const GPUDeviceContext*>(ctx)->stream();
  if (gs == nullptr) {
    return errors::Internal("No gpu stream is available.");
  }
  *stream = gs;
  if (dst != nullptr) {
    if (src.dtype() != dst->dtype()) {
      return errors::Internal("Can't copy a tensor of ",
                              DataTypeString(src.dtype()), " into a tensor of ",
                              DataTypeString(dst->dtype()));
    }
    if (src.TotalBytes() != dst->TotalBytes()) {
      return errors::Internal("Can't copy ", src.TotalBytes(),
                              " bytes of a tensor into another with ",
                              dst->TotalBytes(), " bytes buffer.");
    }
    if ((src.TotalBytes() > 0) && !src.IsInitialized()) {
      return errors::Internal("Src tensor is not initialized.");
    }
    if ((dst->TotalBytes() > 0) && !dst->IsInitialized()) {
      return errors::Internal("Dst tensor is not initialized.");
    }
  }
  if (!DMAHelper::CanUseDMA(&src)) {
    return errors::Internal("GPU copy from non-DMA ",
                            DataTypeString(src.dtype()), " tensor");
  }
  return Status::OK();
}

void* GetBase(const Tensor* src) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_1(mht_1_v, 278, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GetBase");

  return const_cast<void*>(DMAHelper::base(src));
}

void* GetBase(Tensor* dst) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_2(mht_2_v, 285, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GetBase");
 return DMAHelper::base(dst); }

/*static*/
void GPUUtil::SetProtoFromGPU(const Tensor& tensor, Device* dev,
                              const DeviceContext* device_context,
                              TensorProto* proto, bool is_dead,
                              StatusCallback done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_3(mht_3_v, 294, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::SetProtoFromGPU");

  VLOG(1) << "SetProtoFromGPU device_context " << device_context;
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(dev, device_context, tensor, nullptr, &dev_info,
                         &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto send_device_to_host_stream =
      static_cast<const GPUDeviceContext*>(device_context)
          ->device_to_host_stream();
  if (send_device_to_host_stream == nullptr) {
    done(errors::Internal("No send gpu copy-out-stream is available."));
    return;
  }
  // Wait for the sender's main stream to make sure the data are available.
  send_device_to_host_stream->ThenWaitFor(send_stream);

  // Tensor values need to be copied from GPU to CPU ram so that
  // we can build the protobuf response for a RecvTensor RPC.
  // "device context" identifies the stream where the _Send op executed.
  proto->set_dtype(tensor.dtype());
  tensor.shape().AsProto(proto->mutable_tensor_shape());

  // Prepare a proto with the right data buf size, and DMA the data
  // over from the GPU buffer.  Note that 0-size tensors do not have a
  // backing buffer.
  Allocator* alloc = nullptr;
  char* buf = nullptr;
  const int64_t total_bytes = is_dead ? 0 : tensor.TotalBytes();
  if (total_bytes > 0) {
    profiler::ScopedAnnotation annotation("SetProtoFromGPU");
    alloc = GPUProcessState::singleton()->GetGpuHostAllocator(0);
    buf = static_cast<char*>(
        alloc->AllocateRaw(Allocator::kAllocatorAlignment, total_bytes));
    if (LogMemory::IsEnabled()) {
      LogMemory::RecordRawAllocation("SetProtoFromGPU",
                                     LogMemory::PROTO_BUFFER_STEP_ID,
                                     total_bytes, buf, alloc);
    }
    void* src_ptr = GetBase(&tensor);
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    send_device_to_host_stream->ThenMemcpy(buf, gpu_src_ptr, total_bytes);
  }
  // Use of tensor may outlive stack scope, so keep a ref.
  TensorReference tensor_ref(tensor);
  dev_info->event_mgr->ThenExecute(
      send_device_to_host_stream, [send_device_to_host_stream, done, proto, buf,
                                   total_bytes, alloc, tensor_ref]() {
        if (!send_device_to_host_stream->ok()) {
          LOG(FATAL) << "SetProtoFromGPU: GPU Memcpy failed";
        }
        tensor_ref.Unref();
        if (total_bytes > 0) {
          port::CopyFromArray(proto->mutable_tensor_content(), buf,
                              total_bytes);
          if (LogMemory::IsEnabled()) {
            LogMemory::RecordRawDeallocation("SetProtoFromGPU",
                                             LogMemory::PROTO_BUFFER_STEP_ID,
                                             buf, alloc, false);
          }
          alloc->DeallocateRaw(buf);
        }
        done(Status::OK());
      });
}

// static
void GPUUtil::DeviceToDeviceCopy(
    DeviceContext* send_dev_context, DeviceContext* recv_dev_context,
    Device* src, Device* dst, AllocatorAttributes src_alloc_attr,
    AllocatorAttributes dst_alloc_attr, const Tensor* input, Tensor* output,
    int dev_to_dev_stream_index, StatusCallback done) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_4(mht_4_v, 372, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::DeviceToDeviceCopy");

  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(src, send_dev_context, *input, output, &dev_info,
                         &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto send_device_to_device_stream =
      static_cast<const GPUDeviceContext*>(send_dev_context)
          ->device_to_device_stream(dev_to_dev_stream_index);
  if (send_device_to_device_stream == nullptr) {
    done(errors::Internal("No send gpu copy-out-stream is available."));
    return;
  }
  // Wait for the main stream on the sender to make sure the result is
  // available.
  send_device_to_device_stream->ThenWaitFor(send_stream);

  const int64_t total_bytes = input->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(input);
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(output);
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
    auto recv_stream =
        static_cast<const GPUDeviceContext*>(recv_dev_context)->stream();
    if (recv_stream == nullptr) {
      done(errors::Internal("No recv gpu stream is available."));
      return;
    }
    // Since we want to use the memory from recv_stream in the
    // send_device_to_device_stream, add a dependency to make sure the memory is
    // truly free.
    // TODO(zhengxq): remove this dependency when we switch to a better way
    // to make sure the memory is free.
    send_device_to_device_stream->ThenWaitFor(recv_stream);

    VLOG(2) << "src_ptr " << src_ptr << " dst_ptr " << dst_ptr;
    send_device_to_device_stream->ThenMemcpy(&gpu_dst_ptr, gpu_src_ptr,
                                             total_bytes);
  }

  // Use of input may outlive stack scope, so keep a ref.
  TensorReference input_ref(*input);
  dev_info->event_mgr->ThenExecute(
      send_device_to_device_stream,
      [done, send_device_to_device_stream, input_ref]() {
        input_ref.Unref();
        if (!send_device_to_device_stream->ok()) {
          LOG(FATAL) << "GPU->GPU Memcpy failed";
        }
        done(Status::OK());
      });
  send_dev_context->MaintainLifetimeOnStream(input,
                                             send_device_to_device_stream);
}

static CopyTensor::Registration register_gpu_gpu_copy(
    DEVICE_GPU, DEVICE_GPU, GPUUtil::DeviceToDeviceCopy);

namespace {

// Returns whether staging is needed based on tensor buffer's memory type.
bool NeedStaging(const Tensor* tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_5(mht_5_v, 440, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "NeedStaging");

  // Only stage data if the host tensor is on pageable memory.
  // So if the memory type is unknown, it will fallback to GPU driver to handle
  // the staging if needed.
  return tensor->GetMemoryType() == AllocatorMemoryType::kHostPageable;
}

}  // namespace

// static
void GPUUtil::CopyGPUTensorToCPU(Device* gpu_device,
                                 const DeviceContext* device_context,
                                 const Tensor* gpu_tensor, Tensor* cpu_tensor,
                                 StatusCallback done) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_6(mht_6_v, 456, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::CopyGPUTensorToCPU");

  VLOG(1) << "CopyGPUTensorToCPU";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(gpu_device, device_context, *gpu_tensor, cpu_tensor,
                         &dev_info, &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto send_device_to_host_stream =
      static_cast<const GPUDeviceContext*>(device_context)
          ->device_to_host_stream();
  if (send_device_to_host_stream == nullptr) {
    done(errors::Internal("No send gpu copy-out-stream is available."));
    return;
  }
  // Wait for the sender's main stream to make sure the data are available.
  send_device_to_host_stream->ThenWaitFor(send_stream);

  const int64_t total_bytes = gpu_tensor->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(gpu_tensor);
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(cpu_tensor);
    send_device_to_host_stream->ThenMemcpy(dst_ptr, gpu_src_ptr, total_bytes);
  }
  // Use of the input may outlive stack scope, so keep a ref.
  TensorReference input_ref(*gpu_tensor);
  dev_info->event_mgr->ThenExecute(
      send_device_to_host_stream,
      [send_device_to_host_stream, done, input_ref]() {
        if (!send_device_to_host_stream->ok()) {
          LOG(FATAL) << "GPU->CPU Memcpy failed";
        }
        input_ref.Unref();
        done(Status::OK());
      });
}

/*  static */
void GPUUtil::CopyCPUTensorToGPU(const Tensor* cpu_tensor,
                                 const DeviceContext* device_context,
                                 Device* gpu_device, Tensor* gpu_tensor,
                                 StatusCallback done, bool sync_dst_compute) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_7(mht_7_v, 504, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::CopyCPUTensorToGPU");

  VLOG(1) << "CopyCPUTensorToGPU";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* recv_stream = nullptr;
  Status s = PrepareCopy(gpu_device, device_context, *cpu_tensor, gpu_tensor,
                         &dev_info, &recv_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto recv_host_to_device_stream =
      static_cast<const GPUDeviceContext*>(device_context)
          ->host_to_device_stream();
  if (recv_host_to_device_stream == nullptr) {
    done(errors::Internal("No send gpu copy-out-stream is available."));
    return;
  }
  // Wait for the recv-stream to make sure the buffer is truly available.
  if (sync_dst_compute) {
    recv_host_to_device_stream->ThenWaitFor(recv_stream);
  }

  const int64_t total_bytes = cpu_tensor->TotalBytes();

  bool do_staging = false;
  void* staging_buffer = nullptr;
  Allocator* host_memory_allocator = device_context->host_memory_allocator();

  // Use of cpu_tensor may outlive stack scope, so keep a ref.
  TensorReference input_ref(*cpu_tensor);

  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    void* src_ptr = GetBase(cpu_tensor);
    void* dst_ptr = GetBase(gpu_tensor);
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);

    if (NeedStaging(cpu_tensor)) {
      if (host_memory_allocator == nullptr) {
        LOG_FIRST_N(WARNING, 1)
            << "No host memory allocator is available to "
               "stage data for CPU->GPU transfer. Staging will be skipped.";
      } else {
        do_staging = true;
      }
    }

    if (do_staging) {
      staging_buffer = host_memory_allocator->AllocateRaw(
          tensorflow::Allocator::kAllocatorAlignment, total_bytes);
      std::memcpy(staging_buffer, src_ptr, total_bytes);
      input_ref.Unref();

      recv_host_to_device_stream->ThenMemcpy(&gpu_dst_ptr, staging_buffer,
                                             total_bytes);
    } else {
      recv_host_to_device_stream->ThenMemcpy(&gpu_dst_ptr, src_ptr,
                                             total_bytes);
    }
  }

  dev_info->event_mgr->ThenExecute(
      recv_host_to_device_stream,
      [recv_host_to_device_stream, done, input_ref, do_staging, staging_buffer,
       host_memory_allocator]() {
        if (do_staging) {
          host_memory_allocator->DeallocateRaw(staging_buffer);
        } else {
          input_ref.Unref();
        }
        if (!recv_host_to_device_stream->ok()) {
          LOG(FATAL) << "CPU->GPU Memcpy failed";
        }
        done(Status::OK());
      });
}

Status GPUUtil::Sync(Device* gpu_device) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_8(mht_8_v, 585, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::Sync");

  VLOG(1) << "GPUUtil::Sync";
  auto* dev_info = gpu_device->tensorflow_accelerator_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo");
  }
  return dev_info->stream->BlockHostUntilDone();
}

Status GPUUtil::SyncAll(Device* gpu_device) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_9(mht_9_v, 597, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::SyncAll");

  VLOG(1) << "GPUUtil::SyncAll";
  auto* dev_info = gpu_device->tensorflow_accelerator_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo");
  }
  if (!dev_info->stream->parent()->SynchronizeAllActivity() ||
      !dev_info->stream->ok()) {
    return errors::Internal("GPU sync failed");
  }
  return Status::OK();
}

string GPUUtil::MemoryDebugString(const Device* device, Tensor* tensor) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_10(mht_10_v, 613, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::MemoryDebugString");

  string ret;
  CHECK(tensor);
  const int64_t num_bytes = std::min<int64_t>(
      FLAGS_brain_gpu_util_debug_string_maxlen, tensor->TotalBytes());
  void* ptr = (num_bytes > 0) ? GetBase(tensor) : nullptr;
  strings::Appendf(&ret, "%p:", ptr);
  if (num_bytes > 0) {
    auto* dev_info = device->tensorflow_accelerator_device_info();
    if (!dev_info) {
      strings::StrAppend(
          &ret, PrintMemory(reinterpret_cast<const char*>(ptr), num_bytes));
    } else {
      string buf;
      buf.resize(num_bytes);
      DeviceMemoryBase gpu_ptr(ptr, num_bytes);
      auto s = dev_info->stream->parent()->SynchronousMemcpyD2H(
          gpu_ptr, num_bytes, &*buf.begin());
      strings::StrAppend(&ret, PrintMemory(&*buf.begin(), num_bytes));
    }
  }
  return ret;
}

// TODO(pbar) Checksum is called from places without a valid device context.
uint64 GPUUtil::Checksum(Device* gpu_device,
                         const DeviceContext* device_context,
                         const Tensor& tensor) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_11(mht_11_v, 643, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::Checksum");

  Tensor copy(tensor.dtype(), tensor.shape());
  Status s;
  Notification n;
  CopyGPUTensorToCPU(gpu_device, device_context, &tensor, &copy,
                     [&s, &n](Status status) {
                       s.Update(status);
                       n.Notify();
                     });
  n.WaitForNotification();
  CHECK(s.ok()) << s;
  return Checksum(copy);
}

uint64 GPUUtil::Checksum(const Tensor& tensor) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_12(mht_12_v, 660, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::Checksum");

  const float* fptr = reinterpret_cast<const float*>(GetBase(&tensor));
  size_t num_bytes = tensor.TotalBytes();
  size_t num_floats = num_bytes / sizeof(float);
  for (size_t i = 0; i < num_floats; ++i) {
    CHECK(!std::isnan(fptr[i])) << " i " << i;
  }
  // TODO(tucker): consider using crc32c instead.
  return Hash64(reinterpret_cast<const char*>(GetBase(&tensor)),
                tensor.TotalBytes(), 0);
}

// static
void GPUUtil::CopyGPUTensorToSameGPU(Device* gpu_device,
                                     const DeviceContext* device_context,
                                     const Tensor* src_gpu_tensor,
                                     Tensor* dst_gpu_tensor,
                                     StatusCallback done) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_utilDTcc mht_13(mht_13_v, 680, "", "./tensorflow/core/common_runtime/gpu/gpu_util.cc", "GPUUtil::CopyGPUTensorToSameGPU");

  VLOG(1) << "CopyGPUTensorToSameGPU";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(gpu_device, device_context, *src_gpu_tensor,
                         dst_gpu_tensor, &dev_info, &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  const int64_t total_bytes = src_gpu_tensor->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(src_gpu_tensor);
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(dst_gpu_tensor);
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
    send_stream->ThenMemcpy(&gpu_dst_ptr, gpu_src_ptr, total_bytes);
  }

  done(Status::OK());
}

}  // namespace tensorflow
