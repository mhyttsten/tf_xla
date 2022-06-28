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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_context.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.h"
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
#include "tensorflow/core/util/util.h"

// IMPLEMENTATION NOTE:
//
// 1. Within this module, we intentionally LOG(FATAL) if any stream
//    involved in memcpy becomes !stream->ok(), because TF process
//    today (3/2021) can not properly recover from such an error.
//
// 2. When 0-size tensor is being copied, we should not schedule a
//    copy ThenMemcpy since there is no byte to move. However, we must
//    ensure the causal ordering by arranging the copy done callback
//    to happen after all activities scheduled on the given stream being
//    finished.

namespace tensorflow {

using se::DeviceMemoryBase;
using se::Stream;

static Status PrepareCopy(Device* device, const DeviceContext* ctx,
                          const Tensor& src, const Tensor* dst,
                          const DeviceBase::AcceleratorDeviceInfo** dev_info,
                          se::Stream** stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc mht_0(mht_0_v, 228, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.cc", "PrepareCopy");

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
  auto device_stream =
      static_cast<const PluggableDeviceContext*>(ctx)->stream();
  if (device_stream == nullptr) {
    return errors::Internal("No PluggableDevice stream is available.");
  }
  *stream = device_stream;
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
    return errors::Internal("PluggableDevice copy from non-DMA",
                            DataTypeString(src.dtype()), " tensor.");
  }
  return Status::OK();
}

static void* GetBase(const Tensor* src) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc mht_1(mht_1_v, 275, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.cc", "GetBase");

  return const_cast<void*>(DMAHelper::base(src));
}

static void* GetBase(Tensor* dst) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc mht_2(mht_2_v, 282, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.cc", "GetBase");
 return DMAHelper::base(dst); }

// static
void PluggableDeviceUtil::DeviceToDeviceCopy(
    DeviceContext* send_dev_context, DeviceContext* recv_dev_context,
    Device* src, Device* dst, AllocatorAttributes src_alloc_attr,
    AllocatorAttributes dst_alloc_attr, const Tensor* input, Tensor* output,
    int dev_to_dev_stream_index, StatusCallback done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc mht_3(mht_3_v, 292, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.cc", "PluggableDeviceUtil::DeviceToDeviceCopy");

  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(src, send_dev_context, *input, output, &dev_info,
                         &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto send_device_to_device_stream =
      static_cast<const PluggableDeviceContext*>(send_dev_context)
          ->device_to_device_stream(dev_to_dev_stream_index);
  if (send_device_to_device_stream == nullptr) {
    done(errors::Internal(
        "No send PluggableDevice copy-out-stream is available."));
    return;
  }
  // Wait for the main stream on the sender to make sure the result is
  // available.
  send_device_to_device_stream->ThenWaitFor(send_stream);

  const int64_t total_bytes = input->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(input);
    DeviceMemoryBase device_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(output);
    DeviceMemoryBase device_dst_ptr(dst_ptr, total_bytes);
    auto recv_stream =
        static_cast<const PluggableDeviceContext*>(recv_dev_context)->stream();
    if (recv_stream == nullptr) {
      done(errors::Internal("No recv PluggableDevice stream is available."));
      return;
    }
    // Since we want to use the memory from recv_stream in the
    // send_device_to_host_stream, add a dependency to make sure the memory is
    // truly free.
    send_device_to_device_stream->ThenWaitFor(recv_stream);

    VLOG(2) << "src_ptr " << src_ptr << " dst_ptr " << dst_ptr;
    send_device_to_device_stream->ThenMemcpy(&device_dst_ptr, device_src_ptr,
                                             total_bytes);
  }
  // Use of input may outlive stack scope, so keep a ref.
  TensorReference input_ref(*input);
  dev_info->event_mgr->ThenExecute(
      send_device_to_device_stream,
      [done, send_device_to_device_stream, input_ref]() {
        input_ref.Unref();
        if (!send_device_to_device_stream->ok()) {
          LOG(FATAL) << "PluggableDevice->PluggableDevice Memcpy "  // Crash OK
                     << "failed.";
        }
        done(Status::OK());
      });
  send_dev_context->MaintainLifetimeOnStream(input,
                                             send_device_to_device_stream);
}

// static
void PluggableDeviceUtil::CopyPluggableDeviceTensorToCPU(
    Device* device, const DeviceContext* device_context,
    const Tensor* device_tensor, Tensor* cpu_tensor, StatusCallback done) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc mht_4(mht_4_v, 357, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.cc", "PluggableDeviceUtil::CopyPluggableDeviceTensorToCPU");

  VLOG(1) << "CopyPluggableDeviceTensorToCPU";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(device, device_context, *device_tensor, cpu_tensor,
                         &dev_info, &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto send_device_to_host_stream =
      static_cast<const PluggableDeviceContext*>(device_context)
          ->device_to_host_stream();
  if (send_device_to_host_stream == nullptr) {
    done(errors::Internal(
        "No send PluggableDevice copy-out-stream is available."));
    return;
  }
  // Wait for the sender's main stream to make sure that the data are available.
  send_device_to_host_stream->ThenWaitFor(send_stream);

  const int64_t total_bytes = device_tensor->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(device_tensor);
    DeviceMemoryBase device_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(cpu_tensor);
    send_device_to_host_stream->ThenMemcpy(dst_ptr, device_src_ptr,
                                           total_bytes);
  }

  // Use of the input may outlive stack scope, so keep a ref.
  TensorReference input_ref(*device_tensor);
  dev_info->event_mgr->ThenExecute(
      send_device_to_host_stream,
      [send_device_to_host_stream, done, input_ref]() {
        if (!send_device_to_host_stream->ok()) {
          LOG(FATAL) << "PluggableDevice->CPU Memcpy failed.";  // Crash OK
        }
        input_ref.Unref();
        done(Status::OK());
      });
}

// static
void PluggableDeviceUtil::CopyCPUTensorToPluggableDevice(
    const Tensor* cpu_tensor, const DeviceContext* device_context,
    Device* device, Tensor* device_tensor, StatusCallback done,
    bool sync_dst_compute) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc mht_5(mht_5_v, 408, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.cc", "PluggableDeviceUtil::CopyCPUTensorToPluggableDevice");

  VLOG(1) << "CopyCPUTensorToPluggableDevice";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* recv_stream = nullptr;
  Status s = PrepareCopy(device, device_context, *cpu_tensor, device_tensor,
                         &dev_info, &recv_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  auto recv_host_to_device_stream =
      static_cast<const PluggableDeviceContext*>(device_context)
          ->host_to_device_stream();
  if (recv_host_to_device_stream == nullptr) {
    done(errors::Internal(
        "No send PluggableDevice copy-out-stream is available."));
    return;
  }
  // Wait for the recv-stream to make sure the buffer is truly available.
  if (sync_dst_compute) {
    recv_host_to_device_stream->ThenWaitFor(recv_stream);
  }
  const int64_t total_bytes = cpu_tensor->TotalBytes();
  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    void* src_ptr = GetBase(cpu_tensor);
    void* dst_ptr = GetBase(device_tensor);
    DeviceMemoryBase device_dst_ptr(dst_ptr, total_bytes);
    recv_host_to_device_stream->ThenMemcpy(&device_dst_ptr, src_ptr,
                                           total_bytes);
  }
  // Use of cpu_tensor may outlive stack scope, so keep a ref.
  TensorReference input_ref(*cpu_tensor);
  dev_info->event_mgr->ThenExecute(
      recv_host_to_device_stream,
      [recv_host_to_device_stream, done, input_ref]() {
        input_ref.Unref();
        if (!recv_host_to_device_stream->ok()) {
          LOG(FATAL) << "CPU->PluggableDevice Memcpy failed.";  // Crash OK
        }
        done(Status::OK());
      });
}

Status PluggableDeviceUtil::Sync(Device* device) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc mht_6(mht_6_v, 456, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.cc", "PluggableDeviceUtil::Sync");

  VLOG(1) << "PluggableDeviceUtil::Sync";
  auto* dev_info = device->tensorflow_accelerator_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo.");
  }
  return dev_info->stream->BlockHostUntilDone();
}

Status PluggableDeviceUtil::SyncAll(Device* device) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc mht_7(mht_7_v, 468, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.cc", "PluggableDeviceUtil::SyncAll");

  VLOG(1) << "PluggableDeviceUtil::SyncAll";
  auto* dev_info = device->tensorflow_accelerator_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo.");
  }
  if (!dev_info->stream->parent()->SynchronizeAllActivity() ||
      !dev_info->stream->ok()) {
    return errors::Internal("PluggableDevice SyncAll failed.");
  }
  return Status::OK();
}

// static
void PluggableDeviceUtil::CopyPluggableDeviceTensorToSameDevice(
    Device* device, const DeviceContext* device_context,
    const Tensor* src_device_tensor, Tensor* dst_device_tensor,
    StatusCallback done) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_utilDTcc mht_8(mht_8_v, 488, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.cc", "PluggableDeviceUtil::CopyPluggableDeviceTensorToSameDevice");

  VLOG(1) << "CopyPluggableDeviceTensorToSameDevice";
  const DeviceBase::AcceleratorDeviceInfo* dev_info = nullptr;
  se::Stream* send_stream = nullptr;
  Status s = PrepareCopy(device, device_context, *src_device_tensor,
                         dst_device_tensor, &dev_info, &send_stream);
  if (!s.ok()) {
    done(s);
    return;
  }

  const int64_t total_bytes = src_device_tensor->TotalBytes();
  if (total_bytes > 0) {
    void* src_ptr = GetBase(src_device_tensor);
    DeviceMemoryBase device_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = GetBase(dst_device_tensor);
    DeviceMemoryBase device_dst_ptr(dst_ptr, total_bytes);
    send_stream->ThenMemcpy(&device_dst_ptr, device_src_ptr, total_bytes);
  }

  done(Status::OK());
}

}  // namespace tensorflow
