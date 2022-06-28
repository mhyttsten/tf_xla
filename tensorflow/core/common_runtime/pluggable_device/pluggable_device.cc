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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc() {
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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device.h"

#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <list>
#include <map>
#include <tuple>
#include <vector>

#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_context.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_init.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace tensorflow {

// This factory helps ensure that different PluggableDevice objects that
// refer to the same physical device and stream group id use the same stream
// group object (and therefore the same device streams). This is necessary since
// there is a single memory allocator per device (see
// ProcessState::GetPluggableDeviceAllocator) and allocators must not be shared
// across streams.
// TODO(penpornk): Consider refactoring StreamGroupFactory to
// common_runtime/device.
class PluggableDevice::StreamGroupFactory {
 public:
  // Returns the unique stream group for use with the stream defined by
  // {tf_device_id, stream_group_within_device}, creating it if it does not yet
  // exist.
  // This function is thread safe.
  PluggableDevice::StreamGroup* GetOrCreate(const std::string& device_type,
                                            TfDeviceId tf_device_id,
                                            int stream_group_within_device,
                                            se::StreamExecutor* executor,
                                            const GPUOptions& options) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device_type: \"" + device_type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_0(mht_0_v, 250, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "GetOrCreate");

    mutex_lock guard(lock_);
    StreamGroup* group = &streams_[key_type(device_type, tf_device_id.value(),
                                            stream_group_within_device)];
    if (!group->compute) {
      group->compute = new se::Stream(executor);
      group->compute->Init();
      VLOG(2) << "Created stream[" << stream_group_within_device
              << "] = " << group->compute;

      group->host_to_device = new se::Stream(executor);
      group->host_to_device->Init();
      VLOG(2) << "Created host_to_device_stream[" << stream_group_within_device
              << "] = " << group->host_to_device;

      group->device_to_host = new se::Stream(executor);
      group->device_to_host->Init();
      VLOG(2) << "Created device_to_host_stream[" << stream_group_within_device
              << "] = " << group->device_to_host;

      int num_d2d_streams =
          options.experimental().num_dev_to_dev_copy_streams();
      if (num_d2d_streams == 0) num_d2d_streams = 1;
      if (num_d2d_streams < 1 || num_d2d_streams > 4) {
        LOG(ERROR)
            << "Illegal GPUOptions.experimental.num_dev_to_dev_copy_streams="
            << num_d2d_streams << " set to 1 instead.";
        num_d2d_streams = 1;
      }
      for (int i = 0; i < num_d2d_streams; ++i) {
        se::Stream* stream = new se::Stream(executor);
        stream->Init();
        group->device_to_device.push_back(stream);
        VLOG(2) << "Created device_to_device_stream["
                << stream_group_within_device
                << "] = " << group->device_to_device.back();
      }
    }
    return group;
  }

  // Returns a reference to the StreamGroupFactory singleton. Note that this is
  // never destroyed, so the objects it owns are never deleted.
  static StreamGroupFactory& Global() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_1(mht_1_v, 296, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "Global");

    static StreamGroupFactory* instance = new StreamGroupFactory();
    return *instance;
  }

 private:
  mutex lock_;
  using key_type = std::tuple<std::string, int, int>;
  std::map<key_type, StreamGroup> streams_;

  // StreamGroupFactory cannot be created directly; Call
  // StreamGroupFactory::Global to get the global instance.
  StreamGroupFactory() = default;
  TF_DISALLOW_COPY_AND_ASSIGN(StreamGroupFactory);
};

PluggableDevice::PluggableDevice(
    const SessionOptions& options, const std::string& name,
    const std::string& device_type, const std::string& platform_name,
    Bytes memory_limit, const DeviceLocality& locality, TfDeviceId tf_device_id,
    const std::string& physical_device_desc, Allocator* device_allocator,
    Allocator* cpu_allocator, bool sync_every_op)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, device_type.c_str(), memory_limit,
                               locality, physical_device_desc)),
      device_allocator_(device_allocator),
      cpu_allocator_(cpu_allocator),
      tf_device_id_(tf_device_id),
      platform_name_(platform_name),
      sync_every_op_(sync_every_op) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("device_type: \"" + device_type + "\"");
   mht_2_v.push_back("platform_name: \"" + platform_name + "\"");
   mht_2_v.push_back("physical_device_desc: \"" + physical_device_desc + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_2(mht_2_v, 332, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::PluggableDevice");

  if (options.config.has_gpu_options()) {
    force_gpu_compatible_ = options.config.gpu_options().force_gpu_compatible();
  }
  PluggableDeviceProcessState::singleton(device_type, platform_name)
      ->EnablePluggableDevice();
}

PluggableDevice::~PluggableDevice() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_3(mht_3_v, 343, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::~PluggableDevice");

  delete pluggable_device_info_;
  device_context_->Unref();
}

Status PluggableDevice::Init(const SessionOptions& options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_4(mht_4_v, 351, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::Init");

  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  auto executor_status = DeviceIdUtil::ExecutorForTfDeviceId(
      DeviceType(device_type()), platform, tf_device_id_);
  if (!executor_status.status().ok()) {
    return errors::Internal("Failed to get StreamExecutor for device",
                            tf_device_id_.value());
  }
  executor_ = executor_status.ValueOrDie();

  em_ = EventMgrFactory::Singleton()->GetEventMgr(executor_,
                                                  options.config.gpu_options());

  stream_ = StreamGroupFactory::Global().GetOrCreate(
      device_type(), tf_device_id_, 0, executor_, options.config.gpu_options());
  device_context_ = new PluggableDeviceContext(
      0, stream_->compute, stream_->host_to_device, stream_->device_to_host,
      stream_->device_to_device);
  pluggable_device_info_ = new DeviceBase::AcceleratorDeviceInfo;
  pluggable_device_info_->stream = stream_->compute;
  pluggable_device_info_->default_context = device_context_;
  pluggable_device_info_->event_mgr = em_;
  PlatformDeviceId platform_device_id;
  TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
      DeviceType(device_type()), tf_device_id_, &platform_device_id));
  pluggable_device_info_->gpu_id = platform_device_id.value();
  set_tensorflow_accelerator_device_info(pluggable_device_info_);

  // Whether and how the PluggableDevice uses its own threadpool.
  // This option is experimental. Once we confirm the best setting, we
  // may change the default behavior and completely remove this flag.
  // Default values might change in future releases.
  // Possible values:
  //   * global: PluggableDevice uses threads shared with CPU in the main
  //       compute thread-pool. This is currently the default.
  //   * gpu_private: PluggableDevice uses threads dedicated to this device.
  //   * gpu_shared: All PluggableDevices share a dedicated thread pool.

  // TODO(penpornk): Read the following configurations from a PluggableDevice
  // callback instead of GPU environment variables: TF_GPU_THREAD_MODE,
  // TF_GPU_THREAD_COUNT, TF_FORCE_GPU_ALLOC_GROWTH,
  // TF_ENABLE_GPU_GARBAGE_COLLECTION, and TF_GPU_HOST_MEM_LIMIT_IN_MB.
  string device_thread_mode;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar("TF_GPU_THREAD_MODE", "global",
                                          &device_thread_mode));
  device_thread_mode = absl::AsciiStrToLower(device_thread_mode);
  if (device_thread_mode != "global") {
    int64_t device_thread_count = -1;
    // Default to two threads. One for device compute and another for memory
    // copies.
    TF_RETURN_IF_ERROR(
        ReadInt64FromEnvVar("TF_GPU_THREAD_COUNT", 2, &device_thread_count));
    if (device_thread_mode == "gpu_private") {
      thread_pool_.reset(new thread::ThreadPool(
          options.env, ThreadOptions(),
          strings::StrCat("gpu_private_", tf_device_id_.value()),
          static_cast<int32>(device_thread_count),
          !options.config.experimental().disable_thread_spinning(),
          /*allocator=*/nullptr));
      set_tensorflow_device_thread_pool(thread_pool_.get());
    } else if (device_thread_mode == "gpu_shared") {
      static thread::ThreadPool* thread_pool = new thread::ThreadPool(
          options.env, ThreadOptions(), "gpu_shared",
          static_cast<int32>(device_thread_count),
          !options.config.experimental().disable_thread_spinning(),
          /*allocator=*/nullptr);
      set_tensorflow_device_thread_pool(thread_pool);
    } else {
      string error_message =
          strings::StrCat("Invalid gpu_thread_mode: ", device_thread_mode);
      LOG(WARNING) << error_message;
      return errors::InvalidArgument(error_message);
    }
  }

  return Status::OK();
}

Allocator* PluggableDevice::GetAllocator(AllocatorAttributes attr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_5(mht_5_v, 432, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::GetAllocator");

  DCHECK(cpu_allocator_) << "CPU allocator must be set";
  if (attr.on_host()) {
    if (attr.gpu_compatible() || force_gpu_compatible_) {
      PluggableDeviceProcessState* ps =
          PluggableDeviceProcessState::singleton(device_type(), platform_name_);
      return ps->GetPluggableDeviceHostAllocator(0);
    } else {
      return cpu_allocator_;
    }
  } else {
    return device_allocator_;
  }
}

string PluggableDevice::ComputeOpKernelDebugString(const OpKernel& op_kernel,
                                                   const int stream_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_6(mht_6_v, 451, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::ComputeOpKernelDebugString");

  return strings::StrCat(op_kernel.name(), " op ", op_kernel.type_string(),
                         " on ", platform_name_, tf_device_id_.value(),
                         " stream[", stream_id, "]");
}

void PluggableDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_7(mht_7_v, 460, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::Compute");

  PluggableDeviceContext* pluggable_device_context = device_context_;
  if (context->op_device_context() != nullptr) {
    pluggable_device_context =
        static_cast<PluggableDeviceContext*>(context->op_device_context());
  }
  const auto stream_id = pluggable_device_context->stream_id();

  const bool vlog_1 = VLOG_IS_ON(1);

  if (vlog_1) {
    VLOG(1) << "PluggableDevice::ComputeHelper "
            << ComputeOpKernelDebugString(*op_kernel, stream_id);
  }

  op_kernel->Compute(context);
  if (context->status().ok()) {
    if (sync_every_op_) {
      context->SetStatus(PluggableDeviceUtil::Sync(this));
      if (vlog_1) {
        VLOG(1) << "PluggableDevice::ComputeHelper finished"
                << ComputeOpKernelDebugString(*op_kernel, stream_id);
      }
    } else if (vlog_1) {
      VLOG(1) << "PluggableDevice::ComputeHelper scheduled"
              << ComputeOpKernelDebugString(*op_kernel, stream_id);
    }
  } else {
    if (vlog_1) {
      VLOG(1) << "PluggableDevice::ComputeHelper failed to schedule"
              << ComputeOpKernelDebugString(*op_kernel, stream_id);
    }
  }
}

// Based on the semantics of Device::Sync, this call should wait for
// all streams not just the current one.
Status PluggableDevice::Sync() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_8(mht_8_v, 500, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::Sync");
 return PluggableDeviceUtil::SyncAll(this); }

void PluggableDevice::ComputeAsync(AsyncOpKernel* op_kernel,
                                   OpKernelContext* context,
                                   AsyncOpKernel::DoneCallback done) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_9(mht_9_v, 507, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::ComputeAsync");

  PluggableDeviceContext* device_context = device_context_;
  if (context->op_device_context() != nullptr) {
    device_context =
        static_cast<PluggableDeviceContext*>(context->op_device_context());
  }
  const auto stream_id = device_context->stream_id();

  VLOG(1) << "PluggableDevice::ComputeAsync " << op_kernel->name() << " op "
          << op_kernel->type_string() << " on " << device_type()
          << tf_device_id_ << " stream[" << stream_id << "]";
  op_kernel->ComputeAsync(context, std::move(done));
}

Status PluggableDevice::MaybeCopyTensorToPluggableDevice(
    const AllocatorAttributes& alloc_attrs, const Tensor& from, Tensor* to,
    StatusCallback done) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_10(mht_10_v, 526, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::MaybeCopyTensorToPluggableDevice");

  if (alloc_attrs.on_host()) {
    *to = from;
    done(Status::OK());
    return Status::OK();
  } else {
    if (!DMAHelper::CanUseDMA(&from)) {
      Status err = errors::Internal("PluggableDevice copy from non-DMA ",
                                    DataTypeString(from.dtype()), " tensor");
      done(err);
      return err;
    }
    AllocationAttributes allocation_attr;
    auto* copy = new Tensor(GetAllocator(alloc_attrs), from.dtype(),
                            from.shape(), allocation_attr);

    // If the tensor is not initialized, we likely ran out of memory.
    if (!copy->IsInitialized()) {
      delete copy;
      Status err = errors::ResourceExhausted(
          "OOM when allocating tensor of shape ", from.shape().DebugString(),
          " and type ", DataTypeString(from.dtype()));
      done(err);
      return err;
    }

    auto wrapped_done = [to, copy, done = std::move(done)](const Status& s) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_11(mht_11_v, 555, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "lambda");

      if (s.ok()) {
        *to = std::move(*copy);
      }
      delete copy;
      done(s);
    };

    device_context_->CopyCPUTensorToDevice(
        &from, this, copy, std::move(wrapped_done), false /*sync_dst_compute*/);
    return Status::OK();
  }
}

Status PluggableDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_12(mht_12_v, 574, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::MakeTensorFromProto");

  AllocatorAttributes attr;
  attr.set_on_host(true);
  attr.set_gpu_compatible(true);
  Allocator* host_alloc = GetAllocator(attr);
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(host_alloc, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }

  if (parsed.dtype() == DT_VARIANT) {
    const Variant* from = parsed.flat<Variant>().data();
    int numa_node = attributes().locality().numa_node();
    Tensor copy(cpu_allocator(numa_node), DT_VARIANT, parsed.shape());
    Variant* copy_variant = copy.flat<Variant>().data();

    std::list<Notification> notifications;
    Status copy_status;
    auto copier = [this, &alloc_attrs, &notifications, &copy_status](
                      const Tensor& from, Tensor* to) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_13(mht_13_v, 597, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "lambda");

      // Copier isn't run in a multithreaded environment, so we don't
      // have to worry about the notifications list being modified in parallel.
      notifications.emplace_back();
      Notification& n = *notifications.rbegin();
      return MaybeCopyTensorToPluggableDevice(
          alloc_attrs, from, to, [&n, &copy_status](const Status& s) {
            if (copy_status.ok()) {
              copy_status.Update(s);
            }
            n.Notify();
          });
    };
    Status s;
    for (int64_t ix = 0; ix < parsed.NumElements(); ++ix) {
      s = VariantDeviceCopy(VariantDeviceCopyDirection::HOST_TO_DEVICE,
                            from[ix], &copy_variant[ix], copier);
      if (!s.ok()) {
        break;
      }
    }
    for (auto& n : notifications) {
      n.WaitForNotification();
    }
    if (!s.ok()) {
      return s;
    }
    *tensor = std::move(copy);
    return copy_status;
  } else {
    Notification n;
    Status status;
    TF_RETURN_IF_ERROR(MaybeCopyTensorToPluggableDevice(
        alloc_attrs, parsed, tensor, [&n, &status](const Status& s) {
          status = s;
          n.Notify();
        }));
    n.WaitForNotification();
    return status;
  }
}

void PluggableDevice::CopyTensorInSameDevice(
    const Tensor* input_tensor, Tensor* output_tensor,
    const DeviceContext* device_context, StatusCallback done) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_deviceDTcc mht_14(mht_14_v, 644, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device.cc", "PluggableDevice::CopyTensorInSameDevice");

  PluggableDeviceUtil::CopyPluggableDeviceTensorToSameDevice(
      static_cast<Device*>(this), device_context, input_tensor, output_tensor,
      std::move(done));
}

}  // namespace tensorflow
