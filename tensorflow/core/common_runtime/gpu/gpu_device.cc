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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc() {
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

// TODO(opensource): Use a more generic sounding preprocessor name than
// GOOGLE_CUDA
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#endif

#define EIGEN_USE_GPU

#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <list>
#include <map>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_split.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/local_device.h"
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
#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
#endif
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"

#if !defined(PLATFORM_GOOGLE)
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/cuda_config.h"
#endif
#endif

namespace tensorflow {

#if GOOGLE_CUDA

typedef cudaStream_t gpuStream_t;
typedef cudaDeviceProp gpuDeviceProp_t;
#define EIGEN_GPU_SCRATCH_SIZE (Eigen::kGpuScratchSize)
using se::cuda::ScopedActivateExecutorContext;

#elif TENSORFLOW_USE_ROCM

typedef hipStream_t gpuStream_t;
typedef hipDeviceProp_t gpuDeviceProp_t;
#define EIGEN_GPU_SCRATCH_SIZE (Eigen::kGpuScratchSize)
using se::rocm::ScopedActivateExecutorContext;

#endif

// Eigen Ops directly allocate memory only for temporary buffers used
// during OpKernel::Compute().  The recommended way of allocating such
// memory is via OpKernelContext::allocate_temp().  However, Eigen Ops
// don't have access to OpKernelContext, instead they get access to
// memory directly through the device allocator.  As an Open Source
// project, Eigen assumes allocator semantics similar to those of the
// CUDA or ROCm memory allocator, and may not work correctly due to race
// conditions if used with some other allocator.  For safety, we need
// to delay deallocation calls out of Eigen until all events on the
// corresponding stream have completed.  The following two classes
// serve this purpose in two different compilation environments.

class EigenGpuStreamDevice : public ::Eigen::StreamInterface {
 public:
  EigenGpuStreamDevice()
      : scratch_(nullptr), semaphore_(nullptr), context_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_0(mht_0_v, 291, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "EigenGpuStreamDevice");
}
  ~EigenGpuStreamDevice() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_1(mht_1_v, 295, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "~EigenGpuStreamDevice");
}
  void Reinitialize(OpKernelContext* context, const gpuStream_t* gpu_stream,
                    TfDeviceId tf_device_id, ::tensorflow::Allocator* alloc,
                    char* scratch) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_2(mht_2_v, 302, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "Reinitialize");

    if (LogMemory::IsEnabled()) {
      operation_ = context->op_kernel().name() + "/EigenAllocator";
      step_id_ = context->step_id();
    }
    context_ = context;
    scratch_ = scratch;
    semaphore_ =
        reinterpret_cast<unsigned int*>(scratch + Eigen::kGpuScratchSize);
    stream_ = gpu_stream;
    allocator_ = alloc;
    PlatformDeviceId platform_device_id;
    TF_CHECK_OK(
        GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
    device_prop_ = &Eigen::GetGpuDeviceProperties(platform_device_id.value());
  }

  const gpuStream_t& stream() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_3(mht_3_v, 322, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "stream");
 return *stream_; }
  const gpuDeviceProp_t& deviceProperties() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_4(mht_4_v, 326, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "deviceProperties");

    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_5(mht_5_v, 333, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "allocate");

    void* ret = allocator_->AllocateRaw(32 /* alignment */, num_bytes);
    if (ret == nullptr) {
      if (context_) {
        context_->SetStatus(errors::ResourceExhausted(
            strings::StrCat("Ran out of GPU memory when allocating ", num_bytes,
                            " bytes for ", operation_)));
      } else {
        LOG(FATAL)
            << "EigenAllocator for GPU ran out of memory when allocating "
            << num_bytes << ". See error logs for more detailed info.";
      }
    }
    if (LogMemory::IsEnabled() && ret != nullptr) {
      LogMemory::RecordRawAllocation(operation_, step_id_, num_bytes, ret,
                                     allocator_);
    }
    return ret;
  }
  void deallocate(void* buffer) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_6(mht_6_v, 355, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "deallocate");

    if (LogMemory::IsEnabled() && buffer != nullptr) {
      LogMemory::RecordRawDeallocation(operation_, step_id_, buffer, allocator_,
                                       true);
    }
    AsyncFreeData* afData =
        new AsyncFreeData(allocator_, buffer, operation_, step_id_);
#if GOOGLE_CUDA
    cudaError_t err = cudaStreamAddCallback(*stream_, asyncFree, afData, 0);
    CHECK_EQ(err, cudaSuccess);
#elif TENSORFLOW_USE_ROCM
    hipError_t err = hipStreamAddCallback(*stream_, asyncFree, afData, 0);
    CHECK_EQ(err, hipSuccess);
#endif
  }

  // Return a pointer to a per stream scratchpad of 1024 bytes residing
  // in global memory.
  void* scratchpad() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_7(mht_7_v, 376, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "scratchpad");
 return scratch_; }

  // Return a semaphore. The semaphore is initially initialized to 0, and
  // each kernel using it is responsible for resetting to 0 upon completion
  // to maintain the invariant that the semaphore is always equal to 0 upon
  // each kernel start.
  unsigned int* semaphore() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_8(mht_8_v, 385, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "semaphore");
 return semaphore_; }

 private:
  struct AsyncFreeData {
    AsyncFreeData(::tensorflow::Allocator* a, void* p, const string& o,
                  const int64_t s)
        : allocator_(a), address_(p), operation_(o), step_id_(s) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("o: \"" + o + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_9(mht_9_v, 395, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "AsyncFreeData");
}
    ::tensorflow::Allocator* allocator_;
    void* address_;
    const string operation_;
    const int64_t step_id_;
  };

#if GOOGLE_CUDA
  static void CUDART_CB asyncFree(gpuStream_t stream, cudaError_t status,
                                  void* userData)
#elif TENSORFLOW_USE_ROCM
  static void asyncFree(gpuStream_t stream, hipError_t status, void* userData)
#endif
  {
    AsyncFreeData* data = static_cast<AsyncFreeData*>(userData);
    if (LogMemory::IsEnabled()) {
      LogMemory::RecordRawDeallocation(data->operation_, data->step_id_,
                                       data->address_, data->allocator_, false);
    }
    data->allocator_->DeallocateRaw(data->address_);
    delete data;
  }

  string operation_;
  int64_t step_id_;
  const gpuStream_t* stream_;           // Not owned.
  const gpuDeviceProp_t* device_prop_;  // Not owned.
  ::tensorflow::Allocator* allocator_;  // Not owned.
  mutable char* scratch_;
  mutable unsigned int* semaphore_;
  OpKernelContext* context_;

  TF_DISALLOW_COPY_AND_ASSIGN(EigenGpuStreamDevice);
};

// This factory helps to ensure that different GPU device objects that refer to
// the same physical device and stream group id use the same stream group
// object (and therefore the same CUDA streams). This is necessary since there
// is a single memory allocator per device (see ProcessState::GetGPUAllocator)
// and allocators must not be shared across streams.
class BaseGPUDevice::StreamGroupFactory {
 public:
  // Returns the unique stream group for use with the stream defined by
  // {tf_device_id, stream_group_within_gpu}, creating it if it does not yet
  // exist.
  // This function is thread safe.
  BaseGPUDevice::StreamGroup* GetOrCreate(TfDeviceId tf_device_id,
                                          int stream_group_within_gpu,
                                          se::StreamExecutor* executor,
                                          const GPUOptions& options) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_10(mht_10_v, 447, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GetOrCreate");

    mutex_lock guard(lock_);
    StreamGroup* group =
        &streams_[key_type(tf_device_id.value(), stream_group_within_gpu)];
    if (!group->compute) {
      int priority = GetPriority(tf_device_id.value(), options);
      group->priority = priority;
      group->compute = GetStream(executor, priority);
      group->compute->Init();
      VLOG(2) << "Created stream[" << stream_group_within_gpu
              << "] = " << group->compute << " with priority: " << priority;

#if TENSORFLOW_USE_ROCM
      // ROCm streams are lightweight and will not necessarily trigger device
      // queue init until they are first used. For optimal performance,
      // compute and nccl streams must be immediate siblings.
      group->nccl = GetStream(executor, priority);
      group->nccl->Init();
      VLOG(2) << "Created nccl_stream[" << stream_group_within_gpu
              << "] = " << group->nccl;

      // Force underlying resource creation now.
      group->compute->ThenWaitFor(group->nccl);
      group->nccl->ThenWaitFor(group->compute);
#endif

      group->host_to_device = GetStream(executor, priority);
      group->host_to_device->Init();
      VLOG(2) << "Created host_to_device_stream[" << stream_group_within_gpu
              << "] = " << group->host_to_device;

      group->device_to_host = GetStream(executor, priority);
      group->device_to_host->Init();
      VLOG(2) << "Created device_to_host_stream[" << stream_group_within_gpu
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
        se::Stream* stream = GetStream(executor, priority);
        stream->Init();
        group->device_to_device.push_back(stream);
        VLOG(2) << "Created device_to_device_stream[" << stream_group_within_gpu
                << "] = " << group->device_to_device.back();
      }
    }
    return group;
  }

  // Returns a reference to the StreamGroupFactory singleton. Note that this is
  // never destroyed, so the objects it owns are never deleted.
  static StreamGroupFactory& Global() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_11(mht_11_v, 508, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "Global");

    static StreamGroupFactory* instance = new StreamGroupFactory();
    return *instance;
  }

  // Helper method for unit tests to reset the streams. Never use in production.
  void TestOnlyReset() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_12(mht_12_v, 517, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "TestOnlyReset");

    mutex_lock guard(lock_);
    for (auto& item : streams_) {
      auto& stream = item.second;
      if (stream.compute) {
        delete stream.compute;
        stream.compute = nullptr;
      }
#if TENSORFLOW_USE_ROCM
      if (stream.nccl) {
        delete stream.nccl;
        stream.nccl = nullptr;
      }
#endif
      if (stream.host_to_device) {
        delete stream.host_to_device;
        stream.host_to_device = nullptr;
      }
      if (stream.device_to_host) {
        delete stream.device_to_host;
        stream.device_to_host = nullptr;
      }
      while (!stream.device_to_device.empty()) {
        auto back = stream.device_to_device.back();
        if (back) {
          delete back;
        }
        stream.device_to_device.pop_back();
      }
    }
    streams_.clear();
  }

 private:
  // Returns priority for the given virtual GPU id from the session options.
  // Returns 0 if no virtual devices are specified.
  int GetPriority(int tf_device_id, const GPUOptions& options) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_13(mht_13_v, 556, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GetPriority");

    int id = tf_device_id;
    int i = 0;
    int priority = 0;
    while (i < options.experimental().virtual_devices_size()) {
      const int size =
          options.experimental().virtual_devices().Get(i).priority_size();
      if (id >= size) {
        id -= size;
      } else {
        priority =
            options.experimental().virtual_devices().Get(i).priority().Get(id);
        break;
      }
      i++;
    }
    return priority;
  }

  // Returns a Stream with the underlying GPUStream with the given priority.
  se::Stream* GetStream(se::StreamExecutor* executor, int priority) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_14(mht_14_v, 579, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GetStream");

    auto stream = new se::Stream(executor);
    static_cast<stream_executor::gpu::GpuStream*>(stream->implementation())
        ->SetPriority(priority);
    return stream;
  }

  mutex lock_;
  using key_type = std::tuple<int, int>;
  std::map<key_type, StreamGroup> streams_;

  // StreamGroupFactory cannot be created directly; Call
  // StreamGroupFactory::Global() to get the global instance.
  StreamGroupFactory() = default;
  TF_DISALLOW_COPY_AND_ASSIGN(StreamGroupFactory);
};

BaseGPUDevice::BaseGPUDevice(const SessionOptions& options, const string& name,
                             Bytes memory_limit, const DeviceLocality& locality,
                             TfDeviceId tf_device_id,
                             const string& physical_device_desc,
                             Allocator* gpu_allocator, Allocator* cpu_allocator,
                             bool sync_every_op)
    : LocalDevice(options, Device::BuildDeviceAttributes(name, DEVICE_GPU,
                                                         memory_limit, locality,
                                                         physical_device_desc)),
      gpu_allocator_(gpu_allocator),
      cpu_allocator_(cpu_allocator),
      scoped_allocator_mgr_(new ScopedAllocatorMgr(name)),
      tf_device_id_(tf_device_id),
      sync_every_op_(sync_every_op) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   mht_15_v.push_back("physical_device_desc: \"" + physical_device_desc + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_15(mht_15_v, 614, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::BaseGPUDevice");

  // XLA device IDs for GPUs are arbitrary but must be unique, so we hash device
  // names (which include a replica index even for multi-client).
  set_xla_global_id(Fingerprint32(name) % std::numeric_limits<int32_t>::max());
  GPUProcessState::singleton()->EnableGPUDevice();
}

BaseGPUDevice::~BaseGPUDevice() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_16(mht_16_v, 624, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::~BaseGPUDevice");

  delete gpu_device_info_;
  if (scratch_) gpu_allocator_->DeallocateRaw(scratch_);
  device_context_->Unref();
}

// This should be idempotent if already initialized.
Status BaseGPUDevice::InitScratchBuffers() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_17(mht_17_v, 634, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::InitScratchBuffers");

  mutex_lock l(scratch_init_mutex_);
  if (!scratch_) {
    DCHECK(stream_);
    size_t scratch_buffer_size = Eigen::kGpuScratchSize + sizeof(unsigned int);
    profiler::ScopedMemoryDebugAnnotation op_annotation("ScratchBuffer");
    void* scratch_buffer = gpu_allocator_->AllocateRaw(
        Allocator::kAllocatorAlignment, scratch_buffer_size);
    if (scratch_buffer == nullptr) {
      return errors::FailedPrecondition(
          "Failed to allocate scratch buffer for device ",
          tf_device_id_.value());
    }
    se::DeviceMemory<char> mem(
        se::DeviceMemoryBase(scratch_buffer, scratch_buffer_size));
    TF_RETURN_IF_ERROR(executor_->SynchronousMemZero(
        &mem, Eigen::kGpuScratchSize + sizeof(unsigned int)));
    scratch_ = static_cast<char*>(scratch_buffer);
  }
  return Status::OK();
}

Status BaseGPUDevice::Init(const SessionOptions& options) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_18(mht_18_v, 659, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::Init");

  auto executor_status = DeviceIdUtil::ExecutorForTfDeviceId(
      DEVICE_GPU, GPUMachineManager(), tf_device_id_);
  if (!executor_status.status().ok()) {
    return errors::Internal("Failed to get StreamExecutor for device ",
                            tf_device_id_.value());
  }

  executor_ = executor_status.ValueOrDie();

  stream_ = StreamGroupFactory::Global().GetOrCreate(
      tf_device_id_, 0, executor_, options.config.gpu_options());

  // Get an allocator that allocates pinned memory on host.
  AllocatorAttributes attr;
  attr.set_on_host(true);
  attr.set_gpu_compatible(true);
  Allocator* host_memory_allocator = GetAllocator(attr);

  device_context_ =
      new GPUDeviceContext(0, stream_->compute,
#if TENSORFLOW_USE_ROCM
                           stream_->nccl,
#endif
                           stream_->host_to_device, stream_->device_to_host,
                           stream_->device_to_device, host_memory_allocator);

  em_ = EventMgrFactory::Singleton()->GetEventMgr(executor_,
                                                  options.config.gpu_options());

  GPUKernelTracker::Params tracker_params(
      options.config.gpu_options().experimental().kernel_tracker_max_interval(),
      options.config.gpu_options().experimental().kernel_tracker_max_bytes(),
      options.config.gpu_options().experimental().kernel_tracker_max_pending());
  timestamped_allocator_ =
      options.config.gpu_options().experimental().timestamped_allocator();
  pending_cap_ = tracker_params.max_pending;
  if (timestamped_allocator_ ||
      (tracker_params.max_interval > 0 || tracker_params.max_bytes > 0 ||
       tracker_params.max_pending > 0)) {
    SharedCounter* timing_counter = nullptr;
    if (timestamped_allocator_) {
      // In this case the SharedCounter was already created and set in the
      // associated Allocator, with ownership by GPUProcessState.
      // The GPUKernelTracker will use this SharedCounter, instead of
      // owning its own.
      timing_counter =
          GPUProcessState::singleton()->GPUAllocatorCounter(tf_device_id_);
      DCHECK(timing_counter);
    }
    kernel_tracker_.reset(new GPUKernelTracker(
        tracker_params, Env::Default(), stream_->compute, timing_counter,
        timestamped_allocator_ ? gpu_allocator_ : nullptr, em_));
  }

  gpu_device_info_ = new DeviceBase::AcceleratorDeviceInfo;
  gpu_device_info_->stream = stream_->compute;
  gpu_device_info_->default_context = device_context_;
  gpu_device_info_->event_mgr = em_;
  PlatformDeviceId platform_device_id;
  TF_RETURN_IF_ERROR(
      GpuIdManager::TfToPlatformDeviceId(tf_device_id_, &platform_device_id));
  gpu_device_info_->gpu_id = platform_device_id.value();
  set_tensorflow_accelerator_device_info(gpu_device_info_);

  // Whether and how the GPU device uses its own threadpool.
  // This option is experimental. Once we confirm the best setting, we
  // may change the default behavior and completely remove this flag.
  // Default values might change in future releases.
  // Possible values:
  //   * global: GPU uses threads shared with CPU in the main compute
  //          thread-pool. This is currently the default.
  //   * gpu_private: GPU uses threads dedicated to this device.
  //   * gpu_shared: All GPUs share a dedicated thread pool.
  string gpu_thread_mode;
  TF_RETURN_IF_ERROR(
      ReadStringFromEnvVar("TF_GPU_THREAD_MODE", "global", &gpu_thread_mode));
  gpu_thread_mode = absl::AsciiStrToLower(gpu_thread_mode);
  if (gpu_thread_mode != "global") {
    int64_t gpu_thread_count = -1;
    // Default to two threads. One for device compute and another for memory
    // copies.
    TF_RETURN_IF_ERROR(
        ReadInt64FromEnvVar("TF_GPU_THREAD_COUNT", 2, &gpu_thread_count));
    if (gpu_thread_mode == "gpu_private") {
      // TODO(zhengxq): since these threads only serve a single GPU device,
      //   we should set the device context once for each thread, and avoid
      //   setting them for each kernel.
      // TODO(zhengxq): pin the thread to the same socket of the target GPU.
      thread_pool_.reset(new thread::ThreadPool(
          options.env, ThreadOptions(),
          strings::StrCat("gpu_private_", tf_device_id_.value()),
          static_cast<int32>(gpu_thread_count),
          !options.config.experimental().disable_thread_spinning(),
          /*allocator=*/nullptr));
      set_tensorflow_device_thread_pool(thread_pool_.get());
    } else if (gpu_thread_mode == "gpu_shared") {
      static thread::ThreadPool* thread_pool = new thread::ThreadPool(
          options.env, ThreadOptions(), "gpu_shared",
          static_cast<int32>(gpu_thread_count),
          !options.config.experimental().disable_thread_spinning(),
          /*allocator=*/nullptr);
      set_tensorflow_device_thread_pool(thread_pool);
    } else {
      string error_message =
          strings::StrCat("Invalid gpu_thread_mode: ", gpu_thread_mode);
      LOG(WARNING) << error_message;
      return errors::InvalidArgument(error_message);
    }
  }

  TF_ASSIGN_OR_RETURN(
      node_file_writer_,
      NodeFileWriter::GetNodeFileWriterIfEnabled(name(), env()));
  if (node_file_writer_) {
    LOG(INFO) << "Writing NodeDefs to file: " << node_file_writer_->filename();
  }

  return Status::OK();
}

string BaseGPUDevice::ComputeOpKernelDebugString(const OpKernel& op_kernel,
                                                 const int& stream_id) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_19(mht_19_v, 784, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::ComputeOpKernelDebugString");

  return strings::StrCat(op_kernel.name(), " op ", op_kernel.type_string(),
                         " on GPU ", tf_device_id_.value(), " stream[",
                         stream_id, "]");
}

namespace {
const absl::flat_hash_set<std::string>* GetOpsToLogFromEnv() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_20(mht_20_v, 794, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GetOpsToLogFromEnv");

  auto* result = new absl::flat_hash_set<std::string>;
  const char* env = getenv("TF_GPU_DEBUG_OPS_TO_LOG");
  if (!env) {
    return result;
  }

  std::vector<absl::string_view> ops = absl::StrSplit(env, ',');
  LOG(INFO) << "Will log inputs & outputs from the following ops: ";
  for (absl::string_view op : ops) {
    result->insert(std::string(op));
    LOG(INFO) << "  |" << op << "|";
  }

  return result;
}

bool ShouldLogInputsAndOutputs(OpKernel* op_kernel) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_21(mht_21_v, 814, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "ShouldLogInputsAndOutputs");

  static const absl::flat_hash_set<std::string>& ops_to_log =
      *GetOpsToLogFromEnv();
  return ops_to_log.count(op_kernel->type_string());
}
}  // namespace

Tensor BaseGPUDevice::CopyGpuTensorToHostDebugOnly(const Tensor& gpu_tensor) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_22(mht_22_v, 824, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::CopyGpuTensorToHostDebugOnly");

  Tensor host_tensor(gpu_tensor.dtype(), gpu_tensor.shape());
  CHECK(device_context_->stream()
            ->ThenMemcpy(host_tensor.data(),
                         se::DeviceMemoryBase(gpu_tensor.data(),
                                              gpu_tensor.TotalBytes()),
                         gpu_tensor.TotalBytes())
            .BlockHostUntilDone()
            .ok());
  return host_tensor;
}

void BaseGPUDevice::LogInputs(OpKernel* op_kernel, OpKernelContext* context) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_23(mht_23_v, 839, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::LogInputs");

  LOG(INFO) << "Inputs for " << op_kernel->name() << " (total "
            << context->num_inputs() << "):";
  for (int i = 0; i < context->num_inputs(); i++) {
    if (!context->has_input(i)) {
      LOG(INFO) << "input # " << i << " is absent";
      continue;
    }

    Tensor input = context->input_memory_type(i) == DEVICE_MEMORY
                       ? CopyGpuTensorToHostDebugOnly(context->input(i))
                       : context->input(i);

    LOG(INFO) << "input # " << i;
    LOG(INFO) << input.DebugString(-1);
  }
  LOG(INFO) << "";
}

void BaseGPUDevice::LogOutputs(OpKernel* op_kernel, OpKernelContext* context) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_24(mht_24_v, 861, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::LogOutputs");

  if (!context->status().ok()) {
    LOG(INFO) << op_kernel->name()
              << " failed: " << context->status().error_message();
    return;
  }

  LOG(INFO) << "Outputs for " << op_kernel->name() << " (total "
            << context->num_inputs() << "):";
  for (int i = 0; i < context->num_outputs(); i++) {
    Tensor* output_ptr = context->mutable_output(i);
    if (output_ptr == nullptr) {
      LOG(INFO) << "output # " << i << " is null";
      continue;
    }

    Tensor output = context->output_memory_type(i) == DEVICE_MEMORY
                        ? CopyGpuTensorToHostDebugOnly(*output_ptr)
                        : *output_ptr;

    LOG(INFO) << "output # " << i;
    LOG(INFO) << output.DebugString(-1);
  }
  LOG(INFO) << "";
}

void BaseGPUDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_25(mht_25_v, 890, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::Compute");

  // NOTE(tucker): We need to discriminate between Eigen GPU
  // operations and all others.  If an operation is Eigen
  // implemented (or otherwise tries to launch a GPU kernel
  // directly), we need to establish a stacked-scoped environment
  // that directs it to execute on the proper device.  Otherwise we
  // expect the Op to use StreamExecutor directly and correctly.
  GPUDeviceContext* gpu_device_context = device_context_;
  if (context->op_device_context() != nullptr) {
    gpu_device_context =
        static_cast<GPUDeviceContext*>(context->op_device_context());
  }
  se::Stream* stream = gpu_device_context->stream();
  const auto stream_id = gpu_device_context->stream_id();

  const bool vlog_1 = VLOG_IS_ON(1);

  if (vlog_1) {
    VLOG(1) << "GpuDevice::ComputeHelper "
            << ComputeOpKernelDebugString(*op_kernel, stream_id);
  }

  if (kernel_tracker_.get()) {
    context->set_record_memory_consumption(true);
    if (pending_cap_ > 0) {
      kernel_tracker_->PauseWhilePendingExceeds(pending_cap_);
    }
  }
  ScopedActivateExecutorContext scoped_activation{stream->parent()};
  profiler::ScopedMemoryDebugAnnotation op_annotation(
      op_kernel->name_view().data(), context->step_id());
  bool should_log_inputs_and_outputs = ShouldLogInputsAndOutputs(op_kernel);

  if (should_log_inputs_and_outputs) {
    LogInputs(op_kernel, context);
  }

  op_kernel->Compute(context);

  if (should_log_inputs_and_outputs) {
    LogOutputs(op_kernel, context);
  }

  if (context->status().ok()) {
    if (sync_every_op_) {
      // Note: GPUUtil::Sync() only syncs the default stream.
      // We need to either sync the stream used by this op, or
      // all streams.  Given that this flag is typically used for
      // debugging it makes more sense to sync all GPU activity.
      context->SetStatus(GPUUtil::SyncAll(this));
      if (vlog_1) {
        VLOG(1) << "GpuDevice::ComputeHelper finished "
                << ComputeOpKernelDebugString(*op_kernel, stream_id);
      }
    } else if (vlog_1) {
      VLOG(1) << "GpuDevice::ComputeHelper scheduled "
              << ComputeOpKernelDebugString(*op_kernel, stream_id);
    }
    if (kernel_tracker_) {
      GPUKernelTracker* tracker = kernel_tracker_.get();
      DCHECK(tracker);
      uint64 queued_count = tracker->MaybeQueue(context);
      if (queued_count > 0) {
        em_->ThenExecute(stream, [tracker, queued_count]() {
          tracker->RecordTerminated(queued_count);
        });
      }
    }
    if (node_file_writer_) {
      Status s = node_file_writer_->RecordNodeExecution(op_kernel, context);
      if (!s.ok()) {
        LOG(ERROR) << s;
        context->SetStatus(s);
      }
    }
  } else {
    if (vlog_1) {
      VLOG(1) << "GpuDevice::ComputeHelper failed to schedule "
              << ComputeOpKernelDebugString(*op_kernel, stream_id);
    }
  }
}

Status BaseGPUDevice::Sync() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_26(mht_26_v, 976, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::Sync");

  DCHECK_NE(stream_, nullptr);

  // Device::Sync is supposed to block until all operations queued on the device
  // at the time of the call have completed.  On GPUs, only operations enqueued
  // on the compute stream can remain pending after the (Async)OpKernel that
  // enqueued the operation has completed.  We do use other streams for copies
  // and collectives, but in those cases the (Async)OpKernels themselves block
  // until the queued operation has finished.
  return stream_->compute->BlockHostUntilDone();
}

void BaseGPUDevice::ComputeAsync(AsyncOpKernel* op_kernel,
                                 OpKernelContext* context,
                                 AsyncOpKernel::DoneCallback done) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_27(mht_27_v, 993, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::ComputeAsync");

  GPUDeviceContext* gpu_device_context = device_context_;
  if (context->op_device_context() != nullptr) {
    gpu_device_context =
        static_cast<GPUDeviceContext*>(context->op_device_context());
  }
  se::Stream* stream = gpu_device_context->stream();
  const auto stream_id = gpu_device_context->stream_id();

  VLOG(1) << "GpuDevice::ComputeAsync " << op_kernel->name() << " op "
          << op_kernel->type_string() << " on GPU" << tf_device_id_
          << " stream[" << stream_id << "]";

  bool should_log_inputs_and_outputs = ShouldLogInputsAndOutputs(op_kernel);

  if (should_log_inputs_and_outputs) {
    LogInputs(op_kernel, context);
    AsyncOpKernel::DoneCallback parent_done = done;
    done = [this, parent_done, should_log_inputs_and_outputs, op_kernel,
            context]() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_28(mht_28_v, 1015, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "lambda");

      LogOutputs(op_kernel, context);
      parent_done();
    };
  }

  ScopedActivateExecutorContext scoped_activation{stream->parent()};
  op_kernel->ComputeAsync(context, std::move(done));
}

Status BaseGPUDevice::MaybeCopyTensorToGPU(
    const AllocatorAttributes& alloc_attrs, const Tensor& from, Tensor* to,
    StatusCallback done) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_29(mht_29_v, 1030, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::MaybeCopyTensorToGPU");

  if (alloc_attrs.on_host()) {
    *to = from;
    done(Status::OK());
    return Status::OK();
  } else {
    if (!DMAHelper::CanUseDMA(&from)) {
      Status err = errors::Internal("GPU copy from non-DMA ",
                                    DataTypeString(from.dtype()), " tensor");
      done(err);
      return err;
    }
    AllocationAttributes allocation_attr;
    uint64 safe_alloc_frontier = 0;
    std::function<uint64()> freed_by_func = [this, &safe_alloc_frontier]() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_30(mht_30_v, 1047, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "lambda");

      safe_alloc_frontier = SafeAllocFrontier(safe_alloc_frontier);
      return safe_alloc_frontier;
    };
    if (timestamped_allocator_) {
      allocation_attr.freed_by_func = &freed_by_func;
    }
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
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_31(mht_31_v, 1070, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "lambda");

      if (s.ok()) {
        *to = std::move(*copy);
      }
      delete copy;
      done(s);
    };

    profiler::ScopedAnnotation annotation("MakeTensorFromProto");
    device_context_->CopyCPUTensorToDevice(
        &from, this, copy, std::move(wrapped_done),
        !timestamped_allocator_ /*sync_dst_compute*/);
    return Status::OK();
  }
}

Status BaseGPUDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                          const AllocatorAttributes alloc_attrs,
                                          Tensor* tensor) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_32(mht_32_v, 1091, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::MakeTensorFromProto");

  AllocatorAttributes attr;
  attr.set_on_host(true);
  attr.set_gpu_compatible(true);
  Allocator* host_alloc = GetAllocator(attr);
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(host_alloc, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }

  profiler::ScopedMemoryDebugAnnotation op_annotation(
      "MakeTensorFromProto", "dynamic", parsed.dtype(),
      [&parsed]() { return parsed.shape().DebugString(); });
  if (parsed.dtype() == DT_VARIANT) {
    const Variant* from = parsed.flat<Variant>().data();
    int numa_node = attributes().locality().numa_node();
    Tensor copy(cpu_allocator(numa_node), DT_VARIANT, parsed.shape());
    Variant* copy_variant = copy.flat<Variant>().data();

    std::list<Notification> notifications;
    Status copy_status;
    auto copier = [this, &alloc_attrs, &notifications, &copy_status](
                      const Tensor& from, Tensor* to) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_33(mht_33_v, 1117, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "lambda");

      // Copier isn't run in a multithreaded environment, so we don't
      // have to worry about the notifications list being modified in parallel.
      notifications.emplace_back();
      Notification& n = *notifications.rbegin();
      return MaybeCopyTensorToGPU(alloc_attrs, from, to,
                                  [&n, &copy_status](const Status& s) {
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
    TF_RETURN_IF_ERROR(MaybeCopyTensorToGPU(alloc_attrs, parsed, tensor,
                                            [&n, &status](const Status& s) {
                                              status = s;
                                              n.Notify();
                                            }));
    n.WaitForNotification();
    return status;
  }
}

void BaseGPUDevice::CopyTensorInSameDevice(const Tensor* input_tensor,
                                           Tensor* output_tensor,
                                           const DeviceContext* device_context,
                                           StatusCallback done) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_34(mht_34_v, 1165, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::CopyTensorInSameDevice");

  GPUUtil::CopyGPUTensorToSameGPU(static_cast<Device*>(this), device_context,
                                  input_tensor, output_tensor, std::move(done));
}

namespace {
class ConcretePerOpGpuDevice : public PerOpGpuDevice {
 public:
  ConcretePerOpGpuDevice() : device_(&stream_device_) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_35(mht_35_v, 1176, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "ConcretePerOpGpuDevice");
}

  void Reinitialize(OpKernelContext* context, const gpuStream_t* gpu_stream,
                    TfDeviceId tf_device_id, Allocator* base_allocator,
                    char* scratch) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_36(mht_36_v, 1184, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "Reinitialize");

    stream_device_.Reinitialize(context, gpu_stream, tf_device_id,
                                base_allocator, scratch);
  }

  const Eigen::GpuDevice& device() const override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_37(mht_37_v, 1192, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "device");
 return device_; }

 private:
  EigenGpuStreamDevice stream_device_;
  Eigen::GpuDevice device_;
};

Status VerifyVirtualDeviceSettings(
    const size_t num_gpus_to_use, const GPUOptions& gpu_options,
    const std::vector<PlatformDeviceId>& visible_gpu_order,
    const std::vector<PlatformDeviceId>& valid_platform_device_ids,
    const std::map<int, std::pair<int, int>>& supported_priority_ranges) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_38(mht_38_v, 1206, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "VerifyVirtualDeviceSettings");

  const auto& virtual_devices = gpu_options.experimental().virtual_devices();
  CHECK(!virtual_devices.empty());
  if (gpu_options.per_process_gpu_memory_fraction() > 0) {
    return errors::InvalidArgument(
        "It's invalid to set per_process_gpu_memory_fraction when "
        "virtual_devices is set.");
  }
  if (num_gpus_to_use < virtual_devices.size()) {
    return errors::Unknown(
        "Not enough GPUs to create virtual devices."
        " num_gpus_to_use: ",
        num_gpus_to_use, " #virtual_devices: ", virtual_devices.size());
  }
  if (!gpu_options.visible_device_list().empty() &&
      visible_gpu_order.size() != virtual_devices.size()) {
    return errors::InvalidArgument(
        "The number of GPUs in visible_device_list doesn't match the number "
        "of elements in the virtual_devices list.",
        " #GPUs in visible_device_list: ", visible_gpu_order.size(),
        " virtual_devices.size(): ", virtual_devices.size());
  }
  if (valid_platform_device_ids.size() != virtual_devices.size()) {
    return errors::Unknown(
        "The number of valid GPUs doesn't match the number of elements in "
        "the virtual_devices list.",
        " #valid GPUs: ", valid_platform_device_ids.size(),
        " virtual_devices.size(): ", virtual_devices.size());
  }
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  // Check memory_limt_mb and priority sizes match if priority is non-empty.
  bool priority_exists = !virtual_devices.Get(0).priority().empty();
  for (int i = 0; i < virtual_devices.size(); ++i) {
    const auto& memory_limit_mb = virtual_devices.Get(i).memory_limit_mb();
    const auto& priority = virtual_devices.Get(i).priority();
    // If the priority is empty in the first one then treat this as having no
    // priority set in any of the virtual devices for backward compatibility.
    // Either it's set for all or none.
    if (!priority_exists) {
      if (!priority.empty()) {
        return errors::InvalidArgument(
            "Priority must be set for all virtual devices or none. But the "
            "priority is specified for ",
            i,
            " while previous devices didn't "
            "have any set.");
      }
    }
    if (priority_exists && memory_limit_mb.size() != priority.size()) {
      return errors::InvalidArgument(
          "Number of virtual device priorities specified doesn't "
          "match with number of memory_limit_mb specified for GPU# ",
          i, " memory_limit_mb size: ", memory_limit_mb.size(),
          " and priority size: ", priority.size());
    }
    const int gpu_id = valid_platform_device_ids[i].value();
    auto it = supported_priority_ranges.find(gpu_id);
    if (it == supported_priority_ranges.end()) {
      return errors::Internal(
          "Failed to find supported priority range for GPU"
          " device ",
          gpu_id);
    }
    const std::pair<int, int>& priority_range = it->second;
    for (int p : priority) {
      if (p > priority_range.first || p < priority_range.second) {
        return errors::InvalidArgument(
            "Priority ", p,
            " is outside the range of supported priorities "
            "[",
            priority_range.second, ",", priority_range.first,
            "] for virtual device ", i, " on GPU# ", gpu_id);
      }
    }
  }
#endif

  return Status::OK();
}

int64_t MinSystemMemory(int64_t available_memory, int cc_major) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_39(mht_39_v, 1289, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "MinSystemMemory");

  // We use the following heuristic for now:
  //
  // If the available_memory is < 2GiB, we allocate 225MiB to system memory.
  // Otherwise, depending on the capability version assign
  //  500MiB (for cuda_compute_capability <= 6.x) or
  // 1050MiB (for cuda_compute_capability <= 7.x) or
  // 1536MiB (for cuda_compute_capability >= 8.x)
  int64_t min_system_memory;
  if (available_memory < (1LL << 31)) {
    min_system_memory = 225 * 1024 * 1024;
  } else {
    if (cc_major <= 6) {
      min_system_memory = 500 * 1024 * 1024;
    } else if (cc_major <= 7) {
      min_system_memory = 1050 * 1024 * 1024;
    } else {
      min_system_memory = 1536 * 1024 * 1024;
    }
  }
#if defined(__GNUC__) && defined(__OPTIMIZE__)
// Do nothing
#elif !defined(__GNUC__) && defined(NDEBUG)
// Do nothing
#else
  // Double the amount of available GPU memory in non-opt builds (debug
  // builds in windows); because in non-opt builds more system memory
  // is necessary.
  min_system_memory *= 2;
#endif

#if defined(ANDROID_TEGRA)
  // 1GB system mem for NVIDIA Tegra devices since they use the same mem for
  // RAM and Video RAM
  min_system_memory = 1 << 30;
#endif

  VLOG(5) << "available_memory = " << available_memory;
  VLOG(5) << "min_system_memory = " << min_system_memory;
  return min_system_memory;
}

// Get the memory limit for the virtual device being created on GPU with
// 'platform_device_id', when that virtual device is the only virtual device
// being created on that GPU.
Status SingleVirtualDeviceMemoryLimit(const GPUOptions& gpu_options,
                                      PlatformDeviceId platform_device_id,
                                      int64_t* memory_limit) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_40(mht_40_v, 1339, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "SingleVirtualDeviceMemoryLimit");

  int64_t total_memory = 0;
  int64_t available_memory = 0;
  se::StreamExecutor* se = DeviceIdUtil::ExecutorForPlatformDeviceId(
                               GPUMachineManager(), platform_device_id)
                               .ValueOrDie();
  if (!se->DeviceMemoryUsage(&available_memory, &total_memory)) {
    return errors::Unknown("Failed to query available memory for GPU ",
                           platform_device_id.value());
  }

  int64_t allocated_memory = 0;
  const double per_process_gpu_memory_fraction =
      gpu_options.per_process_gpu_memory_fraction();
  se::CudaComputeCapability cc =
      se->GetDeviceDescription().cuda_compute_capability();
  if ((per_process_gpu_memory_fraction > 1.0 ||
       gpu_options.experimental().use_unified_memory()) &&
      !cc.IsAtLeast(se::CudaComputeCapability::PASCAL_)) {
    return errors::Internal(
        "Unified memory on GPUs with compute capability lower than 6.0 "
        "(pre-Pascal class GPUs) does not support oversubscription.");
  }

  if (per_process_gpu_memory_fraction == 0) {
    allocated_memory = available_memory;
    const int64_t min_system_memory =
        MinSystemMemory(available_memory, cc.major);
    if (min_system_memory < allocated_memory) {
      allocated_memory -= min_system_memory;
    }
  } else {
    allocated_memory = total_memory * per_process_gpu_memory_fraction;
  }

  // Override the excluded memory when TF_DEVICE_MIN_SYS_MEMORY_IN_MB is set.
  const char* force_device_reserved_bytes =
      std::getenv("TF_DEVICE_MIN_SYS_MEMORY_IN_MB");
  if (force_device_reserved_bytes != nullptr &&
      strcmp(force_device_reserved_bytes, "") != 0) {
    int64_t reserved_mb;
    if (!strings::safe_strto64(force_device_reserved_bytes, &reserved_mb) ||
        reserved_mb < 0) {
      LOG(WARNING) << "The requested reserved device memory "
                   << force_device_reserved_bytes
                   << " is invalid. The request will be ignored.";
    } else {
      // Convert MBytes to Bytes.
      int64_t allowable_reserved_memory = reserved_mb * 1024 * 1024;
      // TF_DEVICE_MIN_SYS_MEMORY_IN_MB overrides
      // per_process_gpu_memory_fraction.
      if (allowable_reserved_memory <= available_memory) {
        allocated_memory = available_memory - allowable_reserved_memory;
        VLOG(1) << "Setting the GPU reserved bytes to "
                << strings::HumanReadableNumBytes(allocated_memory)
                << " MBytes";
      } else {
        LOG(WARNING) << "The requested reserved device memory "
                     << strings::HumanReadableNumBytes(
                            allowable_reserved_memory)
                     << " is larger than the available memory of "
                     << strings::HumanReadableNumBytes(available_memory)
                     << ". The request is ignored.";
      }
    }
  }
  *memory_limit = allocated_memory;
  return Status::OK();
}
}  // namespace

void BaseGPUDevice::ReinitializeDevice(OpKernelContext* context,
                                       PerOpGpuDevice* device, int stream_id,
                                       Allocator* allocator) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_41(mht_41_v, 1415, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::ReinitializeDevice");

  ConcretePerOpGpuDevice* concrete_device =
      static_cast<ConcretePerOpGpuDevice*>(device);
  DCHECK(concrete_device);
  DCHECK_EQ(stream_id, 0);
  const gpuStream_t* gpu_stream = reinterpret_cast<const gpuStream_t*>(
      stream_->compute->implementation()->GpuStreamMemberHack());
  concrete_device->Reinitialize(context, gpu_stream, tf_device_id_, allocator,
                                scratch_);
}

PerOpGpuDevice* BaseGPUDevice::MakeGpuDevice() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_42(mht_42_v, 1429, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::MakeGpuDevice");

  return new ConcretePerOpGpuDevice();
}

Status BaseGPUDevice::ReinitializeGpuDevice(OpKernelContext* context,
                                            PerOpGpuDevice* device,
                                            DeviceContext* dc,
                                            Allocator* allocator) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_43(mht_43_v, 1439, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::ReinitializeGpuDevice");

  TF_RETURN_IF_ERROR(InitScratchBuffers());
  if (dc) {
    const GPUDeviceContext* gpu_dc = static_cast<GPUDeviceContext*>(dc);
    const int stream_id = gpu_dc->stream_id();
    VLOG(1) << "  eigen_gpu_device(" << dc << ") => stream[" << stream_id
            << "]";
    CHECK_EQ(stream_id, 0);
    ReinitializeDevice(context, device, stream_id, allocator);
  } else {
    ReinitializeDevice(context, device, 0, allocator);
  }
  return Status::OK();
}

Allocator* BaseGPUDevice::GetScopedAllocator(AllocatorAttributes attr,
                                             int64_t step_id) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_44(mht_44_v, 1458, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::GetScopedAllocator");

  if (attr.scope_id > 0) {
    return scoped_allocator_mgr_->GetContainer(step_id)->GetInstance(
        attr.scope_id);
  }
  LOG(FATAL) << "Unexpected call to BaseGPUDevice::GetScopedAllocator "
             << "attr.scope_id = " << attr.scope_id;
  return gpu_allocator_;
}

const int BaseGPUDeviceFactory::InterconnectMap::kSameDeviceStrength = 1000;
const int BaseGPUDeviceFactory::InterconnectMap::kStreamExecutorStrength = 1;

Status BaseGPUDeviceFactory::CacheDeviceIds() {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_45(mht_45_v, 1474, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDeviceFactory::CacheDeviceIds");

  if (!cached_device_ids_.empty()) {
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(ValidateGPUMachineManager());
  se::Platform* gpu_manager = GPUMachineManager();
  if (gpu_manager == nullptr) {
    return Status::OK();
  }

  int device_count = gpu_manager->VisibleDeviceCount();
  if (device_count <= 0) {
    return Status::OK();
  }

  std::vector<PlatformDeviceId> visible_gpu_order(device_count);
  std::iota(visible_gpu_order.begin(), visible_gpu_order.end(), 0);
  TF_RETURN_IF_ERROR(GetValidDeviceIds(visible_gpu_order, &cached_device_ids_));
  return Status::OK();
}

Status BaseGPUDeviceFactory::ListPhysicalDevices(std::vector<string>* devices) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_46(mht_46_v, 1499, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDeviceFactory::ListPhysicalDevices");

  TF_RETURN_IF_ERROR(CacheDeviceIds());
  for (PlatformDeviceId platform_device_id : cached_device_ids_) {
    const string device_name =
        strings::StrCat("/physical_device:GPU:", platform_device_id.value());
    devices->push_back(device_name);
  }

  return Status::OK();
}

Status BaseGPUDeviceFactory::GetDeviceDetails(
    int device_index, std::unordered_map<string, string>* details) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_47(mht_47_v, 1514, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDeviceFactory::GetDeviceDetails");

  TF_RETURN_IF_ERROR(CacheDeviceIds());

  if (device_index < 0 || device_index > cached_device_ids_.size()) {
    return errors::Internal("Invalid device index: ", device_index);
  }
  PlatformDeviceId platform_device_id = cached_device_ids_[device_index];

  TF_RETURN_IF_ERROR(ValidateGPUMachineManager());
  se::Platform* gpu_manager = GPUMachineManager();
  if (gpu_manager == nullptr) {
    return errors::Internal("Cannot get GPUMachineManager");
  }
  auto desc_status =
      gpu_manager->DescriptionForDevice(platform_device_id.value());
  if (!desc_status.ok()) {
    return desc_status.status();
  }

  auto desc = desc_status.ConsumeValueOrDie();
  (*details)["device_name"] = desc->name();
#if GOOGLE_CUDA
  (*details)["compute_capability"] = desc->cuda_compute_capability().ToString();
#endif  // GOOGLE_CUDA
  return Status::OK();
}

Status BaseGPUDeviceFactory::CreateDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
   std::vector<std::string> mht_48_v;
   mht_48_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_48(mht_48_v, 1547, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDeviceFactory::CreateDevices");

  TF_RETURN_IF_ERROR(ValidateGPUMachineManager());
  se::Platform* gpu_manager = GPUMachineManager();
  if (gpu_manager == nullptr) {
    return Status::OK();
  }
  // If there are no GPUs visible, do nothing.
  if (gpu_manager->VisibleDeviceCount() <= 0) {
    return Status::OK();
  }

  size_t num_gpus_to_use = INT_MAX;
  auto iter = options.config.device_count().find("GPU");
  if (iter != options.config.device_count().end()) {
    num_gpus_to_use = iter->second;
  }
  const auto& gpu_options = options.config.gpu_options();
  std::vector<PlatformDeviceId> visible_gpu_order;
  std::vector<PlatformDeviceId> valid_platform_device_ids;
  // If we aren't going to use any GPUs, don't initialize them.
  // We don't want to call ParseVisibleDeviceList if num_gpus_to_use is 0,
  // because it treats an empty gpu_options.visible_device_list as 'all GPUs
  // are visible'.
  if (num_gpus_to_use > 0) {
    TF_RETURN_IF_ERROR(DeviceIdUtil::ParseVisibleDeviceList(
        gpu_options.visible_device_list(), gpu_manager->VisibleDeviceCount(),
        &visible_gpu_order));
    bool new_gpu_found = false;
    for (int i = 0; i < visible_gpu_order.size(); ++i) {
      int visible_gpu_id = visible_gpu_order[i].value();

      // Only perform this once per visible gpu id.
      if (visible_gpu_initialized_[visible_gpu_id]) {
        continue;
      }

      visible_gpu_initialized_[visible_gpu_id] = true;
      new_gpu_found = true;
    }

    // Checking peering and shows matrix if more than one gpu found.
    if (new_gpu_found && visible_gpu_order.size() > 1) {
      // Enable peer access
      TF_RETURN_IF_ERROR(EnablePeerAccess(visible_gpu_order));
    }

    TF_RETURN_IF_ERROR(
        GetValidDeviceIds(visible_gpu_order, &valid_platform_device_ids));
  }
  if (num_gpus_to_use > valid_platform_device_ids.size()) {
    num_gpus_to_use = valid_platform_device_ids.size();
  }
  std::map<int, std::pair<int, int>> supported_priority_ranges;
  if (!valid_platform_device_ids.empty()) {
    // Save the original device.
    int original_device = 0;
#if GOOGLE_CUDA
    cudaError_t err = cudaGetDevice(&original_device);
    if (err != cudaSuccess) {
      return errors::Internal("cudaGetDevice() failed. Status: ",
                              cudaGetErrorString(err));
    }
#elif TENSORFLOW_USE_ROCM
    hipError_t err = hipGetDevice(&original_device);
    if (err != hipSuccess) {
      return errors::Internal("hipGetDevice() failed. Status: ",
                              hipGetErrorString(err));
    }
#endif

    // Force to implicitly initialize CUDA runtime on each valid GPU before
    // CreateGPUDevice().
    for (PlatformDeviceId platform_device_id : valid_platform_device_ids) {
#if GOOGLE_CUDA
      err = cudaSetDevice(platform_device_id.value());
      if (err != cudaSuccess) {
        return errors::Internal(
            "cudaSetDevice() on GPU:", platform_device_id.value(),
            " failed. Status: ", cudaGetErrorString(err));
      }
      int priority_low, priority_high;
      cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
      if (err != cudaSuccess) {
        return errors::Internal(
            "cudaDeviceGetStreamPriorityRange() on GPU:", original_device,
            " failed. Status: ", cudaGetErrorString(err));
      }
      VLOG(1) << "Cuda stream priority range on GPU(" << original_device
              << "): " << priority_high << "," << priority_low;
      supported_priority_ranges.insert(
          std::make_pair(platform_device_id.value(),
                         std::make_pair(priority_low, priority_high)));
#elif TENSORFLOW_USE_ROCM
      err = hipSetDevice(platform_device_id.value());
      if (err != hipSuccess) {
        return errors::Internal(
            "hipSetDevice() on GPU:", platform_device_id.value(),
            " failed. Status: ", hipGetErrorString(err));
      }
      err = hipFree(nullptr);
      if (err != hipSuccess) {
        return errors::Internal("ROCm runtime implicit initialization on GPU:",
                                platform_device_id.value(),
                                " failed. Status: ", hipGetErrorString(err));
      }
      int priority_low, priority_high;
      hipDeviceGetStreamPriorityRange(&priority_low, &priority_high);
      if (err != hipSuccess) {
        return errors::Internal(
            "hipDeviceGetStreamPriorityRange() on GPU:", original_device,
            " failed. Status: ", hipGetErrorString(err));
      }
      VLOG(1) << "HIP stream priority range on GPU(" << original_device
              << "): " << priority_high << "," << priority_low;
      supported_priority_ranges.insert(
          std::make_pair(platform_device_id.value(),
                         std::make_pair(priority_low, priority_high)));
#endif
    }
    // Reset to the original device.
#if GOOGLE_CUDA
    err = cudaSetDevice(original_device);
    if (err != cudaSuccess) {
      return errors::Internal("cudaSetDevice() on GPU:", original_device,
                              " failed. Status: ", cudaGetErrorString(err));
    }
#elif TENSORFLOW_USE_ROCM
    err = hipSetDevice(original_device);
    if (err != hipSuccess) {
      return errors::Internal("hipSetDevice() on GPU:", original_device,
                              " failed. Status: ", hipGetErrorString(err));
    }
#endif

#if GOOGLE_CUDA
    // Log the version of CUDA and cuDNN
    int cuda_major_version = CUDART_VERSION / 1000;
    int cuda_minor_version = (CUDART_VERSION / 10) % 10;
    VLOG(1) << "TensorFlow compiled with CUDA " << cuda_major_version << "."
            << cuda_minor_version << " and cuDNN " << CUDNN_MAJOR << "."
            << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL;
#endif
  }

  std::vector<InterconnectMap> interconnect_maps;
  TF_RETURN_IF_ERROR(
      GetInterconnectMaps(visible_gpu_order, gpu_manager, &interconnect_maps));

  // Print each interconnect map to the log.
  for (const InterconnectMap& im : interconnect_maps) {
    VLOG(1) << "Device interconnect " << im.name << " with strength "
            << im.strength << " edge matrix:";
    string line_buf = "     ";
    for (int i = 0; i < visible_gpu_order.size(); ++i) {
      strings::StrAppend(&line_buf, visible_gpu_order[i].value(), " ");
    }
    VLOG(1) << line_buf;
    for (int i = 0; i < visible_gpu_order.size(); ++i) {
      line_buf = strings::StrCat(visible_gpu_order[i].value(), ":   ");
      PlatformDeviceId gpu_id_i = visible_gpu_order[i];
      for (int j = 0; j < visible_gpu_order.size(); ++j) {
        PlatformDeviceId gpu_id_j = visible_gpu_order[j];
        if (im.directed_links.find({gpu_id_i, gpu_id_j}) !=
            im.directed_links.end()) {
          line_buf.append("Y ");
        } else {
          line_buf.append("N ");
        }
      }
      VLOG(1) << line_buf;
    }
  }

  const auto& virtual_devices = gpu_options.experimental().virtual_devices();
  if (!virtual_devices.empty()) {
    TF_RETURN_IF_ERROR(VerifyVirtualDeviceSettings(
        num_gpus_to_use, gpu_options, visible_gpu_order,
        valid_platform_device_ids, supported_priority_ranges));
    // We've verified that num_gpus_to_use >= virtual_devices.size().
    num_gpus_to_use = virtual_devices.size();
    CHECK(gpu_options.visible_device_list().empty() ||
          valid_platform_device_ids == visible_gpu_order);
  }
  int next_tf_device_id = 0;
  std::vector<int64_t> memory_limit_bytes;
  for (int i = 0; i < num_gpus_to_use; ++i) {
    const PlatformDeviceId platform_device_id = valid_platform_device_ids[i];
    if (virtual_devices.empty() ||
        virtual_devices.Get(i).memory_limit_mb_size() == 0) {
      int64_t single_virtual_device_memory_limit = 0;
      TF_RETURN_IF_ERROR(
          SingleVirtualDeviceMemoryLimit(gpu_options, platform_device_id,
                                         &single_virtual_device_memory_limit));
      memory_limit_bytes.push_back(single_virtual_device_memory_limit);
    } else {
      const auto& memory_limit_mb = virtual_devices.Get(i).memory_limit_mb();
      std::transform(memory_limit_mb.begin(), memory_limit_mb.end(),
                     std::back_inserter(memory_limit_bytes), [](float mb) {
                       return static_cast<int64_t>(mb) * (1ll << 20);
                     });
    }
    while (next_tf_device_id < memory_limit_bytes.size()) {
      TfDeviceId tf_device_id(next_tf_device_id);
      ++next_tf_device_id;
      TF_RETURN_IF_ERROR(GpuIdManager::InsertTfPlatformDeviceIdPair(
          tf_device_id, platform_device_id));
    }
  }
  const int num_tf_gpus = next_tf_device_id;

  LocalityMap device_localities;
  TF_RETURN_IF_ERROR(
      GetDeviceLocalities(num_tf_gpus, interconnect_maps, &device_localities));

  // Build the GPUDevices
  CHECK_EQ(next_tf_device_id, memory_limit_bytes.size());
  for (int di = 0; di < num_tf_gpus; ++di) {
    TfDeviceId tf_device_id(di);
    int64_t bytes = memory_limit_bytes[di];
    auto it = device_localities.find(tf_device_id);
    if (it == device_localities.end()) {
      return errors::Internal("Failed to find DeviceLocality for GPU device ",
                              tf_device_id.value());
    }
    TF_RETURN_IF_ERROR(CreateGPUDevice(options, name_prefix, tf_device_id,
                                       bytes, it->second, num_tf_gpus,
                                       devices));
  }
  return Status::OK();
}

static string GetShortDeviceDescription(PlatformDeviceId platform_device_id,
                                        const se::DeviceDescription& desc) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_49(mht_49_v, 1782, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GetShortDeviceDescription");

#if GOOGLE_CUDA
  // LINT.IfChange
  return strings::StrCat(
      "device: ", platform_device_id.value(), ", name: ", desc.name(),
      ", pci bus id: ", desc.pci_bus_id(),
      ", compute capability: ", desc.cuda_compute_capability().ToString());
  // LINT.ThenChange(//tensorflow/python/framework/gpu_util.py)
#elif TENSORFLOW_USE_ROCM
  return strings::StrCat("device: ", platform_device_id.value(),
                         ", name: ", desc.name(),
                         ", pci bus id: ", desc.pci_bus_id());
#endif
}

Status BaseGPUDeviceFactory::CreateGPUDevice(
    const SessionOptions& options, const string& name_prefix,
    TfDeviceId tf_device_id, int64_t memory_limit,
    const DeviceLocality& dev_locality, size_t num_tf_gpus,
    std::vector<std::unique_ptr<Device>>* devices) {
   std::vector<std::string> mht_50_v;
   mht_50_v.push_back("name_prefix: \"" + name_prefix + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_50(mht_50_v, 1805, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDeviceFactory::CreateGPUDevice");

  CHECK_GE(tf_device_id.value(), 0);
  const string device_name =
      strings::StrCat(name_prefix, "/device:GPU:", tf_device_id.value());
  DeviceIdUtil::CheckValidTfDeviceId(DEVICE_GPU, GPUMachineManager(),
                                     tf_device_id);
  PlatformDeviceId platform_device_id;
  TF_RETURN_IF_ERROR(
      GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
  int numa_node = dev_locality.numa_node();

  se::Platform* gpu_manager = GPUMachineManager();
  auto desc_status =
      gpu_manager->DescriptionForDevice(platform_device_id.value());
  if (!desc_status.ok()) {
    return desc_status.status();
  }
  auto desc = desc_status.ConsumeValueOrDie();

  std::vector<TfDeviceId> peer_gpu_ids;
  peer_gpu_ids.reserve(num_tf_gpus);
  for (int id = 0; id < num_tf_gpus; ++id) {
    TfDeviceId peer_tf_device_id(id);
    if (peer_tf_device_id != tf_device_id) {
      peer_gpu_ids.push_back(peer_tf_device_id);
    }
  }

  GPUProcessState* process_state = GPUProcessState::singleton();
  Allocator* gpu_allocator = process_state->GetGPUAllocator(
      options.config.gpu_options(), tf_device_id, memory_limit, peer_gpu_ids);
  if (gpu_allocator == nullptr) {
    return errors::Internal("Failed to get memory allocator for TF GPU ",
                            tf_device_id.value(), " with ", memory_limit,
                            " bytes of memory.");
  }
  absl::optional<AllocatorStats> stats = gpu_allocator->GetStats();
  if (!stats) {
    return errors::Internal("No allocator statistics");
  }
  // 'memory_limit' is the required memory size, but if the allocator with
  // given tf_device_id was created before, we'll use it instead of creating a
  // new one (as TF gpu device is a shared resource), in which case the actual
  // memory limit represented by 'stats.bytes_limit' used by that allocator
  // may be different (which should be an error).
  //
  // TODO(laigd): report error if memory_limit doesn't match
  // stats->bytes_limit.
  int64_t bytes_limit = stats->bytes_limit ? *stats->bytes_limit : 0;
  std::unique_ptr<BaseGPUDevice> gpu_device = CreateGPUDevice(
      options, device_name, static_cast<Bytes>(bytes_limit), dev_locality,
      tf_device_id, GetShortDeviceDescription(platform_device_id, *desc),
      gpu_allocator, ProcessState::singleton()->GetCPUAllocator(numa_node));
  LOG(INFO) << "Created device " << device_name << " with "
            << (bytes_limit >> 20) << " MB memory: "
            << " -> " << GetShortDeviceDescription(platform_device_id, *desc);
  TF_RETURN_IF_ERROR(gpu_device->Init(options));
  gpu_allocator->SetStreamAndPreallocateMemory(gpu_device->GetStream());
  devices->push_back(std::move(gpu_device));

  return Status::OK();
}

namespace {
std::unique_ptr<std::map<std::pair<PlatformDeviceId, PlatformDeviceId>, bool>>
GetPeerAccessMap(se::Platform* platform,
                 const std::vector<PlatformDeviceId>& visible_gpu_order) {
  std::unique_ptr<std::map<std::pair<PlatformDeviceId, PlatformDeviceId>, bool>>
      map(new std::map<std::pair<PlatformDeviceId, PlatformDeviceId>, bool>);
  for (PlatformDeviceId platform_gpu_i : visible_gpu_order) {
    for (PlatformDeviceId platform_gpu_j : visible_gpu_order) {
      se::StreamExecutor* from =
          DeviceIdUtil::ExecutorForPlatformDeviceId(platform, platform_gpu_i)
              .ValueOrDie();
      se::StreamExecutor* to =
          DeviceIdUtil::ExecutorForPlatformDeviceId(platform, platform_gpu_j)
              .ValueOrDie();
      (*map)[{platform_gpu_i, platform_gpu_j}] =
          from->CanEnablePeerAccessTo(to);
    }
  }

  return map;
}

}  // namespace

Status BaseGPUDeviceFactory::GetInterconnectMaps(
    const std::vector<PlatformDeviceId>& visible_gpu_order,
    se::Platform* gpu_manager, std::vector<InterconnectMap>* maps) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_51(mht_51_v, 1897, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDeviceFactory::GetInterconnectMaps");

  // The default interconnect map is obtained from the StreamExecutor.
  auto access_map = GetPeerAccessMap(gpu_manager, visible_gpu_order);
  maps->resize(1);
  InterconnectMap& imap = maps->at(0);
  imap.name = "StreamExecutor";
  imap.strength = InterconnectMap::kStreamExecutorStrength;
  for (PlatformDeviceId gpu_id_i : visible_gpu_order) {
    for (PlatformDeviceId gpu_id_j : visible_gpu_order) {
      if (gpu_id_i == gpu_id_j) continue;
      if ((*access_map)[{gpu_id_i, gpu_id_j}]) {
        imap.directed_links.insert({gpu_id_i, gpu_id_j});
      }
    }
  }
  return Status::OK();
}

Status BaseGPUDeviceFactory::GetDeviceLocalities(
    int num_tf_gpus, const std::vector<InterconnectMap>& interconnects,
    LocalityMap* localities) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_52(mht_52_v, 1920, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDeviceFactory::GetDeviceLocalities");

  std::vector<TfDeviceId> all_tf_device_ids;
  all_tf_device_ids.reserve(num_tf_gpus);
  for (int i = 0; i < num_tf_gpus; ++i) {
    all_tf_device_ids.push_back(TfDeviceId(i));
  }
  for (TfDeviceId tf_device_id : all_tf_device_ids) {
    PlatformDeviceId platform_device_id;
    TF_RETURN_IF_ERROR(
        GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id));
    // Get GPU bus_id from its reported NUMA affinity.  Because GPUs are
    // virtualized in some environments, we can't just use the GPU id.
    // NUMA locales are indexed from 0, buses are indexed from 1.
    se::Platform* gpu_manager = GPUMachineManager();
    auto desc_status =
        gpu_manager->DescriptionForDevice(platform_device_id.value());
    if (!desc_status.ok()) {
      return desc_status.status();
    }
    auto desc = desc_status.ConsumeValueOrDie();
    int numa_node = desc->numa_node();
    if (numa_node < 0) {
      // For some reason the StreamExecutor couldn't get the NUMA
      // affinity of the GPU.  If this is not a multi-socket mobo with
      // GPUs local to different buses, it doesn't matter.  If it is, we
      // may run into trouble later with data transfer operations.  The
      // trouble may manifest as slower than expected performance, or
      // outright failures.
      LOG(INFO) << "Could not identify NUMA node of platform GPU id "
                << platform_device_id
                << ", defaulting to 0.  Your kernel may not have been built "
                << "with NUMA support.";
      numa_node = 0;
    }
    DeviceLocality dev_locality;
    dev_locality.set_numa_node(numa_node);
    dev_locality.set_bus_id(numa_node + 1);

    // Set LocalLinks from InterconnectMaps.
    LocalLinks* links = dev_locality.mutable_links();
    for (const InterconnectMap& imap : interconnects) {
      for (TfDeviceId tf_gpu_dst : all_tf_device_ids) {
        PlatformDeviceId platform_gpu_dst;
        TF_RETURN_IF_ERROR(
            GpuIdManager::TfToPlatformDeviceId(tf_gpu_dst, &platform_gpu_dst));
        if (imap.directed_links.find({platform_device_id, platform_gpu_dst}) !=
            imap.directed_links.end()) {
          InterconnectLink* ilink = links->add_link();
          ilink->set_device_id(tf_gpu_dst.value());
          ilink->set_type(imap.name);
          ilink->set_strength(imap.strength);
        }
      }
    }

    // If this is one of multiple virtual GPUs on the same physical GPU
    // add high strength links to the others.
    for (TfDeviceId tf_gpu_dst : all_tf_device_ids) {
      if (tf_device_id == tf_gpu_dst) continue;
      PlatformDeviceId platform_gpu_dst;
      TF_RETURN_IF_ERROR(
          GpuIdManager::TfToPlatformDeviceId(tf_gpu_dst, &platform_gpu_dst));
      if (platform_device_id == platform_gpu_dst) {
        InterconnectLink* ilink = links->add_link();
        ilink->set_device_id(tf_gpu_dst.value());
        ilink->set_type("SAME_DEVICE");
        ilink->set_strength(InterconnectMap::kSameDeviceStrength);
      }
    }

    (*localities)[tf_device_id] = dev_locality;
    VLOG(1) << "GPUDevice PlatformDeviceId " << platform_device_id
            << " TfDeviceId " << tf_device_id << " on bus "
            << dev_locality.bus_id() << " numa: " << numa_node
            << " pci: " << desc->pci_bus_id()
            << " DeviceLocality: " << dev_locality.DebugString();
  }
  return Status::OK();
}

static int GetDefaultMinGPUMultiprocessorCount(
    se::Platform* gpu_manager,
    const std::vector<PlatformDeviceId>& visible_gpu_order) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_53(mht_53_v, 2005, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GetDefaultMinGPUMultiprocessorCount");

  static const int kDefaultMinGPUMultiprocessorCount = 8;

  // Find the highest multi-processor count across all visible GPUs.
  int max_count = -1;
  for (int i = 0; i < visible_gpu_order.size(); ++i) {
    int visible_gpu_id = visible_gpu_order[i].value();
    auto description_status = gpu_manager->DescriptionForDevice(visible_gpu_id);
    if (!description_status.ok()) {
      continue;
    }

    auto description = description_status.ConsumeValueOrDie();
    max_count = std::max(max_count, description->core_count());
  }

  if (max_count < 0 || kDefaultMinGPUMultiprocessorCount < max_count) {
    return kDefaultMinGPUMultiprocessorCount;
  } else {
    return max_count;
  }
}

static int GetMinGPUMultiprocessorCount(
    se::Platform* gpu_manager,
    const std::vector<PlatformDeviceId>& visible_gpu_order) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_54(mht_54_v, 2033, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GetMinGPUMultiprocessorCount");

  const char* tf_min_gpu_core_count = getenv("TF_MIN_GPU_MULTIPROCESSOR_COUNT");

  if (tf_min_gpu_core_count == nullptr ||
      strcmp(tf_min_gpu_core_count, "") == 0) {
    return GetDefaultMinGPUMultiprocessorCount(gpu_manager, visible_gpu_order);
  }

  int min_gpu_core_count = -1;
  if (strings::safe_strto32(tf_min_gpu_core_count, &min_gpu_core_count)) {
    if (min_gpu_core_count >= 0) {
      return min_gpu_core_count;
    }
  }

  int count =
      GetDefaultMinGPUMultiprocessorCount(gpu_manager, visible_gpu_order);
  LOG(ERROR) << "Invalid minimum GPU multiprocessor count: ["
             << tf_min_gpu_core_count << "]. "
             << "Using the default value: " << count;
  return count;
}

namespace {

#if GOOGLE_CUDA

se::CudaComputeCapability ComputeCapabilityFromString(
    const std::string& version_name) {
   std::vector<std::string> mht_55_v;
   mht_55_v.push_back("version_name: \"" + version_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_55(mht_55_v, 2065, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "ComputeCapabilityFromString");

  int major_part, minor_part;
  size_t dot_pos = version_name.find('.');
  CHECK(dot_pos != string::npos)
      << "Illegal version name: [" << version_name << "]";
  string major_str = version_name.substr(0, dot_pos);
  CHECK(strings::safe_strto32(major_str, &major_part))
      << "Illegal version name: [" << version_name << "]";
  string minor_str = version_name.substr(dot_pos + 1);
  CHECK(strings::safe_strto32(minor_str, &minor_part))
      << "Illegal version name: [" << version_name << "]";
  return se::CudaComputeCapability{major_part, minor_part};
}

std::vector<se::CudaComputeCapability> GetSupportedCudaComputeCapabilities() {
  std::vector<se::CudaComputeCapability> cuda_caps = {
      ComputeCapabilityFromString("3.5"), ComputeCapabilityFromString("5.2")};
#ifdef TF_EXTRA_CUDA_CAPABILITIES
// TF_EXTRA_CUDA_CAPABILITIES should be defined a sequence separated by commas,
// for example:
//   TF_EXTRA_CUDA_CAPABILITIES=3.0,4.0,5.0
// Use two-level macro expansion for stringification.
#define TF_XSTRING(...) #__VA_ARGS__
#define TF_STRING(s) TF_XSTRING(s)
  string extra_cuda_caps = TF_STRING(TF_EXTRA_CUDA_CAPABILITIES);
#undef TF_STRING
#undef TF_XSTRING
  auto extra_capabilities = str_util::Split(extra_cuda_caps, ',');
  for (const auto& capability : extra_capabilities) {
    cuda_caps.push_back(ComputeCapabilityFromString(capability));
  }
#endif
  return cuda_caps;
}
#endif  // GOOGLE_CUDA

}  // namespace

Status BaseGPUDeviceFactory::EnablePeerAccess(
    const std::vector<PlatformDeviceId>& visible_gpu_order) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_56(mht_56_v, 2107, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDeviceFactory::EnablePeerAccess");

  se::Platform* gpu_manager = GPUMachineManager();
  int possible_peer_count = 0;
  int enabled_peer_count = 0;
  for (int i = 0; i < visible_gpu_order.size(); ++i) {
    const PlatformDeviceId platform_gpu_i = visible_gpu_order[i];
    for (int j = 0; j < visible_gpu_order.size(); ++j) {
      const PlatformDeviceId platform_gpu_j = visible_gpu_order[j];
      // We have already validated that ExecutorForDevice() calls return OK.
      se::StreamExecutor* from =
          DeviceIdUtil::ExecutorForPlatformDeviceId(gpu_manager, platform_gpu_i)
              .ValueOrDie();
      se::StreamExecutor* to =
          DeviceIdUtil::ExecutorForPlatformDeviceId(gpu_manager, platform_gpu_j)
              .ValueOrDie();

      if (from->CanEnablePeerAccessTo(to)) {
        ++possible_peer_count;
        auto status = from->EnablePeerAccessTo(to);
        if (!status.ok()) {
          LOG(WARNING)
              << "Unable to enable peer access between device ordinals "
              << platform_gpu_i << " and " << platform_gpu_j
              << ", status: " << status;
        } else {
          ++enabled_peer_count;
        }
      }
    }
  }

  // Return an error in the extreme failure case where the driver
  // reported that peering was possible but not a single peering was
  // successful.  This is to catch possible system misconfigurations
  // or more fundamental issues.
  if (possible_peer_count > 0 && enabled_peer_count == 0) {
    return errors::Internal(possible_peer_count,
                            " potential peer access pairs were reported by the "
                            "driver, but no peering could be enabled.");
  }
  return Status::OK();
}

Status BaseGPUDeviceFactory::GetValidDeviceIds(
    const std::vector<PlatformDeviceId>& visible_gpu_order,
    std::vector<PlatformDeviceId>* ids) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_57(mht_57_v, 2155, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDeviceFactory::GetValidDeviceIds");

  se::Platform* gpu_manager = GPUMachineManager();
  for (int i = 0; i < visible_gpu_order.size(); ++i) {
    int visible_gpu_id = visible_gpu_order[i].value();
    auto description_status = gpu_manager->DescriptionForDevice(visible_gpu_id);
    if (!description_status.ok()) {
      return description_status.status();
    }

    auto description = description_status.ConsumeValueOrDie();
#if GOOGLE_CUDA
    VLOG(1) << "Found device " << i << " with properties: "
            << "\npciBusID: " << description->pci_bus_id()
            << " name: " << description->name() << " computeCapability: "
            << description->cuda_compute_capability().ToString()
            << "\ncoreClock: " << description->clock_rate_ghz() << "GHz"
            << " coreCount: " << description->core_count()
            << " deviceMemorySize: "
            << strings::HumanReadableNumBytes(description->device_memory_size())
            << " deviceMemoryBandwidth: "
            << strings::HumanReadableNumBytes(description->memory_bandwidth())
            << "/s";
#elif TENSORFLOW_USE_ROCM
    std::string gcn_arch_name =
        description->rocm_compute_capability().gcn_arch_name();
    VLOG(1) << "Found device " << i << " with properties: "
            << "\npciBusID: " << description->pci_bus_id()
            << " name: " << description->name()
            << "     ROCm AMDGPU Arch: " << gcn_arch_name
            << "\ncoreClock: " << description->clock_rate_ghz() << "GHz"
            << " coreCount: " << description->core_count()
            << " deviceMemorySize: "
            << strings::HumanReadableNumBytes(description->device_memory_size())
            << " deviceMemoryBandwidth: "
            << strings::HumanReadableNumBytes(description->memory_bandwidth())
            << "/s";
#endif
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  // Try to dlopen GPU libraries if they are supposed to be dynamically loaded.
  auto handle_or = se::internal::DsoLoader::MaybeTryDlopenGPULibraries();
  if (!handle_or.ok()) {
    LOG(WARNING) << "Cannot dlopen some GPU libraries. Please make sure the "
                    "missing libraries mentioned above are installed properly "
                    "if you would like to use GPU. Follow the guide at "
                    "https://www.tensorflow.org/install/gpu for how to "
                    "download and setup the required libraries for your "
                    "platform.\nSkipping registering "
                    "GPU devices...";
    return Status::OK();
  }
#endif

#if GOOGLE_CUDA
  auto cuda_supported_capabilities = GetSupportedCudaComputeCapabilities();
  if (cuda_supported_capabilities.empty()) {
    return errors::FailedPrecondition(
        "No supported cuda capabilities in binary.");
  }
  se::CudaComputeCapability min_supported_capability = *std::min_element(
      cuda_supported_capabilities.begin(), cuda_supported_capabilities.end());
#endif

  int min_gpu_core_count =
      GetMinGPUMultiprocessorCount(gpu_manager, visible_gpu_order);

  // Filter out devices that don't have the right capability or power.
  for (int i = 0; i < visible_gpu_order.size(); ++i) {
    const PlatformDeviceId visible_gpu_id = visible_gpu_order[i];
    auto description_status =
        gpu_manager->DescriptionForDevice(visible_gpu_id.value());
    if (!description_status.ok()) {
      LOG(INFO) << "Ignoring visible gpu device " << visible_gpu_id
                << " whose executor is in invalid state: "
                << description_status.status().ToString();
      continue;
    }

    auto desc = description_status.ConsumeValueOrDie();

#if GOOGLE_CUDA
    // Only GPUs with no less than the minimum supported compute capability is
    // accepted.
    if (desc->cuda_compute_capability() < min_supported_capability) {
      LOG(INFO) << "Ignoring visible gpu device "
                << "(" << GetShortDeviceDescription(visible_gpu_id, *desc)
                << ") "
                << "with Cuda compute capability "
                << desc->cuda_compute_capability().ToString()
                << ". The minimum required Cuda capability is "
                << min_supported_capability.ToString() << ".";
      continue;
    }
#elif TENSORFLOW_USE_ROCM
    int device_isa;
    // Only GPUs with supported gfx versions are accepted.
    auto rocm_compute_capability = desc->rocm_compute_capability();
    if (!rocm_compute_capability.is_supported_gfx_version()) {
      LOG(INFO) << "Ignoring visible gpu device "
                << "(" << GetShortDeviceDescription(visible_gpu_id, *desc)
                << ") "
                << "with AMDGPU version : "
                << rocm_compute_capability.gfx_version()
                << ". The supported AMDGPU versions are "
                << rocm_compute_capability.supported_gfx_versions_str() << ".";
      continue;
    }
#endif

    // Filter out slow GPUs. By default, GPUs with a lower multiprocessor
    // count than the fastest GPU are filtered out, unless they have 8 or more
    // multiprocessors. If the TF_MIN_GPU_MULTIPROCESSOR_COUNT environment
    // variable is set, its value will be used to filter out GPUs.
    if (desc->core_count() < min_gpu_core_count) {
      LOG(INFO) << "Ignoring visible gpu device "
                << "(" << GetShortDeviceDescription(visible_gpu_id, *desc)
                << ") "
                << "with core count: " << desc->core_count()
                << ". The minimum required count is " << min_gpu_core_count
                << ". You can adjust this requirement with the env var "
                   "TF_MIN_GPU_MULTIPROCESSOR_COUNT.";
      continue;
    }

#if defined(GOOGLE_CUDA) && !defined(PLATFORM_GOOGLE)
    // Compare compute capability of device with list in cuda_config.h and
    // warn about long loading time when we need to JIT kernels from PTX.
    auto compute_capabilities = {TF_CUDA_COMPUTE_CAPABILITIES};
    auto device_capability = desc->cuda_compute_capability();
    if (std::count_if(std::cbegin(compute_capabilities),
                      std::cend(compute_capabilities), [&](int cc) {
                        // CUBINs are backwards compatible within major version.
                        return cc / 10 == device_capability.major &&
                               cc % 10 <= device_capability.minor;
                      }) == 0) {
      LOG(WARNING)
          << "TensorFlow was not built with CUDA kernel binaries "
             "compatible with compute capability "
          << device_capability.ToString() << ". CUDA kernels will be "
          << "jit-compiled from PTX, which could take 30 minutes or longer.";
    }
#endif  // defined(GOOGLE_CUDA) && !defined(PLATFORM_GOOGLE)

    ids->push_back(visible_gpu_id);
  }
  if (!ids->empty()) {
    std::vector<int> raw_ids(ids->size());
    std::transform(ids->begin(), ids->end(), raw_ids.begin(),
                   [](PlatformDeviceId id) -> int { return id.value(); });
    VLOG(1) << "Adding visible gpu devices: " << absl::StrJoin(raw_ids, ", ");
  }

  return Status::OK();
}

uint64 BaseGPUDevice::SafeAllocFrontier(uint64 old_value) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_58(mht_58_v, 2314, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::SafeAllocFrontier");

  if (timestamped_allocator_) {
    return kernel_tracker_->LastTerminatedCount(old_value);
  } else {
    return 0;
  }
}

int BaseGPUDevice::PendingKernels() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_59(mht_59_v, 2325, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::PendingKernels");

  if (kernel_tracker_) {
    return kernel_tracker_->NumPending();
  }
  return 0;
}

void BaseGPUDevice::TestOnlyReset() {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_60(mht_60_v, 2335, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "BaseGPUDevice::TestOnlyReset");

  StreamGroupFactory::Global().TestOnlyReset();
}

uint64 GPUKernelTracker::MaybeQueue(OpKernelContext* ctx) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_61(mht_61_v, 2342, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GPUKernelTracker::MaybeQueue");

  mutex_lock l(mu_);
  ++ops_since_last_;
  int64_t mem_used =
      ctx->persistent_memory_allocated() + ctx->temp_memory_allocated();
  VLOG(2) << "kernel: " << ctx->op_kernel().name() << " mem_used: " << mem_used;
  mem_since_last_ += mem_used;
  int weight = 1;
  // Note that if all {max_bytes, max_interval, max_pending} are zero then
  // we track every single kernel with no pending cap.  This can happen if
  // timestamped_allocator alone was specified.
  if ((mem_since_last_ < params_.max_bytes) &&
      (ops_since_last_ < params_.max_interval)) {
    return 0;
  } else {
    weight = std::min(
        params_.max_pending,
        std::max(1, mem_since_last_ / std::max(16386, params_.max_bytes)));
    mem_since_last_ = 0;
    ops_since_last_ = 0;
  }
  uint64 queued_count = timing_counter_->next();
  RecordQueued(queued_count, weight);
  return queued_count;
}

void GPUKernelTracker::RecordQueued(uint64 queued_count, int weight) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_62(mht_62_v, 2371, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GPUKernelTracker::RecordQueued");

  VLOG(2) << "RecordQueued queued_count=" << queued_count
          << " first_available_=" << first_available_
          << " last_completed_=" << last_completed_
          << " num_pending_=" << num_pending_;
  pending_kernels_[first_available_].queued_count = queued_count;
  pending_kernels_[first_available_].weight = weight;
  pending_kernels_[first_available_].terminated = false;
  ++first_available_;
  num_pending_ += weight;
  if (first_available_ >= pending_kernels_.size()) {
    if (last_completed_ >= 0) {
      // wrap
      first_available_ = 0;
    } else {
      // enlarge the ring buffer
      pending_kernels_.resize(2 * pending_kernels_.size());
    }
  }
  if (first_available_ == last_completed_) {
    // Ring buffer is full: double it.  All of the same valid PendingKernel
    // entries exist after the copy, they are just shifted to begin
    // at index 0 in the new array.
    std::vector<PendingKernel> new_buffer(pending_kernels_.size() * 2);
    for (int i = 0; i < pending_kernels_.size(); ++i) {
      int j = (i + last_completed_) % pending_kernels_.size();
      new_buffer[i] = pending_kernels_[j];
    }
    last_completed_ = 0;
    first_available_ = pending_kernels_.size();
    pending_kernels_.swap(new_buffer);
    VLOG(1) << "last_completed_=" << last_completed_
            << " first_available_=" << first_available_
            << " num_pending_=" << num_pending_;
  }
  DCHECK_NE(first_available_, last_completed_) << "exhausted pending_kernels";
}

// Called by LastTerminatedCount() when new_value is equal to old_value.  This
// case can occur where an allocation failed and waited for memory to be freed,
// then when it retried the safe allocation frontier had not advanced because no
// tracking event had matured.  Maybe GPU progress has stalled waiting on an i/o
// event, or maybe we're tracking at too infrequent an interval.  In any case if
// the GPU compute queue is actually empty it's safe to advance the safe
// frontier so that this request can allocate from unrestricted (and better
// compacted) memory.  So queue an event on the compute stream to ensure the
// frontier does advance.
void GPUKernelTracker::MaybeQueueProgressEvent() {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_63(mht_63_v, 2421, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GPUKernelTracker::MaybeQueueProgressEvent");

  mutex_lock l(mu_);
  if (num_pending_ == 0) {
    uint64 new_count = timing_counter_->next();
    RecordQueued(new_count, 1);
    em_->ThenExecute(stream_,
                     [this, new_count]() { RecordTerminated(new_count); });
  }
}

void GPUKernelTracker::RecordTerminated(uint64 queued_count) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_deviceDTcc mht_64(mht_64_v, 2434, "", "./tensorflow/core/common_runtime/gpu/gpu_device.cc", "GPUKernelTracker::RecordTerminated");

  mutex_lock l(mu_);
  VLOG(2) << this << " RecordTerminated queued_count=" << queued_count
          << " first_available_=" << first_available_
          << " last_completed_=" << last_completed_
          << " num_pending_=" << num_pending_ << " LC="
          << ((last_completed_ >= 0)
                  ? pending_kernels_[last_completed_].queued_count
                  : -1);
  DCHECK_NE(first_available_, last_completed_);
  DCHECK_GT(num_pending_, 0);
  // Starting just past the last completed entry, find the entry with
  // this queued_count and mark it done.
  int index = (last_completed_ + 1) % pending_kernels_.size();
  int weight = 1;
  while (true) {
    if (index == first_available_) {
      // This should never happen.
      LOG(FATAL) << "Failed to find " << queued_count  // Crash OK
                 << " in queue, last_completed_=" << last_completed_
                 << " index=" << index
                 << " first_available_=" << first_available_
                 << " pending_kernels_.size()=" << pending_kernels_.size();
    }
    if (pending_kernels_[index].queued_count == queued_count) {
      pending_kernels_[index].terminated = true;
      weight = pending_kernels_[index].weight;
      break;
    }
    index = (index + 1) % pending_kernels_.size();
  }
  // Next move last_completed_ forward past all completed kernels.  In theory
  // kernels should always complete in queued order so we should be able to
  // advance the completed frontier to the just-completed PendingKernel.  In
  // practice we occasionally see the termination callbacks arrive out of
  // order probably because of thread scheduling.  Eventually we may support
  // out-of- order completion involving multiple compute streams so here we
  // follow a conservative approach and wait for every single callback to
  // arrive before advancing the frontier.
  while (true) {
    int next_index = (last_completed_ + 1) % pending_kernels_.size();
    if (next_index == first_available_) break;
    if (pending_kernels_[next_index].terminated) {
      last_completed_ = next_index;
    } else {
      break;
    }
  }
  if (last_completed_ >= 0) {
    int64_t v = pending_kernels_[last_completed_].queued_count;
    last_terminated_count_ = v;
    if (allocator_) {
      allocator_->SetSafeFrontier(v);
    }
  }
  // Last decrease num_pending before maybe waking a waiter.
  num_pending_ -= weight;
  pending_decreased_.notify_all();
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
