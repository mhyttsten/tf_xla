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

#ifndef TENSORFLOW_CORE_FRAMEWORK_DEVICE_BASE_H_
#define TENSORFLOW_CORE_FRAMEWORK_DEVICE_BASE_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "absl/base/macros.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace Eigen {
struct ThreadPoolDevice;
}  // end namespace Eigen

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace tensorflow {

class Device;
class DeviceAttributes;
class Env;
class EventMgr;
class OpKernelContext;
class ResourceMgr;
class ScopedAllocatorMgr;
class TensorProto;

namespace thread {
class ThreadPool;
}

// A wrapper for an Eigen Gpu Device that includes per-op state. The
// class is defined even for non-GPU devices since the
// OpKernelContext::Params structure wants to fill it in.
class PerOpGpuDevice {
 public:
  virtual ~PerOpGpuDevice() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_0(mht_0_v, 231, "", "./tensorflow/core/framework/device_base.h", "~PerOpGpuDevice");
}
  virtual const Eigen::GpuDevice& device() const = 0;
};

// A class that devices can subclass to pass around
// Device-specific context to OpKernels.
class DeviceContext : public core::RefCounted {
 public:
  ~DeviceContext() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_1(mht_1_v, 242, "", "./tensorflow/core/framework/device_base.h", "~DeviceContext");
}
  virtual stream_executor::Stream* stream() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/framework/device_base.h", "stream");
 return nullptr; }
  virtual void MaintainLifetimeOnStream(const Tensor* t,
                                        stream_executor::Stream* stream) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_3(mht_3_v, 251, "", "./tensorflow/core/framework/device_base.h", "MaintainLifetimeOnStream");

  }

  // "cpu_tensor" is a tensor on a CPU. Copies "cpu_tensor" into
  // "device_tensor" which is on a non-CPU device "device". "device_tensor"
  // must be allocated to be of the same size as "cpu_tensor".
  virtual void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                                     Tensor* device_tensor, StatusCallback done,
                                     bool sync_dst_compute = true) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_4(mht_4_v, 262, "", "./tensorflow/core/framework/device_base.h", "CopyCPUTensorToDevice");

    done(errors::Internal("Unrecognized device type in CPU-to-device Copy"));
  }

  // Same as CopyCPUTensorToDevice, but in a synchronous way.
  Status CopyCPUTensorToDeviceSync(const Tensor* cpu_tensor, Device* device,
                                   Tensor* device_tensor) const;

  // Copies a tensor in this device.
  virtual void CopyTensorInSameDevice(const Tensor* input_tensor,
                                      Device* device, Tensor* output_tensor,
                                      StatusCallback done) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_5(mht_5_v, 276, "", "./tensorflow/core/framework/device_base.h", "CopyTensorInSameDevice");

    done(errors::Unimplemented("Copy in same device not implemented."));
  }

  // "device_tensor" is a tensor on a non-CPU device.  Copies
  // device_tensor into "cpu_tensor".  "cpu_tensor" must be allocated
  // to be of the same size as "device_tensor".
  virtual void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                     StringPiece tensor_name, Device* device,
                                     Tensor* cpu_tensor, StatusCallback done) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_6(mht_6_v, 288, "", "./tensorflow/core/framework/device_base.h", "CopyDeviceTensorToCPU");

    done(errors::Internal("Unrecognized device type in device-to-CPU Copy"));
  }

  // Same as `CopyDeviceTensorToCPU`, but blocks until the copy is done.
  Status CopyDeviceTensorToCPUSync(const Tensor* device_tensor,
                                   StringPiece tensor_name, Device* device,
                                   Tensor* cpu_tensor);

  // If possible, wait for all events on *stream to complete then execute func.
  // A non-OK Status is returned otherwise.  The stream argument should be the
  // one provided by AcceleratorDeviceInfo.  This function is not applicable to
  // devices that don't provide such a value.
  virtual Status ThenExecute(Device* device, stream_executor::Stream* stream,
                             std::function<void()> func) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_7(mht_7_v, 305, "", "./tensorflow/core/framework/device_base.h", "ThenExecute");

    return errors::Internal("ThenExecute not supported by device");
  }

  // check if device is a pluggable device
  virtual bool IsPluggableDevice() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_8(mht_8_v, 313, "", "./tensorflow/core/framework/device_base.h", "IsPluggableDevice");
 return false; }

  // Returns the pinned host memory allocator for the device.
  virtual Allocator* host_memory_allocator() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_9(mht_9_v, 319, "", "./tensorflow/core/framework/device_base.h", "host_memory_allocator");
 return nullptr; }
};

class DeviceBase {
 public:
  explicit DeviceBase(Env* env) : env_(env) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_10(mht_10_v, 327, "", "./tensorflow/core/framework/device_base.h", "DeviceBase");
}
  virtual ~DeviceBase();

  Env* env() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_11(mht_11_v, 333, "", "./tensorflow/core/framework/device_base.h", "env");
 return env_; }

  struct CpuWorkerThreads {
    int num_threads = 0;
    thread::ThreadPool* workers = nullptr;
  };

  // Does not take ownership.
  void set_tensorflow_cpu_worker_threads(CpuWorkerThreads* t) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_12(mht_12_v, 344, "", "./tensorflow/core/framework/device_base.h", "set_tensorflow_cpu_worker_threads");

    cpu_worker_threads_ = t;
  }

  virtual const CpuWorkerThreads* tensorflow_cpu_worker_threads() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_13(mht_13_v, 351, "", "./tensorflow/core/framework/device_base.h", "tensorflow_cpu_worker_threads");

    CHECK(cpu_worker_threads_ != nullptr);
    return cpu_worker_threads_;
  }

  // "stream" is used in special circumstances (such as the
  // constructors of Ops) where there is no available OpKernelContext.
  // "default_context" is used by OpKernelContext whenever a device does not
  // supply a DeviceContext for an op in TryGetDeviceContext() (e.g. when only
  // using a single stream.)
  // "event_mgr" is used to delay deallocation of temporary GPU buffers.
  // TODO(pbar) Work out how to move this out of DeviceBase.
  struct AcceleratorDeviceInfo {
    // Make sure all the defaults are NULL, so we can spot missing assignments.
    stream_executor::Stream* stream = nullptr;
    DeviceContext* default_context = nullptr;
    EventMgr* event_mgr = nullptr;
    int gpu_id = -1;
  };

  // Does not take ownership.
  void set_tensorflow_accelerator_device_info(AcceleratorDeviceInfo* g) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_14(mht_14_v, 375, "", "./tensorflow/core/framework/device_base.h", "set_tensorflow_accelerator_device_info");

    gpu_device_info_ = g;
  }

  virtual const AcceleratorDeviceInfo* tensorflow_gpu_device_info() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_15(mht_15_v, 382, "", "./tensorflow/core/framework/device_base.h", "tensorflow_gpu_device_info");

    return gpu_device_info_;
  }

  virtual const AcceleratorDeviceInfo* tensorflow_accelerator_device_info()
      const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_16(mht_16_v, 390, "", "./tensorflow/core/framework/device_base.h", "tensorflow_accelerator_device_info");

    return gpu_device_info_;
  }

  // The preferred thread pool for this device. If it is nullptr, the system
  // automatically assigns a thread pool for execution.
  virtual thread::ThreadPool* tensorflow_device_thread_pool() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_17(mht_17_v, 399, "", "./tensorflow/core/framework/device_base.h", "tensorflow_device_thread_pool");

    return device_thread_pool_;
  }

  // Does not take ownership.
  void set_eigen_cpu_device(Eigen::ThreadPoolDevice* d);

  // Return the Allocator implementation to use based on the allocator
  // attributes requested.  See allocator.h for more details.
  virtual Allocator* GetAllocator(AllocatorAttributes /*attr*/) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_18(mht_18_v, 411, "", "./tensorflow/core/framework/device_base.h", "GetAllocator");

    LOG(FATAL) << "GetAllocator() is not implemented.";
    return nullptr;
  }

  // This method is provided for backwards compatibility, and will be removed
  // in a future release.
  ABSL_DEPRECATED("Use `this->GetAllocator()` or `this->GetScopedAllocator()`.")
  Allocator* GetStepAllocator(AllocatorAttributes attr, ResourceMgr*) {
    return GetAllocator(attr);
  }

  // Return an Allocator prepared for use in particular places by graph
  // optimization
  virtual Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                        int64_t step_id) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_19(mht_19_v, 429, "", "./tensorflow/core/framework/device_base.h", "GetScopedAllocator");

    LOG(FATAL) << "Device does not implement GetScopedAllocator()";
    return nullptr;
  }

  virtual ScopedAllocatorMgr* GetScopedAllocatorMgr() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_20(mht_20_v, 437, "", "./tensorflow/core/framework/device_base.h", "GetScopedAllocatorMgr");
 return nullptr; }

  virtual bool has_eigen_cpu_device() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_21(mht_21_v, 442, "", "./tensorflow/core/framework/device_base.h", "has_eigen_cpu_device");

    return !eigen_cpu_devices_.empty();
  }

  virtual const Eigen::ThreadPoolDevice* eigen_cpu_device();

  // Caller owns the return value. The OpKernelContext calls this even
  // for devices that do not implement an eigen_gpu_device. Overridden
  // by GPU devices to return a derived type.
  virtual PerOpGpuDevice* MakeGpuDevice() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_22(mht_22_v, 454, "", "./tensorflow/core/framework/device_base.h", "MakeGpuDevice");
 return nullptr; }

  virtual DeviceBase* UnderlyingDevice() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_23(mht_23_v, 459, "", "./tensorflow/core/framework/device_base.h", "UnderlyingDevice");
 return this; }
  virtual const DeviceBase* UnderlyingDevice() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_24(mht_24_v, 463, "", "./tensorflow/core/framework/device_base.h", "UnderlyingDevice");
 return this; }

  // This is overridden by GPU devices to reinitialize the derived
  // type returned by MakeGpuDevice.
  virtual Status ReinitializeGpuDevice(OpKernelContext* /*context*/,
                                       PerOpGpuDevice* /*device*/,
                                       DeviceContext* /*dc*/,
                                       Allocator* /*allocator*/) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_25(mht_25_v, 473, "", "./tensorflow/core/framework/device_base.h", "ReinitializeGpuDevice");

    return Status::OK();
  }

  // Unimplemented by default
  virtual const DeviceAttributes& attributes() const;
  virtual int NumaNode() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_26(mht_26_v, 482, "", "./tensorflow/core/framework/device_base.h", "NumaNode");
 return attributes().locality().numa_node(); }
  virtual const std::string& name() const;
  virtual const DeviceNameUtils::ParsedName& parsed_name() const;

  // Updates `attributes()`, indicating the XLA global ID associated with this
  // device. This ID is unique across clients in a multi-client setup. For TPUs
  // this does not happen until the TPU system has been initialized.
  //
  // Implemented in Device.
  virtual void set_xla_global_id(int64_t id) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_27(mht_27_v, 494, "", "./tensorflow/core/framework/device_base.h", "set_xla_global_id");
}

  // Materializes the given TensorProto into 'tensor' stored in Device
  // memory.  Most devices will want to override this.
  //
  // TODO(vrv): We should be able to put this function into
  // OpKernelContext and handle the copies from device memory via send
  // and receive nodes, instead of requiring that each device handle
  // the copies here as well as in copy ops.
  virtual Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                     const AllocatorAttributes alloc_attrs,
                                     Tensor* tensor) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_28(mht_28_v, 508, "", "./tensorflow/core/framework/device_base.h", "MakeTensorFromProto");

    return errors::Internal("Device does not implement MakeTensorFromProto()");
  }

  // Some devices (i.e. GPUs) may free device memory prior to its actual use
  // being completed on the assumption that subsequent allocations can only be
  // used serially with respect to pending uses.  If this function returns a
  // non-zero value it is the value of a device-specific counter such that any
  // device memory tagged with an earlier freed-at count is really unencumbered
  // by pending uses.  For this to be useful the device memory allocator must
  // be tagging deallocated memory chunks using the same counter.
  virtual uint64 SafeAllocFrontier(uint64 old_value) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_29(mht_29_v, 522, "", "./tensorflow/core/framework/device_base.h", "SafeAllocFrontier");
 return 0; }

  // Copies `input_tensor` to `output_tensor`, where both tensors are on this
  // device. This function assumes that `output_tensor` has already been
  // allocated with a buffer that is large enough to hold `input_tensor`'s data.
  // Calls `done` from a device-specific thread after copy is finished, which
  // may be the same as calling thread.
  //
  // NOTE(ayushd): This function is for TensorFlow internal use only.  Deep copy
  // is discouraged and should not be used in OpKernels.
  virtual void CopyTensorInSameDevice(const Tensor* input_tensor,
                                      Tensor* output_tensor,
                                      const DeviceContext* device_context,
                                      StatusCallback done) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_30(mht_30_v, 538, "", "./tensorflow/core/framework/device_base.h", "CopyTensorInSameDevice");

    done(errors::Internal("Device ", name(), " does not implement ",
                          "CopyTensorInSameDevice"));
  }

 protected:
  // Does not take ownership.
  void set_tensorflow_device_thread_pool(thread::ThreadPool* thread_pool) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTh mht_31(mht_31_v, 548, "", "./tensorflow/core/framework/device_base.h", "set_tensorflow_device_thread_pool");

    device_thread_pool_ = thread_pool;
  }

 private:
  Env* const env_;
  CpuWorkerThreads* cpu_worker_threads_ = nullptr;
  // Set by GPUs as well as by TPU devices.
  AcceleratorDeviceInfo* gpu_device_info_ = nullptr;
  thread::ThreadPool* device_thread_pool_ = nullptr;
  std::vector<Eigen::ThreadPoolDevice*> eigen_cpu_devices_;
};

// Methods to create and check for Symbolic execution devices.
// Such devices are mostly used for TF-XLA bridge. TF should not treat these as
// normal devices.
void AddSymbolicExecutionDevice(absl::string_view device_name);
bool IsSymbolicExecutionDevice(absl::string_view device_name);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DEVICE_BASE_H_
