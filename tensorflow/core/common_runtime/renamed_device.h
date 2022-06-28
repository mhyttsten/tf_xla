/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_RENAMED_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_RENAMED_DEVICE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh() {
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


#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/lib/core/threadpool_interface.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

// Wraps a device with a new name, delegating work to the wrapped device.
//
// This class is used to wrap local devices when using clusterspec propagation
// where the name of a particular device may change in the context of a given
// session.
class RenamedDevice : public Device {
 public:
  static std::unique_ptr<Device> NewRenamedDevice(
      const string& new_base, Device* underlying, bool owns_underlying,
      bool isolate_session_state,
      thread::ThreadPoolInterface* underlying_threadpool = nullptr);

  ~RenamedDevice() override;

  const DeviceBase* UnderlyingDevice() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_0(mht_0_v, 208, "", "./tensorflow/core/common_runtime/renamed_device.h", "UnderlyingDevice");

    return underlying_device_->UnderlyingDevice();
  }
  DeviceBase* UnderlyingDevice() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_1(mht_1_v, 214, "", "./tensorflow/core/common_runtime/renamed_device.h", "UnderlyingDevice");

    return underlying_device_->UnderlyingDevice();
  }

  const CpuWorkerThreads* tensorflow_cpu_worker_threads() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_2(mht_2_v, 221, "", "./tensorflow/core/common_runtime/renamed_device.h", "tensorflow_cpu_worker_threads");

    if (underlying_threadpool_) {
      return Device::tensorflow_cpu_worker_threads();
    }
    return underlying_device_->tensorflow_cpu_worker_threads();
  }

  const DeviceBase::AcceleratorDeviceInfo* tensorflow_accelerator_device_info()
      const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_3(mht_3_v, 232, "", "./tensorflow/core/common_runtime/renamed_device.h", "tensorflow_accelerator_device_info");

    return underlying_device_->tensorflow_accelerator_device_info();
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_4(mht_4_v, 239, "", "./tensorflow/core/common_runtime/renamed_device.h", "GetAllocator");

    return underlying_device_->GetAllocator(attr);
  }

  Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                int64_t step_id) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_5(mht_5_v, 247, "", "./tensorflow/core/common_runtime/renamed_device.h", "GetScopedAllocator");

    return underlying_device_->GetScopedAllocator(attr, step_id);
  }

  ScopedAllocatorMgr* GetScopedAllocatorMgr() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_6(mht_6_v, 254, "", "./tensorflow/core/common_runtime/renamed_device.h", "GetScopedAllocatorMgr");

    return underlying_device_->GetScopedAllocatorMgr();
  }

  const Eigen::ThreadPoolDevice* eigen_cpu_device() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_7(mht_7_v, 261, "", "./tensorflow/core/common_runtime/renamed_device.h", "eigen_cpu_device");

    // Use the underlying threadpool only if the underlying device supports
    // eigen_cpu_device.
    if (underlying_threadpool_ && underlying_device_->has_eigen_cpu_device()) {
      return Device::eigen_cpu_device();
    }
    return underlying_device_->eigen_cpu_device();
  }

  thread::ThreadPool* tensorflow_device_thread_pool() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_8(mht_8_v, 273, "", "./tensorflow/core/common_runtime/renamed_device.h", "tensorflow_device_thread_pool");

    // Use the underlying threadpool instead of tensorflow_device_thread_pool
    // of the underlying device only if tensorflow_device_thread_pool is defined
    // for the underlying device.
    if (underlying_threadpool_ &&
        underlying_device_->tensorflow_device_thread_pool() != nullptr) {
      return Device::tensorflow_device_thread_pool();
    }
    return underlying_device_->tensorflow_device_thread_pool();
  }

  bool has_eigen_cpu_device() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_9(mht_9_v, 287, "", "./tensorflow/core/common_runtime/renamed_device.h", "has_eigen_cpu_device");

    return underlying_device_->has_eigen_cpu_device();
  }


  PerOpGpuDevice* MakeGpuDevice() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_10(mht_10_v, 295, "", "./tensorflow/core/common_runtime/renamed_device.h", "MakeGpuDevice");

    return underlying_device_->MakeGpuDevice();
  }

  Status ReinitializeGpuDevice(OpKernelContext* context, PerOpGpuDevice* device,
                               DeviceContext* dc,
                               Allocator* allocator) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_11(mht_11_v, 304, "", "./tensorflow/core/common_runtime/renamed_device.h", "ReinitializeGpuDevice");

    return underlying_device_->ReinitializeGpuDevice(context, device, dc,
                                                     allocator);
  }

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_12(mht_12_v, 314, "", "./tensorflow/core/common_runtime/renamed_device.h", "MakeTensorFromProto");

    return underlying_device_->MakeTensorFromProto(tensor_proto, alloc_attrs,
                                                   tensor);
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_13(mht_13_v, 324, "", "./tensorflow/core/common_runtime/renamed_device.h", "CopyTensorInSameDevice");

    underlying_device_->CopyTensorInSameDevice(input_tensor, output_tensor,
                                               device_context, std::move(done));
  }

  // Below are virtual methods defined on Device

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_14(mht_14_v, 334, "", "./tensorflow/core/common_runtime/renamed_device.h", "Compute");

    underlying_device_->Compute(op_kernel, context);
  }

  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_15(mht_15_v, 342, "", "./tensorflow/core/common_runtime/renamed_device.h", "ComputeAsync");

    underlying_device_->ComputeAsync(op_kernel, context, std::move(done));
  }

  Status Sync() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_16(mht_16_v, 349, "", "./tensorflow/core/common_runtime/renamed_device.h", "Sync");
 return underlying_device_->Sync(); }

  Status MaybeRewriteGraph(std::unique_ptr<Graph>* graph) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_17(mht_17_v, 354, "", "./tensorflow/core/common_runtime/renamed_device.h", "MaybeRewriteGraph");

    return underlying_device_->MaybeRewriteGraph(graph);
  }

  Status TryGetDeviceContext(DeviceContext** out_context) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_18(mht_18_v, 361, "", "./tensorflow/core/common_runtime/renamed_device.h", "TryGetDeviceContext");

    return underlying_device_->TryGetDeviceContext(out_context);
  }

  // Returns the resource manager associated w/ this device.
  ResourceMgr* resource_manager() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_19(mht_19_v, 369, "", "./tensorflow/core/common_runtime/renamed_device.h", "resource_manager");

    if (isolate_session_state_) {
      return Device::resource_manager();
    } else {
      return underlying_device_->resource_manager();
    }
  }

  bool IsLocal() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_20(mht_20_v, 380, "", "./tensorflow/core/common_runtime/renamed_device.h", "IsLocal");
 return underlying_device_->IsLocal(); }

  bool IsRemoteCallAllowed() const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTh mht_21(mht_21_v, 385, "", "./tensorflow/core/common_runtime/renamed_device.h", "IsRemoteCallAllowed");

    return underlying_device_->IsRemoteCallAllowed();
  }

 private:
  RenamedDevice(Device* underlying, const DeviceAttributes& attributes,
                bool owns_underlying, bool isolate_session_state,
                thread::ThreadPoolInterface* underlying_threadpool);
  Device* const underlying_device_;
  const bool owns_underlying_device_;
  const bool isolate_session_state_;

  std::unique_ptr<thread::ThreadPool> underlying_threadpool_;
  // eigen_worker_threads_ is stored here so that we can pass the pointer
  // of eigen_worker_threads_.workers to the parent class.
  DeviceBase::CpuWorkerThreads eigen_worker_threads_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_RENAMED_DEVICE_H_
