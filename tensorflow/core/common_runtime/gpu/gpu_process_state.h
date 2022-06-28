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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_PROCESS_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_PROCESS_STATE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTh() {
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


#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class Allocator;
class GPUBFCAllocator;
class PoolAllocator;
class SharedCounter;

// Singleton that manages per-process state when GPUs are present.
class GPUProcessState {
 public:
  // If ps == nullptr, returns pointer to the single instance of this class to
  // be used within this process.
  //
  // If ps != nullptrs, accepts a value to be returned by all subsequent calls.
  // A non-null ps may ONLY be provided during program static storage
  // initialization.  Must not be called more than once with a non-null ps.
  //
  // If a derived class of GPUProcessState is ever used in a process, it must
  // always be used in place of this class.  In order to ensure that existing
  // calls to GPUProcessState::singleton() all resolve to the derived instance
  // instead, this function must be called once during startup, supplying the
  // derived instance value, prior to any accessor call to this function.
  static GPUProcessState* singleton(GPUProcessState* ps = nullptr);

  // Query whether any GPU device has been created so far.
  // Disable thread safety analysis since a race is benign here.
  bool HasGPUDevice() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return gpu_device_enabled_;
  }

  // Set the flag to indicate a GPU device has been created.
  // Disable thread safety analysis since a race is benign here.
  void EnableGPUDevice() TF_NO_THREAD_SAFETY_ANALYSIS {
    gpu_device_enabled_ = true;
  }

  // Returns the one GPU allocator used for the indexed GPU.
  // Note that this is a system GPU index, not (necessarily) a brain
  // device index.
  //
  // 'total_bytes' is the total number of bytes that should be made
  // available to the allocator.  The first call to this function for
  // a given tf_device_id creates the allocator, so only the total_bytes
  // used on that first call is used.
  //
  // "Allocator type" describes the type of algorithm to use for the
  // underlying allocator.  REQUIRES: Must be a valid type (see
  // config.proto for the list of supported strings.).
  //
  // REQUIRES: tf_device_id must be a valid id for a BaseGPUDevice available in
  // the current system environment.  Otherwise returns nullptr.
  virtual Allocator* GetGPUAllocator(
      const GPUOptions& options, TfDeviceId tf_device_id, size_t total_bytes,
      const std::vector<TfDeviceId>& peer_gpu_ids);

  Allocator* GetGPUAllocator(TfDeviceId tf_device_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTh mht_0(mht_0_v, 257, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.h", "GetGPUAllocator");

    return GetGPUAllocator(/*options=*/{}, tf_device_id, /*total_bytes=*/0,
                           /*peer_gpu_ids=*/{});
  }

  int NumGPUAllocators() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTh mht_1(mht_1_v, 265, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.h", "NumGPUAllocators");

    mutex_lock l(mu_);
    return gpu_allocators_.size();
  }

  virtual Allocator* GetGpuHostAllocator(int numa_node);

  // Registers a Visitor to be invoked on new chunks of memory allocated by the
  // SubAllocator of every GPU proximate to the specified bus.  The AllocVisitor
  // is provided with a memory pointer, a GPU id, and the size of the area it
  // identifies.  The pointer is not guaranteed to be valid after the call
  // terminates.  The intention is for this interface to be used for network
  // device memory registration.  "bus_id" is platform-specific.  On many
  // platforms it should be 0.  On machines with multiple PCIe buses, it should
  // be the index of one of the PCIe buses (maybe the NUMA node at which the
  // PCIe is rooted).  If the bus_id is invalid, results are undefined.
  virtual void AddGPUAllocVisitor(int bus_id,
                                  const SubAllocator::Visitor& visitor);

  // Registers a Visitor to be invoked on new chunks of memory allocated by
  // the SubAllocator of the GpuHostAllocator for the given numa_node.
  virtual void AddGpuHostAllocVisitor(int numa_node,
                                      const SubAllocator::Visitor& visitor);

  // Registers a Visitor to be invoked on each chunk handed back for freeing to
  // the SubAllocator of the GpuHostAllocator for the given numa_node.
  virtual void AddGpuHostFreeVisitor(int numa_node,
                                     const SubAllocator::Visitor& visitor);

  // Returns bus_id for the given GPU id.
  virtual int BusIdForGPU(TfDeviceId tf_device_id);

  SharedCounter* GPUAllocatorCounter(TfDeviceId tf_device_id);

 protected:
  // GPUProcessState is a singleton that should not normally be deleted except
  // at process shutdown.
  GPUProcessState();
  virtual ~GPUProcessState() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTh mht_2(mht_2_v, 306, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.h", "~GPUProcessState");
}
  friend class GPUDeviceTest;

  // Helper method for unit tests to reset the ProcessState singleton by
  // cleaning up everything. Never use in production.
  virtual void TestOnlyReset();

  ProcessState::MDMap* mem_desc_map() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpuPSgpu_process_stateDTh mht_3(mht_3_v, 316, "", "./tensorflow/core/common_runtime/gpu/gpu_process_state.h", "mem_desc_map");

    if (process_state_) return &process_state_->mem_desc_map_;
    return nullptr;
  }

  static GPUProcessState* instance_;
  ProcessState* process_state_;  // Not owned.
  bool gpu_device_enabled_;

  mutex mu_;

  struct AllocatorParts {
    std::unique_ptr<Allocator> allocator;
    std::unique_ptr<SharedCounter> counter;
    GPUBFCAllocator* bfc_allocator;
    SubAllocator* sub_allocator;  // owned by allocator
    std::unique_ptr<Allocator> recording_allocator;
  };
  std::vector<AllocatorParts> gpu_allocators_ TF_GUARDED_BY(mu_);
  std::vector<std::vector<SubAllocator::Visitor>> gpu_visitors_
      TF_GUARDED_BY(mu_);

  std::vector<AllocatorParts> gpu_host_allocators_ TF_GUARDED_BY(mu_);
  std::vector<std::vector<SubAllocator::Visitor>> gpu_host_alloc_visitors_
      TF_GUARDED_BY(mu_);
  std::vector<std::vector<SubAllocator::Visitor>> gpu_host_free_visitors_
      TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_PROCESS_STATE_H_
