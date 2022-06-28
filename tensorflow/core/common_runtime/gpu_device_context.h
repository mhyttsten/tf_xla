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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh() {
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
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace tensorflow {

class GPUDeviceContext : public DeviceContext {
 public:
  // Does not take ownership of streams.
  GPUDeviceContext(int stream_id, se::Stream* stream,
#if TENSORFLOW_USE_ROCM
                   se::Stream* nccl_stream,
#endif
                   se::Stream* host_to_device_stream,
                   se::Stream* device_to_host_stream,
                   gtl::InlinedVector<se::Stream*, 4> device_to_device_stream,
                   Allocator* host_memory_allocator)
      : stream_id_(stream_id),
        stream_(stream),
#if TENSORFLOW_USE_ROCM
        nccl_stream_(nccl_stream),
#endif
        host_to_device_stream_(host_to_device_stream),
        device_to_host_stream_(device_to_host_stream),
        device_to_device_stream_(device_to_device_stream),
        host_memory_allocator_(host_memory_allocator) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_0(mht_0_v, 217, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "GPUDeviceContext");

  }

  ~GPUDeviceContext() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_1(mht_1_v, 223, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "~GPUDeviceContext");
}

  se::Stream* stream() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_2(mht_2_v, 228, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "stream");
 return stream_; }
#if TENSORFLOW_USE_ROCM
  se::Stream* nccl_stream() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_3(mht_3_v, 233, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "nccl_stream");
 return nccl_stream_; }
#endif
  se::Stream* host_to_device_stream() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_4(mht_4_v, 238, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "host_to_device_stream");
 return host_to_device_stream_; }
  se::Stream* device_to_host_stream() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_5(mht_5_v, 242, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "device_to_host_stream");
 return device_to_host_stream_; }
  se::Stream* device_to_device_stream(int index) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_6(mht_6_v, 246, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "device_to_device_stream");

    return device_to_device_stream_[index % device_to_device_stream_.size()];
  }
  int stream_id() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_7(mht_7_v, 252, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "stream_id");
 return stream_id_; }
  Allocator* host_memory_allocator() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_8(mht_8_v, 256, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "host_memory_allocator");

    return host_memory_allocator_;
  }

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override;

  void MaintainLifetimeOnStream(const Tensor* t,
                                se::Stream* stream) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSgpu_device_contextDTh mht_9(mht_9_v, 276, "", "./tensorflow/core/common_runtime/gpu_device_context.h", "MaintainLifetimeOnStream");
}

  Status ThenExecute(Device* device, se::Stream* stream,
                     std::function<void()> func) override;

 private:
  int stream_id_;
  // The default primary stream to use for this context.
  // All the memory belongs to this stream.
  se::Stream* stream_;
#if TENSORFLOW_USE_ROCM
  // The stream to use for nccl operations.
  se::Stream* nccl_stream_;
#endif
  // The stream to use for copying data from host into GPU.
  se::Stream* host_to_device_stream_;
  // The stream to use for copying data from GPU to host.
  se::Stream* device_to_host_stream_;
  // Streams to use for copying data between GPUs.
  gtl::InlinedVector<se::Stream*, 4> device_to_device_stream_;
  // The allocator to use for allocating pinned host memory.
  // Not owned.
  Allocator* host_memory_allocator_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
