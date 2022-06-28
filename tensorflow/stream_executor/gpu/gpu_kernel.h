/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// The CUDA implementation of the StreamExecutorInterface functionality.
// CUDA inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the CUDA streams
// programming model provided by the libcuda.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh() {
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


#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace gpu {

// Wraps a GpuFunctionHandle to implement the platform-independent
// KernelInterface.
class GpuKernel : public internal::KernelInterface {
 public:
  GpuKernel()
      : gpu_function_(nullptr),
        arity_(0),
        preferred_cache_config_(KernelCacheConfig::kNoPreference) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_0(mht_0_v, 210, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "GpuKernel");
}

  // Note that the function is unloaded when the module is unloaded, and the
  // module that the function is contained in is owned by the GpuExecutor.
  ~GpuKernel() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_1(mht_1_v, 217, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "~GpuKernel");
}

  // As arity cannot be reflected upon using the CUDA API, the arity is
  // explicitly set during the GpuExecutor::GetKernel initialization process.
  void set_arity(unsigned arity) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_2(mht_2_v, 224, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "set_arity");
 arity_ = arity; }
  unsigned Arity() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_3(mht_3_v, 228, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "Arity");
 return arity_; }

  // Returns the GpuFunctionHandle value for passing to the CUDA API.
  GpuFunctionHandle AsGpuFunctionHandle() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_4(mht_4_v, 234, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "AsGpuFunctionHandle");

    DCHECK(gpu_function_ != nullptr);
    return const_cast<GpuFunctionHandle>(gpu_function_);
  }

  // Returns the slot that the GpuFunctionHandle is stored within for this
  // object, for the CUDA API which wants to load into a GpuFunctionHandle*.
  GpuFunctionHandle* gpu_function_ptr() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_5(mht_5_v, 244, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "gpu_function_ptr");
 return &gpu_function_; }

  // CUDA supports setting the preferred cache configuration of a
  // GpuFunctionHandle (more-or-less equivalent to a GpuKernel). We support this
  // via the below functions; users can set a preference, and that is applied
  // when the kernel is [lazy-]loaded (in GpuExecutor::Launch). The alternative
  // would be to load the kernel & set the preference when the user calls the
  // setter below; either approach is valid. Sets the current kernel cache
  // configuration preference.
  void SetPreferredCacheConfig(KernelCacheConfig config) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_6(mht_6_v, 256, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "SetPreferredCacheConfig");

    preferred_cache_config_ = config;
  }

  // Returns the current kernel cache configuration preference.
  KernelCacheConfig GetPreferredCacheConfig() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_7(mht_7_v, 264, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "GetPreferredCacheConfig");

    return preferred_cache_config_;
  }

  // Returns the current kernel cache configuration preference as a
  // CUfunc_cache.
  GpuFuncCachePreference GetGpuCacheConfig() const;

 private:
  GpuFunctionHandle gpu_function_;  // Wrapped CUDA kernel handle.
  unsigned arity_;  // Number of formal parameters the kernel takes.

  // Preferred (but not required) cache configuration for this kernel.
  KernelCacheConfig preferred_cache_config_;
};

// Given a platform-independent kernel datatype, returns the (const) internal
// CUDA platform implementation pointer.
inline const GpuKernel* AsGpuKernel(const KernelBase* kernel) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_8(mht_8_v, 285, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "AsGpuKernel");

  return static_cast<const GpuKernel*>(kernel->implementation());
}

// Given a platform-independent kernel datatype, returns the (non-const)
// internal CUDA platform implementation pointer.
inline GpuKernel* AsGpuKernel(KernelBase* kernel) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_kernelDTh mht_9(mht_9_v, 294, "", "./tensorflow/stream_executor/gpu/gpu_kernel.h", "AsGpuKernel");

  return static_cast<GpuKernel*>(kernel->implementation());
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
