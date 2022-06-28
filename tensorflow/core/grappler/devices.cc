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
class MHTracer_DTPStensorflowPScorePSgrapplerPSdevicesDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSdevicesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSdevicesDTcc() {
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

#include <memory>

#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace grappler {

int GetNumAvailableGPUs(
    const std::pair<int, int>& min_cuda_compute_capability) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSdevicesDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/grappler/devices.cc", "GetNumAvailableGPUs");

  int num_eligible_gpus = 0;

#if TENSORFLOW_USE_ROCM
  if (min_cuda_compute_capability.first != 0 ||
      min_cuda_compute_capability.second != 0) {
    LOG(ERROR) << "GetNumAvailableGPUs() should receive zero "
                  "min_cuda_compute_capability";
    return 0;
  }
#endif
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (ValidateGPUMachineManager().ok()) {
    se::Platform* gpu_manager = GPUMachineManager();
    if (gpu_manager != nullptr) {
      int num_gpus = gpu_manager->VisibleDeviceCount();
      for (int i = 0; i < num_gpus; i++) {
#if GOOGLE_CUDA
        auto desc = gpu_manager->DescriptionForDevice(i);
        if (desc.ok()) {
          int min_gpu_core_count = 8;
          if ((*desc)->core_count() >= min_gpu_core_count &&
              (*desc)->cuda_compute_capability().IsAtLeast(
                  min_cuda_compute_capability.first,
                  min_cuda_compute_capability.second)) {
            num_eligible_gpus++;
          }
        }
#else
        num_eligible_gpus++;
#endif
      }
    }
  }
#if GOOGLE_CUDA
  LOG(INFO)
      << "Number of eligible GPUs (core count >= 8, compute capability >= "
      << min_cuda_compute_capability.first << "."
      << min_cuda_compute_capability.second << "): " << num_eligible_gpus;
#else
  LOG(INFO) << "Number of eligible GPUs: " << num_eligible_gpus;
#endif

#else   // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  LOG(INFO)
      << "Number of eligible GPUs (core count >= 8, compute capability >= "
      << min_cuda_compute_capability.first << "."
      << min_cuda_compute_capability.second << "): " << num_eligible_gpus
      << " (Note: TensorFlow was not compiled with CUDA or ROCm support)";
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return num_eligible_gpus;
}

int64_t AvailableGPUMemory(int gpu_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSdevicesDTcc mht_1(mht_1_v, 256, "", "./tensorflow/core/grappler/devices.cc", "AvailableGPUMemory");

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  // Look up the device, to see its attributes.
  se::Platform* gpu_platform = GPUMachineManager();
  CHECK_LT(gpu_id, gpu_platform->VisibleDeviceCount());
  se::StreamExecutor* se = gpu_platform->ExecutorForDevice(gpu_id).ValueOrDie();
  int64_t total_memory, available_memory;
  CHECK(se->DeviceMemoryUsage(&available_memory, &total_memory));

  return available_memory;
#else
  return 0;
#endif
}

int GetNumAvailableLogicalCPUCores() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSdevicesDTcc mht_2(mht_2_v, 274, "", "./tensorflow/core/grappler/devices.cc", "GetNumAvailableLogicalCPUCores");
 return port::NumSchedulableCPUs(); }

}  // end namespace grappler
}  // end namespace tensorflow
