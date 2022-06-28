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
class MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSutilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSutilsDTcc() {
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

#include "tensorflow/core/grappler/clusters/utils.h"

#include "third_party/eigen3/Eigen/Core"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cudnn/cudnn.h"
#endif

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#endif

#ifdef TENSORFLOW_USE_LIBXSMM
#include "include/libxsmm.h"
#endif

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {
namespace grappler {

DeviceProperties GetLocalCPUInfo() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSutilsDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/grappler/clusters/utils.cc", "GetLocalCPUInfo");

  DeviceProperties device;
  device.set_type("CPU");

  device.set_vendor(port::CPUVendorIDString());
  // Combine cpu family and model into the model string.
  device.set_model(
      strings::StrCat((port::CPUFamily() << 4) + port::CPUModelNum()));
  device.set_frequency(port::NominalCPUFrequency() * 1e-6);
  device.set_num_cores(port::NumSchedulableCPUs());
  device.set_l1_cache_size(Eigen::l1CacheSize());
  device.set_l2_cache_size(Eigen::l2CacheSize());
  device.set_l3_cache_size(Eigen::l3CacheSize());

  int64_t free_mem = port::AvailableRam();
  if (free_mem < INT64_MAX) {
    device.set_memory_size(free_mem);
  }

  (*device.mutable_environment())["cpu_instruction_set"] =
      Eigen::SimdInstructionSetsInUse();

  (*device.mutable_environment())["eigen"] = strings::StrCat(
      EIGEN_WORLD_VERSION, ".", EIGEN_MAJOR_VERSION, ".", EIGEN_MINOR_VERSION);
#ifdef TENSORFLOW_USE_LIBXSMM
  (*device.mutable_environment())["libxsmm"] = LIBXSMM_VERSION;
#endif

  return device;
}

DeviceProperties GetLocalGPUInfo(PlatformDeviceId platform_device_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSutilsDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/grappler/clusters/utils.cc", "GetLocalGPUInfo");

  DeviceProperties device;
  device.set_type("GPU");

#if GOOGLE_CUDA
  cudaDeviceProp properties;
  cudaError_t error =
      cudaGetDeviceProperties(&properties, platform_device_id.value());
  if (error != cudaSuccess) {
    device.set_type("UNKNOWN");
    LOG(ERROR) << "Failed to get device properties, error code: " << error;
    return device;
  }

  device.set_vendor("NVIDIA");
  device.set_model(properties.name);
  device.set_frequency(properties.clockRate * 1e-3);
  device.set_num_cores(properties.multiProcessorCount);
  device.set_num_registers(properties.regsPerMultiprocessor);
  // For compute capability less than 5, l1 cache size is configurable to
  // either 16 KB or 48 KB. We use the initial configuration 16 KB here. For
  // compute capability larger or equal to 5, l1 cache (unified with texture
  // cache) size is 24 KB. This number may need to be updated for future
  // compute capabilities.
  device.set_l1_cache_size((properties.major < 5) ? 16 * 1024 : 24 * 1024);
  device.set_l2_cache_size(properties.l2CacheSize);
  device.set_l3_cache_size(0);
  device.set_shared_memory_size_per_multiprocessor(
      properties.sharedMemPerMultiprocessor);
  device.set_memory_size(properties.totalGlobalMem);
  // 8 is the number of bits per byte. 2 is accounted for
  // double data rate (DDR).
  device.set_bandwidth(properties.memoryBusWidth / 8 *
                       properties.memoryClockRate * 2);

  (*device.mutable_environment())["architecture"] =
      strings::StrCat(properties.major, ".", properties.minor);
  (*device.mutable_environment())["cuda"] = strings::StrCat(CUDA_VERSION);
  (*device.mutable_environment())["cudnn"] = strings::StrCat(CUDNN_VERSION);

#elif TENSORFLOW_USE_ROCM
  hipDeviceProp_t properties;
  hipError_t error =
      hipGetDeviceProperties(&properties, platform_device_id.value());
  if (error != hipSuccess) {
    device.set_type("UNKNOWN");
    LOG(ERROR) << "Failed to get device properties, error code: " << error;
    return device;
  }

  // ROCM TODO review if numbers here are valid
  device.set_vendor("Advanced Micro Devices, Inc");
  device.set_model(properties.name);
  device.set_frequency(properties.clockRate * 1e-3);
  device.set_num_cores(properties.multiProcessorCount);
  device.set_num_registers(properties.regsPerBlock);
  device.set_l1_cache_size(16 * 1024);
  device.set_l2_cache_size(properties.l2CacheSize);
  device.set_l3_cache_size(0);
  device.set_shared_memory_size_per_multiprocessor(
      properties.maxSharedMemoryPerMultiProcessor);
  device.set_memory_size(properties.totalGlobalMem);
  // 8 is the number of bits per byte. 2 is accounted for
  // double data rate (DDR).
  device.set_bandwidth(properties.memoryBusWidth / 8 *
                       properties.memoryClockRate * 2);

  (*device.mutable_environment())["architecture"] =
      strings::StrCat("gfx", properties.gcnArch);
#endif

  return device;
}

DeviceProperties GetDeviceInfo(const DeviceNameUtils::ParsedName& device) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSutilsDTcc mht_2(mht_2_v, 326, "", "./tensorflow/core/grappler/clusters/utils.cc", "GetDeviceInfo");

  DeviceProperties unknown;
  unknown.set_type("UNKNOWN");

  if (device.type == "CPU") {
    return GetLocalCPUInfo();
  } else if (device.type == "GPU") {
    if (device.has_id) {
      TfDeviceId tf_device_id(device.id);
      PlatformDeviceId platform_device_id;
      Status s =
          GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id);
      if (!s.ok()) {
        LOG(ERROR) << s;
        return unknown;
      }
      return GetLocalGPUInfo(platform_device_id);
    } else {
      return GetLocalGPUInfo(PlatformDeviceId(0));
    }
  }
  return unknown;
}

}  // end namespace grappler
}  // end namespace tensorflow
