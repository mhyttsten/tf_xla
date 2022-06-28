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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_deviceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_deviceDTcc() {
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

#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"

#import <sys/utsname.h>

#include <string>

namespace tflite {
namespace gpu {
namespace metal {
namespace {
GpuInfo CreateGpuInfoFromMetalDevice(id<MTLDevice> device) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_deviceDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/gpu/metal/metal_device.cc", "CreateGpuInfoFromMetalDevice");

  std::string device_name = std::string([[device name] UTF8String]);
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(device_name, GpuApi::kMetal, &gpu_info);

  if (gpu_info.apple_info.gpu_type == AppleGpu::kA15) {
    struct utsname system_info;
    uname(&system_info);
    const std::string gadget_name(system_info.machine);
    // iPhone 13 mini(iPhone14,4) and iPhone 13(iPhone14,5) have A15 with 4 core
    // GPU.
    // In general A15 GPU has 5 cores.
    if (gadget_name == "iPhone14,4" || gadget_name == "iPhone14,5") {
      gpu_info.apple_info.SetComputeUnits(4);
    }
  }

  const bool a7_or_a8 =
      gpu_info.IsApple() && (gpu_info.apple_info.IsA7GenerationGpu() ||
                             gpu_info.apple_info.IsA8GenerationGpu());
  gpu_info.metal_info.image2d_max_width = a7_or_a8 ? 1024 * 8 : 1024 * 16;
  gpu_info.metal_info.image2d_max_height = a7_or_a8 ? 1024 * 8 : 1024 * 16;
  gpu_info.metal_info.image_array_max_layers = 2048;
  gpu_info.metal_info.image3d_max_width = 2048;
  gpu_info.metal_info.image3d_max_height = 2048;
  gpu_info.metal_info.image3d_max_depth = 2048;

  if (@available(macOS 10.11, iOS 9.0, tvOS 9.0, *)) {
    MTLSize threadsPerGroup = [device maxThreadsPerThreadgroup];
    gpu_info.metal_info.max_work_group_size_x = threadsPerGroup.width;
    gpu_info.metal_info.max_work_group_size_y = threadsPerGroup.height;
    gpu_info.metal_info.max_work_group_size_z = threadsPerGroup.depth;
  } else {
    gpu_info.metal_info.max_work_group_size_x = 256;
    gpu_info.metal_info.max_work_group_size_y = 256;
    gpu_info.metal_info.max_work_group_size_z = 64;
  }

  if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    gpu_info.metal_info.buffer_max_size = [device maxBufferLength];
  } else {
    // 256 MB
    gpu_info.metal_info.buffer_max_size = 256 * 1024 * 1024;
  }

  if (@available(macOS 11.0, iOS 14.0, tvOS 14.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal2_3;
  } else if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal2_2;
  } else if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal2_1;
  } else if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal2_0;
  } else if (@available(macOS 10.12, iOS 10.0, tvOS 10.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal1_2;
  } else if (@available(macOS 10.11, iOS 9.0, tvOS 9.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal1_1;
  } else {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal1_0;
  }

  return gpu_info;
}
}  // namespace

MetalDevice::MetalDevice() : device_(MTLCreateSystemDefaultDevice()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_deviceDTcc mht_1(mht_1_v, 263, "", "./tensorflow/lite/delegates/gpu/metal/metal_device.cc", "MetalDevice::MetalDevice");

  info_ = CreateGpuInfoFromMetalDevice(device_);
}
MetalDevice::MetalDevice(id<MTLDevice> device) : device_(device) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_deviceDTcc mht_2(mht_2_v, 269, "", "./tensorflow/lite/delegates/gpu/metal/metal_device.cc", "MetalDevice::MetalDevice");

  info_ = CreateGpuInfoFromMetalDevice(device_);
}

bool MetalDevice::IsLanguageVersion2orHigher() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_deviceDTcc mht_3(mht_3_v, 276, "", "./tensorflow/lite/delegates/gpu/metal/metal_device.cc", "MetalDevice::IsLanguageVersion2orHigher");

  auto version = info_.metal_info.language_version;
  return version != MetalLanguageVersion::kMetal1_0 &&
         version != MetalLanguageVersion::kMetal1_1 &&
         version != MetalLanguageVersion::kMetal1_2;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
