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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/environment.h"

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
absl::Status CreateEnvironment(Environment* result, bool shared,
                               cl_context_properties egl_context,
                               cl_context_properties egl_display) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "CreateEnvironment");

  CLDevice gpu;
  RETURN_IF_ERROR(CreateDefaultGPUDevice(&gpu));

  CLContext context;
  if (shared) {
    RETURN_IF_ERROR(CreateCLGLContext(gpu, egl_context, egl_display, &context));
  } else {
    RETURN_IF_ERROR(CreateCLContext(gpu, &context));
  }
  CLCommandQueue queue;
  RETURN_IF_ERROR(CreateCLCommandQueue(gpu, context, &queue));
  ProfilingCommandQueue profiling_queue;
  RETURN_IF_ERROR(CreateProfilingCommandQueue(gpu, context, &profiling_queue));

  *result = Environment(std::move(gpu), std::move(context), std::move(queue),
                        std::move(profiling_queue));

  return result->Init();
}

bool IsGpuSupportsStorageType(const GpuInfo& gpu_info,
                              TensorStorageType storage_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "IsGpuSupportsStorageType");

  switch (storage_type) {
    case TensorStorageType::TEXTURE_2D:
      return !gpu_info.IsAMD();
    case TensorStorageType::BUFFER:
      return true;
    case TensorStorageType::TEXTURE_ARRAY:
      return !gpu_info.IsAMD() && gpu_info.SupportsTextureArray();
    case TensorStorageType::IMAGE_BUFFER:
      return (gpu_info.IsAdreno() || gpu_info.IsAMD() || gpu_info.IsNvidia()) &&
             gpu_info.SupportsImageBuffer();
    case TensorStorageType::TEXTURE_3D:
      return !gpu_info.IsAMD() && gpu_info.SupportsImage3D();
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return false;
    case TensorStorageType::UNKNOWN:
      return false;
  }
  return false;
}

bool IsGpuSupportsPrecision(const GpuInfo& gpu_info,
                            CalculationsPrecision precision) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_2(mht_2_v, 249, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "IsGpuSupportsPrecision");

  switch (precision) {
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      return gpu_info.SupportsFP16();
    case CalculationsPrecision::F32:
      return true;
  }
}

}  // namespace

Environment::Environment(CLDevice&& device, CLContext&& context,
                         CLCommandQueue&& queue,
                         ProfilingCommandQueue&& profiling_queue)
    : device_(std::move(device)),
      context_(std::move(context)),
      queue_(std::move(queue)),
      profiling_queue_(std::move(profiling_queue)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_3(mht_3_v, 270, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::Environment");
}

Environment::Environment(Environment&& environment)
    : device_(std::move(environment.device_)),
      context_(std::move(environment.context_)),
      queue_(std::move(environment.queue_)),
      profiling_queue_(std::move(environment.profiling_queue_)),
      program_cache_(std::move(environment.program_cache_)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_4(mht_4_v, 280, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::Environment");
}

Environment& Environment::operator=(Environment&& environment) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_5(mht_5_v, 285, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "=");

  if (this != &environment) {
    device_ = std::move(environment.device_);
    context_ = std::move(environment.context_);
    queue_ = std::move(environment.queue_);
    profiling_queue_ = std::move(environment.profiling_queue_);
    program_cache_ = std::move(environment.program_cache_);
  }
  return *this;
}

absl::Status Environment::Init() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_6(mht_6_v, 299, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::Init");

  if (device().GetInfo().IsAdreno() &&
      device().GetInfo().SupportsTextureArray()) {
    const auto& adreno_info = device().info_.adreno_info;
    // Some Adreno < 600 have bug with one layer texture array. b/131099086
    // If we have one layer texture array and will write smt from kernel to this
    // texture, we will get zeroes instead of actual values.
    // The same kernel will work, if we use texture array with more than one
    // layer.
    if (adreno_info.IsAdreno3xx() || adreno_info.IsAdreno4xx() ||
        adreno_info.IsAdreno5xx()) {
      GetDevicePtr()->DisableOneLayerTextureArray();
    }
  }
  return absl::OkStatus();
}

void Environment::SetHighPerformance() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_7(mht_7_v, 319, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::SetHighPerformance");

  // TODO(sorokin) use cl_perf_hint if available
}

void Environment::SetDefaultPerformance() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_8(mht_8_v, 326, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::SetDefaultPerformance");

  // TODO(sorokin) use cl_perf_hint if available
}

void Environment::SetLowPerformance() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_9(mht_9_v, 333, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::SetLowPerformance");

  // TODO(sorokin) use cl_perf_hint if available
}

std::vector<CalculationsPrecision> Environment::GetSupportedPrecisions() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_10(mht_10_v, 340, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::GetSupportedPrecisions");

  std::vector<CalculationsPrecision> precisions;
  for (CalculationsPrecision precision :
       {CalculationsPrecision::F32, CalculationsPrecision::F32_F16,
        CalculationsPrecision::F16}) {
    if (IsSupported(precision)) {
      precisions.push_back(precision);
    }
  }
  return precisions;
}

bool Environment::IsSupported(CalculationsPrecision precision) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_11(mht_11_v, 355, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::IsSupported");

  return IsGpuSupportsPrecision(device_.GetInfo(), precision);
}

std::vector<TensorStorageType> Environment::GetSupportedStorages() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_12(mht_12_v, 362, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::GetSupportedStorages");

  std::vector<TensorStorageType> storage_types;
  for (auto storage_type :
       {TensorStorageType::TEXTURE_2D, TensorStorageType::BUFFER,
        TensorStorageType::TEXTURE_ARRAY, TensorStorageType::IMAGE_BUFFER,
        TensorStorageType::TEXTURE_3D}) {
    if (IsSupported(storage_type)) {
      storage_types.push_back(storage_type);
    }
  }
  return storage_types;
}

std::vector<TensorStorageType>
Environment::GetSupportedStoragesWithHWZeroClampSupport() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_13(mht_13_v, 379, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::GetSupportedStoragesWithHWZeroClampSupport");

  std::vector<TensorStorageType> storage_types;
  for (auto storage_type :
       {TensorStorageType::TEXTURE_2D, TensorStorageType::TEXTURE_ARRAY,
        TensorStorageType::TEXTURE_3D}) {
    if (IsSupported(storage_type)) {
      storage_types.push_back(storage_type);
    }
  }
  return storage_types;
}

bool Environment::IsSupported(TensorStorageType storage_type) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_14(mht_14_v, 394, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "Environment::IsSupported");

  return IsGpuSupportsStorageType(device_.GetInfo(), storage_type);
}

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_15(mht_15_v, 401, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "GetFastestStorageType");

  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno6xxOrHigher() &&
        !gpu_info.opencl_info.IsImage2dFromBufferSupported()) {
      return TensorStorageType::TEXTURE_ARRAY;
    } else {
      return TensorStorageType::TEXTURE_2D;
    }
  } else if (gpu_info.IsPowerVR()) {
    return TensorStorageType::TEXTURE_2D;
  } else if (gpu_info.IsMali()) {
    const MaliInfo mali_info = gpu_info.mali_info;
    if (mali_info.IsMaliT8xx() || mali_info.IsBifrostGen3() ||
        mali_info.IsValhall()) {
      return TensorStorageType::TEXTURE_2D;
    } else {
      return TensorStorageType::BUFFER;
    }
  } else if (gpu_info.IsNvidia()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsAMD()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsIntel()) {
    return TensorStorageType::BUFFER;
  }
  return TensorStorageType::BUFFER;
}

TensorStorageType GetStorageTypeWithMinimalMemoryConsumption(
    const GpuInfo& gpu_info) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_16(mht_16_v, 435, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "GetStorageTypeWithMinimalMemoryConsumption");

  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx() ||
        gpu_info.adreno_info.IsAdreno4xx()) {
      return TensorStorageType::BUFFER;
    } else {
      if (gpu_info.opencl_info.IsImage2dFromBufferSupported()) {
        return TensorStorageType::TEXTURE_2D;
      } else {
        return TensorStorageType::IMAGE_BUFFER;
      }
    }
  } else if (gpu_info.IsPowerVR()) {
    if (gpu_info.opencl_info.IsImage2dFromBufferSupported() &&
        CanUseSubBufferForImage2d(gpu_info)) {
      return TensorStorageType::TEXTURE_2D;
    } else {
      return TensorStorageType::BUFFER;
    }
  } else if (gpu_info.IsMali()) {
    const MaliInfo mali_info = gpu_info.mali_info;
    if (mali_info.IsMaliT8xx() || mali_info.IsBifrostGen3() ||
        mali_info.IsValhall()) {
      if (gpu_info.opencl_info.IsImage2dFromBufferSupported() &&
          CanUseSubBufferForImage2d(gpu_info)) {
        return TensorStorageType::TEXTURE_2D;
      } else {
        return TensorStorageType::BUFFER;
      }
    } else {
      return TensorStorageType::BUFFER;
    }
  } else if (gpu_info.IsNvidia()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsAMD()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsIntel()) {
    return TensorStorageType::BUFFER;
  }
  return TensorStorageType::BUFFER;
}

bool CanUseSubBufferForImage2d(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_17(mht_17_v, 482, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "CanUseSubBufferForImage2d");

  if (!gpu_info.IsCL11OrHigher()) {
    return false;
  }
  if (gpu_info.IsPowerVR()) {
    // driver issue
    return false;
  }
  if (gpu_info.IsMali() &&
      (gpu_info.mali_info.IsBifrost() || gpu_info.mali_info.IsMidgard())) {
    // Known driver issue on some G72 (Bifrost), G76 (Bifrost), T830 (Midgard),
    // and T880 (Midgard) devices.
    return false;
  }
  return true;
}

absl::Status CreateEnvironment(Environment* result) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPSenvironmentDTcc mht_18(mht_18_v, 502, "", "./tensorflow/lite/delegates/gpu/cl/environment.cc", "CreateEnvironment");

  CLDevice gpu;
  RETURN_IF_ERROR(CreateDefaultGPUDevice(&gpu));

  CLContext context;
  RETURN_IF_ERROR(CreateCLContext(gpu, &context));
  CLCommandQueue queue;
  RETURN_IF_ERROR(CreateCLCommandQueue(gpu, context, &queue));
  ProfilingCommandQueue profiling_queue;
  RETURN_IF_ERROR(CreateProfilingCommandQueue(gpu, context, &profiling_queue));

  *result = Environment(std::move(gpu), std::move(context), std::move(queue),
                        std::move(profiling_queue));
  return result->Init();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
