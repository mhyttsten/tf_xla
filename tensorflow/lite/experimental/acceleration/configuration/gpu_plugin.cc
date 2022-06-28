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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/configuration/gpu_plugin.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"

namespace tflite {
namespace delegates {

int GpuPlugin::GetDelegateErrno(TfLiteDelegate* from_delegate) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc mht_0(mht_0_v, 194, "", "./tensorflow/lite/experimental/acceleration/configuration/gpu_plugin.cc", "GpuPlugin::GetDelegateErrno");
 return 0; }

std::unique_ptr<DelegatePluginInterface> GpuPlugin::New(
    const TFLiteSettings& acceleration) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc mht_1(mht_1_v, 200, "", "./tensorflow/lite/experimental/acceleration/configuration/gpu_plugin.cc", "GpuPlugin::New");

  return absl::make_unique<GpuPlugin>(acceleration);
}

#if TFLITE_SUPPORTS_GPU_DELEGATE

namespace {

TfLiteGpuInferencePriority ConvertInferencePriority(
    GPUInferencePriority priority) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc mht_2(mht_2_v, 212, "", "./tensorflow/lite/experimental/acceleration/configuration/gpu_plugin.cc", "ConvertInferencePriority");

  switch (priority) {
    case GPUInferencePriority_GPU_PRIORITY_AUTO:
      return TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    case GPUInferencePriority_GPU_PRIORITY_MAX_PRECISION:
      return TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    case GPUInferencePriority_GPU_PRIORITY_MIN_LATENCY:
      return TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    case GPUInferencePriority_GPU_PRIORITY_MIN_MEMORY_USAGE:
      return TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
  }
}

}  // namespace

TfLiteDelegatePtr GpuPlugin::Create() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc mht_3(mht_3_v, 230, "", "./tensorflow/lite/experimental/acceleration/configuration/gpu_plugin.cc", "GpuPlugin::Create");

  return TfLiteDelegatePtr(TfLiteGpuDelegateV2Create(&options_),
                           TfLiteGpuDelegateV2Delete);
}

GpuPlugin::GpuPlugin(const TFLiteSettings& tflite_settings)
    : options_(TfLiteGpuDelegateOptionsV2Default()) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc mht_4(mht_4_v, 239, "", "./tensorflow/lite/experimental/acceleration/configuration/gpu_plugin.cc", "GpuPlugin::GpuPlugin");

  if (tflite_settings.max_delegated_partitions() >= 0) {
    options_.max_delegated_partitions =
        tflite_settings.max_delegated_partitions();
  }

  const auto* gpu_settings = tflite_settings.gpu_settings();
  if (!gpu_settings) return;

  options_.inference_preference = gpu_settings->inference_preference();

  if (gpu_settings->inference_priority1() > 0) {
    // User has specified their own inference priorities, so just copy over.
    options_.inference_priority1 =
        ConvertInferencePriority(gpu_settings->inference_priority1());
    options_.inference_priority2 =
        ConvertInferencePriority(gpu_settings->inference_priority2());
    options_.inference_priority3 =
        ConvertInferencePriority(gpu_settings->inference_priority3());
  } else {
    options_.inference_priority1 =
        gpu_settings->is_precision_loss_allowed()
            ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
            : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  }

  if (gpu_settings->enable_quantized_inference()) {
    options_.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  }
  if (gpu_settings->force_backend() == GPUBackend_OPENCL) {
    options_.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
  } else if (gpu_settings->force_backend() == GPUBackend_OPENGL) {
    options_.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
  }
  if (gpu_settings->cache_directory() &&
      gpu_settings->cache_directory()->size() > 0 &&
      gpu_settings->model_token() && gpu_settings->model_token()->size()) {
    cache_dir_ = gpu_settings->cache_directory()->str();
    model_token_ = gpu_settings->model_token()->str();
    options_.serialization_dir = cache_dir_.c_str();
    options_.model_token = model_token_.c_str();
    options_.experimental_flags |=
        TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
  }
}

#elif defined(REAL_IPHONE_DEVICE)

TfLiteDelegatePtr GpuPlugin::Create() {
  return TfLiteDelegatePtr(TFLGpuDelegateCreate(&options_),
                           &TFLGpuDelegateDelete);
}

GpuPlugin::GpuPlugin(const TFLiteSettings& tflite_settings) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc mht_5(mht_5_v, 295, "", "./tensorflow/lite/experimental/acceleration/configuration/gpu_plugin.cc", "GpuPlugin::GpuPlugin");

  options_ = {0};
  const auto* gpu_settings = tflite_settings.gpu_settings();
  if (!gpu_settings) return;

  options_.allow_precision_loss = gpu_settings->is_precision_loss_allowed();
  options_.enable_quantization = gpu_settings->enable_quantized_inference();
}

#else

TfLiteDelegatePtr GpuPlugin::Create() {
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

// In case GPU acceleration is not supported for this platform, we still need to
// construct an empty object so that Create() can later be called on it.
GpuPlugin::GpuPlugin(const TFLiteSettings& tflite_settings) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSgpu_pluginDTcc mht_6(mht_6_v, 315, "", "./tensorflow/lite/experimental/acceleration/configuration/gpu_plugin.cc", "GpuPlugin::GpuPlugin");
}

#endif

TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(GpuPlugin, GpuPlugin::New);

}  // namespace delegates
}  // namespace tflite
