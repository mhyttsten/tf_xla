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
class MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSgpu_delegate_providerDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSgpu_delegate_providerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSgpu_delegate_providerDTcc() {
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
#include <string>
#include <utility>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#if TFLITE_SUPPORTS_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#elif defined(__APPLE__)
#include "TargetConditionals.h"
#if (TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR) || \
    (TARGET_OS_OSX && TARGET_CPU_ARM64)
// Only enable metal delegate when using a real iPhone device or Apple Silicon.
#define REAL_IPHONE_DEVICE
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif
#endif

namespace tflite {
namespace tools {

class GpuDelegateProvider : public DelegateProvider {
 public:
  GpuDelegateProvider() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSgpu_delegate_providerDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/tools/delegates/gpu_delegate_provider.cc", "GpuDelegateProvider");

    default_params_.AddParam("use_gpu", ToolParam::Create<bool>(false));
#if TFLITE_SUPPORTS_GPU_DELEGATE || defined(REAL_IPHONE_DEVICE)
    default_params_.AddParam("gpu_precision_loss_allowed",
                             ToolParam::Create<bool>(true));
    default_params_.AddParam("gpu_experimental_enable_quant",
                             ToolParam::Create<bool>(true));
#endif
#if TFLITE_SUPPORTS_GPU_DELEGATE
    default_params_.AddParam("gpu_inference_for_sustained_speed",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("gpu_backend", ToolParam::Create<std::string>(""));
#endif
#if defined(REAL_IPHONE_DEVICE)
    default_params_.AddParam("gpu_wait_type",
                             ToolParam::Create<std::string>(""));
#endif
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSgpu_delegate_providerDTcc mht_1(mht_1_v, 236, "", "./tensorflow/lite/tools/delegates/gpu_delegate_provider.cc", "GetName");
 return "GPU"; }
};
REGISTER_DELEGATE_PROVIDER(GpuDelegateProvider);

std::vector<Flag> GpuDelegateProvider::CreateFlags(ToolParams* params) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSgpu_delegate_providerDTcc mht_2(mht_2_v, 243, "", "./tensorflow/lite/tools/delegates/gpu_delegate_provider.cc", "GpuDelegateProvider::CreateFlags");

  std::vector<Flag> flags = {
    CreateFlag<bool>("use_gpu", params, "use gpu"),
#if TFLITE_SUPPORTS_GPU_DELEGATE || defined(REAL_IPHONE_DEVICE)
    CreateFlag<bool>("gpu_precision_loss_allowed", params,
                     "Allow to process computation in lower precision than "
                     "FP32 in GPU. By default, it's enabled."),
    CreateFlag<bool>("gpu_experimental_enable_quant", params,
                     "Whether to enable the GPU delegate to run quantized "
                     "models or not. By default, it's enabled."),
#endif
#if TFLITE_SUPPORTS_GPU_DELEGATE
    CreateFlag<bool>("gpu_inference_for_sustained_speed", params,
                     "Whether to prefer maximizing the throughput. This mode "
                     "will help when the same delegate will be used repeatedly "
                     "on multiple inputs. This is supported on non-iOS "
                     "platforms. By default, it's disabled."),
    CreateFlag<std::string>(
        "gpu_backend", params,
        "Force the GPU delegate to use a particular backend for execution, and "
        "fail if unsuccessful. Should be one of: cl, gl"),
#endif
#if defined(REAL_IPHONE_DEVICE)
    CreateFlag<std::string>(
        "gpu_wait_type", params,
        "GPU wait type. Should be one of the following: passive, active, "
        "do_not_wait, aggressive"),
#endif
  };
  return flags;
}

void GpuDelegateProvider::LogParams(const ToolParams& params,
                                    bool verbose) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSgpu_delegate_providerDTcc mht_3(mht_3_v, 279, "", "./tensorflow/lite/tools/delegates/gpu_delegate_provider.cc", "GpuDelegateProvider::LogParams");

  LOG_TOOL_PARAM(params, bool, "use_gpu", "Use gpu", verbose);
#if TFLITE_SUPPORTS_GPU_DELEGATE || defined(REAL_IPHONE_DEVICE)
  LOG_TOOL_PARAM(params, bool, "gpu_precision_loss_allowed",
                 "Allow lower precision in gpu", verbose);
  LOG_TOOL_PARAM(params, bool, "gpu_experimental_enable_quant",
                 "Enable running quant models in gpu", verbose);
#endif
#if TFLITE_SUPPORTS_GPU_DELEGATE
  LOG_TOOL_PARAM(params, bool, "gpu_inference_for_sustained_speed",
                 "Prefer maximizing the throughput in gpu", verbose);
  LOG_TOOL_PARAM(params, std::string, "gpu_backend", "GPU backend", verbose);
#endif
#if defined(REAL_IPHONE_DEVICE)
  LOG_TOOL_PARAM(params, std::string, "gpu_wait_type", "GPU delegate wait type",
                 verbose);
#endif
}

TfLiteDelegatePtr GpuDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSgpu_delegate_providerDTcc mht_4(mht_4_v, 302, "", "./tensorflow/lite/tools/delegates/gpu_delegate_provider.cc", "GpuDelegateProvider::CreateTfLiteDelegate");

  TfLiteDelegatePtr delegate = CreateNullDelegate();

  if (params.Get<bool>("use_gpu")) {
#if TFLITE_SUPPORTS_GPU_DELEGATE
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    if (params.Get<bool>("gpu_precision_loss_allowed")) {
      gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      gpu_opts.inference_priority2 =
          TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
      gpu_opts.inference_priority3 =
          TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    }
    if (params.Get<bool>("gpu_experimental_enable_quant")) {
      gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
    }
    if (params.Get<bool>("gpu_inference_for_sustained_speed")) {
      gpu_opts.inference_preference =
          TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    }
    std::string gpu_backend = params.Get<std::string>("gpu_backend");
    if (!gpu_backend.empty()) {
      if (gpu_backend == "cl") {
        gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
      } else if (gpu_backend == "gl") {
        gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
      }
    }
    gpu_opts.max_delegated_partitions =
        params.Get<int>("max_delegated_partitions");

    // Serialization.
    std::string serialize_dir =
        params.Get<std::string>("delegate_serialize_dir");
    std::string serialize_token =
        params.Get<std::string>("delegate_serialize_token");
    if (!serialize_dir.empty() && !serialize_token.empty()) {
      gpu_opts.experimental_flags =
          gpu_opts.experimental_flags |
          TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
      gpu_opts.serialization_dir = serialize_dir.c_str();
      gpu_opts.model_token = serialize_token.c_str();
    }

    delegate = evaluation::CreateGPUDelegate(&gpu_opts);
#elif defined(REAL_IPHONE_DEVICE)
    TFLGpuDelegateOptions gpu_opts = {0};
    gpu_opts.allow_precision_loss =
        params.Get<bool>("gpu_precision_loss_allowed");
    gpu_opts.enable_quantization =
        params.Get<bool>("gpu_experimental_enable_quant");

    std::string string_gpu_wait_type = params.Get<std::string>("gpu_wait_type");
    if (!string_gpu_wait_type.empty()) {
      TFLGpuDelegateWaitType wait_type = TFLGpuDelegateWaitTypePassive;
      if (string_gpu_wait_type == "passive") {
        wait_type = TFLGpuDelegateWaitTypePassive;
      } else if (string_gpu_wait_type == "active") {
        wait_type = TFLGpuDelegateWaitTypeActive;
      } else if (string_gpu_wait_type == "do_not_wait") {
        wait_type = TFLGpuDelegateWaitTypeDoNotWait;
      } else if (string_gpu_wait_type == "aggressive") {
        wait_type = TFLGpuDelegateWaitTypeAggressive;
      }
      gpu_opts.wait_type = wait_type;
    }
    delegate = TfLiteDelegatePtr(TFLGpuDelegateCreate(&gpu_opts),
                                 &TFLGpuDelegateDelete);
#else
    TFLITE_LOG(WARN) << "The GPU delegate compile options are only supported "
                        "on Android or iOS platforms or when the tool was "
                        "built with -DCL_DELEGATE_NO_GL.";
    delegate = evaluation::CreateGPUDelegate();
#endif

    if (!delegate.get()) {
      TFLITE_LOG(WARN) << "GPU acceleration is unsupported on this platform.";
    }
  }
  return delegate;
}

std::pair<TfLiteDelegatePtr, int>
GpuDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr), params.GetPosition<bool>("use_gpu"));
}
}  // namespace tools
}  // namespace tflite
