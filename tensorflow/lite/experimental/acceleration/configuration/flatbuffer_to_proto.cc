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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.h"

#include "tensorflow/lite/experimental/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {
proto::ExecutionPreference ConvertExecutionPreference(
    ExecutionPreference preference) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_0(mht_0_v, 193, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertExecutionPreference");

  switch (preference) {
    case ExecutionPreference_ANY:
      return proto::ExecutionPreference::ANY;
    case ExecutionPreference_LOW_LATENCY:
      return proto::ExecutionPreference::LOW_LATENCY;
    case ExecutionPreference_LOW_POWER:
      return proto::ExecutionPreference::LOW_POWER;
    case ExecutionPreference_FORCE_CPU:
      return proto::ExecutionPreference::FORCE_CPU;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for ExecutionPreference: %d", preference);
  return proto::ExecutionPreference::ANY;
}

proto::Delegate ConvertDelegate(Delegate delegate) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertDelegate");

  switch (delegate) {
    case Delegate_NONE:
      return proto::Delegate::NONE;
    case Delegate_NNAPI:
      return proto::Delegate::NNAPI;
    case Delegate_GPU:
      return proto::Delegate::GPU;
    case Delegate_HEXAGON:
      return proto::Delegate::HEXAGON;
    case Delegate_XNNPACK:
      return proto::Delegate::XNNPACK;
    case Delegate_EDGETPU:
      return proto::Delegate::EDGETPU;
    case Delegate_EDGETPU_CORAL:
      return proto::Delegate::EDGETPU_CORAL;
    case Delegate_CORE_ML:
      return proto::Delegate::CORE_ML;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for Delegate: %d",
                  delegate);
  return proto::Delegate::NONE;
}

proto::NNAPIExecutionPreference ConvertNNAPIExecutionPreference(
    NNAPIExecutionPreference preference) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_2(mht_2_v, 240, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertNNAPIExecutionPreference");

  switch (preference) {
    case NNAPIExecutionPreference_UNDEFINED:
      return proto::NNAPIExecutionPreference::UNDEFINED;
    case NNAPIExecutionPreference_NNAPI_LOW_POWER:
      return proto::NNAPIExecutionPreference::NNAPI_LOW_POWER;
    case NNAPIExecutionPreference_NNAPI_FAST_SINGLE_ANSWER:
      return proto::NNAPIExecutionPreference::NNAPI_FAST_SINGLE_ANSWER;
    case NNAPIExecutionPreference_NNAPI_SUSTAINED_SPEED:
      return proto::NNAPIExecutionPreference::NNAPI_SUSTAINED_SPEED;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for NNAPIExecutionPreference: %d",
                  preference);
  return proto::NNAPIExecutionPreference::UNDEFINED;
}

proto::NNAPIExecutionPriority ConvertNNAPIExecutionPriority(
    NNAPIExecutionPriority priority) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_3(mht_3_v, 261, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertNNAPIExecutionPriority");

  switch (priority) {
    case NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED:
      return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_UNDEFINED;
    case NNAPIExecutionPriority_NNAPI_PRIORITY_LOW:
      return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_LOW;
    case NNAPIExecutionPriority_NNAPI_PRIORITY_MEDIUM:
      return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_MEDIUM;
    case NNAPIExecutionPriority_NNAPI_PRIORITY_HIGH:
      return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_HIGH;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for NNAPIExecutionPriority: %d", priority);
  return proto::NNAPIExecutionPriority::NNAPI_PRIORITY_UNDEFINED;
}

proto::GPUBackend ConvertGPUBackend(GPUBackend backend) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_4(mht_4_v, 280, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertGPUBackend");

  switch (backend) {
    case GPUBackend_UNSET:
      return proto::GPUBackend::UNSET;
    case GPUBackend_OPENCL:
      return proto::GPUBackend::OPENCL;
    case GPUBackend_OPENGL:
      return proto::GPUBackend::OPENGL;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for GPUBackend: %d",
                  backend);
  return proto::GPUBackend::UNSET;
}

proto::GPUInferenceUsage ConvertGPUInferenceUsage(
    GPUInferenceUsage preference) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_5(mht_5_v, 298, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertGPUInferenceUsage");

  switch (preference) {
    case GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER:
      return proto::GPUInferenceUsage::
          GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    case GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED:
      return proto::GPUInferenceUsage::GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for GPUInferenceUsage: %d", preference);
  return proto::GPUInferenceUsage::GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
}

proto::GPUInferencePriority ConvertGPUInferencePriority(
    GPUInferencePriority priority) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_6(mht_6_v, 315, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertGPUInferencePriority");

  switch (priority) {
    case GPUInferencePriority_GPU_PRIORITY_AUTO:
      return proto::GPUInferencePriority::GPU_PRIORITY_AUTO;
    case GPUInferencePriority_GPU_PRIORITY_MAX_PRECISION:
      return proto::GPUInferencePriority::GPU_PRIORITY_MAX_PRECISION;
    case GPUInferencePriority_GPU_PRIORITY_MIN_LATENCY:
      return proto::GPUInferencePriority::GPU_PRIORITY_MIN_LATENCY;
    case GPUInferencePriority_GPU_PRIORITY_MIN_MEMORY_USAGE:
      return proto::GPUInferencePriority::GPU_PRIORITY_MIN_MEMORY_USAGE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for GPUInferencePriority: %d", priority);
  return proto::GPUInferencePriority::GPU_PRIORITY_AUTO;
}

proto::EdgeTpuPowerState ConvertEdgeTpuPowerState(EdgeTpuPowerState state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_7(mht_7_v, 334, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertEdgeTpuPowerState");

  switch (state) {
    case EdgeTpuPowerState_UNDEFINED_POWERSTATE:
      return proto::EdgeTpuPowerState::UNDEFINED_POWERSTATE;
    case EdgeTpuPowerState_TPU_CORE_OFF:
      return proto::EdgeTpuPowerState::TPU_CORE_OFF;
    case EdgeTpuPowerState_READY:
      return proto::EdgeTpuPowerState::READY;
    case EdgeTpuPowerState_ACTIVE_MIN_POWER:
      return proto::EdgeTpuPowerState::ACTIVE_MIN_POWER;
    case EdgeTpuPowerState_ACTIVE_VERY_LOW_POWER:
      return proto::EdgeTpuPowerState::ACTIVE_VERY_LOW_POWER;
    case EdgeTpuPowerState_ACTIVE_LOW_POWER:
      return proto::EdgeTpuPowerState::ACTIVE_LOW_POWER;
    case EdgeTpuPowerState_ACTIVE:
      return proto::EdgeTpuPowerState::ACTIVE;
    case EdgeTpuPowerState_OVER_DRIVE:
      return proto::EdgeTpuPowerState::OVER_DRIVE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for EdgeTpuSettings::PowerState: %d",
                  state);
  return proto::EdgeTpuPowerState::UNDEFINED_POWERSTATE;
}

proto::FallbackSettings ConvertFallbackSettings(
    const FallbackSettings& settings) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_8(mht_8_v, 363, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertFallbackSettings");

  proto::FallbackSettings proto_settings;
  proto_settings.set_allow_automatic_fallback_on_compilation_error(
      settings.allow_automatic_fallback_on_compilation_error());
  proto_settings.set_allow_automatic_fallback_on_execution_error(
      settings.allow_automatic_fallback_on_execution_error());
  return proto_settings;
}

proto::NNAPISettings ConvertNNAPISettings(const NNAPISettings& settings) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_9(mht_9_v, 375, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertNNAPISettings");

  proto::NNAPISettings proto_settings;
  if (settings.accelerator_name() != nullptr) {
    proto_settings.set_accelerator_name(settings.accelerator_name()->str());
  }
  if (settings.cache_directory() != nullptr) {
    proto_settings.set_cache_directory(settings.cache_directory()->str());
  }
  if (settings.model_token() != nullptr) {
    proto_settings.set_model_token(settings.model_token()->str());
  }
  proto_settings.set_execution_preference(
      ConvertNNAPIExecutionPreference(settings.execution_preference()));
  proto_settings.set_no_of_nnapi_instances_to_cache(
      settings.no_of_nnapi_instances_to_cache());
  if (settings.fallback_settings() != nullptr) {
    *(proto_settings.mutable_fallback_settings()) =
        ConvertFallbackSettings(*settings.fallback_settings());
  }
  proto_settings.set_allow_nnapi_cpu_on_android_10_plus(
      settings.allow_nnapi_cpu_on_android_10_plus());
  proto_settings.set_execution_priority(
      ConvertNNAPIExecutionPriority(settings.execution_priority()));
  proto_settings.set_allow_dynamic_dimensions(
      settings.allow_dynamic_dimensions());
  proto_settings.set_allow_fp16_precision_for_fp32(
      settings.allow_fp16_precision_for_fp32());
  proto_settings.set_use_burst_computation(settings.use_burst_computation());
  proto_settings.set_support_library_handle(settings.support_library_handle());

  return proto_settings;
}

proto::GPUSettings ConvertGPUSettings(const GPUSettings& settings) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_10(mht_10_v, 411, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertGPUSettings");

  proto::GPUSettings proto_settings;
  proto_settings.set_is_precision_loss_allowed(
      settings.is_precision_loss_allowed());
  proto_settings.set_enable_quantized_inference(
      settings.enable_quantized_inference());
  proto_settings.set_force_backend(ConvertGPUBackend(settings.force_backend()));
  proto_settings.set_inference_priority1(
      ConvertGPUInferencePriority(settings.inference_priority1()));
  proto_settings.set_inference_priority2(
      ConvertGPUInferencePriority(settings.inference_priority2()));
  proto_settings.set_inference_priority3(
      ConvertGPUInferencePriority(settings.inference_priority3()));
  proto_settings.set_inference_preference(
      ConvertGPUInferenceUsage(settings.inference_preference()));
  if (settings.cache_directory() != nullptr) {
    proto_settings.set_cache_directory(settings.cache_directory()->str());
  }
  if (settings.model_token() != nullptr) {
    proto_settings.set_model_token(settings.model_token()->str());
  }
  return proto_settings;
}

proto::HexagonSettings ConvertHexagonSettings(const HexagonSettings& settings) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_11(mht_11_v, 438, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertHexagonSettings");

  proto::HexagonSettings proto_settings;
  proto_settings.set_debug_level(settings.debug_level());
  proto_settings.set_powersave_level(settings.powersave_level());
  proto_settings.set_print_graph_profile(settings.print_graph_profile());
  proto_settings.set_print_graph_debug(settings.print_graph_debug());
  return proto_settings;
}

proto::XNNPackSettings ConvertXNNPackSettings(const XNNPackSettings& settings) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_12(mht_12_v, 450, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertXNNPackSettings");

  proto::XNNPackSettings proto_settings;
  proto_settings.set_num_threads(settings.num_threads());
  return proto_settings;
}

proto::CoreMLSettings ConvertCoreMLSettings(const CoreMLSettings& settings) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_13(mht_13_v, 459, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertCoreMLSettings");

  proto::CoreMLSettings proto_settings;
  switch (settings.enabled_devices()) {
    case CoreMLSettings_::EnabledDevices_DEVICES_ALL:
      proto_settings.set_enabled_devices(proto::CoreMLSettings::DEVICES_ALL);
      break;
    case CoreMLSettings_::EnabledDevices_DEVICES_WITH_NEURAL_ENGINE:
      proto_settings.set_enabled_devices(
          proto::CoreMLSettings::DEVICES_WITH_NEURAL_ENGINE);
      break;
    default:
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Invalid devices enum: %d",
                      settings.enabled_devices());
  }
  proto_settings.set_coreml_version(settings.coreml_version());
  proto_settings.set_max_delegated_partitions(
      settings.max_delegated_partitions());
  proto_settings.set_min_nodes_per_partition(
      settings.min_nodes_per_partition());
  return proto_settings;
}

proto::CPUSettings ConvertCPUSettings(const CPUSettings& settings) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_14(mht_14_v, 484, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertCPUSettings");

  proto::CPUSettings proto_settings;
  proto_settings.set_num_threads(settings.num_threads());
  return proto_settings;
}

proto::EdgeTpuDeviceSpec ConvertEdgeTpuDeviceSpec(
    const EdgeTpuDeviceSpec& device_spec) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_15(mht_15_v, 494, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertEdgeTpuDeviceSpec");

  proto::EdgeTpuDeviceSpec proto_settings;

  if (device_spec.device_paths() != nullptr) {
    for (int i = 0; i < device_spec.device_paths()->size(); ++i) {
      auto device_path = device_spec.device_paths()->Get(i);
      proto_settings.add_device_paths(device_path->str());
    }
  }

  proto_settings.set_platform_type(
      static_cast<proto::EdgeTpuDeviceSpec::PlatformType>(
          device_spec.platform_type()));
  proto_settings.set_num_chips(device_spec.num_chips());
  proto_settings.set_chip_family(device_spec.chip_family());

  return proto_settings;
}

proto::EdgeTpuSettings ConvertEdgeTpuSettings(const EdgeTpuSettings& settings) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_16(mht_16_v, 516, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertEdgeTpuSettings");

  proto::EdgeTpuSettings proto_settings;
  proto_settings.set_inference_power_state(
      ConvertEdgeTpuPowerState(settings.inference_power_state()));
  proto_settings.set_inference_priority(settings.inference_priority());
  if (settings.model_token() != nullptr) {
    proto_settings.set_model_token(settings.model_token()->str());
  }
  if (settings.edgetpu_device_spec() != nullptr) {
    *(proto_settings.mutable_edgetpu_device_spec()) =
        ConvertEdgeTpuDeviceSpec(*settings.edgetpu_device_spec());
  }
  proto_settings.set_float_truncation_type(
      static_cast<proto::EdgeTpuSettings::FloatTruncationType>(
          settings.float_truncation_type()));

  auto inactive_powre_configs = settings.inactive_power_configs();
  if (inactive_powre_configs != nullptr) {
    for (int i = 0; i < inactive_powre_configs->size(); ++i) {
      auto config = inactive_powre_configs->Get(i);
      auto proto_config = proto_settings.add_inactive_power_configs();
      proto_config->set_inactive_power_state(
          ConvertEdgeTpuPowerState(config->inactive_power_state()));
      proto_config->set_inactive_timeout_us(config->inactive_timeout_us());
    }
  }

  return proto_settings;
}

proto::CoralSettings ConvertCoralSettings(const CoralSettings& settings) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_17(mht_17_v, 549, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertCoralSettings");

  proto::CoralSettings proto_settings;
  if (settings.device() != nullptr) {
    proto_settings.set_device(settings.device()->str());
  }
  proto_settings.set_performance(
      static_cast<proto::CoralSettings::Performance>(settings.performance()));
  proto_settings.set_usb_always_dfu(settings.usb_always_dfu());
  proto_settings.set_usb_max_bulk_in_queue_length(
      settings.usb_max_bulk_in_queue_length());
  return proto_settings;
}

proto::TFLiteSettings ConvertTfliteSettings(const TFLiteSettings& settings) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_18(mht_18_v, 565, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertTfliteSettings");

  proto::TFLiteSettings proto_settings;
  proto_settings.set_delegate(ConvertDelegate(settings.delegate()));
  if (settings.nnapi_settings() != nullptr) {
    *(proto_settings.mutable_nnapi_settings()) =
        ConvertNNAPISettings(*settings.nnapi_settings());
  }
  if (settings.gpu_settings() != nullptr) {
    *(proto_settings.mutable_gpu_settings()) =
        ConvertGPUSettings(*settings.gpu_settings());
  }
  if (settings.hexagon_settings() != nullptr) {
    *(proto_settings.mutable_hexagon_settings()) =
        ConvertHexagonSettings(*settings.hexagon_settings());
  }

  if (settings.xnnpack_settings() != nullptr) {
    *(proto_settings.mutable_xnnpack_settings()) =
        ConvertXNNPackSettings(*settings.xnnpack_settings());
  }

  if (settings.coreml_settings() != nullptr) {
    *(proto_settings.mutable_coreml_settings()) =
        ConvertCoreMLSettings(*settings.coreml_settings());
  }

  if (settings.cpu_settings() != nullptr) {
    *(proto_settings.mutable_cpu_settings()) =
        ConvertCPUSettings(*settings.cpu_settings());
  }

  proto_settings.set_max_delegated_partitions(
      settings.max_delegated_partitions());
  if (settings.edgetpu_settings() != nullptr) {
    *(proto_settings.mutable_edgetpu_settings()) =
        ConvertEdgeTpuSettings(*settings.edgetpu_settings());
  }
  if (settings.coral_settings() != nullptr) {
    *(proto_settings.mutable_coral_settings()) =
        ConvertCoralSettings(*settings.coral_settings());
  }
  if (settings.fallback_settings() != nullptr) {
    *(proto_settings.mutable_fallback_settings()) =
        ConvertFallbackSettings(*settings.fallback_settings());
  }
  return proto_settings;
}

proto::ModelFile ConvertModelFile(const ModelFile& model_file) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_19(mht_19_v, 616, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertModelFile");

  proto::ModelFile proto_settings;
  if (model_file.filename() != nullptr) {
    proto_settings.set_filename(model_file.filename()->str());
  }
  proto_settings.set_fd(model_file.fd());
  proto_settings.set_offset(model_file.offset());
  proto_settings.set_length(model_file.length());
  return proto_settings;
}

proto::BenchmarkStoragePaths ConvertBenchmarkStoragePaths(
    const BenchmarkStoragePaths& storage_paths) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_20(mht_20_v, 631, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBenchmarkStoragePaths");

  proto::BenchmarkStoragePaths proto_settings;
  if (storage_paths.storage_file_path() != nullptr) {
    proto_settings.set_storage_file_path(
        storage_paths.storage_file_path()->str());
  }
  if (storage_paths.data_directory_path() != nullptr) {
    proto_settings.set_data_directory_path(
        storage_paths.data_directory_path()->str());
  }
  return proto_settings;
}

proto::MinibenchmarkSettings ConvertMinibenchmarkSettings(
    const MinibenchmarkSettings& settings) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_21(mht_21_v, 648, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertMinibenchmarkSettings");

  proto::MinibenchmarkSettings proto_settings;
  if (settings.settings_to_test() != nullptr &&
      settings.settings_to_test()->size() > 0) {
    for (int i = 0; i < settings.settings_to_test()->size(); ++i) {
      auto tflite_setting = settings.settings_to_test()->Get(i);
      auto proto_tflite_setting = proto_settings.add_settings_to_test();
      *proto_tflite_setting = ConvertTfliteSettings(*tflite_setting);
    }
  }
  if (settings.model_file() != nullptr) {
    *(proto_settings.mutable_model_file()) =
        ConvertModelFile(*settings.model_file());
  }
  if (settings.storage_paths() != nullptr) {
    *(proto_settings.mutable_storage_paths()) =
        ConvertBenchmarkStoragePaths(*settings.storage_paths());
  }
  return proto_settings;
}

proto::BenchmarkEventType ConvertBenchmarkEventType(BenchmarkEventType type) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_22(mht_22_v, 672, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBenchmarkEventType");

  switch (type) {
    case BenchmarkEventType_UNDEFINED_BENCHMARK_EVENT_TYPE:
      return proto::BenchmarkEventType::UNDEFINED_BENCHMARK_EVENT_TYPE;
    case BenchmarkEventType_START:
      return proto::BenchmarkEventType::START;
    case BenchmarkEventType_END:
      return proto::BenchmarkEventType::END;
    case BenchmarkEventType_ERROR:
      return proto::BenchmarkEventType::ERROR;
    case BenchmarkEventType_LOGGED:
      return proto::BenchmarkEventType::LOGGED;
    case BenchmarkEventType_RECOVERED_ERROR:
      return proto::BenchmarkEventType::RECOVERED_ERROR;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for BenchmarkEventType: %d", type);
  return proto::BenchmarkEventType::UNDEFINED_BENCHMARK_EVENT_TYPE;
}

proto::BenchmarkMetric ConvertBenchmarkMetric(const BenchmarkMetric& metric) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_23(mht_23_v, 695, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBenchmarkMetric");

  proto::BenchmarkMetric proto_metric;
  if (metric.name() != nullptr) {
    proto_metric.set_name(metric.name()->str());
  }
  auto values = metric.values();
  if (values != nullptr) {
    for (int i = 0; i < values->size(); ++i) {
      proto_metric.add_values(values->Get(i));
    }
  }
  return proto_metric;
}

proto::BenchmarkResult ConvertBenchmarkResult(const BenchmarkResult& result) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_24(mht_24_v, 712, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBenchmarkResult");

  proto::BenchmarkResult proto_result;
  auto initialization_time_us = result.initialization_time_us();
  if (initialization_time_us != nullptr) {
    for (int i = 0; i < initialization_time_us->size(); ++i) {
      proto_result.add_initialization_time_us(initialization_time_us->Get(i));
    }
  }
  auto inference_time_us = result.inference_time_us();
  if (inference_time_us != nullptr) {
    for (int i = 0; i < inference_time_us->size(); ++i) {
      proto_result.add_inference_time_us(inference_time_us->Get(i));
    }
  }
  proto_result.set_max_memory_kb(result.max_memory_kb());
  proto_result.set_ok(result.ok());
  auto metrics = result.metrics();
  if (metrics != nullptr) {
    for (int i = 0; i < metrics->size(); ++i) {
      *proto_result.add_metrics() = ConvertBenchmarkMetric(*metrics->Get(i));
    }
  }
  return proto_result;
}

proto::BenchmarkStage ConvertBenchmarkStage(BenchmarkStage stage) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_25(mht_25_v, 740, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBenchmarkStage");

  switch (stage) {
    case BenchmarkStage_UNKNOWN:
      return proto::BenchmarkStage::UNKNOWN;
    case BenchmarkStage_INITIALIZATION:
      return proto::BenchmarkStage::INITIALIZATION;
    case BenchmarkStage_INFERENCE:
      return proto::BenchmarkStage::INFERENCE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for BenchmarkStage: %d",
                  stage);
  return proto::BenchmarkStage::UNKNOWN;
}

proto::ErrorCode ConvertBenchmarkErrorCode(const ErrorCode& code) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_26(mht_26_v, 757, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBenchmarkErrorCode");

  proto::ErrorCode proto_code;
  proto_code.set_source(ConvertDelegate(code.source()));
  proto_code.set_tflite_error(code.tflite_error());
  proto_code.set_underlying_api_error(code.underlying_api_error());
  return proto_code;
}

proto::BenchmarkError ConvertBenchmarkError(const BenchmarkError& error) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_27(mht_27_v, 768, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBenchmarkError");

  proto::BenchmarkError proto_error;
  proto_error.set_stage(ConvertBenchmarkStage(error.stage()));
  proto_error.set_exit_code(error.exit_code());
  proto_error.set_signal(error.signal());
  auto error_codes = error.error_code();
  if (error_codes != nullptr) {
    for (int i = 0; i < error_codes->size(); ++i) {
      *proto_error.add_error_code() =
          ConvertBenchmarkErrorCode(*error_codes->Get(i));
    }
  }
  proto_error.set_mini_benchmark_error_code(error.mini_benchmark_error_code());
  return proto_error;
}

proto::BenchmarkEvent ConvertBenchmarkEvent(const BenchmarkEvent& event) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_28(mht_28_v, 787, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBenchmarkEvent");

  proto::BenchmarkEvent proto_event;
  if (event.tflite_settings() != nullptr) {
    *proto_event.mutable_tflite_settings() =
        ConvertTfliteSettings(*event.tflite_settings());
  }
  proto_event.set_event_type(ConvertBenchmarkEventType(event.event_type()));
  if (event.result() != nullptr) {
    *proto_event.mutable_result() = ConvertBenchmarkResult(*event.result());
  }
  if (event.error() != nullptr) {
    *proto_event.mutable_error() = ConvertBenchmarkError(*event.error());
  }
  proto_event.set_boottime_us(event.boottime_us());
  proto_event.set_wallclock_us(event.wallclock_us());
  return proto_event;
}

proto::BestAccelerationDecision ConvertBestAccelerationDecision(
    const BestAccelerationDecision& decision) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_29(mht_29_v, 809, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBestAccelerationDecision");

  proto::BestAccelerationDecision proto_decision;
  proto_decision.set_number_of_source_events(
      decision.number_of_source_events());
  if (decision.min_latency_event() != nullptr) {
    *proto_decision.mutable_min_latency_event() =
        ConvertBenchmarkEvent(*decision.min_latency_event());
  }
  proto_decision.set_min_inference_time_us(decision.min_inference_time_us());
  return proto_decision;
}

proto::BenchmarkInitializationFailure ConvertBenchmarkInitializationFailure(
    const BenchmarkInitializationFailure& init_failure) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_30(mht_30_v, 825, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertBenchmarkInitializationFailure");

  proto::BenchmarkInitializationFailure proto_init_failure;
  proto_init_failure.set_initialization_status(
      init_failure.initialization_status());
  return proto_init_failure;
}

}  // namespace

proto::ComputeSettings ConvertFromFlatbuffer(
    const ComputeSettings& settings, bool skip_mini_benchmark_settings) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_31(mht_31_v, 838, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertFromFlatbuffer");

  proto::ComputeSettings proto_settings;

  proto_settings.set_preference(
      ConvertExecutionPreference(settings.preference()));
  if (settings.tflite_settings() != nullptr) {
    *(proto_settings.mutable_tflite_settings()) =
        ConvertTfliteSettings(*settings.tflite_settings());
  }
  if (settings.model_namespace_for_statistics() != nullptr) {
    proto_settings.set_model_namespace_for_statistics(
        settings.model_namespace_for_statistics()->str());
  }
  if (settings.model_identifier_for_statistics() != nullptr) {
    proto_settings.set_model_identifier_for_statistics(
        settings.model_identifier_for_statistics()->str());
  }
  if (!skip_mini_benchmark_settings &&
      settings.settings_to_test_locally() != nullptr) {
    *(proto_settings.mutable_settings_to_test_locally()) =
        ConvertMinibenchmarkSettings(*settings.settings_to_test_locally());
  }

  return proto_settings;
}

proto::ComputeSettings ConvertFromFlatbuffer(
    const ComputeSettingsT& settings, bool skip_mini_benchmark_settings) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_32(mht_32_v, 868, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertFromFlatbuffer");

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(ComputeSettings::Pack(fbb, &settings));
  auto settings_fbb =
      flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer());
  return ConvertFromFlatbuffer(*settings_fbb, skip_mini_benchmark_settings);
}

proto::MiniBenchmarkEvent ConvertFromFlatbuffer(
    const MiniBenchmarkEvent& event) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_33(mht_33_v, 880, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertFromFlatbuffer");

  proto::MiniBenchmarkEvent proto_event;
  proto_event.set_is_log_flushing_event(event.is_log_flushing_event());
  if (event.best_acceleration_decision() != nullptr) {
    *proto_event.mutable_best_acceleration_decision() =
        ConvertBestAccelerationDecision(*event.best_acceleration_decision());
  }
  if (event.initialization_failure() != nullptr) {
    *proto_event.mutable_initialization_failure() =
        ConvertBenchmarkInitializationFailure(*event.initialization_failure());
  }

  if (event.benchmark_event() != nullptr) {
    *proto_event.mutable_benchmark_event() =
        ConvertBenchmarkEvent(*event.benchmark_event());
  }

  return proto_event;
}

proto::MiniBenchmarkEvent ConvertFromFlatbuffer(
    const MiniBenchmarkEventT& event) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSflatbuffer_to_protoDTcc mht_34(mht_34_v, 904, "", "./tensorflow/lite/experimental/acceleration/configuration/flatbuffer_to_proto.cc", "ConvertFromFlatbuffer");

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(MiniBenchmarkEvent::Pack(fbb, &event));
  auto event_fbb =
      flatbuffers::GetRoot<MiniBenchmarkEvent>(fbb.GetBufferPointer());
  return ConvertFromFlatbuffer(*event_fbb);
}
}  // namespace tflite
