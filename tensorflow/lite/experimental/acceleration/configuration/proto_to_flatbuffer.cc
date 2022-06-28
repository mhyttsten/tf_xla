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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.h"

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {

using ::flatbuffers::FlatBufferBuilder;
using ::flatbuffers::Offset;
using ::flatbuffers::String;
using ::flatbuffers::Vector;

ExecutionPreference ConvertExecutionPreference(
    proto::ExecutionPreference preference) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertExecutionPreference");

  switch (preference) {
    case proto::ExecutionPreference::ANY:
      return ExecutionPreference_ANY;
    case proto::ExecutionPreference::LOW_LATENCY:
      return ExecutionPreference_LOW_LATENCY;
    case proto::ExecutionPreference::LOW_POWER:
      return ExecutionPreference_LOW_POWER;
    case proto::ExecutionPreference::FORCE_CPU:
      return ExecutionPreference_FORCE_CPU;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for ExecutionPreference: %d", preference);
  return ExecutionPreference_ANY;
}

Delegate ConvertDelegate(proto::Delegate delegate) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_1(mht_1_v, 218, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertDelegate");

  switch (delegate) {
    case proto::Delegate::NONE:
      return Delegate_NONE;
    case proto::Delegate::NNAPI:
      return Delegate_NNAPI;
    case proto::Delegate::GPU:
      return Delegate_GPU;
    case proto::Delegate::HEXAGON:
      return Delegate_HEXAGON;
    case proto::Delegate::XNNPACK:
      return Delegate_XNNPACK;
    case proto::Delegate::EDGETPU:
      return Delegate_EDGETPU;
    case proto::Delegate::EDGETPU_CORAL:
      return Delegate_EDGETPU_CORAL;
    case proto::Delegate::CORE_ML:
      return Delegate_CORE_ML;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for Delegate: %d",
                  delegate);
  return Delegate_NONE;
}

NNAPIExecutionPreference ConvertNNAPIExecutionPreference(
    proto::NNAPIExecutionPreference preference) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_2(mht_2_v, 246, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertNNAPIExecutionPreference");

  switch (preference) {
    case proto::NNAPIExecutionPreference::UNDEFINED:
      return NNAPIExecutionPreference_UNDEFINED;
    case proto::NNAPIExecutionPreference::NNAPI_LOW_POWER:
      return NNAPIExecutionPreference_NNAPI_LOW_POWER;
    case proto::NNAPIExecutionPreference::NNAPI_FAST_SINGLE_ANSWER:
      return NNAPIExecutionPreference_NNAPI_FAST_SINGLE_ANSWER;
    case proto::NNAPIExecutionPreference::NNAPI_SUSTAINED_SPEED:
      return NNAPIExecutionPreference_NNAPI_SUSTAINED_SPEED;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for NNAPIExecutionPreference: %d",
                  preference);
  return NNAPIExecutionPreference_UNDEFINED;
}

NNAPIExecutionPriority ConvertNNAPIExecutionPriority(
    proto::NNAPIExecutionPriority priority) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_3(mht_3_v, 267, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertNNAPIExecutionPriority");

  switch (priority) {
    case proto::NNAPIExecutionPriority::NNAPI_PRIORITY_UNDEFINED:
      return NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED;
    case proto::NNAPIExecutionPriority::NNAPI_PRIORITY_LOW:
      return NNAPIExecutionPriority_NNAPI_PRIORITY_LOW;
    case proto::NNAPIExecutionPriority::NNAPI_PRIORITY_MEDIUM:
      return NNAPIExecutionPriority_NNAPI_PRIORITY_MEDIUM;
    case proto::NNAPIExecutionPriority::NNAPI_PRIORITY_HIGH:
      return NNAPIExecutionPriority_NNAPI_PRIORITY_HIGH;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for NNAPIExecutionPriority: %d", priority);
  return NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED;
}

GPUBackend ConvertGPUBackend(proto::GPUBackend backend) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_4(mht_4_v, 286, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertGPUBackend");

  switch (backend) {
    case proto::GPUBackend::UNSET:
      return GPUBackend_UNSET;
    case proto::GPUBackend::OPENCL:
      return GPUBackend_OPENCL;
    case proto::GPUBackend::OPENGL:
      return GPUBackend_OPENGL;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Unexpected value for GPUBackend: %d",
                  backend);
  return GPUBackend_UNSET;
}

GPUInferenceUsage ConvertGPUInferenceUsage(
    proto::GPUInferenceUsage preference) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_5(mht_5_v, 304, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertGPUInferenceUsage");

  switch (preference) {
    case proto::GPUInferenceUsage::GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER:
      return GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    case proto::GPUInferenceUsage::GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED:
      return GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for GPUInferenceUsage: %d", preference);
  return GPUInferenceUsage_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
}

GPUInferencePriority ConvertGPUInferencePriority(
    proto::GPUInferencePriority priority) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_6(mht_6_v, 320, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertGPUInferencePriority");

  switch (priority) {
    case proto::GPUInferencePriority::GPU_PRIORITY_AUTO:
      return GPUInferencePriority_GPU_PRIORITY_AUTO;
    case proto::GPUInferencePriority::GPU_PRIORITY_MAX_PRECISION:
      return GPUInferencePriority_GPU_PRIORITY_MAX_PRECISION;
    case proto::GPUInferencePriority::GPU_PRIORITY_MIN_LATENCY:
      return GPUInferencePriority_GPU_PRIORITY_MIN_LATENCY;
    case proto::GPUInferencePriority::GPU_PRIORITY_MIN_MEMORY_USAGE:
      return GPUInferencePriority_GPU_PRIORITY_MIN_MEMORY_USAGE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for GPUInferencePriority: %d", priority);
  return GPUInferencePriority_GPU_PRIORITY_AUTO;
}

EdgeTpuPowerState ConvertEdgeTpuPowerState(proto::EdgeTpuPowerState state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_7(mht_7_v, 339, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertEdgeTpuPowerState");

  switch (state) {
    case proto::EdgeTpuPowerState::UNDEFINED_POWERSTATE:
      return EdgeTpuPowerState_UNDEFINED_POWERSTATE;
    case proto::EdgeTpuPowerState::TPU_CORE_OFF:
      return EdgeTpuPowerState_TPU_CORE_OFF;
    case proto::EdgeTpuPowerState::READY:
      return EdgeTpuPowerState_READY;
    case proto::EdgeTpuPowerState::ACTIVE_MIN_POWER:
      return EdgeTpuPowerState_ACTIVE_MIN_POWER;
    case proto::EdgeTpuPowerState::ACTIVE_VERY_LOW_POWER:
      return EdgeTpuPowerState_ACTIVE_VERY_LOW_POWER;
    case proto::EdgeTpuPowerState::ACTIVE_LOW_POWER:
      return EdgeTpuPowerState_ACTIVE_LOW_POWER;
    case proto::EdgeTpuPowerState::ACTIVE:
      return EdgeTpuPowerState_ACTIVE;
    case proto::EdgeTpuPowerState::OVER_DRIVE:
      return EdgeTpuPowerState_OVER_DRIVE;
  }
  TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                  "Unexpected value for EdgeTpuSettings::PowerState: %d",
                  state);
  return EdgeTpuPowerState_UNDEFINED_POWERSTATE;
}

Offset<FallbackSettings> ConvertFallbackSettings(
    const proto::FallbackSettings& settings, FlatBufferBuilder* builder) {
  return CreateFallbackSettings(
      *builder, /*allow_automatic_fallback_on_compilation_error=*/
      settings.allow_automatic_fallback_on_compilation_error(),
      /*allow_automatic_fallback_on_execution_error=*/
      settings.allow_automatic_fallback_on_execution_error());
}

Offset<NNAPISettings> ConvertNNAPISettings(const proto::NNAPISettings& settings,
                                           FlatBufferBuilder* builder) {
  return CreateNNAPISettings(
      *builder,
      /*accelerator_name=*/builder->CreateString(settings.accelerator_name()),
      /*cache_directory=*/builder->CreateString(settings.cache_directory()),
      /*model_token=*/builder->CreateString(settings.model_token()),
      ConvertNNAPIExecutionPreference(settings.execution_preference()),
      /*no_of_nnapi_instances_to_cache=*/
      settings.no_of_nnapi_instances_to_cache(),
      ConvertFallbackSettings(settings.fallback_settings(), builder),
      /*allow_nnapi_cpu_on_android_10_plus=*/
      settings.allow_nnapi_cpu_on_android_10_plus(),
      ConvertNNAPIExecutionPriority(settings.execution_priority()),
      /*allow_dynamic_dimensions=*/
      settings.allow_dynamic_dimensions(),
      /*allow_fp16_precision_for_fp32=*/
      settings.allow_fp16_precision_for_fp32(),
      /*use_burst_computation=*/
      settings.use_burst_computation(),
      /*support_library_handle=*/
      settings.support_library_handle());
}

Offset<GPUSettings> ConvertGPUSettings(const proto::GPUSettings& settings,
                                       FlatBufferBuilder* builder) {
  return CreateGPUSettings(
      *builder,
      /*is_precision_loss_allowed=*/settings.is_precision_loss_allowed(),
      /*enable_quantized_inference=*/settings.enable_quantized_inference(),
      ConvertGPUBackend(settings.force_backend()),
      ConvertGPUInferencePriority(settings.inference_priority1()),
      ConvertGPUInferencePriority(settings.inference_priority2()),
      ConvertGPUInferencePriority(settings.inference_priority3()),
      ConvertGPUInferenceUsage(settings.inference_preference()),
      /*cache_directory=*/builder->CreateString(settings.cache_directory()),
      /*model_token=*/builder->CreateString(settings.model_token()));
}

Offset<HexagonSettings> ConvertHexagonSettings(
    const proto::HexagonSettings& settings, FlatBufferBuilder* builder) {
  return CreateHexagonSettings(
      *builder,
      /*debug_level=*/settings.debug_level(),
      /*powersave_level=*/settings.powersave_level(),
      /*print_graph_profile=*/settings.print_graph_profile(),
      /*print_graph_debug=*/settings.print_graph_debug());
}

Offset<XNNPackSettings> ConvertXNNPackSettings(
    const proto::XNNPackSettings& settings, FlatBufferBuilder* builder) {
  return CreateXNNPackSettings(*builder,
                               /*num_threads=*/settings.num_threads());
}

Offset<CoreMLSettings> ConvertCoreMLSettings(
    const proto::CoreMLSettings& settings, FlatBufferBuilder* builder) {
  tflite::CoreMLSettings_::EnabledDevices enabled_devices =
      tflite::CoreMLSettings_::EnabledDevices_DEVICES_ALL;
  switch (settings.enabled_devices()) {
    case proto::CoreMLSettings::DEVICES_ALL:
      enabled_devices = tflite::CoreMLSettings_::EnabledDevices_DEVICES_ALL;
      break;
    case proto::CoreMLSettings::DEVICES_WITH_NEURAL_ENGINE:
      enabled_devices =
          tflite::CoreMLSettings_::EnabledDevices_DEVICES_WITH_NEURAL_ENGINE;
      break;
    default:
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Invalid devices enum: %d",
                      settings.enabled_devices());
  }

  return CreateCoreMLSettings(
      *builder, enabled_devices, settings.coreml_version(),
      settings.max_delegated_partitions(), settings.min_nodes_per_partition());
}

Offset<CPUSettings> ConvertCPUSettings(const proto::CPUSettings& settings,
                                       FlatBufferBuilder* builder) {
  return CreateCPUSettings(*builder,
                           /*num_threads=*/settings.num_threads());
}

Offset<tflite::EdgeTpuDeviceSpec> ConvertEdgeTpuDeviceSpec(
    FlatBufferBuilder* builder, const proto::EdgeTpuDeviceSpec& device_spec) {
  Offset<Vector<Offset<String>>> device_paths_fb = 0;
  if (device_spec.device_paths_size() > 0) {
    std::vector<Offset<String>> device_paths;
    for (const auto& device_path : device_spec.device_paths()) {
      auto device_path_fb = builder->CreateString(device_path);
      device_paths.push_back(device_path_fb);
    }
    device_paths_fb = builder->CreateVector(device_paths);
  }

  return tflite::CreateEdgeTpuDeviceSpec(
      *builder,
      static_cast<tflite::EdgeTpuDeviceSpec_::PlatformType>(
          device_spec.platform_type()),
      device_spec.num_chips(), device_paths_fb, device_spec.chip_family());
}

Offset<EdgeTpuSettings> ConvertEdgeTpuSettings(
    const proto::EdgeTpuSettings& settings, FlatBufferBuilder* builder) {
  Offset<Vector<Offset<tflite::EdgeTpuInactivePowerConfig>>>
      inactive_power_configs = 0;

  // Uses std vector to first construct the list and creates the flatbuffer
  // offset from it later.
  std::vector<Offset<tflite::EdgeTpuInactivePowerConfig>>
      inactive_power_configs_std;
  if (settings.inactive_power_configs_size() > 0) {
    for (const auto& config : settings.inactive_power_configs()) {
      inactive_power_configs_std.push_back(
          tflite::CreateEdgeTpuInactivePowerConfig(
              *builder,
              static_cast<tflite::EdgeTpuPowerState>(
                  config.inactive_power_state()),
              config.inactive_timeout_us()));
    }

    inactive_power_configs =
        builder->CreateVector<Offset<tflite::EdgeTpuInactivePowerConfig>>(
            inactive_power_configs_std);
  }

  Offset<tflite::EdgeTpuDeviceSpec> edgetpu_device_spec = 0;
  if (settings.has_edgetpu_device_spec()) {
    edgetpu_device_spec =
        ConvertEdgeTpuDeviceSpec(builder, settings.edgetpu_device_spec());
  }

  Offset<String> model_token = 0;
  if (settings.has_model_token()) {
    model_token = builder->CreateString(settings.model_token());
  }

  return CreateEdgeTpuSettings(
      *builder, ConvertEdgeTpuPowerState(settings.inference_power_state()),
      inactive_power_configs, settings.inference_priority(),
      edgetpu_device_spec, model_token,
      static_cast<tflite::EdgeTpuSettings_::FloatTruncationType>(
          settings.float_truncation_type()),
      static_cast<tflite::EdgeTpuSettings_::QosClass>(settings.qos_class()));
}

Offset<CoralSettings> ConvertCoralSettings(const proto::CoralSettings& settings,
                                           FlatBufferBuilder* builder) {
  return CreateCoralSettings(
      *builder, builder->CreateString(settings.device()),
      static_cast<tflite::CoralSettings_::Performance>(settings.performance()),
      settings.usb_always_dfu(), settings.usb_max_bulk_in_queue_length());
}

Offset<TFLiteSettings> ConvertTfliteSettings(
    const proto::TFLiteSettings& settings, FlatBufferBuilder* builder) {
  return CreateTFLiteSettings(
      *builder, ConvertDelegate(settings.delegate()),
      ConvertNNAPISettings(settings.nnapi_settings(), builder),
      ConvertGPUSettings(settings.gpu_settings(), builder),
      ConvertHexagonSettings(settings.hexagon_settings(), builder),
      ConvertXNNPackSettings(settings.xnnpack_settings(), builder),
      ConvertCoreMLSettings(settings.coreml_settings(), builder),
      ConvertCPUSettings(settings.cpu_settings(), builder),
      /*max_delegated_partitions=*/settings.max_delegated_partitions(),
      ConvertEdgeTpuSettings(settings.edgetpu_settings(), builder),
      ConvertCoralSettings(settings.coral_settings(), builder),
      ConvertFallbackSettings(settings.fallback_settings(), builder));
}

Offset<ModelFile> ConvertModelFile(const proto::ModelFile& model_file,
                                   FlatBufferBuilder* builder) {
  return CreateModelFile(*builder, builder->CreateString(model_file.filename()),
                         model_file.fd(), model_file.offset(),
                         model_file.length());
}

Offset<BenchmarkStoragePaths> ConvertBenchmarkStoragePaths(
    const proto::BenchmarkStoragePaths& storage_paths,
    FlatBufferBuilder* builder) {
  return CreateBenchmarkStoragePaths(
      *builder, builder->CreateString(storage_paths.storage_file_path()),
      builder->CreateString(storage_paths.data_directory_path()));
}

Offset<MinibenchmarkSettings> ConvertMinibenchmarkSettings(
    const proto::MinibenchmarkSettings& settings, FlatBufferBuilder* builder) {
  Offset<Vector<Offset<TFLiteSettings>>> settings_to_test = 0;
  std::vector<Offset<TFLiteSettings>> settings_to_test_vec;
  if (settings.settings_to_test_size() > 0) {
    for (const auto& one : settings.settings_to_test()) {
      settings_to_test_vec.push_back(ConvertTfliteSettings(one, builder));
    }
    settings_to_test =
        builder->CreateVector<Offset<TFLiteSettings>>(settings_to_test_vec);
  }

  return CreateMinibenchmarkSettings(
      *builder, settings_to_test,
      ConvertModelFile(settings.model_file(), builder),
      ConvertBenchmarkStoragePaths(settings.storage_paths(), builder));
}

const ComputeSettings* ConvertFromProto(
    const proto::ComputeSettings& proto_settings, FlatBufferBuilder* builder) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_8(mht_8_v, 580, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertFromProto");

  auto settings = CreateComputeSettings(
      *builder, ConvertExecutionPreference(proto_settings.preference()),
      ConvertTfliteSettings(proto_settings.tflite_settings(), builder),
      builder->CreateString(proto_settings.model_namespace_for_statistics()),
      builder->CreateString(proto_settings.model_identifier_for_statistics()),
      ConvertMinibenchmarkSettings(proto_settings.settings_to_test_locally(),
                                   builder));
  return flatbuffers::GetTemporaryPointer(*builder, settings);
}

const MinibenchmarkSettings* ConvertFromProto(
    const proto::MinibenchmarkSettings& proto_settings,
    flatbuffers::FlatBufferBuilder* builder) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSproto_to_flatbufferDTcc mht_9(mht_9_v, 596, "", "./tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.cc", "ConvertFromProto");

  auto settings = ConvertMinibenchmarkSettings(proto_settings, builder);
  return flatbuffers::GetTemporaryPointer(*builder, settings);
}

}  // namespace tflite
