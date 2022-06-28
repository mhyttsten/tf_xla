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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSinternal_api_samplesDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSinternal_api_samplesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSinternal_api_samplesDTcc() {
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

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <iostream>
#include <string>

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
void FillInputTensors(tflite::Interpreter* interpreter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSinternal_api_samplesDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/gpu/cl/testing/internal_api_samples.cc", "FillInputTensors");

  for (int k = 0; k < interpreter->inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr = interpreter->tensor(interpreter->inputs()[k]);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);
    if (tensor_ptr->type == kTfLiteFloat32) {
      float* p = interpreter->typed_input_tensor<float>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = std::sin(i);
      }
    } else {
      std::cout << "No support of non Float32 input/output tensors"
                << std::endl;
    }
  }
}

void CompareCPUGPUResults(tflite::Interpreter* cpu,
                          const std::vector<int64_t>& outputs,
                          const std::vector<std::vector<float>>& gpu,
                          float eps) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSinternal_api_samplesDTcc mht_1(mht_1_v, 228, "", "./tensorflow/lite/delegates/gpu/cl/testing/internal_api_samples.cc", "CompareCPUGPUResults");

  for (int i = 0; i < gpu.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu->tensor(outputs[i]);
    const float* cpu_out = tensor_ptr->data.f;
    const float* gpu_out = gpu[i].data();
    const int kMaxPrint = 10;
    int printed = 0;
    int total_different = 0;
    for (int k = 0; k < tensor_ptr->bytes / 4; ++k) {
      const float abs_diff = fabs(cpu_out[k] - gpu_out[k]);
      if (abs_diff > eps) {
        total_different++;
        if (printed < kMaxPrint) {
          std::cout << "Output #" << i << ": element #" << k << ": CPU value - "
                    << cpu_out[k] << ", GPU value - " << gpu_out[k]
                    << ", abs diff - " << abs_diff << std::endl;
          printed++;
        }
        if (printed == kMaxPrint) {
          std::cout << "Printed " << kMaxPrint
                    << " different elements, threshhold - " << eps
                    << ", next different elements skipped" << std::endl;
          printed++;
        }
      }
    }
    std::cout << "Total " << total_different
              << " different elements, for output #" << i << ", threshhold - "
              << eps << std::endl;
  }
}
}  // namespace

absl::Status RunModelSampleWithInternalAPISerializedKernels(
    const std::string& model_name, const std::vector<uint8_t>& kernel_cache);

absl::Status RunModelSampleWithInternalAPISerialized(
    tflite::Interpreter* cpu, const std::vector<uint8_t>& kernel_cache,
    const std::vector<uint8_t>& serialized_model);

// Run Jet with OpenCL internal API and compares correctness with TFLite CPU
absl::Status RunModelSampleWithInternalAPI(const std::string& model_name,
                                           std::vector<uint8_t>* kernel_cache) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("model_name: \"" + model_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSinternal_api_samplesDTcc mht_2(mht_2_v, 274, "", "./tensorflow/lite/delegates/gpu/cl/testing/internal_api_samples.cc", "RunModelSampleWithInternalAPI");

  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());

  ops::builtin::BuiltinOpResolver op_resolver;
  InterpreterBuilder tfl_builder(*flatbuffer, op_resolver);

  // CPU.
  std::unique_ptr<tflite::Interpreter> cpu_inference;
  tfl_builder(&cpu_inference);
  if (!cpu_inference) {
    return absl::InternalError("Failed to build CPU inference.");
  }
  auto status = cpu_inference->AllocateTensors();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to AllocateTensors for CPU inference.");
  }
  for (int k = 0; k < cpu_inference->inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr =
        cpu_inference->tensor(cpu_inference->inputs()[k]);
    if (tensor_ptr->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "Internal api supports only F32 input tensors");
    }
  }
  for (int k = 0; k < cpu_inference->outputs().size(); ++k) {
    TfLiteTensor* tensor_ptr =
        cpu_inference->tensor(cpu_inference->outputs()[k]);
    if (tensor_ptr->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "Internal api supports only F32 output tensors");
    }
  }
  FillInputTensors(cpu_inference.get());
  status = cpu_inference->Invoke();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to Invoke CPU inference.");
  }

  const auto start = std::chrono::high_resolution_clock::now();
  GraphFloat32 graph_cl;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl));

  auto inputs = graph_cl.inputs();
  auto outputs = graph_cl.outputs();
  std::vector<int64_t> in_refs(inputs.size());
  std::vector<int64_t> out_refs(outputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    in_refs[i] = inputs[i]->tensor.ref;
  }
  for (int i = 0; i < outputs.size(); ++i) {
    out_refs[i] = outputs[i]->tensor.ref;
  }

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  std::unique_ptr<InferenceEnvironment> inf_env;
  // Initializes environment.
  InferenceEnvironmentOptions env_options;
  env_options.device = env.device().id();
  env_options.context = env.context().context();
  env_options.command_queue = env.queue()->queue();
  RETURN_IF_ERROR(NewInferenceEnvironment(env_options, &inf_env, nullptr));

  std::unique_ptr<InferenceBuilder> builder;
  // Initializes builder.
  InferenceOptions options;
  options.priority1 = InferencePriority::MIN_LATENCY;
  options.priority2 = InferencePriority::MIN_MEMORY_USAGE;
  options.priority3 = InferencePriority::MAX_PRECISION;
  options.usage = InferenceUsage::SUSTAINED_SPEED;

  RETURN_IF_ERROR(
      inf_env->NewInferenceBuilder(options, std::move(graph_cl), &builder));

  // Sets input/output object def for builder_.
  ObjectDef obj_def;
  obj_def.data_type = DataType::FLOAT32;
  obj_def.data_layout = DataLayout::BHWC;
  obj_def.object_type = ObjectType::CPU_MEMORY;
  obj_def.user_provided = true;
  for (int i = 0; i < in_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetInputObjectDef(i, obj_def));
  }
  for (int i = 0; i < out_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetOutputObjectDef(i, obj_def));
  }

  std::unique_ptr<::tflite::gpu::InferenceRunner> runner;
  // Builds runner.
  RETURN_IF_ERROR(builder->Build(&runner));

  const auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Initialization total time - " << (end - start).count() * 1e-6f
            << "ms" << std::endl;

  if (kernel_cache) {
    *kernel_cache = inf_env->GetSerializedBinaryCache();
    std::cout << "Kernel cache size - " << kernel_cache->size() << std::endl;
  }

  // Sets the input/output object.
  for (int i = 0; i < in_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu_inference->tensor(in_refs[i]);
    RETURN_IF_ERROR(runner->SetInputObject(
        i, CpuMemory{tensor_ptr->data.data, tensor_ptr->bytes}));
  }

  std::vector<std::vector<float>> output_tensors(out_refs.size());
  for (int i = 0; i < out_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu_inference->tensor(out_refs[i]);
    output_tensors[i].resize(tensor_ptr->bytes / 4);
    RETURN_IF_ERROR(runner->SetOutputObject(
        i, CpuMemory{output_tensors[i].data(), tensor_ptr->bytes}));
  }

  RETURN_IF_ERROR(runner->Run());

  CompareCPUGPUResults(cpu_inference.get(), out_refs, output_tensors, 1e-4f);

  return absl::OkStatus();
}

absl::Status RunModelSampleWithInternalAPISerializedKernels(
    const std::string& model_name, const std::vector<uint8_t>& kernel_cache) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("model_name: \"" + model_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSinternal_api_samplesDTcc mht_3(mht_3_v, 402, "", "./tensorflow/lite/delegates/gpu/cl/testing/internal_api_samples.cc", "RunModelSampleWithInternalAPISerializedKernels");

  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());

  ops::builtin::BuiltinOpResolver op_resolver;
  InterpreterBuilder tfl_builder(*flatbuffer, op_resolver);

  // CPU.
  std::unique_ptr<tflite::Interpreter> cpu_inference;
  tfl_builder(&cpu_inference);
  if (!cpu_inference) {
    return absl::InternalError("Failed to build CPU inference.");
  }
  auto status = cpu_inference->AllocateTensors();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to AllocateTensors for CPU inference.");
  }
  for (int k = 0; k < cpu_inference->inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr =
        cpu_inference->tensor(cpu_inference->inputs()[k]);
    if (tensor_ptr->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "Internal api supports only F32 input tensors");
    }
  }
  for (int k = 0; k < cpu_inference->outputs().size(); ++k) {
    TfLiteTensor* tensor_ptr =
        cpu_inference->tensor(cpu_inference->outputs()[k]);
    if (tensor_ptr->type != kTfLiteFloat32) {
      return absl::InvalidArgumentError(
          "Internal api supports only F32 output tensors");
    }
  }
  FillInputTensors(cpu_inference.get());
  status = cpu_inference->Invoke();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to Invoke CPU inference.");
  }

  const auto start = std::chrono::high_resolution_clock::now();
  GraphFloat32 graph_cl;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl));

  auto inputs = graph_cl.inputs();
  auto outputs = graph_cl.outputs();
  std::vector<int64_t> in_refs(inputs.size());
  std::vector<int64_t> out_refs(outputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    in_refs[i] = inputs[i]->tensor.ref;
  }
  for (int i = 0; i < outputs.size(); ++i) {
    out_refs[i] = outputs[i]->tensor.ref;
  }

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  std::unique_ptr<InferenceEnvironment> inf_env;
  // Initializes environment.
  InferenceEnvironmentOptions env_options;
  env_options.device = env.device().id();
  env_options.context = env.context().context();
  env_options.command_queue = env.queue()->queue();
  env_options.serialized_binary_cache =
      absl::MakeSpan(kernel_cache.data(), kernel_cache.size());
  RETURN_IF_ERROR(NewInferenceEnvironment(env_options, &inf_env, nullptr));

  InferenceOptions options;
  options.priority1 = InferencePriority::MIN_LATENCY;
  options.priority2 = InferencePriority::MIN_MEMORY_USAGE;
  options.priority3 = InferencePriority::MAX_PRECISION;
  options.usage = InferenceUsage::SUSTAINED_SPEED;

  std::vector<uint8_t> serialized_model;
  RETURN_IF_ERROR(inf_env->BuildSerializedModel(options, std::move(graph_cl),
                                                &serialized_model));
  std::unique_ptr<InferenceBuilder> builder;
  RETURN_IF_ERROR(inf_env->NewInferenceBuilder(serialized_model, &builder));

  // Sets input/output object def for builder_.
  ObjectDef obj_def;
  obj_def.data_type = DataType::FLOAT32;
  obj_def.data_layout = DataLayout::BHWC;
  obj_def.object_type = ObjectType::CPU_MEMORY;
  obj_def.user_provided = true;
  for (int i = 0; i < in_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetInputObjectDef(i, obj_def));
  }
  for (int i = 0; i < out_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetOutputObjectDef(i, obj_def));
  }

  std::unique_ptr<::tflite::gpu::InferenceRunner> runner;
  // Builds runner.
  RETURN_IF_ERROR(builder->Build(&runner));

  const auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Initialization total time";
  if (!kernel_cache.empty()) {
    std::cout << "(with kernel cache)";
  }
  std::cout << " - " << (end - start).count() * 1e-6f << "ms" << std::endl;

  // Sets the input/output object.
  for (int i = 0; i < in_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu_inference->tensor(in_refs[i]);
    RETURN_IF_ERROR(runner->SetInputObject(
        i, CpuMemory{tensor_ptr->data.data, tensor_ptr->bytes}));
  }

  std::vector<std::vector<float>> output_tensors(out_refs.size());
  for (int i = 0; i < out_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu_inference->tensor(out_refs[i]);
    output_tensors[i].resize(tensor_ptr->bytes / 4);
    RETURN_IF_ERROR(runner->SetOutputObject(
        i, CpuMemory{output_tensors[i].data(), tensor_ptr->bytes}));
  }

  RETURN_IF_ERROR(runner->Run());

  CompareCPUGPUResults(cpu_inference.get(), out_refs, output_tensors, 1e-4f);

  RETURN_IF_ERROR(RunModelSampleWithInternalAPISerialized(
      cpu_inference.get(), kernel_cache, serialized_model));

  return absl::OkStatus();
}

absl::Status RunModelSampleWithInternalAPISerialized(
    tflite::Interpreter* cpu, const std::vector<uint8_t>& kernel_cache,
    const std::vector<uint8_t>& serialized_model) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSinternal_api_samplesDTcc mht_4(mht_4_v, 534, "", "./tensorflow/lite/delegates/gpu/cl/testing/internal_api_samples.cc", "RunModelSampleWithInternalAPISerialized");

  FillInputTensors(cpu);
  auto status = cpu->Invoke();
  if (status != kTfLiteOk) {
    return absl::InternalError("Failed to Invoke CPU inference.");
  }

  const auto start = std::chrono::high_resolution_clock::now();

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  std::unique_ptr<InferenceEnvironment> inf_env;
  // Initializes environment.
  InferenceEnvironmentOptions env_options;
  env_options.device = env.device().id();
  env_options.context = env.context().context();
  env_options.command_queue = env.queue()->queue();
  env_options.serialized_binary_cache =
      absl::MakeSpan(kernel_cache.data(), kernel_cache.size());
  RETURN_IF_ERROR(NewInferenceEnvironment(env_options, &inf_env, nullptr));

  std::vector<int64_t> in_refs;
  std::vector<int64_t> out_refs;
  RETURN_IF_ERROR(GetInOutRefs(serialized_model, &in_refs, &out_refs));
  std::unique_ptr<InferenceBuilder> builder;
  RETURN_IF_ERROR(inf_env->NewInferenceBuilder(serialized_model, &builder));

  // Sets input/output object def for builder_.
  ObjectDef obj_def;
  obj_def.data_type = DataType::FLOAT32;
  obj_def.data_layout = DataLayout::BHWC;
  obj_def.object_type = ObjectType::CPU_MEMORY;
  obj_def.user_provided = true;
  for (int i = 0; i < in_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetInputObjectDef(i, obj_def));
  }
  for (int i = 0; i < out_refs.size(); ++i) {
    RETURN_IF_ERROR(builder->SetOutputObjectDef(i, obj_def));
  }

  std::unique_ptr<::tflite::gpu::InferenceRunner> runner;
  // Builds runner.
  RETURN_IF_ERROR(builder->Build(&runner));

  const auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Serialized initialization total time";
  if (kernel_cache.empty()) {
    std::cout << "(without kernel cache)";
  } else {
    std::cout << "(with kernel cache)";
  }
  std::cout << " - " << (end - start).count() * 1e-6f << "ms" << std::endl;

  // Sets the input/output object.
  for (int i = 0; i < in_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu->tensor(in_refs[i]);
    RETURN_IF_ERROR(runner->SetInputObject(
        i, CpuMemory{tensor_ptr->data.data, tensor_ptr->bytes}));
  }

  std::vector<std::vector<float>> output_tensors(out_refs.size());
  for (int i = 0; i < out_refs.size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu->tensor(out_refs[i]);
    output_tensors[i].resize(tensor_ptr->bytes / 4);
    RETURN_IF_ERROR(runner->SetOutputObject(
        i, CpuMemory{output_tensors[i].data(), tensor_ptr->bytes}));
  }

  RETURN_IF_ERROR(runner->Run());

  std::cout << "Comparing results second time:" << std::endl;

  CompareCPUGPUResults(cpu, out_refs, output_tensors, 1e-4f);

  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

int main(int argc, char** argv) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSinternal_api_samplesDTcc mht_5(mht_5_v, 619, "", "./tensorflow/lite/delegates/gpu/cl/testing/internal_api_samples.cc", "main");

  if (argc <= 1) {
    std::cerr << "Expected model path as second argument.";
    return -1;
  }

  auto load_status = tflite::gpu::cl::LoadOpenCL();
  if (!load_status.ok()) {
    std::cerr << load_status.message();
    return -1;
  }

  std::vector<uint8_t> kernel_cache;
  auto run_status =
      tflite::gpu::cl::RunModelSampleWithInternalAPI(argv[1], &kernel_cache);
  if (!run_status.ok()) {
    std::cerr << run_status.message();
    return -1;
  }
  run_status = tflite::gpu::cl::RunModelSampleWithInternalAPISerializedKernels(
      argv[1], kernel_cache);
  if (!run_status.ok()) {
    std::cerr << run_status.message();
    return -1;
  }

  // The same with empty kernels cache.
  run_status = tflite::gpu::cl::RunModelSampleWithInternalAPISerializedKernels(
      argv[1], {});
  if (!run_status.ok()) {
    std::cerr << run_status.message();
    return -1;
  }

  return EXIT_SUCCESS;
}
