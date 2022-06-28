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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSperformance_profilingDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSperformance_profilingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSperformance_profilingDTcc() {
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

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <iostream>
#include <string>

#include "absl/time/time.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace gpu {
namespace cl {

absl::Status RunPredefinedLayoutSample(const std::string& model_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("model_name: \"" + model_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSperformance_profilingDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/delegates/gpu/cl/testing/performance_profiling.cc", "RunPredefinedLayoutSample");

  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  GraphFloat32 graph_cl;
  ops::builtin::BuiltinOpResolver op_resolver;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl,
                                      /*allow_quant_ops=*/true));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  CreateGpuModelInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device().GetInfo());
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  {
    // Example of adding predefined descriptor
    // Assumed that graph has first input with batch = 1.
    auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
    create_info.predefined[graph_cl.inputs()[0]->id] =
        TensorDescriptor{data_type, TensorStorageType::BUFFER, Layout::HWC};
  }
  std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
  std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
  InferenceContext context;
  RETURN_IF_ERROR(
      context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

  // After initialization we can receive input tensor
  // in_ten will have TensorStorageType::BUFFER storage type
  Tensor* in_ten = context.GetTensor(graph_cl.inputs()[0]->id);
  if (in_ten->GetStorageType() != TensorStorageType::BUFFER) {
    return absl::InternalError("Failed preconditiion");
  }

  RETURN_IF_ERROR(context.AddToQueue(env.queue()));

  std::cout << "Finished RunPredefinedLayoutSample." << std::endl;

  return absl::OkStatus();
}

absl::Status RunExternalImmutableSample(const std::string& model_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("model_name: \"" + model_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSperformance_profilingDTcc mht_1(mht_1_v, 251, "", "./tensorflow/lite/delegates/gpu/cl/testing/performance_profiling.cc", "RunExternalImmutableSample");

  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  GraphFloat32 graph_cl;
  ops::builtin::BuiltinOpResolver op_resolver;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl,
                                      /*allow_quant_ops*/ true));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  CreateGpuModelInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device().GetInfo());
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  // Example of external immutable tensors:
  std::vector<Tensor> outputs(graph_cl.outputs().size());
  for (int i = 0; i < graph_cl.outputs().size(); ++i) {
    // Assumed that graph outputs have batch size = 1.
    auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
    RETURN_IF_ERROR(CreateTensor(
        env.context(), graph_cl.outputs()[i]->tensor.shape,
        TensorDescriptor{data_type, TensorStorageType::TEXTURE_ARRAY,
                         Layout::HWC},
        &outputs[i]));
    create_info.external_immutable_tensors[graph_cl.outputs()[i]->id] =
        &outputs[i];
  }
  std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
  std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
  InferenceContext context;
  RETURN_IF_ERROR(
      context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

  RETURN_IF_ERROR(context.AddToQueue(env.queue()));

  // outputs can be used here. But AddToQueue do not have cpu
  // syncronization.
  RETURN_IF_ERROR(env.queue()->WaitForCompletion());

  const auto dst_shape = BHWC(outputs[0].Batch(), outputs[0].Height(),
                              outputs[0].Width(), outputs[0].Channels());
  TensorFloat32 cpu_tensor;
  cpu_tensor.shape = dst_shape;
  cpu_tensor.data.resize(dst_shape.DimensionsProduct());
  RETURN_IF_ERROR(outputs[0].ReadData(env.queue(), &cpu_tensor));
  std::cout << "First tensor data at index 0 - " << cpu_tensor.data[0]
            << std::endl;

  return absl::OkStatus();
}

absl::Status RunSerializedTest(const std::string& model_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("model_name: \"" + model_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSperformance_profilingDTcc mht_2(mht_2_v, 309, "", "./tensorflow/lite/delegates/gpu/cl/testing/performance_profiling.cc", "RunSerializedTest");

  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  GraphFloat32 graph_cl;
  ops::builtin::BuiltinOpResolver op_resolver;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl,
                                      /*allow_quant_ops*/ true));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  CreateGpuModelInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device().GetInfo());
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);

  {  // calculating time without building serialized model
    InferenceContext test_context;
    const auto start = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(
        test_context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    std::cout << "Inference context initialization total time - "
              << total_time_ms << "ms" << std::endl;
  }
  InferenceContext context;
  std::vector<uint8_t> serialized_model;
  RETURN_IF_ERROR(context.InitFromGraphWithTransforms(create_info, &graph_cl,
                                                      &env, &serialized_model));

  std::vector<TensorFloat32> src_tensors(graph_cl.inputs().size());
  for (int i = 0; i < graph_cl.inputs().size(); ++i) {
    src_tensors[i].id = graph_cl.inputs()[i]->id;
    src_tensors[i].shape = graph_cl.inputs()[i]->tensor.shape;
    src_tensors[i].data.resize(src_tensors[i].shape.DimensionsProduct());
    for (int j = 0; j < src_tensors[i].data.size(); ++j) {
      src_tensors[i].data[j] = std::sin(j);
    }
  }
  for (int i = 0; i < graph_cl.inputs().size(); ++i) {
    RETURN_IF_ERROR(context.SetInputTensor(graph_cl.inputs()[i]->id,
                                           src_tensors[i], env.queue()));
  }
  RETURN_IF_ERROR(context.AddToQueue(env.queue()));
  RETURN_IF_ERROR(env.queue()->WaitForCompletion());

  std::vector<TensorFloat32> dst_tensors(graph_cl.outputs().size());
  for (int i = 0; i < graph_cl.outputs().size(); ++i) {
    RETURN_IF_ERROR(context.GetOutputTensor(graph_cl.outputs()[i]->id,
                                            env.queue(), &dst_tensors[i]));
  }

  Environment env_v2;
  RETURN_IF_ERROR(CreateEnvironment(&env_v2));
  InferenceContext serialized_context;
  {
    const auto start = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(
        serialized_context.RestoreDeserialized(serialized_model, &env_v2));
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    std::cout << "Serialized inference context initialization total time - "
              << total_time_ms << "ms" << std::endl;
  }
  for (int i = 0; i < graph_cl.inputs().size(); ++i) {
    RETURN_IF_ERROR(serialized_context.SetInputTensor(
        graph_cl.inputs()[i]->id, src_tensors[i], env_v2.queue()));
  }

  RETURN_IF_ERROR(serialized_context.AddToQueue(env_v2.queue()));
  RETURN_IF_ERROR(env_v2.queue()->WaitForCompletion());

  std::vector<TensorFloat32> dst_tensors_v2(graph_cl.outputs().size());
  for (int i = 0; i < graph_cl.outputs().size(); ++i) {
    RETURN_IF_ERROR(serialized_context.GetOutputTensor(
        graph_cl.outputs()[i]->id, env_v2.queue(), &dst_tensors_v2[i]));
  }

  for (int i = 0; i < graph_cl.outputs().size(); ++i) {
    if (dst_tensors[i].data.size() != dst_tensors_v2[i].data.size()) {
      std::cout << "Different sizes for " << i << " output tensor" << std::endl;
      break;
    }
    for (int j = 0; j < dst_tensors[i].data.size(); ++j) {
      if (dst_tensors[i].data[j] != dst_tensors_v2[i].data[j]) {
        std::cout << "Different elements for " << j << " element in " << i
                  << " tensor: " << dst_tensors[i].data[j] << " - "
                  << dst_tensors_v2[i].data[j] << std::endl;
        break;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status RunModelSample(const std::string& model_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("model_name: \"" + model_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSperformance_profilingDTcc mht_3(mht_3_v, 411, "", "./tensorflow/lite/delegates/gpu/cl/testing/performance_profiling.cc", "RunModelSample");

  auto flatbuffer = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  GraphFloat32 graph_cl;
  ops::builtin::BuiltinOpResolver op_resolver;
  RETURN_IF_ERROR(BuildFromFlatBuffer(*flatbuffer, op_resolver, &graph_cl,
                                      /*allow_quant_ops*/ true));

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));

  CreateGpuModelInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device().GetInfo());
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  std::cout << "Precision: " << ToString(create_info.precision) << std::endl;
  std::cout << "Storage type: " << ToString(create_info.storage_type)
            << std::endl;
  InferenceContext context;
  RETURN_IF_ERROR(
      context.InitFromGraphWithTransforms(create_info, &graph_cl, &env));

  auto* queue = env.profiling_queue();
  ProfilingInfo profiling_info;
  RETURN_IF_ERROR(context.Profile(queue, &profiling_info));
  std::cout << profiling_info.GetDetailedReport() << std::endl;
  const uint64_t runtime_mem_bytes =
      context.GetSizeOfMemoryAllocatedForIntermediateTensors();
  std::cout << "Memory for intermediate tensors - "
            << runtime_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
  const uint64_t const_mem_bytes = context.GetConstantTensorsSize();
  std::cout << "Memory for constant tensors - "
            << const_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
  std::cout << "Total tensors memory(const + intermediate) - "
            << (const_mem_bytes + runtime_mem_bytes) / 1024.0 / 1024.0 << " MB"
            << std::endl;

  const int num_runs_per_sec = std::max(
      1, static_cast<int>(1000.0f / absl::ToDoubleMilliseconds(
                                        profiling_info.GetTotalTime())));

  const int kNumRuns = 10;
  for (int i = 0; i < kNumRuns; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < num_runs_per_sec; ++k) {
      RETURN_IF_ERROR(context.AddToQueue(env.queue()));
    }
    RETURN_IF_ERROR(env.queue()->WaitForCompletion());
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_time_ms = (end - start).count() * 1e-6f;
    const double average_inference_time = total_time_ms / num_runs_per_sec;
    std::cout << "Total time - " << average_inference_time << "ms" << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

int main(int argc, char** argv) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSperformance_profilingDTcc mht_4(mht_4_v, 476, "", "./tensorflow/lite/delegates/gpu/cl/testing/performance_profiling.cc", "main");

  if (argc <= 1) {
    std::cerr << "Expected model path as second argument.";
    return -1;
  }

  auto load_status = tflite::gpu::cl::LoadOpenCL();
  if (!load_status.ok()) {
    std::cerr << load_status.message();
    return -1;
  }

  auto run_status = tflite::gpu::cl::RunModelSample(argv[1]);
  if (!run_status.ok()) {
    std::cerr << run_status.message();
    return -1;
  }

  bool run_serialized_test = false;
  if (run_serialized_test) {
    run_status = tflite::gpu::cl::RunSerializedTest(argv[1]);
    if (!run_status.ok()) {
      std::cerr << run_status.message();
      return -1;
    }
  }

  bool run_with_external_immutable_tensors = false;
  if (run_with_external_immutable_tensors) {
    run_status = tflite::gpu::cl::RunExternalImmutableSample(argv[1]);
    if (!run_status.ok()) {
      std::cerr << run_status.message();
      return -1;
    }
  }

  bool run_with_predefined_layout = false;
  if (run_with_predefined_layout) {
    run_status = tflite::gpu::cl::RunPredefinedLayoutSample(argv[1]);
    if (!run_status.ok()) {
      std::cerr << run_status.message();
      return -1;
    }
  }

  return EXIT_SUCCESS;
}
