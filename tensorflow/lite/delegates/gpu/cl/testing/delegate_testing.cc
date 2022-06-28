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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSdelegate_testingDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSdelegate_testingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSdelegate_testingDTcc() {
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
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "absl/time/time.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_api_delegate.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

namespace {

void FillInputTensor(tflite::Interpreter* interpreter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSdelegate_testingDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/delegates/gpu/cl/testing/delegate_testing.cc", "FillInputTensor");

  for (int k = 0; k < interpreter->inputs().size(); ++k) {
    TfLiteTensor* tensor_ptr = interpreter->tensor(interpreter->inputs()[k]);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);
    if (tensor_ptr->type == kTfLiteFloat32) {
      float* p = interpreter->typed_input_tensor<float>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = std::sin(i);
      }
    }
    if (tensor_ptr->type == kTfLiteInt32) {
      int* p = interpreter->typed_input_tensor<int>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 32;
      }
    }
    if (tensor_ptr->type == kTfLiteInt8) {
      int8_t* p = interpreter->typed_input_tensor<int8_t>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 256 - 128;
      }
    }
    if (tensor_ptr->type == kTfLiteUInt8) {
      uint8_t* p = interpreter->typed_input_tensor<uint8_t>(k);
      for (int i = 0; i < tensor_elements_count; ++i) {
        p[i] = i % 256;
      }
    }
  }
}

void CompareCPUGPUResults(tflite::Interpreter* cpu, tflite::Interpreter* gpu,
                          float eps) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSdelegate_testingDTcc mht_1(mht_1_v, 237, "", "./tensorflow/lite/delegates/gpu/cl/testing/delegate_testing.cc", "CompareCPUGPUResults");

  for (int i = 0; i < cpu->outputs().size(); ++i) {
    TfLiteTensor* tensor_ptr = cpu->tensor(cpu->outputs()[i]);
    const auto tensor_elements_count = tflite::NumElements(tensor_ptr);

    std::cout << "Output " << tensor_ptr->name << ":" << std::endl;

    const int kMaxPrint = 10;
    int printed = 0;
    int total_different = 0;
    for (int k = 0; k < tensor_elements_count; ++k) {
      float cpu_val = 0.0f;
      float gpu_val = 0.0f;
      if (tensor_ptr->type == kTfLiteFloat32) {
        const float* cpu_out = cpu->typed_output_tensor<float>(i);
        const float* gpu_out = gpu->typed_output_tensor<float>(i);
        cpu_val = cpu_out[k];
        gpu_val = gpu_out[k];
      }
      if (tensor_ptr->type == kTfLiteInt8) {
        int8_t* cpu_out = cpu->typed_output_tensor<int8_t>(i);
        int8_t* gpu_out = gpu->typed_output_tensor<int8_t>(i);
        cpu_val = cpu_out[k];
        gpu_val = gpu_out[k];
      }
      if (tensor_ptr->type == kTfLiteUInt8) {
        uint8_t* cpu_out = cpu->typed_output_tensor<uint8_t>(i);
        uint8_t* gpu_out = gpu->typed_output_tensor<uint8_t>(i);
        cpu_val = cpu_out[k];
        gpu_val = gpu_out[k];
      }
      const float abs_diff = fabs(cpu_val - gpu_val);
      if (abs_diff > eps) {
        total_different++;
        if (printed < kMaxPrint) {
          std::cout << "Element #" << k << ": CPU value - " << cpu_val
                    << ", GPU value - " << gpu_val << ", abs diff - "
                    << abs_diff << std::endl;
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

int main(int argc, char** argv) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPStestingPSdelegate_testingDTcc mht_2(mht_2_v, 296, "", "./tensorflow/lite/delegates/gpu/cl/testing/delegate_testing.cc", "main");

  if (argc <= 1) {
    std::cerr << "Expected model path as second argument." << std::endl;
    return -1;
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
  if (!model) {
    std::cerr << "FlatBufferModel::BuildFromFile failed, model path - "
              << argv[1] << std::endl;
    return -1;
  }
  tflite::ops::builtin::BuiltinOpResolver op_resolver;
  tflite::InterpreterBuilder builder(*model, op_resolver);

  // CPU.
  std::unique_ptr<tflite::Interpreter> cpu_inference;
  builder(&cpu_inference);
  if (!cpu_inference) {
    std::cerr << "Failed to build CPU inference." << std::endl;
    return -1;
  }
  auto status = cpu_inference->AllocateTensors();
  if (status != kTfLiteOk) {
    std::cerr << "Failed to AllocateTensors for CPU inference." << std::endl;
    return -1;
  }
  FillInputTensor(cpu_inference.get());
  status = cpu_inference->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Failed to Invoke CPU inference." << std::endl;
    return -1;
  }

  // GPU.
  std::unique_ptr<tflite::Interpreter> gpu_inference;
  builder(&gpu_inference);
  if (!gpu_inference) {
    std::cerr << "Failed to build GPU inference." << std::endl;
    return -1;
  }
  TfLiteGpuDelegateOptionsV2 options;
  options.is_precision_loss_allowed = -1;
  options.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
  options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
  options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
  options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  options.max_delegated_partitions = 1;
  auto* gpu_delegate = TfLiteGpuDelegateV2Create(&options);
  status = gpu_inference->ModifyGraphWithDelegate(gpu_delegate);
  if (status != kTfLiteOk) {
    std::cerr << "ModifyGraphWithDelegate failed." << std::endl;
    return -1;
  }
  FillInputTensor(gpu_inference.get());
  status = gpu_inference->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Failed to Invoke GPU inference." << std::endl;
    return -1;
  }

  CompareCPUGPUResults(cpu_inference.get(), gpu_inference.get(), 1e-4f);

  // CPU inference latency.
  auto start = std::chrono::high_resolution_clock::now();
  cpu_inference->Invoke();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "CPU time - " << (end - start).count() * 1e-6f << "ms"
            << std::endl;

  // GPU inference latency.
  start = std::chrono::high_resolution_clock::now();
  gpu_inference->Invoke();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "GPU time(CPU->GPU->CPU) - " << (end - start).count() * 1e-6f
            << "ms" << std::endl;

  TfLiteGpuDelegateV2Delete(gpu_delegate);
  return EXIT_SUCCESS;
}
