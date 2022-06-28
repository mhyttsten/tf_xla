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
class MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc() {
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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace random_uniform {

struct OpData {
  // This implementation uses a random generator from the standard C++ library
  // on the platform where TFLite is build. This is different from the TF
  // version of the kernel that uses custom implementations of random
  // generator, different for different hardware.
  std::default_random_engine rng;
};

namespace {

template <typename T, typename dist_type>
void RandomUniformSample(std::default_random_engine& rng, T* buffer,
                         size_t buffer_size, T min_value, T max_value) {
  dist_type dist(min_value, max_value);
  std::generate(buffer, buffer + buffer_size, [&]() { return dist(rng); });
}

TfLiteIntArray* CreateDimensionsFromTensor(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/kernels/random_uniform_custom.cc", "CreateDimensionsFromTensor");

  const int output_dims = tflite::SizeOfDimension(tensor, 0);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(output_dims);
  for (int i = 0; i < output_dims; i++) {
    if (tensor->type == kTfLiteInt32) {
      output_shape->data[i] = tensor->data.i32[i];
    } else {
      output_shape->data[i] = tensor->data.i64[i];
    }
  }
  return output_shape;
}
}  // namespace
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/kernels/random_uniform_custom.cc", "Init");

  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc mht_2(mht_2_v, 240, "", "./tensorflow/lite/kernels/random_uniform_custom.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc mht_3(mht_3_v, 247, "", "./tensorflow/lite/kernels/random_uniform_custom.cc", "Prepare");

  // TODO(b/169611265): Handle optional seed input.
  TF_LITE_ENSURE(context, tflite::NumInputs(node) >= 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  // Input is a shape tensor.
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteInt32 || input->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(input), 1);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  if (!IsConstantTensor(input)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return context->ResizeTensor(context, output,
                               CreateDimensionsFromTensor(input));
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc mht_4(mht_4_v, 269, "", "./tensorflow/lite/kernels/random_uniform_custom.cc", "EvalFloat");

  OpData* params = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  if (IsDynamicTensor(output)) {
    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output,
                                            CreateDimensionsFromTensor(input)));
  }
  const size_t output_size = tflite::NumElements(output);
  switch (output->type) {
    case kTfLiteFloat32:
      RandomUniformSample<float, std::uniform_real_distribution<float>>(
          params->rng, GetTensorData<float>(output), output_size, 0.f, 1.f);
      break;
    case kTfLiteFloat64:
      RandomUniformSample<double, std::uniform_real_distribution<double>>(
          params->rng, GetTensorData<double>(output), output_size, 0.f, 1.f);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported output datatype for RandomUniform: %s",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

int64_t IntValueFromTensor(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc mht_5(mht_5_v, 303, "", "./tensorflow/lite/kernels/random_uniform_custom.cc", "IntValueFromTensor");

  switch (tensor->type) {
    case kTfLiteInt8:
      return *GetTensorData<int8_t>(tensor);
    case kTfLiteInt32:
      return *GetTensorData<int32_t>(tensor);
    case kTfLiteInt64:
      return *GetTensorData<int64_t>(tensor);
    default:
      return -1;
  }
}

TfLiteStatus EvalInt(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc mht_6(mht_6_v, 319, "", "./tensorflow/lite/kernels/random_uniform_custom.cc", "EvalInt");

  OpData* params = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  TF_LITE_ENSURE(context, tflite::NumInputs(node) >= 3);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  if (IsDynamicTensor(output)) {
    const TfLiteTensor* input = tflite::GetInput(context, node, 0);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output,
                                            CreateDimensionsFromTensor(input)));
  }
  int64_t min_value = IntValueFromTensor(tflite::GetInput(context, node, 1));
  int64_t max_value = IntValueFromTensor(tflite::GetInput(context, node, 2));
  TF_LITE_ENSURE(context, min_value < max_value);
  size_t output_size = tflite::NumElements(output);
  switch (output->type) {
    case kTfLiteInt8:
      RandomUniformSample<int8_t, std::uniform_int_distribution<int32_t>>(
          params->rng, GetTensorData<int8_t>(output), output_size, min_value,
          max_value);
      break;
    case kTfLiteInt32:
      RandomUniformSample<int32_t, std::uniform_int_distribution<int32_t>>(
          params->rng, GetTensorData<int32_t>(output), output_size, min_value,
          max_value);
      break;
    case kTfLiteInt64:
      RandomUniformSample<int64_t, std::uniform_int_distribution<int64_t>>(
          params->rng, GetTensorData<int64_t>(output), output_size, min_value,
          max_value);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported output datatype for RandomUniformInt: %s",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace random_uniform

TfLiteRegistration* Register_RANDOM_UNIFORM() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc mht_7(mht_7_v, 366, "", "./tensorflow/lite/kernels/random_uniform_custom.cc", "Register_RANDOM_UNIFORM");

  static TfLiteRegistration r = {random_uniform::Init, random_uniform::Free,
                                 random_uniform::Prepare,
                                 random_uniform::EvalFloat};
  return &r;
}

TfLiteRegistration* Register_RANDOM_UNIFORM_INT() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_uniform_customDTcc mht_8(mht_8_v, 376, "", "./tensorflow/lite/kernels/random_uniform_custom.cc", "Register_RANDOM_UNIFORM_INT");

  static TfLiteRegistration r = {random_uniform::Init, random_uniform::Free,
                                 random_uniform::Prepare,
                                 random_uniform::EvalInt};
  return &r;
}
}  // namespace custom
}  // namespace ops
}  // namespace tflite
