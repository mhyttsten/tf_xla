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
class MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc() {
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
#include <random>

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions_utils.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace random {

namespace {

using Generator = ::tensorflow::random::PhiloxRandom;

enum RandomType { kRandomUniform, kRandomStandardNormal, kMultinomial };

struct OpData {
  Generator rng;
};

// Initialize the OpData based on the seed and seed2 values.
void InitializeOpData(TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_0(mht_0_v, 211, "", "./tensorflow/lite/kernels/random_ops.cc", "InitializeOpData");

  static std::mt19937_64* seed_generator = []() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_1(mht_1_v, 215, "", "./tensorflow/lite/kernels/random_ops.cc", "lambda");

    std::random_device device("/dev/urandom");
    return new std::mt19937_64(device());
  }();
  auto* params = static_cast<TfLiteRandomParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  int64_t seed = params->seed;
  int64_t seed2 = params->seed2;
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, generate non-deterministic random numbers.
    seed = (*seed_generator)();
    seed2 = (*seed_generator)();
  }
  Generator rng(seed, seed2);
  data->rng = rng;
}

// Generates random numbers following a uniform distribution.
// Source: third_party/tensorflow/core/kernels/random_op.cc
void GenerateRandomUniformNumbers(
    Generator& rng, float* buffer, size_t buffer_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_2(mht_2_v, 238, "", "./tensorflow/lite/kernels/random_ops.cc", "GenerateRandomUniformNumbers");

  size_t current_size = 0;
  size_t rng_size = Generator::kResultElementCount;

  while (current_size < buffer_size) {
    typename Generator::ResultType samples = rng();
    const int rng_net_size = std::min(rng_size, buffer_size - current_size);
    for (int i = 0; i < rng_net_size; i++) {
      buffer[current_size + i] = tensorflow::random::Uint32ToFloat(samples[i]);
    }
    current_size += rng_net_size;
  }
}

// Generates random numbers following a standard normal distribution.
// Source: third_party/tensorflow/core/kernels/random_op.cc
void GenerateRandomStandardNormalNumbers(
    Generator& rng, float* buffer, size_t buffer_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_3(mht_3_v, 258, "", "./tensorflow/lite/kernels/random_ops.cc", "GenerateRandomStandardNormalNumbers");

  size_t current_size = 0;
  size_t rng_size = Generator::kResultElementCount;

  while (current_size < buffer_size) {
    typename Generator::ResultType samples = rng();
    const int rng_net_size = std::min(rng_size, buffer_size - current_size);
    for (int i = 0; i < rng_net_size; i += 2) {
      tensorflow::random::BoxMullerFloat(samples[i], samples[i + 1],
                                         &buffer[current_size + i],
                                         &buffer[current_size + i + 1]);
    }
    current_size += rng_net_size;
  }
}

// Generates random numbers following a multinomial distribution.
// Source: third_party/tensorflow/core/kernels/multinomial_op.cc
template <typename IntType>
void GenerateMultinomialNumbers(Generator& rng, int batch_size,
                                const float* logits, size_t logits_size,
                                IntType* output, size_t num_samples) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_4(mht_4_v, 282, "", "./tensorflow/lite/kernels/random_ops.cc", "GenerateMultinomialNumbers");

  // Skip a large fixed number of samples in the rng (random number generator)
  // for each op invoke to ensure that the output is always unique. (Make a copy
  // of the rng before skipping samples to use it in the current op invoke)
  // Context: This feature (to skip fixed samples) was added in TF as some
  // versions of the Multinomial op draw an unknown number of samples from the
  // rng. Though the TFLite version below only draws a fixed number of samples,
  // we still need to keep this feature to maintain parity with the TF op.
  Generator rng_copy = rng;
  rng.Skip(batch_size * ((num_samples + 3) / 4 * 4) * 2 *
           256);  // Round to a multiple of 4, 2x is for CPU and 256 is a
                  // conservative multiplier

  // Variables to store intermediate results between batches.
  typename Generator::ResultType rng_results;
  int used_rng_results_index = Generator::kResultElementCount;
  typename Generator::ResultElementType x0, x1;

  // Iterate over all batches to compute the outputs.
  for (int batch = 0; batch < batch_size; ++batch) {
    const float* logits_row = logits + batch * logits_size;
    IntType* output_row = output + batch * num_samples;

    // Compute the maximum logit.
    float max = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < logits_size; i++) {
      if (std::isfinite(logits_row[i])) {
        max = std::max(max, logits_row[i]);
      }
    }
    const double max_logit = static_cast<double>(max);

    // Compute the (unnormalized) cumulative probability distribution.
    // For numerical stability (as the exponential function grows very fast),
    // subtract the maximum logit. Though you can subtract any value without
    // changing the output, we use the maximum logit for convenience.
    std::vector<double> cdf(logits_size);
    double cumulative_total = 0.0f;
    for (size_t i = 0; i < logits_size; i++) {
      if (std::isfinite(logits_row[i])) {
        cumulative_total += exp(logits_row[i] - max_logit);
      }
      cdf[i] = cumulative_total;
    }

    // Generate random categorical numbers and populate the output.
    for (int64_t j = 0; j < num_samples; ++j) {
      if (used_rng_results_index == Generator::kResultElementCount) {
        rng_results = rng_copy();
        used_rng_results_index = 0;
      }
      x0 = rng_results[used_rng_results_index];
      x1 = rng_results[used_rng_results_index + 1];
      used_rng_results_index += 2;
      const double to_find =
          (tensorflow::random::Uint64ToDouble(x0, x1) * cumulative_total);
      auto found_iter = std::upper_bound(cdf.begin(), cdf.end(), to_find);
      output_row[j] = std::distance(cdf.begin(), found_iter);
    }
  }
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_5(mht_5_v, 350, "", "./tensorflow/lite/kernels/random_ops.cc", "Init");

  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_6(mht_6_v, 357, "", "./tensorflow/lite/kernels/random_ops.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_7(mht_7_v, 364, "", "./tensorflow/lite/kernels/random_ops.cc", "Prepare");

  // Validate number of inputs and outputs
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // 'shape' is a 1-D int array
  const TfLiteTensor* shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &shape));
  TF_LITE_ENSURE_EQ(context, shape->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(shape), 1);

  // Initialize the random number generator
  InitializeOpData(node);

  TfLiteTensor* output = GetOutput(context, node, 0);
  if (!IsConstantTensor(shape)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  TfLiteIntArray* output_shape;
  TF_LITE_ENSURE_OK(context,
                    GetOutputShapeFromInput(context, shape, &output_shape));
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus PrepareMultinomial(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_8(mht_8_v, 392, "", "./tensorflow/lite/kernels/random_ops.cc", "PrepareMultinomial");

  // Validate number of inputs and outputs
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // 'logits' is a 2-D input float matrix with shape [batch_size, num_classes]
  const TfLiteTensor* logits;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &logits));
  TF_LITE_ENSURE(context, logits->type == kTfLiteFloat32);

  // 'num_samples' is a 0-D input int scalar
  const TfLiteTensor* num_samples;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &num_samples));
  TF_LITE_ENSURE_EQ(context, num_samples->type, kTfLiteInt32);

  // Initialize the random number generator
  InitializeOpData(node);

  TfLiteTensor* output = GetOutput(context, node, 0);
  if (!IsConstantTensor(logits) || !IsConstantTensor(num_samples)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }

  // 'output' is a 2-D int64 matrix with shape [batch_size, num_samples]
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(2);
  output_shape->data[0] = SizeOfDimension(logits, 0);  // batch_size
  output_shape->data[1] = *num_samples->data.i32;      // num_samples
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus EvalRandomType(
    TfLiteContext* context, TfLiteNode* node, RandomType random_type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_9(mht_9_v, 427, "", "./tensorflow/lite/kernels/random_ops.cc", "EvalRandomType");

  TfLiteTensor* output = GetOutput(context, node, 0);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const size_t output_size = NumElements(output);
  switch (random_type) {
    case kRandomUniform:
      GenerateRandomUniformNumbers(
        data->rng, GetTensorData<float>(output), output_size);
      break;
    case kRandomStandardNormal:
      GenerateRandomStandardNormalNumbers(
        data->rng, GetTensorData<float>(output), output_size);
      break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <RandomType rtype>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_10(mht_10_v, 450, "", "./tensorflow/lite/kernels/random_ops.cc", "Eval");

  TfLiteTensor* output = GetOutput(context, node, 0);

  if (IsDynamicTensor(output)) {
    const TfLiteTensor* shape = GetInput(context, node, 0);
    TfLiteIntArray* output_shape;
    TF_LITE_ENSURE_OK(context,
                      GetOutputShapeFromInput(context, shape, &output_shape));
    context->ResizeTensor(context, output, output_shape);
  }

  switch (output->type) {
    case kTfLiteFloat32:
        EvalRandomType(context, node, rtype);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Unsupported output datatype for %s op: %s",
          rtype == kRandomUniform? "RandomUniform": "RandomStandardNormal",
          TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMultinomial(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_11(mht_11_v, 478, "", "./tensorflow/lite/kernels/random_ops.cc", "EvalMultinomial");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // 'logits' is a 2-D float matrix with shape [batch_size, num_classes]
  const TfLiteTensor* logits_tensor = GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(logits_tensor), 2);
  const float* logits = GetTensorData<float>(logits_tensor);
  const int batch_size = SizeOfDimension(logits_tensor, 0);
  const int num_classes = SizeOfDimension(logits_tensor, 1);
  TF_LITE_ENSURE(context, num_classes > 0);

  // 'num_samples' is an int scalar
  const TfLiteTensor* num_samples_tensor = GetInput(context, node, 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(num_samples_tensor), 0);
  const int num_samples = *num_samples_tensor->data.i32;
  TF_LITE_ENSURE(context, num_samples >= 0);

  TfLiteTensor* output_tensor = GetOutput(context, node, 0);
  if (IsDynamicTensor(output_tensor)) {
    // 'output' is a 2-D int64 matrix with shape [batch_size, num_samples]
    TfLiteIntArray* output_shape = TfLiteIntArrayCreate(2);
    output_shape->data[0] = batch_size;
    output_shape->data[1] = num_samples;
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, output_tensor, output_shape));
  }

  switch (output_tensor->type) {
    case kTfLiteInt64:
      GenerateMultinomialNumbers<int64_t>(
          data->rng, batch_size, logits, num_classes,
          GetTensorData<int64_t>(output_tensor), num_samples);
      break;
    case kTfLiteInt32:
      GenerateMultinomialNumbers<int32_t>(
          data->rng, batch_size, logits, num_classes,
          GetTensorData<int32_t>(output_tensor), num_samples);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported output datatype for Multinomial op: %s",
                         TfLiteTypeGetName(output_tensor->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace random

TfLiteRegistration* Register_RANDOM_UNIFORM() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_12(mht_12_v, 530, "", "./tensorflow/lite/kernels/random_ops.cc", "Register_RANDOM_UNIFORM");

  static TfLiteRegistration r = {random::Init, random::Free, random::Prepare,
                                 random::Eval<random::kRandomUniform>};
  return &r;
}

TfLiteRegistration* Register_RANDOM_STANDARD_NORMAL() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_13(mht_13_v, 539, "", "./tensorflow/lite/kernels/random_ops.cc", "Register_RANDOM_STANDARD_NORMAL");

  static TfLiteRegistration r = {random::Init, random::Free, random::Prepare,
                                 random::Eval<random::kRandomStandardNormal>};
  return &r;
}

TfLiteRegistration* Register_MULTINOMIAL() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSrandom_opsDTcc mht_14(mht_14_v, 548, "", "./tensorflow/lite/kernels/random_ops.cc", "Register_MULTINOMIAL");

  static TfLiteRegistration r = {random::Init, random::Free,
                                 random::PrepareMultinomial,
                                 random::EvalMultinomial};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
