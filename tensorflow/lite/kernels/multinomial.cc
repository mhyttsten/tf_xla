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
class MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc() {
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
namespace multinomial {

struct MultinomialParams {
  std::default_random_engine rng;
};

// Draws a sample from a categorical distribution.
template <typename FloatType, typename IntegralType>
TfLiteStatus MultinomialSample(std::default_random_engine& rng,
                               const FloatType* logits, int logits_size,
                               IntegralType* outputs, int output_size) {
  // Computes arg_max(cumsum(exp(logits)) > rand()).
  // TODO(b/169166131): Remove hard-coded double for constrained use-cases.
  std::vector<double> cumulative_odds;
  cumulative_odds.reserve(logits_size);
  double last_odds = 0.0;

  // Compute max logit for numerical stability.
  FloatType max_logit = std::numeric_limits<FloatType>::lowest();
  for (int i = 0; i < logits_size; i++) {
    max_logit = std::max(max_logit, logits[i]);
  }

  for (int i = 0; i < logits_size; i++) {
    FloatType odds = std::exp(logits[i] - max_logit) + last_odds;
    cumulative_odds.push_back(odds);
    last_odds = odds;
  }

  std::uniform_real_distribution<double> distribution{0.0,
                                                      cumulative_odds.back()};

  for (int i = 0; i < output_size; i++) {
    double sample = distribution(rng);
    auto it = std::lower_bound(cumulative_odds.begin(), cumulative_odds.end(),
                               sample);
    if (it == cumulative_odds.end()) {
      // This should be impossible by construction.
      return kTfLiteError;
    }
    *outputs++ = static_cast<IntegralType>(it - cumulative_odds.begin());
  }
  return kTfLiteOk;
}

template <typename FloatType>
TfLiteStatus MultinomialSample(TfLiteContext* context,
                               std::default_random_engine& rng,
                               const FloatType* logits, int logits_size,
                               TfLiteTensor* output, int outputs_offset,
                               int output_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc mht_0(mht_0_v, 247, "", "./tensorflow/lite/kernels/multinomial.cc", "MultinomialSample");

  switch (output->type) {
    case kTfLiteInt32:
      return MultinomialSample<FloatType, int32_t>(
          rng, logits, logits_size,
          GetTensorData<int32_t>(output) + outputs_offset, output_size);
      break;
    case kTfLiteInt64:
      return MultinomialSample<FloatType, int64_t>(
          rng, logits, logits_size,
          GetTensorData<int64_t>(output) + outputs_offset, output_size);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported datatype for multinomial output: %s",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
}

TfLiteStatus MultinomialSample(TfLiteContext* context,
                               std::default_random_engine& rng,
                               const TfLiteTensor* logits, int logits_offset,
                               int logits_size, TfLiteTensor* output,
                               int outputs_offset, int output_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc mht_1(mht_1_v, 274, "", "./tensorflow/lite/kernels/multinomial.cc", "MultinomialSample");

  switch (logits->type) {
    case kTfLiteFloat16:
      TF_LITE_KERNEL_LOG(context, "TfLiteFloat16 is currently not supported.");
      return kTfLiteError;
      break;
    case kTfLiteFloat32:
      TF_LITE_ENSURE_OK(
          context,
          MultinomialSample<float>(
              context, rng, GetTensorData<float>(logits) + logits_offset,
              logits_size, output, outputs_offset, output_size));
      break;
    case kTfLiteFloat64:
      TF_LITE_ENSURE_OK(
          context,
          MultinomialSample<double>(
              context, rng, GetTensorData<double>(logits) + logits_offset,
              logits_size, output, outputs_offset, output_size));
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported datatype for multinomial logit input: %s",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc mht_2(mht_2_v, 307, "", "./tensorflow/lite/kernels/multinomial.cc", "Init");

  return new MultinomialParams();
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc mht_3(mht_3_v, 314, "", "./tensorflow/lite/kernels/multinomial.cc", "Free");

  delete reinterpret_cast<MultinomialParams*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc mht_4(mht_4_v, 321, "", "./tensorflow/lite/kernels/multinomial.cc", "Prepare");

  // TODO(b/169166131): Handle optional seed input.
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  // 'logits' is a float matrix [batch_size, num_categories]
  const TfLiteTensor* logits_input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(logits_input), 2);
  int batch_size = tflite::SizeOfDimension(logits_input, 0);

  // 'num_samples' is an int scalar.
  const TfLiteTensor* num_samples_input = tflite::GetInput(context, node, 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(num_samples_input), 0);
  // TODO(b/169166131): Allow different integer input types.
  TF_LITE_ENSURE_EQ(context, num_samples_input->type, kTfLiteInt32);
  // TODO(b/169166131): Support dynamic output tensors.
  TF_LITE_ENSURE(context, IsConstantTensor(num_samples_input));

  int num_samples = *num_samples_input->data.i32;

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(2);
  output_shape->data[0] = batch_size;
  output_shape->data[1] = num_samples;

  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  // ResizeTensor takes ownership of output_shape.
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc mht_5(mht_5_v, 353, "", "./tensorflow/lite/kernels/multinomial.cc", "Eval");

  // TODO(b/169166131): Handle optional seed input.
  MultinomialParams* params =
      reinterpret_cast<MultinomialParams*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  const TfLiteTensor* logits = tflite::GetInput(context, node, 0);
  int batch_size = tflite::SizeOfDimension(logits, 0);
  int logits_size = tflite::SizeOfDimension(logits, 1);

  const TfLiteTensor* num_samples_input = tflite::GetInput(context, node, 1);
  int output_size = *num_samples_input->data.i32;

  TfLiteTensor* output = tflite::GetOutput(context, node, 0);

  for (int batch = 0; batch < batch_size; ++batch) {
    int logits_offset = logits_size * batch;
    int output_offset = output_size * batch;

    TF_LITE_ENSURE_OK(
        context,
        MultinomialSample(context, params->rng, logits, logits_offset,
                          logits_size, output, output_offset, output_size));
  }

  return kTfLiteOk;
}

}  // namespace multinomial

TfLiteRegistration* Register_MULTINOMIAL() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmultinomialDTcc mht_6(mht_6_v, 386, "", "./tensorflow/lite/kernels/multinomial.cc", "Register_MULTINOMIAL");

  static TfLiteRegistration r = {multinomial::Init, multinomial::Free,
                                 multinomial::Prepare, multinomial::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
