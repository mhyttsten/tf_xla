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
class MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc() {
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
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/dequantize.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace numeric_verify {

static constexpr const char kToleranceStr[] = "tolerance";
static constexpr const char kLogIfFailedStr[] = "log_if_failed";
static constexpr const int kTemporaryDequantizedTensor = 0;
static constexpr const int kOutputTensor = 0;

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc mht_0(mht_0_v, 215, "", "./tensorflow/lite/kernels/numeric_verify.cc", "OpContext");

    input = GetInput(context, node, 0);
    ref = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* ref;
  TfLiteTensor* output;
};

const int kTensorNotAllocated = -1;

struct OpData {
  // The percentage of the tensor value range. Must be a number less than 1.0.
  float tolerance;
  // This boolean value is only used when the input tensor is constant.
  bool float_input_initialized;
  int cache_tensor_id = kTensorNotAllocated;
  // This boolean value is for controlling the behavior of numeric verify op.
  bool log_if_failed;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc mht_1(mht_1_v, 241, "", "./tensorflow/lite/kernels/numeric_verify.cc", "Init");

  auto* op_data = new OpData();
  op_data->float_input_initialized = false;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  const float tolerance = m[kToleranceStr].AsFloat();
  const bool log_if_failed = m[kLogIfFailedStr].AsBool();
  op_data->tolerance = tolerance;
  op_data->log_if_failed = log_if_failed;

  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc mht_2(mht_2_v, 258, "", "./tensorflow/lite/kernels/numeric_verify.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc mht_3(mht_3_v, 265, "", "./tensorflow/lite/kernels/numeric_verify.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  OpContext op_context(context, node);

  TF_LITE_ENSURE(context, op_context.input->type == kTfLiteUInt8 ||
                              op_context.input->type == kTfLiteInt8 ||
                              op_context.input->type == kTfLiteInt16 ||
                              op_context.input->type == kTfLiteFloat16);
  TF_LITE_ENSURE(context, op_context.ref->type == kTfLiteFloat32);

  // Allocate tensor to store the dequantized inputs.
  if (op_data->cache_tensor_id == kTensorNotAllocated) {
    TF_LITE_ENSURE_OK(
        context, context->AddTensors(context, 1, &op_data->cache_tensor_id));
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(1);
  node->temporaries->data[0] = op_data->cache_tensor_id;

  TfLiteTensor* dequantized;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kTemporaryDequantizedTensor,
                                     &dequantized));
  dequantized->type = op_context.ref->type;
  dequantized->allocation_type = kTfLiteDynamic;

  TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                 context, dequantized,
                                 TfLiteIntArrayCopy(op_context.input->dims)));

  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputTensor, &op_context.output));
  op_context.output->type = kTfLiteFloat32;
  op_context.output->allocation_type = kTfLiteArenaRwPersistent;
  return context->ResizeTensor(context, op_context.output,
                               TfLiteIntArrayCopy(op_context.input->dims));
}

static int32_t GetQuantizedValue(const OpContext& op_context, int index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc mht_4(mht_4_v, 310, "", "./tensorflow/lite/kernels/numeric_verify.cc", "GetQuantizedValue");

  switch (op_context.input->type) {
    case kTfLiteUInt8:
      return GetTensorData<uint8_t>(op_context.input)[index];
    case kTfLiteInt8:
      return GetTensorData<int8_t>(op_context.input)[index];
    case kTfLiteInt16:
      return GetTensorData<int16_t>(op_context.input)[index];
    default:
      return 0;
  }
}

template <builtin::dequantize::KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc mht_5(mht_5_v, 327, "", "./tensorflow/lite/kernels/numeric_verify.cc", "Eval");

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  OpContext op_context(context, node);
  if (IsConstantTensor(op_context.input) && op_data->float_input_initialized) {
    return kTfLiteOk;
  }

  // Dequantize the input
  TfLiteTensor* dequantized;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, kTemporaryDequantizedTensor,
                                     &dequantized));
  auto status = builtin::dequantize::DequantizeImpl<kernel_type>(
      context, node, op_context.input, dequantized);
  if (status != kTfLiteOk) {
    return status;
  }

  if (IsConstantTensor(op_context.input)) {
    op_data->float_input_initialized = true;
  }

  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kOutputTensor, &op_context.output));
  auto output_data = GetTensorData<float>(op_context.output);

  // If log_if_failed is on, calculate differences between float and
  // quantized values, their statistics and output logs.
  // Throw errors if any diff greater than tolerance exists.
  const int n = NumElements(dequantized);
  if (op_data->log_if_failed && op_data->tolerance >= 0.1) {
    // Verify the dequantized output.
    auto max_diff = op_data->tolerance * op_context.input->params.scale;
    for (int i = 0; i < n; ++i) {
      int32_t value = GetQuantizedValue(op_context, i);
      float dequant = GetTensorData<float>(dequantized)[i];
      float reference = GetTensorData<float>(op_context.ref)[i];
      output_data[i] = dequant - reference;
      float diff = std::abs(output_data[i]);
      if (diff > max_diff) {
        TF_LITE_KERNEL_LOG(
            context,
            "Mismatch: %f is quantized to %d with (%f, %d). "
            "abs(%f - %f) = %f > %f (tolerance) range percentage %f.\n",
            reference, value, op_context.input->params.scale,
            op_context.input->params.zero_point, reference, dequant, diff,
            max_diff, op_data->tolerance);
        return kTfLiteError;
      }
    }
  } else {
    // If tolerance is small or log_if_failed is off, then we only care about
    // statistics.
    // These statistics logging was added to identify some errors in practice.
    std::vector<double> diffs, temp;
    diffs.reserve(n);
    temp.reserve(n);
    diffs.resize(n);
    temp.resize(n);
    for (int i = 0; i < n; ++i) {
      float dequant = GetTensorData<float>(dequantized)[i];
      float reference = GetTensorData<float>(op_context.ref)[i];
      diffs[i] = static_cast<double>(dequant - reference);
      output_data[i] = dequant - reference;
    }
    double mean =
        std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();
    double max_diff = 0.0;
    std::transform(diffs.begin(), diffs.end(), temp.begin(),
                   [mean, &max_diff](double x) {
                     max_diff = std::max(max_diff, std::abs(x));
                     return x - mean;
                   });
    double sq_sum =
        std::inner_product(temp.begin(), temp.end(), temp.begin(), 0.0);
    double std = std::sqrt(sq_sum / diffs.size());
    TF_LITE_KERNEL_LOG(
        context,
        "std: %f, mean: %f, max_diff: %f (scale: %f, zero_point: %d).\n", std,
        mean, max_diff, op_context.input->params.scale,
        op_context.input->params.zero_point);
  }
  return kTfLiteOk;
}

}  // namespace numeric_verify

TfLiteRegistration* Register_NUMERIC_VERIFY_OPT() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc mht_6(mht_6_v, 417, "", "./tensorflow/lite/kernels/numeric_verify.cc", "Register_NUMERIC_VERIFY_OPT");

  static TfLiteRegistration r = {
      numeric_verify::Init, numeric_verify::Free, numeric_verify::Prepare,
      numeric_verify::Eval<builtin::dequantize::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_NUMERIC_VERIFY_REF() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc mht_7(mht_7_v, 427, "", "./tensorflow/lite/kernels/numeric_verify.cc", "Register_NUMERIC_VERIFY_REF");

  static TfLiteRegistration r = {
      numeric_verify::Init, numeric_verify::Free, numeric_verify::Prepare,
      numeric_verify::Eval<builtin::dequantize::kReference>};
  return &r;
}

TfLiteRegistration* Register_NUMERIC_VERIFY() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSnumeric_verifyDTcc mht_8(mht_8_v, 437, "", "./tensorflow/lite/kernels/numeric_verify.cc", "Register_NUMERIC_VERIFY");

#ifdef USE_NEON
  return Register_NUMERIC_VERIFY_OPT();
#else
  return Register_NUMERIC_VERIFY_REF();
#endif
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
