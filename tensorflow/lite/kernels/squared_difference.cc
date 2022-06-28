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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <stddef.h>
#include <stdint.h>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace squared_difference {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  bool requires_broadcast;
  ArithmeticParams arithmetic_params;
};

template <typename T>
T SquaredDifference(T input1, T input2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc mht_0(mht_0_v, 213, "", "./tensorflow/lite/kernels/squared_difference.cc", "SquaredDifference");

  const T difference = input1 - input2;
  return difference * difference;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc mht_1(mht_1_v, 222, "", "./tensorflow/lite/kernels/squared_difference.cc", "Init");

  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc mht_2(mht_2_v, 231, "", "./tensorflow/lite/kernels/squared_difference.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc mht_3(mht_3_v, 238, "", "./tensorflow/lite/kernels/squared_difference.cc", "Prepare");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  output->type = input2->type;

  // Ensure the quantization parameters are equivalent.
  if (input1->type == kTfLiteInt8) {
    const auto& input1_quantization_params = input1->params;
    const auto& input2_quantization_params = input2->params;
    const auto& output_quantization_params = output->params;
    const int32_t integer_type_min = std::numeric_limits<int8_t>::min();
    const int32_t integer_type_max = std::numeric_limits<int8_t>::max();
    TF_LITE_ENSURE(context,
                   input1_quantization_params.zero_point >= integer_type_min);
    TF_LITE_ENSURE(context,
                   input1_quantization_params.zero_point <= integer_type_max);
    TF_LITE_ENSURE(context,
                   input2_quantization_params.zero_point >= integer_type_min);
    TF_LITE_ENSURE(context,
                   input2_quantization_params.zero_point <= integer_type_max);
    TF_LITE_ENSURE(context,
                   output_quantization_params.zero_point >= integer_type_min);
    TF_LITE_ENSURE(context,
                   output_quantization_params.zero_point <= integer_type_max);
    data->arithmetic_params.input1_offset =
        -input1_quantization_params.zero_point;
    data->arithmetic_params.input2_offset =
        -input2_quantization_params.zero_point;
    data->arithmetic_params.output_offset =
        output_quantization_params.zero_point;

    // shift to make integer for scales.
    data->arithmetic_params.left_shift = 7;
    const double twice_max_input_scale =
        2 * std::max(input1_quantization_params.scale,
                     input2_quantization_params.scale);
    const double real_input1_multiplier =
        input1_quantization_params.scale / twice_max_input_scale;
    double real_input2_multiplier =
        input2_quantization_params.scale / twice_max_input_scale;
    const double real_output_multiplier =
        (twice_max_input_scale * twice_max_input_scale) /
        ((1 << data->arithmetic_params.left_shift * 2) *
         output_quantization_params.scale);
    tflite::QuantizeMultiplierSmallerThanOneExp(
        real_input1_multiplier, &data->arithmetic_params.input1_multiplier,
        &data->arithmetic_params.input1_shift);
    tflite::QuantizeMultiplierSmallerThanOneExp(
        real_input2_multiplier, &data->arithmetic_params.input2_multiplier,
        &data->arithmetic_params.input2_shift);
    tflite::QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &data->arithmetic_params.output_multiplier,
        &data->arithmetic_params.output_shift);
    data->arithmetic_params.quantized_activation_min =
        std::numeric_limits<int8_t>::min();
    data->arithmetic_params.quantized_activation_max =
        std::numeric_limits<int8_t>::max();
  }

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

inline int8_t SquaredDifference(int8_t x, int8_t y,
                                const ArithmeticParams& params) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc mht_4(mht_4_v, 328, "", "./tensorflow/lite/kernels/squared_difference.cc", "SquaredDifference");

  const int32_t input1_val = params.input1_offset + x;
  const int32_t input2_val = params.input2_offset + y;
  const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
  const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
  const int32_t scaled_input1_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input1_val, params.input1_multiplier, params.input1_shift);
  const int32_t scaled_input2_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input2_val, params.input2_multiplier, params.input2_shift);
  const int32_t raw_diff = scaled_input1_val - scaled_input2_val;

  // Max of this is 255^2 * (1 << 14), so won't overflow 32 bits.
  const int32_t squared_raw_diff = raw_diff * raw_diff;
  const int32_t raw_output =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          squared_raw_diff, params.output_multiplier, params.output_shift) +
      params.output_offset;
  const int32_t clamped_output =
      std::min(params.quantized_activation_max,
               std::max(params.quantized_activation_min, raw_output));
  return static_cast<int8_t>(clamped_output);
}

template <typename T>
void EvalQuantizedSquaredDifference(TfLiteContext* context, TfLiteNode* node,
                                    const OpData* data,
                                    const TfLiteTensor* input1,
                                    const TfLiteTensor* input2,
                                    TfLiteTensor* output) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc mht_5(mht_5_v, 361, "", "./tensorflow/lite/kernels/squared_difference.cc", "EvalQuantizedSquaredDifference");

  const auto* op_data = static_cast<const OpData*>(node->user_data);
  if (data->requires_broadcast) {
    reference_integer_ops::BroadcastBinaryFunction4DSlow(
        op_data->arithmetic_params, GetTensorShape(input1),
        GetTensorData<T>(input1), GetTensorShape(input2),
        GetTensorData<T>(input2), GetTensorShape(output),
        GetTensorData<T>(output), reference_integer_ops::CheckArithmeticParams,
        SquaredDifference);
  } else {
    const int flat_size = GetTensorShape(input1).FlatSize();
    reference_integer_ops::ElementWise(
        flat_size, op_data->arithmetic_params, GetTensorData<int8_t>(input1),
        GetTensorData<int8_t>(input2), GetTensorData<int8_t>(output),
        reference_integer_ops::CheckArithmeticParams, SquaredDifference);
  }
}

template <typename T>
void EvalSquaredDifference(TfLiteContext* context, TfLiteNode* node,
                           const OpData* data, const TfLiteTensor* input1,
                           const TfLiteTensor* input2, TfLiteTensor* output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc mht_6(mht_6_v, 385, "", "./tensorflow/lite/kernels/squared_difference.cc", "EvalSquaredDifference");

  if (data->requires_broadcast) {
    reference_ops::BroadcastBinaryFunction4DSlow<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output), SquaredDifference<T>);
  } else {
    reference_ops::BinaryFunction<T, T, T>(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output), SquaredDifference<T>);
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc mht_7(mht_7_v, 402, "", "./tensorflow/lite/kernels/squared_difference.cc", "Eval");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  ruy::profiler::ScopeLabel label("SquaredDifference");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (output->type == kTfLiteFloat32) {
    EvalSquaredDifference<float>(context, node, data, input1, input2, output);
  } else if (output->type == kTfLiteInt32) {
    EvalSquaredDifference<int32_t>(context, node, data, input1, input2, output);
  } else if (output->type == kTfLiteInt8) {
    EvalQuantizedSquaredDifference<int8_t>(context, node, data, input1, input2,
                                           output);
  } else {
    context->ReportError(
        context,
        "SquaredDifference only supports FLOAT32 and INT32 now, got %d.",
        output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace squared_difference

TfLiteRegistration* Register_SQUARED_DIFFERENCE() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsquared_differenceDTcc mht_8(mht_8_v, 439, "", "./tensorflow/lite/kernels/squared_difference.cc", "Register_SQUARED_DIFFERENCE");

  static TfLiteRegistration r = {
      squared_difference::Init, squared_difference::Free,
      squared_difference::Prepare, squared_difference::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
