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
class MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc() {
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

#include <stdint.h>
#include <stdlib.h>

#include <cmath>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace elementwise {
namespace {

const char kAbsName[] = "Abs";
const char kRsqrtName[] = "Rsqrt";

struct OpData {
  int32_t multiplier;
  int32_t shift;
  int input_offset;
  int output_offset;
  bool needs_rescale;
};

bool IsNumericSupportedType(const TfLiteType type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/kernels/elementwise.cc", "IsNumericSupportedType");

  return type == kTfLiteFloat32;
}

bool IsLogicalSupportedType(const TfLiteType type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_1(mht_1_v, 223, "", "./tensorflow/lite/kernels/elementwise.cc", "IsLogicalSupportedType");

  return type == kTfLiteBool;
}

bool IsAbsSupportedType(const TfLiteType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/kernels/elementwise.cc", "IsAbsSupportedType");

  return type == kTfLiteFloat32 || type == kTfLiteInt8 || type == kTfLiteInt16;
}

bool IsRsqrtSupportedType(const TfLiteType type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_3(mht_3_v, 237, "", "./tensorflow/lite/kernels/elementwise.cc", "IsRsqrtSupportedType");

  return type == kTfLiteFloat32 || type == kTfLiteInt8;
}

inline void SetAbsOutputMultiplier(const float input_scale,
                                   const float output_scale,
                                   int32_t* multiplier, int32_t* shift) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_4(mht_4_v, 246, "", "./tensorflow/lite/kernels/elementwise.cc", "SetAbsOutputMultiplier");

  QuantizeMultiplier(input_scale / output_scale, multiplier, shift);
}

inline void SetRsqrtOutputMultiplier(const float input_scale,
                                     const float output_scale,
                                     int32_t* multiplier, int32_t* shift) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_5(mht_5_v, 255, "", "./tensorflow/lite/kernels/elementwise.cc", "SetRsqrtOutputMultiplier");

  const double scale = 1. / (std::sqrt(input_scale) * output_scale);
  QuantizeMultiplier(scale, multiplier, shift);
}

typedef bool (*IsSupportedType)(TfLiteType);
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node,
                            IsSupportedType is_supported_type,
                            const char* op_name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_6(mht_6_v, 267, "", "./tensorflow/lite/kernels/elementwise.cc", "GenericPrepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (!is_supported_type(input->type)) {
    TF_LITE_UNSUPPORTED_TYPE(context, input->type, op_name);
  }
  // For int16 type input, we support both quantized and non-quantized
  // evaluation.
  if (input->type == kTfLiteInt8 ||
      (input->type == kTfLiteInt16 &&
       input->quantization.type != kTfLiteNoQuantization)) {
    TfLiteTensor* output = GetOutput(context, node, 0);
    auto* op_data = static_cast<OpData*>(node->user_data);
    TF_LITE_ENSURE_EQ(context, input->quantization.type,
                      kTfLiteAffineQuantization);
    TF_LITE_ENSURE_EQ(context, output->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* input_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    const auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);
    TF_LITE_ENSURE(context, input_params != nullptr);
    TF_LITE_ENSURE(context, input_params->scale != nullptr);
    TF_LITE_ENSURE(context, input_params->scale->size > 0);
    TF_LITE_ENSURE(context, input_params->zero_point->size > 0);
    TF_LITE_ENSURE(context, output_params != nullptr);
    TF_LITE_ENSURE(context, output_params->scale != nullptr);
    TF_LITE_ENSURE(context, output_params->scale->size > 0);
    TF_LITE_ENSURE(context, output_params->zero_point->size > 0);
    op_data->input_offset = input_params->zero_point->data[0];
    op_data->output_offset = output_params->zero_point->data[0];
    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, op_data->input_offset, 0);
      TF_LITE_ENSURE_EQ(context, op_data->output_offset, 0);
    }
    const float input_scale = input_params->scale->data[0];
    const float output_scale = output_params->scale->data[0];
    op_data->needs_rescale = input_scale != output_scale;
    if (op_name == kAbsName && op_data->needs_rescale) {
      SetAbsOutputMultiplier(input_scale, output_scale, &op_data->multiplier,
                             &op_data->shift);
    } else if (op_name == kRsqrtName) {
      SetRsqrtOutputMultiplier(input_scale, output_scale, &op_data->multiplier,
                               &op_data->shift);
    }
  }
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <typename T>
inline TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                             std::function<T(T)> func,
                             std::function<TfLiteStatus(T)> validate_input_func,
                             TfLiteType expected_type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_7(mht_7_v, 329, "", "./tensorflow/lite/kernels/elementwise.cc", "EvalImpl");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, expected_type);
  const int64_t num_elements = NumElements(input);
  const T* in_data = GetTensorData<T>(input);
  T* out_data = GetTensorData<T>(output);
  for (int64_t i = 0; i < num_elements; ++i) {
    if (validate_input_func) {
      TF_LITE_ENSURE_OK(context, validate_input_func(in_data[i]));
    }
    out_data[i] = func(in_data[i]);
  }
  return kTfLiteOk;
}

// Non-quantized evaluation of Abs op when input is int16.
inline TfLiteStatus AbsInt16EvalImpl(TfLiteContext* context, TfLiteNode* node,
                                     TfLiteType expected_type) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_8(mht_8_v, 352, "", "./tensorflow/lite/kernels/elementwise.cc", "AbsInt16EvalImpl");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, expected_type);
  const int64_t num_elements = NumElements(input);
  const int16_t* in_data = GetTensorData<int16_t>(input);
  int16_t* out_data = GetTensorData<int16_t>(output);
  for (int64_t i = 0; i < num_elements; ++i) {
    out_data[i] = static_cast<int16_t>(
        std::abs<int32_t>(static_cast<int32_t>(in_data[i])));
  }
  return kTfLiteOk;
}

template <typename T>
inline TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                             std::function<T(T)> func,
                             TfLiteType expected_type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_9(mht_9_v, 374, "", "./tensorflow/lite/kernels/elementwise.cc", "EvalImpl");

  return EvalImpl<T>(context, node, func, /*validate_input_func=*/nullptr,
                     expected_type);
}

inline TfLiteStatus EvalNumeric(TfLiteContext* context, TfLiteNode* node,
                                float float_func(float)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_10(mht_10_v, 383, "", "./tensorflow/lite/kernels/elementwise.cc", "EvalNumeric");

  return EvalImpl<float>(context, node, float_func, kTfLiteFloat32);
}

inline TfLiteStatus EvalLogical(TfLiteContext* context, TfLiteNode* node,
                                bool bool_func(bool)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_11(mht_11_v, 391, "", "./tensorflow/lite/kernels/elementwise.cc", "EvalLogical");

  return EvalImpl<bool>(context, node, bool_func, kTfLiteBool);
}

void* ElementWiseQuantizedInit(TfLiteContext* context, const char* buffer,
                               size_t length) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_12(mht_12_v, 400, "", "./tensorflow/lite/kernels/elementwise.cc", "ElementWiseQuantizedInit");

  return new OpData();
}

void ElementWiseQuantizedFree(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_13(mht_13_v, 407, "", "./tensorflow/lite/kernels/elementwise.cc", "ElementWiseQuantizedFree");

  delete static_cast<OpData*>(buffer);
}

template <typename T>
TfLiteStatus AbsEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLiteType type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_14(mht_14_v, 416, "", "./tensorflow/lite/kernels/elementwise.cc", "AbsEvalQuantized");

  const auto* op_data = static_cast<const OpData*>(node->user_data);
  const int kMin = std::numeric_limits<T>::min();
  const int kMax = std::numeric_limits<T>::max();

  std::function<T(T)> func = [&](T i) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_15(mht_15_v, 424, "", "./tensorflow/lite/kernels/elementwise.cc", "lambda");

    const int32_t value = std::abs(i - op_data->input_offset);
    if (!op_data->needs_rescale) {
      return static_cast<T>(
          std::min(std::max(value + op_data->output_offset, kMin), kMax));
    }
    const int32_t output = MultiplyByQuantizedMultiplier(
                               value, op_data->multiplier, op_data->shift) +
                           op_data->output_offset;
    return static_cast<T>(std::min(std::max(output, kMin), kMax));
  };

  return EvalImpl<T>(context, node, func, type);
}

TfLiteStatus AbsEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_16(mht_16_v, 442, "", "./tensorflow/lite/kernels/elementwise.cc", "AbsEval");

  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteType type = input->type;
  switch (type) {
    case kTfLiteFloat32:
      return EvalImpl<float>(context, node, std::abs<float>, type);
    case kTfLiteInt8:
      return AbsEvalQuantized<int8_t>(context, node, type);
    case kTfLiteInt16:
      return input->quantization.type == kTfLiteNoQuantization
                 ? AbsInt16EvalImpl(context, node, type)
                 : AbsEvalQuantized<int16_t>(context, node, type);
    default:
      TF_LITE_KERNEL_LOG(context, "Current data type %s is not supported.",
                         TfLiteTypeGetName(type));
      return kTfLiteError;
  }
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_17(mht_17_v, 464, "", "./tensorflow/lite/kernels/elementwise.cc", "SinEval");

  return EvalNumeric(context, node, std::sin);
}

TfLiteStatus CosEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_18(mht_18_v, 471, "", "./tensorflow/lite/kernels/elementwise.cc", "CosEval");

  return EvalNumeric(context, node, std::cos);
}

TfLiteStatus LogEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_19(mht_19_v, 478, "", "./tensorflow/lite/kernels/elementwise.cc", "LogEval");

  return EvalNumeric(context, node, std::log);
}

TfLiteStatus SqrtEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_20(mht_20_v, 485, "", "./tensorflow/lite/kernels/elementwise.cc", "SqrtEval");

  return EvalNumeric(context, node, std::sqrt);
}

TfLiteStatus RsqrtEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                                TfLiteType type) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_21(mht_21_v, 493, "", "./tensorflow/lite/kernels/elementwise.cc", "RsqrtEvalQuantized");

  const auto* op_data = static_cast<const OpData*>(node->user_data);
  const int kMin = std::numeric_limits<int8_t>::min();
  const int kMax = std::numeric_limits<int8_t>::max();
  std::function<TfLiteStatus(int8_t)> validate_input_func = [&](int8_t i) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_22(mht_22_v, 500, "", "./tensorflow/lite/kernels/elementwise.cc", "lambda");

    TF_LITE_ENSURE_MSG(context, i >= op_data->input_offset,
                       "Rsqrt is only defined for positive values");
    return kTfLiteOk;
  };

  std::function<int8_t(int8_t)> func = [&](int8_t i) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_23(mht_23_v, 509, "", "./tensorflow/lite/kernels/elementwise.cc", "lambda");

    const int32_t value = (i - op_data->input_offset);
    const int32_t kShift = 20;  // Shift to keep value integer.
    if (value == 0) {
      // Assume that any value close to 0 represents the max output value.
      return static_cast<int8_t>(kMax);
    }
    int32_t inv_sqrt_multiplier;
    int inv_sqrt_shift;
    GetInvSqrtQuantizedMultiplierExp(value, kReverseShift, &inv_sqrt_multiplier,
                                     &inv_sqrt_shift);
    const int32_t data = MultiplyByQuantizedMultiplier(1, inv_sqrt_multiplier,
                                                       inv_sqrt_shift + kShift);
    const int32_t output =
        MultiplyByQuantizedMultiplier(data, op_data->multiplier,
                                      op_data->shift - kShift) +
        op_data->output_offset;
    return static_cast<int8_t>(std::min(std::max(output, kMin), kMax));
  };

  return EvalImpl<int8_t>(context, node, func, validate_input_func, type);
}

TfLiteStatus RsqrtEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_24(mht_24_v, 535, "", "./tensorflow/lite/kernels/elementwise.cc", "RsqrtEval");

  const TfLiteType type = GetInput(context, node, 0)->type;
  switch (type) {
    case kTfLiteFloat32:
      return EvalImpl<float>(
          context, node, [](float f) { return 1.f / std::sqrt(f); }, type);
    case kTfLiteInt8:
      return RsqrtEvalQuantized(context, node, type);
    default:
      TF_LITE_KERNEL_LOG(context, "Current data type %s is not supported.",
                         TfLiteTypeGetName(type));
      return kTfLiteError;
  }
}

TfLiteStatus SquareEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_25(mht_25_v, 553, "", "./tensorflow/lite/kernels/elementwise.cc", "SquareEval");

  return EvalNumeric(context, node, [](float f) { return f * f; });
}

TfLiteStatus LogicalNotEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSelementwiseDTcc mht_26(mht_26_v, 560, "", "./tensorflow/lite/kernels/elementwise.cc", "LogicalNotEval");

  return EvalLogical(context, node, [](bool v) { return !v; });
}

}  // namespace
}  // namespace elementwise

// Given a function...
// template<int T>
// int Foo(int b)
//
// typedef int(*Bar)(int);
//
// MSVC2015 will not see Foo<10> as the same type as Bar.
//
// This works around the issue by instantiating wrapper methods around
// elementwise::GenericPrepare() rather than using a templated
// elementwise::GenericPrepare method.
#define GENERIC_PREPARE(function_name, is_supported_type_function, type_name)  \
  static TfLiteStatus function_name(TfLiteContext* context,                    \
                                    TfLiteNode* node) {                        \
    return elementwise::GenericPrepare(context, node,                          \
                                       is_supported_type_function, type_name); \
  }

GENERIC_PREPARE(PrepareAbs, elementwise::IsAbsSupportedType,
                elementwise::kAbsName)

TfLiteRegistration* Register_ABS() {
  static TfLiteRegistration r = {elementwise::ElementWiseQuantizedInit,
                                 elementwise::ElementWiseQuantizedFree,
                                 PrepareAbs, elementwise::AbsEval};
  return &r;
}

GENERIC_PREPARE(PrepareSin, elementwise::IsNumericSupportedType, "Sin")

TfLiteRegistration* Register_SIN() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr, PrepareSin,
                                 elementwise::SinEval};
  return &r;
}

GENERIC_PREPARE(PrepareCos, elementwise::IsNumericSupportedType, "Cos")

TfLiteRegistration* Register_COS() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr, PrepareCos,
                                 elementwise::CosEval};
  return &r;
}

GENERIC_PREPARE(PrepareLog, elementwise::IsNumericSupportedType, "Log")

TfLiteRegistration* Register_LOG() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr, PrepareLog,
                                 elementwise::LogEval};
  return &r;
}

GENERIC_PREPARE(PrepareSqrt, elementwise::IsNumericSupportedType, "Sqrt")

TfLiteRegistration* Register_SQRT() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 PrepareSqrt, elementwise::SqrtEval};
  return &r;
}

GENERIC_PREPARE(PrepareRsqrt, elementwise::IsRsqrtSupportedType,
                elementwise::kRsqrtName)

TfLiteRegistration* Register_RSQRT() {
  static TfLiteRegistration r = {elementwise::ElementWiseQuantizedInit,
                                 elementwise::ElementWiseQuantizedFree,
                                 PrepareRsqrt, elementwise::RsqrtEval};
  return &r;
}

GENERIC_PREPARE(PrepareSquare, elementwise::IsNumericSupportedType, "Square")

TfLiteRegistration* Register_SQUARE() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 PrepareSquare, elementwise::SquareEval};
  return &r;
}

GENERIC_PREPARE(PrepareNot, elementwise::IsLogicalSupportedType, "Not")

TfLiteRegistration* Register_LOGICAL_NOT() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr, PrepareNot,
                                 elementwise::LogicalNotEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
