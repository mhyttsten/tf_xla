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
class MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc() {
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
#include "tensorflow/lite/kernels/internal/reference/comparisons.h"

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace comparisons {
namespace {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus ComparisonPrepareCommon(TfLiteContext* context, TfLiteNode* node,
                                     bool is_string_allowed) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/kernels/comparisons.cc", "ComparisonPrepareCommon");

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

  // Don't support string.
  if (!is_string_allowed) {
    TF_LITE_ENSURE(context, input1->type != kTfLiteString);
  }
  // Currently only support tensors have the same type.
  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  output->type = kTfLiteBool;

  bool requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus ComparisonPrepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_1(mht_1_v, 247, "", "./tensorflow/lite/kernels/comparisons.cc", "ComparisonPrepare");

  return ComparisonPrepareCommon(context, node, false);
}

TfLiteStatus ComparisonPrepareStringAllowed(TfLiteContext* context,
                                            TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_2(mht_2_v, 255, "", "./tensorflow/lite/kernels/comparisons.cc", "ComparisonPrepareStringAllowed");

  return ComparisonPrepareCommon(context, node, true);
}

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* left_shift) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_3(mht_3_v, 263, "", "./tensorflow/lite/kernels/comparisons.cc", "QuantizeMultiplier");

  if (double_multiplier < 1.0) {
    QuantizeMultiplierSmallerThanOneExp(double_multiplier, quantized_multiplier,
                                        left_shift);
  } else {
    QuantizeMultiplierGreaterThanOne(double_multiplier, quantized_multiplier,
                                     left_shift);
  }
}

template <typename input_dtype, reference_ops::ComparisonFn<int32> opname>
void ComparisonQuantized(const TfLiteTensor* input1, const TfLiteTensor* input2,
                         TfLiteTensor* output, bool requires_broadcast) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_4(mht_4_v, 278, "", "./tensorflow/lite/kernels/comparisons.cc", "ComparisonQuantized");

  if (input1->type == kTfLiteUInt8 || input1->type == kTfLiteInt8) {
    auto input1_offset = -input1->params.zero_point;
    auto input2_offset = -input2->params.zero_point;
    const int left_shift = 8;

    int32 input1_multiplier;
    int32 input2_multiplier;
    int input1_shift;
    int input2_shift;
    QuantizeMultiplier(input1->params.scale, &input1_multiplier, &input1_shift);
    QuantizeMultiplier(input2->params.scale, &input2_multiplier, &input2_shift);

    ComparisonParams op_params;
    op_params.left_shift = left_shift;
    op_params.input1_offset = input1_offset;
    op_params.input1_multiplier = input1_multiplier;
    op_params.input1_shift = input1_shift;
    op_params.input2_offset = input2_offset;
    op_params.input2_multiplier = input2_multiplier;
    op_params.input2_shift = input2_shift;
    if (requires_broadcast) {
      reference_ops::BroadcastComparison4DSlowWithScaling<input_dtype, opname>(
          op_params, GetTensorShape(input1), GetTensorData<input_dtype>(input1),
          GetTensorShape(input2), GetTensorData<input_dtype>(input2),
          GetTensorShape(output), GetTensorData<bool>(output));
    } else {
      reference_ops::ComparisonWithScaling<input_dtype, opname>(
          op_params, GetTensorShape(input1), GetTensorData<input_dtype>(input1),
          GetTensorShape(input2), GetTensorData<input_dtype>(input2),
          GetTensorShape(output), GetTensorData<bool>(output));
    }
  }
}

template <typename T, reference_ops::ComparisonFn<T> opname>
void Comparison(const TfLiteTensor* input1, const TfLiteTensor* input2,
                TfLiteTensor* output, bool requires_broadcast) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_5(mht_5_v, 318, "", "./tensorflow/lite/kernels/comparisons.cc", "Comparison");

  ComparisonParams op_params;
  requires_broadcast
      ? reference_ops::BroadcastComparison4DSlowImpl<T, opname>(
            op_params, GetTensorShape(input1), GetTensorData<T>(input1),
            GetTensorShape(input2), GetTensorData<T>(input2),
            GetTensorShape(output), GetTensorData<bool>(output))
      : reference_ops::ComparisonImpl<T, opname>(
            op_params, GetTensorShape(input1), GetTensorData<T>(input1),
            GetTensorShape(input2), GetTensorData<T>(input2),
            GetTensorShape(output), GetTensorData<bool>(output));
}

void ComparisonString(bool (*opname)(const StringRef&, const StringRef&),
                      const TfLiteTensor* input1, const TfLiteTensor* input2,
                      TfLiteTensor* output, bool requires_broadcast) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_6(mht_6_v, 336, "", "./tensorflow/lite/kernels/comparisons.cc", "ComparisonString");

  bool* output_data = GetTensorData<bool>(output);
  if (requires_broadcast) {
    reference_ops::BroadcastComparison4DSlowStringImpl(
        opname, GetTensorShape(input1), input1, GetTensorShape(input2), input2,
        GetTensorShape(output), output_data);
  } else {
    reference_ops::ComparisonStringImpl(opname, GetTensorShape(input1), input1,
                                        GetTensorShape(input2), input2,
                                        GetTensorShape(output), output_data);
  }
}

TfLiteStatus EqualEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_7(mht_7_v, 352, "", "./tensorflow/lite/kernels/comparisons.cc", "EqualEval");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      Comparison<bool, reference_ops::EqualFn>(input1, input2, output,
                                               requires_broadcast);
      break;
    case kTfLiteFloat32:
      Comparison<float, reference_ops::EqualFn>(input1, input2, output,
                                                requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::EqualFn>(input1, input2, output,
                                                  requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::EqualFn>(input1, input2, output,
                                                  requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::EqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::EqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteString:
      ComparisonString(reference_ops::StringRefEqualFn, input1, input2, output,
                       requires_broadcast);
      break;
    default:
      context->ReportError(
          context,
          "Does not support type %d, requires bool|float|int|uint8|string",
          input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus NotEqualEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_8(mht_8_v, 405, "", "./tensorflow/lite/kernels/comparisons.cc", "NotEqualEval");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteBool:
      Comparison<bool, reference_ops::NotEqualFn>(input1, input2, output,
                                                  requires_broadcast);
      break;
    case kTfLiteFloat32:
      Comparison<float, reference_ops::NotEqualFn>(input1, input2, output,
                                                   requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::NotEqualFn>(input1, input2, output,
                                                     requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::NotEqualFn>(input1, input2, output,
                                                     requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::NotEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::NotEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteString:
      ComparisonString(reference_ops::StringRefNotEqualFn, input1, input2,
                       output, requires_broadcast);
      break;
    default:
      context->ReportError(
          context,
          "Does not support type %d, requires bool|float|int|uint8|string",
          input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GreaterEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_9(mht_9_v, 458, "", "./tensorflow/lite/kernels/comparisons.cc", "GreaterEval");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      Comparison<float, reference_ops::GreaterFn>(input1, input2, output,
                                                  requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::GreaterFn>(input1, input2, output,
                                                    requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::GreaterFn>(input1, input2, output,
                                                    requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::GreaterFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::GreaterFn>(
          input1, input2, output, requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus GreaterEqualEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_10(mht_10_v, 502, "", "./tensorflow/lite/kernels/comparisons.cc", "GreaterEqualEval");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      Comparison<float, reference_ops::GreaterEqualFn>(input1, input2, output,
                                                       requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::GreaterEqualFn>(input1, input2, output,
                                                         requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::GreaterEqualFn>(input1, input2, output,
                                                         requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::GreaterEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::GreaterEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus LessEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_11(mht_11_v, 546, "", "./tensorflow/lite/kernels/comparisons.cc", "LessEval");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      Comparison<float, reference_ops::LessFn>(input1, input2, output,
                                               requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::LessFn>(input1, input2, output,
                                                 requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::LessFn>(input1, input2, output,
                                                 requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::LessFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::LessFn>(input1, input2, output,
                                                         requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus LessEqualEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_12(mht_12_v, 590, "", "./tensorflow/lite/kernels/comparisons.cc", "LessEqualEval");

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  bool requires_broadcast = !HaveSameShapes(input1, input2);
  switch (input1->type) {
    case kTfLiteFloat32:
      Comparison<float, reference_ops::LessEqualFn>(input1, input2, output,
                                                    requires_broadcast);
      break;
    case kTfLiteInt32:
      Comparison<int32_t, reference_ops::LessEqualFn>(input1, input2, output,
                                                      requires_broadcast);
      break;
    case kTfLiteInt64:
      Comparison<int64_t, reference_ops::LessEqualFn>(input1, input2, output,
                                                      requires_broadcast);
      break;
    case kTfLiteUInt8:
      ComparisonQuantized<uint8_t, reference_ops::LessEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    case kTfLiteInt8:
      ComparisonQuantized<int8_t, reference_ops::LessEqualFn>(
          input1, input2, output, requires_broadcast);
      break;
    default:
      context->ReportError(context,
                           "Does not support type %d, requires float|int|uint8",
                           input1->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace comparisons

TfLiteRegistration* Register_EQUAL() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_13(mht_13_v, 637, "", "./tensorflow/lite/kernels/comparisons.cc", "Register_EQUAL");

  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepareStringAllowed,
                                 comparisons::EqualEval};
  return &r;
}

TfLiteRegistration* Register_NOT_EQUAL() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_14(mht_14_v, 647, "", "./tensorflow/lite/kernels/comparisons.cc", "Register_NOT_EQUAL");

  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepareStringAllowed,
                                 comparisons::NotEqualEval};
  return &r;
}

TfLiteRegistration* Register_GREATER() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_15(mht_15_v, 657, "", "./tensorflow/lite/kernels/comparisons.cc", "Register_GREATER");

  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepare,
                                 comparisons::GreaterEval};
  return &r;
}

TfLiteRegistration* Register_GREATER_EQUAL() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_16(mht_16_v, 667, "", "./tensorflow/lite/kernels/comparisons.cc", "Register_GREATER_EQUAL");

  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepare,
                                 comparisons::GreaterEqualEval};
  return &r;
}

TfLiteRegistration* Register_LESS() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_17(mht_17_v, 677, "", "./tensorflow/lite/kernels/comparisons.cc", "Register_LESS");

  static TfLiteRegistration r = {
      nullptr, nullptr, comparisons::ComparisonPrepare, comparisons::LessEval};
  return &r;
}

TfLiteRegistration* Register_LESS_EQUAL() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScomparisonsDTcc mht_18(mht_18_v, 686, "", "./tensorflow/lite/kernels/comparisons.cc", "Register_LESS_EQUAL");

  static TfLiteRegistration r = {nullptr, nullptr,
                                 comparisons::ComparisonPrepare,
                                 comparisons::LessEqualEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
