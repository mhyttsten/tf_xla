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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/sub.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/sub.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace sub {

// This file has three implementation of Sub.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  bool requires_broadcast;

  // These fields are used in both the general 8-bit -> 8bit quantized path,
  // and the special 16-bit -> 16bit quantized path
  int input1_shift;
  int input2_shift;
  int32 output_activation_min;
  int32 output_activation_max;

  // These fields are used only in the general 8-bit -> 8bit quantized path
  int32 input1_multiplier;
  int32 input2_multiplier;
  int32 output_multiplier;
  int output_shift;
  int left_shift;
  int32 input1_offset;
  int32 input2_offset;
  int32 output_offset;

  // This parameter is used to indicate whether
  // parameter scale is power of two.
  // It is used in 16-bit -> 16-bit quantization.
  bool pot_scale_int16;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_0(mht_0_v, 252, "", "./tensorflow/lite/kernels/sub.cc", "Init");

  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_1(mht_1_v, 261, "", "./tensorflow/lite/kernels/sub.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus PrepareGeneralSubOp(TfLiteContext* context,
                                 const TfLiteTensor* input_1,
                                 const TfLiteTensor* input_2,
                                 TfLiteTensor* output, TfLiteSubParams* params,
                                 OpData* op_params) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_2(mht_2_v, 272, "", "./tensorflow/lite/kernels/sub.cc", "PrepareGeneralSubOp");

  TF_LITE_ENSURE(context, output->type == kTfLiteUInt8 ||
                              output->type == kTfLiteInt8 ||
                              output->type == kTfLiteInt16);
  const auto& input1_quantization_params = input_1->params;
  const auto& input2_quantization_params = input_2->params;
  const auto& output_quantization_params = output->params;
  int32_t integer_type_min = 0;
  int32_t integer_type_max = 0;
  if (output->type == kTfLiteUInt8) {
    integer_type_min = std::numeric_limits<uint8_t>::min();
    integer_type_max = std::numeric_limits<uint8_t>::max();
  } else if (output->type == kTfLiteInt16) {
    integer_type_min = std::numeric_limits<int16_t>::min();
    integer_type_max = std::numeric_limits<int16_t>::max();
  } else {
    // output->type == kTfLiteInt8
    integer_type_min = std::numeric_limits<int8_t>::min();
    integer_type_max = std::numeric_limits<int8_t>::max();
  }

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

  op_params->input1_offset = -input1_quantization_params.zero_point;
  op_params->input2_offset = -input2_quantization_params.zero_point;
  op_params->output_offset = output_quantization_params.zero_point;

  // The shift is set to 15 in case of 16-bit and 20 in case of 8-bit,
  // accordingly. In case of 16-bit we have 65535 << 15 which is less than 1 <<
  // 31, therefore the addition will still fit in a 32 bit accumulator.
  op_params->left_shift = output->type == kTfLiteInt16 ? 15 : 20;
  const double twice_max_input_scale =
      2 * std::max(input1_quantization_params.scale,
                   input2_quantization_params.scale);
  const double real_input1_multiplier =
      input1_quantization_params.scale / twice_max_input_scale;
  const double real_input2_multiplier =
      input2_quantization_params.scale / twice_max_input_scale;
  const double real_output_multiplier =
      twice_max_input_scale /
      ((1 << op_params->left_shift) * output_quantization_params.scale);

  tflite::QuantizeMultiplierSmallerThanOneExp(real_input1_multiplier,
                                              &op_params->input1_multiplier,
                                              &op_params->input1_shift);
  tflite::QuantizeMultiplierSmallerThanOneExp(real_input2_multiplier,
                                              &op_params->input2_multiplier,
                                              &op_params->input2_shift);
  tflite::QuantizeMultiplierSmallerThanOneExp(real_output_multiplier,
                                              &op_params->output_multiplier,
                                              &op_params->output_shift);

  TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
      context, params->activation, output, &op_params->output_activation_min,
      &op_params->output_activation_max));

  return kTfLiteOk;
}

TfLiteStatus PrepareInt16SubOpPOT(TfLiteContext* context,
                                  const TfLiteTensor* input1,
                                  const TfLiteTensor* input2,
                                  TfLiteTensor* output, TfLiteSubParams* params,
                                  OpData* data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_3(mht_3_v, 349, "", "./tensorflow/lite/kernels/sub.cc", "PrepareInt16SubOpPOT");

  // 16bit -> 16bit special quantized path, supporting only a rather
  // narrow case of quantization parameters: zero_points must all be 0
  // ("symmetric quantization") and scales must be power-of-two (which
  // we abbreviate as "POT" below). The intended use case for this path
  // is in LSTM cells, where, due to the constraints of implementing
  // some of the math in these LSTM cells in fixed-point arithmetic,
  // we need to have such symmetric, power-of-two quantization
  // (Fixed-point formats are inherently symmetric, power-of-two).
  TF_LITE_ENSURE_EQ(context, input1->params.zero_point, 0);
  TF_LITE_ENSURE_EQ(context, input2->params.zero_point, 0);
  TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

  int input1_scale_log2_rounded;
  bool input1_scale_is_pot =
      CheckedLog2(input1->params.scale, &input1_scale_log2_rounded);
  TF_LITE_ENSURE(context, input1_scale_is_pot);

  int input2_scale_log2_rounded;
  bool input2_scale_is_pot =
      CheckedLog2(input2->params.scale, &input2_scale_log2_rounded);
  TF_LITE_ENSURE(context, input2_scale_is_pot);

  int output_scale_log2_rounded;
  bool output_scale_is_pot =
      CheckedLog2(output->params.scale, &output_scale_log2_rounded);
  TF_LITE_ENSURE(context, output_scale_is_pot);

  data->input1_shift = input1_scale_log2_rounded - output_scale_log2_rounded;
  data->input2_shift = input2_scale_log2_rounded - output_scale_log2_rounded;

  // Shifting of one input is supported. The graph quantization should ensure
  // that the other input matches the output.
  TF_LITE_ENSURE(context, data->input1_shift == 0 || data->input2_shift == 0);
  TF_LITE_ENSURE(context, data->input1_shift <= 0);
  TF_LITE_ENSURE(context, data->input2_shift <= 0);

  TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
      context, params->activation, output, &data->output_activation_min,
      &data->output_activation_max));
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_4(mht_4_v, 395, "", "./tensorflow/lite/kernels/sub.cc", "Prepare");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteSubParams*>(node->builtin_data);

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

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  // 8bit -> 8bit general quantized path, with general rescalings
  // as well as, 16bit -> 16bit with general rescalings

  // There are two implementations of SUB operator in case of
  // 16bit input depending on whether the scale parameter is
  // the power of 2 or not. Currently only implementation for
  // general case is used, but we need to use another implementation
  // for older versions.
  bool general_scale_int16 = false;

  bool input1_scale_is_pot = false;
  bool input2_scale_is_pot = false;
  bool output_scale_is_pot = false;

  int input1_scale_log2_rounded{0};
  int input2_scale_log2_rounded{0};
  int output_scale_log2_rounded{0};

  if (input1->type == kTfLiteInt16 && input2->type == kTfLiteInt16 &&
      output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input1->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, input2->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    general_scale_int16 = !params || !params->pot_scale_int16;

    if (!general_scale_int16) {
      // Do preparation in the case of the scale parameter is power of 2.
      input1_scale_is_pot =
          CheckedLog2(input1->params.scale, &input1_scale_log2_rounded);

      input2_scale_is_pot =
          CheckedLog2(input2->params.scale, &input2_scale_log2_rounded);

      output_scale_is_pot =
          CheckedLog2(output->params.scale, &output_scale_log2_rounded);

      general_scale_int16 =
          !input1_scale_is_pot || !input2_scale_is_pot || !output_scale_is_pot;
    }
  }

  data->pot_scale_int16 = !general_scale_int16;

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8 ||
      general_scale_int16) {
    TF_LITE_ENSURE_OK(context, PrepareGeneralSubOp(context, input1, input2,
                                                   output, params, data));
  } else if (output->type == kTfLiteInt16) {
    // LSTM-special case with scale parameter of POT
    TF_LITE_ENSURE_OK(context, PrepareInt16SubOpPOT(context, input1, input2,
                                                    output, params, data));
  }

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type, typename data_type>
void EvalSubImpl(TfLiteContext* context, TfLiteNode* node,
                 TfLiteSubParams* params, const OpData* data,
                 const TfLiteTensor* input1, const TfLiteTensor* input2,
                 bool requires_broadcast, TfLiteTensor* output) {
  data_type output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  switch (kernel_type) {
    case kReference:
      if (requires_broadcast) {
        reference_ops::BroadcastSubSlow(
            op_params, GetTensorShape(input1), GetTensorData<data_type>(input1),
            GetTensorShape(input2), GetTensorData<data_type>(input2),
            GetTensorShape(output), GetTensorData<data_type>(output));
      } else {
        reference_ops::SubWithActivation(
            op_params, GetTensorShape(input1), GetTensorData<data_type>(input1),
            GetTensorShape(input2), GetTensorData<data_type>(input2),
            GetTensorShape(output), GetTensorData<data_type>(output));
      }
      break;
    case kGenericOptimized:
    case kNeonOptimized:
      if (requires_broadcast) {
        optimized_ops::BroadcastSubSlow(
            op_params, GetTensorShape(input1), GetTensorData<data_type>(input1),
            GetTensorShape(input2), GetTensorData<data_type>(input2),
            GetTensorShape(output), GetTensorData<data_type>(output));
      } else {
        optimized_ops::SubWithActivation(
            op_params, GetTensorShape(input1), GetTensorData<data_type>(input1),
            GetTensorShape(input2), GetTensorData<data_type>(input2),
            GetTensorShape(output), GetTensorData<data_type>(output));
      }
      break;
  }
}

template <KernelType kernel_type>
void EvalSub(TfLiteContext* context, TfLiteNode* node, TfLiteSubParams* params,
             const OpData* data, const TfLiteTensor* input1,
             const TfLiteTensor* input2, TfLiteTensor* output) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_5(mht_5_v, 530, "", "./tensorflow/lite/kernels/sub.cc", "EvalSub");

  const bool requires_broadcast = data->requires_broadcast;
  switch (output->type) {
    case kTfLiteInt32:
      EvalSubImpl<kernel_type, int32_t>(context, node, params, data, input1,
                                        input2, requires_broadcast, output);
      break;
    case kTfLiteFloat32:
      EvalSubImpl<kernel_type, float>(context, node, params, data, input1,
                                      input2, requires_broadcast, output);
      break;
    case kTfLiteInt64:
      EvalSubImpl<kernel_type, int64_t>(context, node, params, data, input1,
                                        input2, requires_broadcast, output);
      break;

    default:
      TF_LITE_KERNEL_LOG(context, "output type %s is not supported.",
                         TfLiteTypeGetName(output->type));
  }
}

template <KernelType kernel_type>
void EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                   TfLiteSubParams* params, const OpData* data,
                   const TfLiteTensor* input1, const TfLiteTensor* input2,
                   TfLiteTensor* output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_6(mht_6_v, 559, "", "./tensorflow/lite/kernels/sub.cc", "EvalQuantized");

  tflite::ArithmeticParams op_params;
  op_params.left_shift = data->left_shift;
  op_params.input1_offset = data->input1_offset;
  op_params.input1_multiplier = data->input1_multiplier;
  op_params.input1_shift = data->input1_shift;
  op_params.input2_offset = data->input2_offset;
  op_params.input2_multiplier = data->input2_multiplier;
  op_params.input2_shift = data->input2_shift;
  op_params.output_offset = data->output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  SetActivationParams(data->output_activation_min, data->output_activation_max,
                      &op_params);

  const bool need_broadcast = optimized_ops::ProcessBroadcastShapes(
      GetTensorShape(input1), GetTensorShape(input2), &op_params);

#define TF_LITE_SUB(type, opname, data_type)                             \
  type::opname(op_params, GetTensorShape(input1),                        \
               GetTensorData<data_type>(input1), GetTensorShape(input2), \
               GetTensorData<data_type>(input2), GetTensorShape(output), \
               GetTensorData<data_type>(output))
  if (output->type == kTfLiteInt8) {
    if (need_broadcast) {
      TF_LITE_SUB(reference_ops, BroadcastQuantSubSlow, int8_t);
    } else {
      TF_LITE_SUB(reference_ops, Sub, int8_t);
    }
  } else if (!data->pot_scale_int16) {
    if (kernel_type == kReference) {
      if (need_broadcast) {
        TF_LITE_SUB(reference_ops, BroadcastQuantSubSlow, int16_t);
      } else {
        TF_LITE_SUB(reference_ops, Sub, int16_t);
      }
    } else {
      if (need_broadcast) {
        TF_LITE_SUB(optimized_integer_ops, BroadcastSubDispatch, int16_t);
      } else {
        TF_LITE_SUB(optimized_integer_ops, Sub, int16_t);
      }
    }
  } else if (output->type == kTfLiteUInt8) {
    if (need_broadcast) {
      TF_LITE_SUB(reference_ops, BroadcastQuantSubSlow, uint8_t);
    } else {
      TF_LITE_SUB(reference_ops, Sub, uint8_t);
    }
  } else {
    if (kernel_type == kReference) {
      if (need_broadcast) {
        TF_LITE_SUB(reference_ops, BroadcastSub16POTSlow, int16_t);
      } else {
        TF_LITE_SUB(reference_ops, Sub16, int16_t);
      }
    } else {
      if (need_broadcast) {
        TF_LITE_SUB(optimized_ops, BroadcastSub16POTSlow, int16_t);
      } else {
        TF_LITE_SUB(optimized_ops, Sub16, int16_t);
      }
    }
  }
#undef TF_LITE_SUB
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_7(mht_7_v, 630, "", "./tensorflow/lite/kernels/sub.cc", "Eval");

  auto* params = reinterpret_cast<TfLiteSubParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (output->type == kTfLiteFloat32 || output->type == kTfLiteInt32 ||
      output->type == kTfLiteInt64) {
    EvalSub<kernel_type>(context, node, params, data, input1, input2, output);
  } else if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8 ||
             output->type == kTfLiteInt16) {
    EvalQuantized<kernel_type>(context, node, params, data, input1, input2,
                               output);
  } else {
    context->ReportError(
        context,
        "output type %d is not supported, requires float|uint8|int32 types.",
        output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace sub

TfLiteRegistration* Register_SUB_REF() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_8(mht_8_v, 667, "", "./tensorflow/lite/kernels/sub.cc", "Register_SUB_REF");

  static TfLiteRegistration r = {sub::Init, sub::Free, sub::Prepare,
                                 sub::Eval<sub::kReference>};
  return &r;
}

TfLiteRegistration* Register_SUB_GENERIC_OPT() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_9(mht_9_v, 676, "", "./tensorflow/lite/kernels/sub.cc", "Register_SUB_GENERIC_OPT");

  static TfLiteRegistration r = {sub::Init, sub::Free, sub::Prepare,
                                 sub::Eval<sub::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_SUB_NEON_OPT() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_10(mht_10_v, 685, "", "./tensorflow/lite/kernels/sub.cc", "Register_SUB_NEON_OPT");

  static TfLiteRegistration r = {sub::Init, sub::Free, sub::Prepare,
                                 sub::Eval<sub::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_SUB() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubDTcc mht_11(mht_11_v, 694, "", "./tensorflow/lite/kernels/sub.cc", "Register_SUB");

#ifdef USE_NEON
  return Register_SUB_NEON_OPT();
#else
  return Register_SUB_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
