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
class MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc() {
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
#include "tensorflow/lite/kernels/internal/reference/quantize.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace quantize {

// This file has two implementation of Quantize.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct OpData {
  int32_t output_multiplier;
  int output_shift;
};

inline bool IsQuantizedPerChannel(const TfLiteTensor* input) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc mht_0(mht_0_v, 215, "", "./tensorflow/lite/kernels/quantize.cc", "IsQuantizedPerChannel");

  if (input->quantization.type == kTfLiteAffineQuantization &&
      input->quantization.params) {
    auto* quant_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    return (quant_params->scale && quant_params->scale->size > 1);
  }
  return false;
}

namespace {
template <KernelType kernel_type, typename output_type>
static inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                                  const RuntimeShape& input_shape,
                                  const float* input_data,
                                  const RuntimeShape& output_shape,
                                  output_type* output_data) {
  if (kernel_type == kReference) {
    reference_ops::AffineQuantize(op_params, input_shape, input_data,
                                  output_shape, output_data);
  } else {
    optimized_ops::AffineQuantize(op_params, input_shape, input_data,
                                  output_shape, output_data);
  }
}

template <KernelType kernel_type, typename input_type, typename output_type>
static inline void Requantize(const input_type* input_data, int32_t size,
                              int32_t effective_scale_multiplier,
                              int32_t effective_scale_shift,
                              int32_t input_zeropoint, int32_t output_zeropoint,
                              output_type* output_data) {
  if (kernel_type == kReference) {
    reference_ops::Requantize(input_data, size, effective_scale_multiplier,
                              effective_scale_shift, input_zeropoint,
                              output_zeropoint, output_data);
  } else {
    optimized_ops::Requantize(input_data, size, effective_scale_multiplier,
                              effective_scale_shift, input_zeropoint,
                              output_zeropoint, output_data);
  }
}

void ReportError(TfLiteContext* context, TfLiteType input_type,
                 TfLiteType output_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc mht_1(mht_1_v, 262, "", "./tensorflow/lite/kernels/quantize.cc", "ReportError");

  context->ReportError(
      context, "Input type %s with Output type %s is not currently supported.",
      TfLiteTypeGetName(input_type), TfLiteTypeGetName(output_type));
}
}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc mht_2(mht_2_v, 273, "", "./tensorflow/lite/kernels/quantize.cc", "Init");

  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc mht_3(mht_3_v, 280, "", "./tensorflow/lite/kernels/quantize.cc", "Free");

  delete static_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc mht_4(mht_4_v, 287, "", "./tensorflow/lite/kernels/quantize.cc", "Prepare");

  OpData* data = static_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  // Currently this only support affine quantization.
  TF_LITE_ENSURE_EQ(context, output->quantization.type,
                    kTfLiteAffineQuantization);

  if (input->type == kTfLiteFloat32) {
    // Quantize use case.
    TF_LITE_ENSURE(context, output->type == kTfLiteUInt8 ||
                                output->type == kTfLiteInt8 ||
                                output->type == kTfLiteInt16);
  } else {
    // Requantize use case.
    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE(context, output->type == kTfLiteInt8 ||
                                  output->type == kTfLiteInt16 ||
                                  output->type == kTfLiteInt32);
    } else if (input->type == kTfLiteInt32) {
      TF_LITE_ENSURE(
          context, output->type == kTfLiteInt8 || output->type == kTfLiteInt16);
    } else {
      TF_LITE_ENSURE(context,
                     input->type == kTfLiteInt8 || input->type == kTfLiteUInt8);
      TF_LITE_ENSURE(
          context, output->type == kTfLiteUInt8 || output->type == kTfLiteInt8);
    }
    const double effective_output_scale =
        static_cast<double>(input->params.scale) /
        static_cast<double>(output->params.scale);
    QuantizeMultiplier(effective_output_scale, &data->output_multiplier,
                       &data->output_shift);
  }

  if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc mht_5(mht_5_v, 341, "", "./tensorflow/lite/kernels/quantize.cc", "Eval");

  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape output_shape = GetTensorShape(output);

  switch (input->type) {
    case kTfLiteFloat32: {
      // Float to int8, uint8, int16.
      const float* input_data = GetTensorData<float>(input);

      if (IsQuantizedPerChannel(output)) {
        // Per-channel quantization: one scale and zero point for each channel.
        const auto* quantization_params =
            reinterpret_cast<const TfLiteAffineQuantization*>(
                output->quantization.params);
        PerChannelQuantizationParams per_channel_op_params;
        per_channel_op_params.quantized_dimension =
            quantization_params->quantized_dimension;
        per_channel_op_params.scale = quantization_params->scale->data;
        per_channel_op_params.zero_point =
            quantization_params->zero_point->data;

        switch (output->type) {
          case kTfLiteInt8:
            reference_ops::PerChannelQuantize(
                per_channel_op_params, input_shape, input_data, output_shape,
                GetTensorData<int8_t>(output));
            return kTfLiteOk;
          case kTfLiteUInt8:
            reference_ops::PerChannelQuantize(
                per_channel_op_params, input_shape, input_data, output_shape,
                GetTensorData<uint8_t>(output));
            return kTfLiteOk;
          case kTfLiteInt16:
            reference_ops::PerChannelQuantize(
                per_channel_op_params, input_shape, input_data, output_shape,
                GetTensorData<int16_t>(output));
            return kTfLiteOk;
          default:
            ReportError(context, input->type, output->type);
            return kTfLiteError;
        }
      } else {
        // Per-node quantization: single scale and zero point for all channels.
        tflite::QuantizationParams op_params;
        op_params.zero_point = output->params.zero_point;
        op_params.scale = output->params.scale;

        switch (output->type) {
          case kTfLiteInt8:
            AffineQuantize<kernel_type>(op_params, input_shape, input_data,
                                        output_shape,
                                        GetTensorData<int8_t>(output));
            return kTfLiteOk;
          case kTfLiteUInt8:
            AffineQuantize<kernel_type>(op_params, input_shape, input_data,
                                        output_shape,
                                        GetTensorData<uint8_t>(output));
            return kTfLiteOk;
          case kTfLiteInt16:
            AffineQuantize<kernel_type>(op_params, input_shape, input_data,
                                        output_shape,
                                        GetTensorData<int16_t>(output));
            return kTfLiteOk;
          default:
            ReportError(context, input->type, output->type);
            return kTfLiteError;
        }
      }
    }
    // This case is not supported by the converter or other TFLite tools. The
    // only use case is for applications that take quantized int32 inference
    // inputs.
    case kTfLiteInt32: {
      // int32 to int8 or int16.
      switch (output->type) {
        case kTfLiteInt8:
          Requantize<kernel_type>(GetTensorData<int32_t>(input),
                                  MatchingFlatSize(input_shape, output_shape),
                                  data->output_multiplier, data->output_shift,
                                  input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int8_t>(output));
          return kTfLiteOk;
        case kTfLiteInt16:
          Requantize<kernel_type>(GetTensorData<int32_t>(input),
                                  MatchingFlatSize(input_shape, output_shape),
                                  data->output_multiplier, data->output_shift,
                                  input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int16_t>(output));
          return kTfLiteOk;
        default:
          ReportError(context, input->type, output->type);
          return kTfLiteError;
      }
    }
    case kTfLiteInt16: {
      // int16 to int8 or int16.
      switch (output->type) {
        case kTfLiteInt8:
          Requantize<kernel_type>(GetTensorData<int16_t>(input),
                                  MatchingFlatSize(input_shape, output_shape),
                                  data->output_multiplier, data->output_shift,
                                  input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int8_t>(output));
          return kTfLiteOk;
        case kTfLiteInt16:
          Requantize<kernel_type>(GetTensorData<int16_t>(input),
                                  MatchingFlatSize(input_shape, output_shape),
                                  data->output_multiplier, data->output_shift,
                                  input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int16_t>(output));
          return kTfLiteOk;
        case kTfLiteInt32:
          // This case is not supported by the converter or other TFLite tools.
          // The only use case is for applications that take quantized int32
          // inference outputs.
          Requantize<kernel_type>(GetTensorData<int16_t>(input),
                                  MatchingFlatSize(input_shape, output_shape),
                                  data->output_multiplier, data->output_shift,
                                  input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int32_t>(output));
          return kTfLiteOk;
        default:
          ReportError(context, input->type, output->type);
          return kTfLiteError;
      }
    }
    case kTfLiteInt8: {
      // int8 to int8, uint8.
      const int32_t size = MatchingFlatSize(input_shape, output_shape);
      const int8_t* input_data = GetTensorData<int8_t>(input);
      switch (output->type) {
        case kTfLiteInt8:
          Requantize<kernel_type>(input_data, size, data->output_multiplier,
                                  data->output_shift, input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int8_t>(output));
          return kTfLiteOk;
        case kTfLiteUInt8:
          Requantize<kernel_type>(input_data, size, data->output_multiplier,
                                  data->output_shift, input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<uint8_t>(output));
          return kTfLiteOk;
        default:
          ReportError(context, input->type, output->type);
          return kTfLiteError;
      }
    }
    case kTfLiteUInt8: {
      // uint8 to int8, uint8.
      const int32_t size = MatchingFlatSize(input_shape, output_shape);
      const uint8_t* input_data = GetTensorData<uint8_t>(input);
      switch (output->type) {
        case kTfLiteInt8:
          Requantize<kernel_type>(input_data, size, data->output_multiplier,
                                  data->output_shift, input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int8_t>(output));
          return kTfLiteOk;
        case kTfLiteUInt8:
          Requantize<kernel_type>(input_data, size, data->output_multiplier,
                                  data->output_shift, input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<uint8_t>(output));
          return kTfLiteOk;
        default:
          ReportError(context, input->type, output->type);
          return kTfLiteError;
      }
    }
    default:
      ReportError(context, input->type, output->type);
      return kTfLiteError;
  }
}

}  // namespace quantize

// This Op (QUANTIZE) quantizes the input and produces quantized output.
// The input can be either float or quantized. If the input is float,
// AffineQuantize takes scale and zero point and quantize the float value to
// quantized output, in int8 or uint8 format. If the input is quantized value,
// the op requantize the input (of a certain type, with a given scale and zero
// point) to the output of the same or different type with a same or different
// scale and zero point.
TfLiteRegistration* Register_QUANTIZE_OPT() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc mht_6(mht_6_v, 541, "", "./tensorflow/lite/kernels/quantize.cc", "Register_QUANTIZE_OPT");

  static TfLiteRegistration r = {quantize::Init, quantize::Free,
                                 quantize::Prepare,
                                 quantize::Eval<quantize::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_QUANTIZE_REF() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc mht_7(mht_7_v, 551, "", "./tensorflow/lite/kernels/quantize.cc", "Register_QUANTIZE_REF");

  static TfLiteRegistration r = {quantize::Init, quantize::Free,
                                 quantize::Prepare,
                                 quantize::Eval<quantize::kReference>};
  return &r;
}

TfLiteRegistration* Register_QUANTIZE() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSquantizeDTcc mht_8(mht_8_v, 561, "", "./tensorflow/lite/kernels/quantize.cc", "Register_QUANTIZE");
 return Register_QUANTIZE_OPT(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
