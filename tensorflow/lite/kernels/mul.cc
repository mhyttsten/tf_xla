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
class MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc() {
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
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/mul.h"

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace mul {

// This file has three implementation of Mul.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  // Parameters used in the quantized paths where the output is 8bit
  int32 output_activation_min;
  int32 output_activation_max;

  // Parameters used in all quantized paths
  int32_t output_multiplier;
  int output_shift;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_0(mht_0_v, 232, "", "./tensorflow/lite/kernels/mul.cc", "Init");

  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_1(mht_1_v, 240, "", "./tensorflow/lite/kernels/mul.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/kernels/mul.cc", "Prepare");

  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);
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

  const bool requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8 ||
      output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
    double real_multiplier =
        input1->params.scale * input2->params.scale / output->params.scale;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  }

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
void EvalMul(TfLiteContext* context, TfLiteNode* node, TfLiteMulParams* params,
             const OpData* data, const TfLiteTensor* input1,
             const TfLiteTensor* input2, TfLiteTensor* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_3(mht_3_v, 296, "", "./tensorflow/lite/kernels/mul.cc", "EvalMul");

  tflite::ArithmeticParams op_params;
  const bool need_broadcast = optimized_ops::ProcessBroadcastShapes(
      GetTensorShape(input1), GetTensorShape(input2), &op_params);
#define TF_LITE_MUL(type, opname, data_type)                             \
  data_type output_activation_min, output_activation_max;                \
  CalculateActivationRange(params->activation, &output_activation_min,   \
                           &output_activation_max);                      \
  SetActivationParams(output_activation_min, output_activation_max,      \
                      &op_params);                                       \
  type::opname(op_params, GetTensorShape(input1),                        \
               GetTensorData<data_type>(input1), GetTensorShape(input2), \
               GetTensorData<data_type>(input2), GetTensorShape(output), \
               GetTensorData<data_type>(output))

  if (output->type == kTfLiteInt32) {
    if (kernel_type == kReference) {
      if (need_broadcast) {
        TF_LITE_MUL(reference_ops, BroadcastMul4DSlow, int32_t);
      } else {
        TF_LITE_MUL(reference_ops, Mul, int32_t);
      }
    } else {
      if (need_broadcast) {
        TF_LITE_MUL(optimized_ops, BroadcastMul4DSlow, int32_t);
      } else {
        TF_LITE_MUL(optimized_ops, Mul, int32_t);
      }
    }
  } else if (output->type == kTfLiteFloat32) {
    if (kernel_type == kReference) {
      if (need_broadcast) {
        TF_LITE_MUL(reference_ops, BroadcastMul4DSlow, float);
      } else {
        TF_LITE_MUL(reference_ops, Mul, float);
      }
    } else {
      if (need_broadcast) {
        TF_LITE_MUL(optimized_ops, BroadcastMulDispatch, float);
      } else {
        TF_LITE_MUL(optimized_ops, Mul, float);
      }
    }
  } else if (output->type == kTfLiteInt64) {
    if (need_broadcast) {
      TF_LITE_MUL(reference_ops, BroadcastMul4DSlow, int64_t);
    } else {
      TF_LITE_MUL(reference_ops, Mul, int64_t);
    }
  }
#undef TF_LITE_MUL
}

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteMulParams* params, const OpData* data,
                           const TfLiteTensor* input1,
                           const TfLiteTensor* input2, TfLiteTensor* output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_4(mht_4_v, 356, "", "./tensorflow/lite/kernels/mul.cc", "EvalQuantized");

  if (input1->type == input2->type && input1->type == output->type &&
      (input1->type == kTfLiteUInt8 || input1->type == kTfLiteInt8 ||
       input1->type == kTfLiteInt16)) {
    tflite::ArithmeticParams op_params;
    SetActivationParams(data->output_activation_min,
                        data->output_activation_max, &op_params);
    op_params.input1_offset = -input1->params.zero_point;
    op_params.input2_offset = -input2->params.zero_point;
    op_params.output_offset = output->params.zero_point;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    bool need_broadcast = optimized_ops::ProcessBroadcastShapes(
        GetTensorShape(input1), GetTensorShape(input2), &op_params);
#define TF_LITE_MUL(type, opname, dtype)                             \
  type::opname(op_params, GetTensorShape(input1),                    \
               GetTensorData<dtype>(input1), GetTensorShape(input2), \
               GetTensorData<dtype>(input2), GetTensorShape(output), \
               GetTensorData<dtype>(output))
    if (input1->type == kTfLiteInt8) {
      if (kernel_type == kReference) {
        if (need_broadcast) {
          TF_LITE_MUL(reference_integer_ops, BroadcastMul4DSlow, int8_t);
        } else {
          TF_LITE_MUL(reference_integer_ops, Mul, int8_t);
        }
      } else {
        if (need_broadcast) {
          TF_LITE_MUL(optimized_integer_ops, BroadcastMulDispatch, int8_t);
        } else {
          TF_LITE_MUL(optimized_integer_ops, Mul, int8_t);
        }
      }
    } else if (input1->type == kTfLiteInt16) {
      // We have this check, because in case of int16
      // input1_val*input2_val can overflow int32:
      // see MulElementwise -
      // tensorflow/lite/kernels/internal/reference/integer_ops/mul.h in case of
      // 16-bit this function is used in symmetric quantization, so offset
      // should be zero.
      TF_LITE_ENSURE_EQ(context, op_params.input1_offset, 0.0);
      TF_LITE_ENSURE_EQ(context, op_params.input2_offset, 0.0);
      TF_LITE_ENSURE_EQ(context, op_params.output_offset, 0.0);

      if (need_broadcast) {
        TF_LITE_MUL(reference_integer_ops, BroadcastMul4DSlow, int16_t);
      } else {
        TF_LITE_MUL(reference_integer_ops, Mul, int16_t);
      }
    } else {
      // type == kTfLiteUInt8
      if (kernel_type == kReference) {
        if (need_broadcast) {
          TF_LITE_MUL(reference_ops, BroadcastMul4DSlow, uint8_t);
        } else {
          TF_LITE_MUL(reference_ops, Mul, uint8_t);
        }
      } else {
        if (need_broadcast) {
          TF_LITE_MUL(optimized_ops, BroadcastMulDispatch, uint8_t);
        } else {
          TF_LITE_MUL(optimized_ops, Mul, uint8_t);
        }
      }
    }
#undef TF_LITE_MUL
  } else if (input1->type == kTfLiteInt16 && input2->type == kTfLiteInt16 &&
             (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8)) {
#define TF_LITE_MUL(type, opname, output_dtype)                        \
  tflite::ArithmeticParams op_params;                                  \
  SetActivationParams(data->output_activation_min,                     \
                      data->output_activation_max, &op_params);        \
  op_params.output_offset = output->params.zero_point;                 \
  type::opname(op_params, GetTensorShape(input1),                      \
               GetTensorData<int16_t>(input1), GetTensorShape(input2), \
               GetTensorData<int16_t>(input2), GetTensorShape(output), \
               GetTensorData<output_dtype>(output))
    if (output->type == kTfLiteInt8) {
      TF_LITE_MUL(reference_integer_ops, Mul, int8_t);
    } else {
      if (kernel_type == kReference) {
        TF_LITE_MUL(reference_ops, Mul, uint8_t);
      } else {
        TF_LITE_MUL(optimized_ops, Mul, uint8_t);
      }
    }
#undef TF_LITE_MUL
  } else {
    context->ReportError(
        context, "Unsupported combination of input and output types in Mul.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_5(mht_5_v, 455, "", "./tensorflow/lite/kernels/mul.cc", "Eval");

  auto* params = reinterpret_cast<TfLiteMulParams*>(node->builtin_data);
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
    EvalMul<kernel_type>(context, node, params, data, input1, input2, output);
  } else if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8 ||
             output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_OK(
        context, EvalQuantized<kernel_type>(context, node, params, data, input1,
                                            input2, output));
  } else {
    context->ReportError(context,
                         "Mul only supports FLOAT32, INT32 and quantized UINT8,"
                         " INT8 and INT16 now, got %d.",
                         output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace mul

TfLiteRegistration* Register_MUL_REF() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_6(mht_6_v, 493, "", "./tensorflow/lite/kernels/mul.cc", "Register_MUL_REF");

  static TfLiteRegistration r = {mul::Init, mul::Free, mul::Prepare,
                                 mul::Eval<mul::kReference>};
  return &r;
}

TfLiteRegistration* Register_MUL_GENERIC_OPT() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_7(mht_7_v, 502, "", "./tensorflow/lite/kernels/mul.cc", "Register_MUL_GENERIC_OPT");

  static TfLiteRegistration r = {mul::Init, mul::Free, mul::Prepare,
                                 mul::Eval<mul::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_MUL_NEON_OPT() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_8(mht_8_v, 511, "", "./tensorflow/lite/kernels/mul.cc", "Register_MUL_NEON_OPT");

  static TfLiteRegistration r = {mul::Init, mul::Free, mul::Prepare,
                                 mul::Eval<mul::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_MUL() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSmulDTcc mht_9(mht_9_v, 520, "", "./tensorflow/lite/kernels/mul.cc", "Register_MUL");

#ifdef USE_NEON
  return Register_MUL_NEON_OPT();
#else
  return Register_MUL_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
