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
class MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc() {
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
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/pooling.h"

#include <stddef.h>
#include <stdint.h>

#include <cstdlib>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pooling {

// This file has two implementation of each pooling op.
enum KernelType {
  kReference,
  kGenericOptimized,
};

enum PoolType {
  kAverage,
  kMax,
  kL2,
};

struct OpData {
  TfLitePaddingValues padding;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_0(mht_0_v, 226, "", "./tensorflow/lite/kernels/pooling.cc", "Init");

  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_1(mht_1_v, 236, "", "./tensorflow/lite/kernels/pooling.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

template <PoolType pool_type>
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_2(mht_2_v, 244, "", "./tensorflow/lite/kernels/pooling.cc", "GenericPrepare");

  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int out_width, out_height;

  // Prevent division by 0 in optimized pooling implementations
  TF_LITE_ENSURE(context, params->stride_height > 0);
  TF_LITE_ENSURE(context, params->stride_width > 0);

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      params->filter_height, params->filter_width, padding, &out_height,
      &out_width);

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    if (pool_type == kAverage || pool_type == kMax) {
      TFLITE_DCHECK_LE(std::abs(input->params.scale - output->params.scale),
                       1.0e-6);
      TFLITE_DCHECK_EQ(input->params.zero_point, output->params.zero_point);
    }
    if (pool_type == kL2) {
      // We currently don't have a quantized implementation of L2Pool
      TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
    }
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus AverageEvalFloat(TfLiteContext* context, TfLiteNode* node,
                              TfLitePoolParams* params, OpData* data,
                              const TfLiteTensor* input, TfLiteTensor* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_3(mht_3_v, 301, "", "./tensorflow/lite/kernels/pooling.cc", "AverageEvalFloat");

  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_AVERAGE_POOL(type)                                            \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.float_activation_min = activation_min;                            \
  op_params.float_activation_max = activation_max;                            \
  TF_LITE_ENSURE(context, type::AveragePool(op_params, GetTensorShape(input), \
                                            GetTensorData<float>(input),      \
                                            GetTensorShape(output),           \
                                            GetTensorData<float>(output)))
  if (kernel_type == kReference) {
    TF_LITE_AVERAGE_POOL(reference_ops);
  } else {
    TF_LITE_AVERAGE_POOL(optimized_ops);
  }
#undef TF_LITE_AVERAGE_POOL
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus AverageEvalQuantizedUint8(TfLiteContext* context, TfLiteNode* node,
                                       TfLitePoolParams* params, OpData* data,
                                       const TfLiteTensor* input,
                                       TfLiteTensor* output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_4(mht_4_v, 335, "", "./tensorflow/lite/kernels/pooling.cc", "AverageEvalQuantizedUint8");

  int32_t activation_min;
  int32_t activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);
#define TF_LITE_AVERAGE_POOL(type)                                            \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.quantized_activation_min = activation_min;                        \
  op_params.quantized_activation_max = activation_max;                        \
  TF_LITE_ENSURE(context, type::AveragePool(op_params, GetTensorShape(input), \
                                            GetTensorData<uint8_t>(input),    \
                                            GetTensorShape(output),           \
                                            GetTensorData<uint8_t>(output)))
  if (kernel_type == kReference) {
    TF_LITE_AVERAGE_POOL(reference_ops);
  } else {
    TF_LITE_AVERAGE_POOL(optimized_ops);
  }
#undef TF_LITE_AVERAGE_POOL
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus AverageEvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                                      TfLitePoolParams* params, OpData* data,
                                      const TfLiteTensor* input,
                                      TfLiteTensor* output) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_5(mht_5_v, 370, "", "./tensorflow/lite/kernels/pooling.cc", "AverageEvalQuantizedInt8");

  int32_t activation_min;
  int32_t activation_max;

  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);
#define TF_LITE_AVERAGE_POOL(type)                                            \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.quantized_activation_min = activation_min;                        \
  op_params.quantized_activation_max = activation_max;                        \
  TF_LITE_ENSURE(context, type::AveragePool(op_params, GetTensorShape(input), \
                                            GetTensorData<int8_t>(input),     \
                                            GetTensorShape(output),           \
                                            GetTensorData<int8_t>(output)))
  if (kernel_type == kReference) {
    TF_LITE_AVERAGE_POOL(reference_integer_ops);
  } else {
    TF_LITE_AVERAGE_POOL(optimized_integer_ops);
  }
#undef TF_LITE_AVERAGE_POOL
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus AverageEvalQuantizedInt16(TfLiteContext* context, TfLiteNode* node,
                                       TfLitePoolParams* params, OpData* data,
                                       const TfLiteTensor* input,
                                       TfLiteTensor* output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_6(mht_6_v, 406, "", "./tensorflow/lite/kernels/pooling.cc", "AverageEvalQuantizedInt16");

  int32_t activation_min;
  int32_t activation_max;
  CalculateActivationRangeQuantized(context, params->activation, output,
                                    &activation_min, &activation_max);
#define TF_LITE_AVERAGE_POOL(type)                                            \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.quantized_activation_min = activation_min;                        \
  op_params.quantized_activation_max = activation_max;                        \
  TF_LITE_ENSURE(context, type::AveragePool(op_params, GetTensorShape(input), \
                                            GetTensorData<int16_t>(input),    \
                                            GetTensorShape(output),           \
                                            GetTensorData<int16_t>(output)))
  TF_LITE_AVERAGE_POOL(reference_integer_ops);
#undef TF_LITE_AVERAGE_POOL
  return kTfLiteOk;
}

template <KernelType kernel_type>
void MaxEvalFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLitePoolParams* params, OpData* data,
                  const TfLiteTensor* input, TfLiteTensor* output) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_7(mht_7_v, 436, "", "./tensorflow/lite/kernels/pooling.cc", "MaxEvalFloat");

  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_MAX_POOL(type)                                                 \
  tflite::PoolParams op_params;                                                \
  op_params.stride_height = params->stride_height;                             \
  op_params.stride_width = params->stride_width;                               \
  op_params.filter_height = params->filter_height;                             \
  op_params.filter_width = params->filter_width;                               \
  op_params.padding_values.height = data->padding.height;                      \
  op_params.padding_values.width = data->padding.width;                        \
  op_params.float_activation_min = activation_min;                             \
  op_params.float_activation_max = activation_max;                             \
  type::MaxPool(op_params, GetTensorShape(input), GetTensorData<float>(input), \
                GetTensorShape(output), GetTensorData<float>(output))
  if (kernel_type == kReference) {
    TF_LITE_MAX_POOL(reference_ops);
  } else {
    TF_LITE_MAX_POOL(optimized_ops);
  }
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void MaxEvalQuantizedUInt8(TfLiteContext* context, TfLiteNode* node,
                           TfLitePoolParams* params, OpData* data,
                           const TfLiteTensor* input, TfLiteTensor* output) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_8(mht_8_v, 466, "", "./tensorflow/lite/kernels/pooling.cc", "MaxEvalQuantizedUInt8");

  int32_t activation_min;
  int32_t activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);
#define TF_LITE_MAX_POOL(type)                                         \
  tflite::PoolParams op_params;                                        \
  op_params.stride_height = params->stride_height;                     \
  op_params.stride_width = params->stride_width;                       \
  op_params.filter_height = params->filter_height;                     \
  op_params.filter_width = params->filter_width;                       \
  op_params.padding_values.height = data->padding.height;              \
  op_params.padding_values.width = data->padding.width;                \
  op_params.quantized_activation_min = activation_min;                 \
  op_params.quantized_activation_max = activation_max;                 \
  type::MaxPool(op_params, GetTensorShape(input),                      \
                GetTensorData<uint8_t>(input), GetTensorShape(output), \
                GetTensorData<uint8_t>(output))
  if (kernel_type == kReference) {
    TF_LITE_MAX_POOL(reference_ops);
  } else {
    TF_LITE_MAX_POOL(optimized_ops);
  }
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void MaxEvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                          TfLitePoolParams* params, OpData* data,
                          const TfLiteTensor* input, TfLiteTensor* output) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_9(mht_9_v, 498, "", "./tensorflow/lite/kernels/pooling.cc", "MaxEvalQuantizedInt8");

  int32_t activation_min;
  int32_t activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);
#define TF_LITE_MAX_POOL(type)                                        \
  tflite::PoolParams op_params;                                       \
  op_params.stride_height = params->stride_height;                    \
  op_params.stride_width = params->stride_width;                      \
  op_params.filter_height = params->filter_height;                    \
  op_params.filter_width = params->filter_width;                      \
  op_params.padding_values.height = data->padding.height;             \
  op_params.padding_values.width = data->padding.width;               \
  op_params.quantized_activation_min = activation_min;                \
  op_params.quantized_activation_max = activation_max;                \
  type::MaxPool(op_params, GetTensorShape(input),                     \
                GetTensorData<int8_t>(input), GetTensorShape(output), \
                GetTensorData<int8_t>(output))
  if (kernel_type == kReference) {
    TF_LITE_MAX_POOL(reference_integer_ops);
  } else {
    TF_LITE_MAX_POOL(optimized_integer_ops);
  }
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void MaxEvalQuantizedInt16(TfLiteContext* context, TfLiteNode* node,
                           TfLitePoolParams* params, OpData* data,
                           const TfLiteTensor* input, TfLiteTensor* output) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_10(mht_10_v, 530, "", "./tensorflow/lite/kernels/pooling.cc", "MaxEvalQuantizedInt16");

  int32_t activation_min;
  int32_t activation_max;
  CalculateActivationRangeQuantized(context, params->activation, output,
                                    &activation_min, &activation_max);
#define TF_LITE_MAX_POOL(type)                                         \
  tflite::PoolParams op_params;                                        \
  op_params.stride_height = params->stride_height;                     \
  op_params.stride_width = params->stride_width;                       \
  op_params.filter_height = params->filter_height;                     \
  op_params.filter_width = params->filter_width;                       \
  op_params.padding_values.height = data->padding.height;              \
  op_params.padding_values.width = data->padding.width;                \
  op_params.quantized_activation_min = activation_min;                 \
  op_params.quantized_activation_max = activation_max;                 \
  type::MaxPool(op_params, GetTensorShape(input),                      \
                GetTensorData<int16_t>(input), GetTensorShape(output), \
                GetTensorData<int16_t>(output))
  TF_LITE_MAX_POOL(reference_integer_ops);
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void L2EvalFloat(TfLiteContext* context, TfLiteNode* node,
                 TfLitePoolParams* params, OpData* data,
                 const TfLiteTensor* input, TfLiteTensor* output) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_11(mht_11_v, 558, "", "./tensorflow/lite/kernels/pooling.cc", "L2EvalFloat");

  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_L2_POOL(type)                                                 \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.float_activation_min = activation_min;                            \
  op_params.float_activation_max = activation_max;                            \
  type::L2Pool(op_params, GetTensorShape(input), GetTensorData<float>(input), \
               GetTensorShape(output), GetTensorData<float>(output))
  if (kernel_type == kReference) {
    TF_LITE_L2_POOL(reference_ops);
  } else {
    TF_LITE_L2_POOL(optimized_ops);
  }
#undef TF_LITE_L2_POOL
}

#undef TF_LITE_KERNEL_TYPE_DISPATCH

template <KernelType kernel_type>
TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return AverageEvalFloat<kernel_type>(context, node, params, data, input,
                                           output);
    case kTfLiteUInt8:
      return AverageEvalQuantizedUint8<kernel_type>(context, node, params, data,
                                                    input, output);
    case kTfLiteInt8:
      return AverageEvalQuantizedInt8<kernel_type>(context, node, params, data,
                                                   input, output);
    case kTfLiteInt16:
      return AverageEvalQuantizedInt16<kernel_type>(context, node, params, data,
                                                    input, output);
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_12(mht_12_v, 618, "", "./tensorflow/lite/kernels/pooling.cc", "MaxEval");

  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      MaxEvalFloat<kernel_type>(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
      MaxEvalQuantizedUInt8<kernel_type>(context, node, params, data, input,
                                         output);
      break;
    case kTfLiteInt8:
      MaxEvalQuantizedInt8<kernel_type>(context, node, params, data, input,
                                        output);
      break;
    case kTfLiteInt16:
      MaxEvalQuantizedInt16<kernel_type>(context, node, params, data, input,
                                         output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus L2Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_13(mht_13_v, 654, "", "./tensorflow/lite/kernels/pooling.cc", "L2Eval");

  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      L2EvalFloat<kernel_type>(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
    // We don't have a quantized implementation, so just fall through to the
    // 'default' case.
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace pooling

TfLiteRegistration* Register_AVERAGE_POOL_REF() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_14(mht_14_v, 682, "", "./tensorflow/lite/kernels/pooling.cc", "Register_AVERAGE_POOL_REF");

  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kAverage>,
                                 pooling::AverageEval<pooling::kReference>};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_REF() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_15(mht_15_v, 692, "", "./tensorflow/lite/kernels/pooling.cc", "Register_MAX_POOL_REF");

  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kMax>,
                                 pooling::MaxEval<pooling::kReference>};
  return &r;
}

TfLiteRegistration* Register_L2_POOL_REF() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_16(mht_16_v, 702, "", "./tensorflow/lite/kernels/pooling.cc", "Register_L2_POOL_REF");

  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kL2>,
                                 pooling::L2Eval<pooling::kReference>};
  return &r;
}

TfLiteRegistration* Register_AVERAGE_POOL_GENERIC_OPT() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_17(mht_17_v, 712, "", "./tensorflow/lite/kernels/pooling.cc", "Register_AVERAGE_POOL_GENERIC_OPT");

  static TfLiteRegistration r = {
      pooling::Init, pooling::Free, pooling::GenericPrepare<pooling::kAverage>,
      pooling::AverageEval<pooling::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_GENERIC_OPT() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_18(mht_18_v, 722, "", "./tensorflow/lite/kernels/pooling.cc", "Register_MAX_POOL_GENERIC_OPT");

  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kMax>,
                                 pooling::MaxEval<pooling::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_L2_POOL_GENERIC_OPT() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_19(mht_19_v, 732, "", "./tensorflow/lite/kernels/pooling.cc", "Register_L2_POOL_GENERIC_OPT");

  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kL2>,
                                 pooling::L2Eval<pooling::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_AVERAGE_POOL_2D() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_20(mht_20_v, 742, "", "./tensorflow/lite/kernels/pooling.cc", "Register_AVERAGE_POOL_2D");

  return Register_AVERAGE_POOL_GENERIC_OPT();
}

TfLiteRegistration* Register_MAX_POOL_2D() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_21(mht_21_v, 749, "", "./tensorflow/lite/kernels/pooling.cc", "Register_MAX_POOL_2D");

  return Register_MAX_POOL_GENERIC_OPT();
}

TfLiteRegistration* Register_L2_POOL_2D() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSpoolingDTcc mht_22(mht_22_v, 756, "", "./tensorflow/lite/kernels/pooling.cc", "Register_L2_POOL_2D");

  return Register_L2_POOL_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
