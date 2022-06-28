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
class MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc() {
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
#include "tensorflow/lite/kernels/internal/reference/reduce.h"

#include <stddef.h>

#include <cstdint>
#include <limits>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reduce {

// This file has reference implementation of reduce_* operators.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct OpData {
  int32_t multiplier;
  int shift;
  // The index of the temporary tensor where the quantized inputs are cached.
  int scratch_tensor_index;
};

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_0(mht_0_v, 227, "", "./tensorflow/lite/kernels/reduce.cc", "OpContext");

    params = reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);
    input = GetInput(context, node, 0);
    axis = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
  }
  TfLiteReducerParams* params;
  const TfLiteTensor* input;
  const TfLiteTensor* axis;
  TfLiteTensor* output;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_1(mht_1_v, 243, "", "./tensorflow/lite/kernels/reduce.cc", "Init");

  // Creates two temp tensors to store index and axis for internal
  // implementation only.
  auto* op_data = new OpData();
  context->AddTensors(context, 3, &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_2(mht_2_v, 254, "", "./tensorflow/lite/kernels/reduce.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

// Resizes the temp tensor that stores resolved axis.
TfLiteStatus ResizeTempAxis(TfLiteContext* context, OpContext* op_context,
                            TfLiteTensor* resolved_axis) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_3(mht_3_v, 263, "", "./tensorflow/lite/kernels/reduce.cc", "ResizeTempAxis");

  TfLiteIntArray* axis_size = TfLiteIntArrayCreate(1);
  axis_size->data[0] = static_cast<int>(NumElements(op_context->axis));
  return context->ResizeTensor(context, resolved_axis, axis_size);
}

// Resizes the temp tensor that stores temp sum of reduced elements.
TfLiteStatus ResizeTempAccum(TfLiteContext* context, OpContext* op_context,
                             TfLiteTensor* temp_accum) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_4(mht_4_v, 274, "", "./tensorflow/lite/kernels/reduce.cc", "ResizeTempAccum");

  TfLiteIntArray* size = TfLiteIntArrayCreate(1);
  size->data[0] = static_cast<int>(NumElements(op_context->output));
  return context->ResizeTensor(context, temp_accum, size);
}

// Resizes output array based on the input size and resolved axis.
TfLiteStatus ResizeOutputTensor(TfLiteContext* context, OpContext* op_context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_5(mht_5_v, 284, "", "./tensorflow/lite/kernels/reduce.cc", "ResizeOutputTensor");

  size_t num_axis = NumElements(op_context->axis);
  const TfLiteIntArray* input_dims = op_context->input->dims;
  int input_num_dims = NumDimensions(op_context->input);
  if (input_num_dims == 0) {
    return context->ResizeTensor(context, op_context->output,
                                 TfLiteIntArrayCreate(0));
  }
  const int* axis = GetTensorData<int>(op_context->axis);
  if (op_context->params->keep_dims) {
    TfLiteIntArray* output_dims = TfLiteIntArrayCreate(input_num_dims);
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          is_axis = true;
          break;
        }
      }
      if (is_axis) {
        output_dims->data[idx] = 1;
      } else {
        output_dims->data[idx] = input_dims->data[idx];
      }
    }
    return context->ResizeTensor(context, op_context->output, output_dims);
  } else {
    // Calculates size of reducing axis.
    int num_reduce_axis = num_axis;
    for (int i = 0; i < num_axis; ++i) {
      int current = axis[i];
      if (current < 0) {
        current += input_num_dims;
      }
      TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims);
      for (int j = 0; j < i; ++j) {
        int previous = axis[j];
        if (previous < 0) {
          previous += input_num_dims;
        }
        if (current == previous) {
          --num_reduce_axis;
          break;
        }
      }
    }
    // Determines output dimensions.
    TfLiteIntArray* output_dims =
        TfLiteIntArrayCreate(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis) {
        output_dims->data[idx - num_skip_axis] = input_dims->data[idx];
      }
    }
    return context->ResizeTensor(context, op_context->output, output_dims);
  }
}

// Initializes temp tensors to store index and resolved axis.
TfLiteStatus InitializeTemporaries(TfLiteContext* context, TfLiteNode* node,
                                   OpContext* op_context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_6(mht_6_v, 356, "", "./tensorflow/lite/kernels/reduce.cc", "InitializeTemporaries");

  // Creates a temp index to iterate through input data.
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(3);
  node->temporaries->data[0] = op_data->scratch_tensor_index;
  TfLiteTensor* scratch_tensor;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &scratch_tensor));
  scratch_tensor->type = kTfLiteInt32;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TfLiteIntArray* index_size = TfLiteIntArrayCreate(1);
  index_size->data[0] = NumDimensions(op_context->input);
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, scratch_tensor, index_size));

  // Creates a temp tensor to store resolved axis given input data.
  node->temporaries->data[1] = op_data->scratch_tensor_index + 1;
  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  resolved_axis->type = kTfLiteInt32;
  // Creates a temporary accumulation tensor to store temp sums when calculating
  // mean or temp prod when calculating reduce prod.
  node->temporaries->data[2] = op_data->scratch_tensor_index + 2;
  TfLiteTensor* temp_accum;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_accum));
  switch (op_context->input->type) {
    case kTfLiteFloat32:
      temp_accum->type = kTfLiteFloat32;
      break;
    case kTfLiteInt32:
      temp_accum->type = kTfLiteInt64;
      break;
    case kTfLiteInt64:
      temp_accum->type = kTfLiteInt64;
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt16:
      temp_accum->type = kTfLiteInt32;
      break;
    case kTfLiteBool:
      temp_accum->type = kTfLiteBool;
      break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus PrepareSimple(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_7(mht_7_v, 411, "", "./tensorflow/lite/kernels/reduce.cc", "PrepareSimple");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpContext op_context(context, node);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.axis->type, kTfLiteInt32);
  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, &op_context));

  if (op_context.input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, op_context.input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point, 0);
  }

  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  // Leaves work to Eval if axis is not constant; else resizes output.
  if (!IsConstantTensor(op_context.axis)) {
    SetTensorToDynamic(op_context.output);
    SetTensorToDynamic(resolved_axis);
    return kTfLiteOk;
  }
  resolved_axis->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    ResizeTempAxis(context, &op_context, resolved_axis));
  TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  return kTfLiteOk;
}

TfLiteStatus PrepareAllOrAny(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_8(mht_8_v, 443, "", "./tensorflow/lite/kernels/reduce.cc", "PrepareAllOrAny");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteBool);
  return PrepareSimple(context, node);
}

TfLiteStatus PrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_9(mht_9_v, 454, "", "./tensorflow/lite/kernels/reduce.cc", "PrepareMeanOrSum");

  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node));
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // reduce_mean requires a buffer to store intermediate sum result.
  OpContext op_context(context, node);
  if (op_context.input->type == kTfLiteInt8 ||
      op_context.input->type == kTfLiteUInt8 ||
      op_context.input->type == kTfLiteInt16) {
    const double real_multiplier =
        static_cast<double>(op_context.input->params.scale) /
        static_cast<double>(op_context.output->params.scale);
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->multiplier, &exponent);
    data->shift = exponent;
  }

  if (op_context.input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, op_context.input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point, 0);
  }

  TfLiteTensor* temp_sum;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_sum));
  if (!IsConstantTensor(op_context.axis)) {
    SetTensorToDynamic(temp_sum);
    return kTfLiteOk;
  }
  temp_sum->allocation_type = kTfLiteArenaRw;
  return ResizeTempAccum(context, &op_context, temp_sum);
}

double GetQuantProdScaling(double input_scale, double output_scale,
                           int reduced_axis_size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_10(mht_10_v, 491, "", "./tensorflow/lite/kernels/reduce.cc", "GetQuantProdScaling");

  // The scaling after taking the product of all the quantized values should
  // be (input_scale**reduced_axis_size)/output_scale but to avoid overflowing
  // the accumulator we instead scale each multiplication by
  // input_scale/nth_root(output_scale, reduced_axis_size).
  return input_scale / std::pow(output_scale, 1.0 / reduced_axis_size);
}

TfLiteStatus PrepareProd(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_11(mht_11_v, 502, "", "./tensorflow/lite/kernels/reduce.cc", "PrepareProd");

  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node));

  OpContext op_context(context, node);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* temp_prod;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_prod));

  if (op_context.input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, op_context.input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point, 0);
  }

  if (!IsConstantTensor(op_context.axis)) {
    SetTensorToDynamic(temp_prod);
    return kTfLiteOk;
  }

  const int input_size = GetTensorShape(op_context.input).FlatSize();
  const int output_size = GetTensorShape(op_context.output).FlatSize();
  // We support both quantized and non-quantized int8/int16 inputs
  if (op_context.input->quantization.type != kTfLiteNoQuantization &&
      (op_context.input->type == kTfLiteInt8 ||
       op_context.input->type == kTfLiteInt16) &&
      input_size != 0 && output_size != 0) {
    const int reduced_axis_size = input_size / output_size;
    const double scaling = GetQuantProdScaling(
        static_cast<double>(op_context.input->params.scale),
        static_cast<double>(op_context.output->params.scale),
        reduced_axis_size);
    QuantizeMultiplier(scaling, &data->multiplier, &data->shift);
  }

  temp_prod->allocation_type = kTfLiteArenaRw;
  return ResizeTempAccum(context, &op_context, temp_prod);
}

void ResolveAxis(const int* axis_data, int axis_count,
                 tflite::MeanParams* op_params) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_12(mht_12_v, 545, "", "./tensorflow/lite/kernels/reduce.cc", "ResolveAxis");

  int i = 0;
  for (; i < axis_count; ++i) {
    op_params->axis[i] = static_cast<int16>(axis_data[i]);
  }
  for (; i < 4; ++i) {
    op_params->axis[i] = 1;
  }
}

template <typename integer_type>
TfLiteStatus EvalMeanReferenceOps(TfLiteContext* context,
                                  const OpContext& op_context, int num_axis,
                                  OpData* data, TfLiteTensor* temp_index,
                                  TfLiteTensor* resolved_axis,
                                  TfLiteTensor* temp_sum) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_13(mht_13_v, 563, "", "./tensorflow/lite/kernels/reduce.cc", "EvalMeanReferenceOps");

  tflite::MeanParams op_params;
  op_params.axis_count = num_axis;
  ResolveAxis(GetTensorData<int>(op_context.axis), num_axis, &op_params);
  const TfLiteTensor* input = op_context.input;

  // TODO(b/139102329): Handle all the cases in the combined reference
  // method.
  if (op_context.params->keep_dims && NumDimensions(input) == 4 &&
      op_params.axis_count == 2 &&
      ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
       (op_params.axis[0] == 2 && op_params.axis[1] == 1))) {
    if (std::is_same<integer_type, uint8_t>::value) {
      reference_ops::Mean(op_params, GetTensorShape(op_context.input),
                          GetTensorData<uint8_t>(op_context.input),
                          op_context.input->params.zero_point,
                          op_context.input->params.scale,
                          GetTensorShape(op_context.output),
                          GetTensorData<uint8_t>(op_context.output),
                          op_context.output->params.zero_point,
                          op_context.output->params.scale);
    } else {
      reference_integer_ops::Mean(
          op_params, data->multiplier, data->shift, GetTensorShape(input),
          GetTensorData<integer_type>(input),
          op_context.input->params.zero_point,
          GetTensorShape(op_context.output),
          GetTensorData<integer_type>(op_context.output),
          op_context.output->params.zero_point);
    }
  } else if (input->params.zero_point == op_context.output->params.zero_point &&
             input->params.scale == op_context.output->params.scale) {
    TF_LITE_ENSURE(
        context,
        reference_ops::Mean(
            GetTensorData<integer_type>(input), input->dims->data,
            input->dims->size, GetTensorData<integer_type>(op_context.output),
            op_context.output->dims->data, op_context.output->dims->size,
            GetTensorData<int>(op_context.axis), num_axis,
            op_context.params->keep_dims, GetTensorData<int>(temp_index),
            GetTensorData<int>(resolved_axis), GetTensorData<int>(temp_sum)));
  } else {
    TF_LITE_ENSURE(
        context,
        reference_ops::QuantizedMeanOrSum<>(
            GetTensorData<integer_type>(input), input->params.zero_point,
            input->params.scale, input->dims->data, input->dims->size,
            GetTensorData<integer_type>(op_context.output),
            op_context.output->params.zero_point,
            op_context.output->params.scale, op_context.output->dims->data,
            op_context.output->dims->size, GetTensorData<int>(op_context.axis),
            num_axis, op_context.params->keep_dims,
            GetTensorData<int>(temp_index), GetTensorData<int>(resolved_axis),
            GetTensorData<int>(temp_sum),
            /*compute_sum=*/false));
  }
  return kTfLiteOk;
}

template <typename T>
void InitializeMeanOutputTyped(TfLiteTensor* output) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_14(mht_14_v, 626, "", "./tensorflow/lite/kernels/reduce.cc", "InitializeMeanOutputTyped");

  RuntimeShape output_shape = GetTensorShape(output);
  const size_t flat_size = output_shape.FlatSize();
  T* output_data = GetTensorData<T>(output);
  T nan_value = std::numeric_limits<T>::quiet_NaN();
  for (int idx = 0; idx < flat_size; ++idx) {
    *output_data++ = nan_value;
  }
}

TfLiteStatus InitializeMeanOutput(TfLiteTensor* output) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_15(mht_15_v, 639, "", "./tensorflow/lite/kernels/reduce.cc", "InitializeMeanOutput");

  switch (output->type) {
    case kTfLiteFloat32:
      InitializeMeanOutputTyped<float>(output);
      break;
    case kTfLiteInt32:
      InitializeMeanOutputTyped<int>(output);
      break;
    case kTfLiteInt64:
      InitializeMeanOutputTyped<int64_t>(output);
      break;
    case kTfLiteUInt8:
      InitializeMeanOutputTyped<uint8_t>(output);
      break;
    case kTfLiteInt8:
      InitializeMeanOutputTyped<int8_t>(output);
      break;
    case kTfLiteInt16:
      InitializeMeanOutputTyped<int16_t>(output);
      break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalMean(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_16(mht_16_v, 669, "", "./tensorflow/lite/kernels/reduce.cc", "EvalMean");

  OpContext op_context(context, node);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  int num_axis = static_cast<int>(NumElements(op_context.axis));
  TfLiteTensor* temp_index;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/0, &temp_index));
  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  TfLiteTensor* temp_sum;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_sum));
  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeTempAxis(context, &op_context, resolved_axis));
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
    TF_LITE_ENSURE_OK(context, ResizeTempAccum(context, &op_context, temp_sum));
  }

  // Return early when input is empty.
  const TfLiteTensor* input = op_context.input;
  RuntimeShape input_shape = GetTensorShape(input);
  if (input_shape.FlatSize() == 0) {
    TF_LITE_ENSURE_OK(context, InitializeMeanOutput(op_context.output));
    return kTfLiteOk;
  }

  if (kernel_type == kGenericOptimized) {
    // Use optimized ops if available.
    switch (input->type) {
      case kTfLiteInt8: {
        tflite::MeanParams op_params;
        op_params.axis_count = num_axis;
        ResolveAxis(GetTensorData<int>(op_context.axis), num_axis, &op_params);
        if (op_context.params->keep_dims && NumDimensions(input) == 4 &&
            op_params.axis_count == 2 &&
            ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
             (op_params.axis[0] == 2 && op_params.axis[1] == 1))) {
          optimized_integer_ops::Mean(
              op_params, input_shape, GetTensorData<int8_t>(input),
              input->params.zero_point, input->params.scale,
              GetTensorShape(op_context.output),
              GetTensorData<int8_t>(op_context.output),
              op_context.output->params.zero_point,
              op_context.output->params.scale,
              CpuBackendContext::GetFromContext(context));
          return kTfLiteOk;
        }
      } break;
      case kTfLiteUInt8: {
        tflite::MeanParams op_params;
        op_params.axis_count = num_axis;
        ResolveAxis(GetTensorData<int>(op_context.axis), num_axis, &op_params);
        if (op_context.params->keep_dims && NumDimensions(input) == 4 &&
            op_params.axis_count == 2 &&
            ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
             (op_params.axis[0] == 2 && op_params.axis[1] == 1))) {
          optimized_ops::Mean(op_params, input_shape,
                              GetTensorData<uint8_t>(input),
                              input->params.zero_point, input->params.scale,
                              GetTensorShape(op_context.output),
                              GetTensorData<uint8_t>(op_context.output),
                              op_context.output->params.zero_point,
                              op_context.output->params.scale,
                              CpuBackendContext::GetFromContext(context));
          return kTfLiteOk;
        }
      } break;
      default:
        break;
    }
  }

  // From here, it uses the reference implementations.
  // TODO(b/139102329): Clean up the function signatures to merge the variations
  // and handle the specialized cases in the combined reference implementations
  // per each op.
  switch (op_context.input->type) {
    case kTfLiteFloat32: {
      tflite::MeanParams op_params;
      op_params.axis_count = num_axis;
      ResolveAxis(GetTensorData<int>(op_context.axis), num_axis, &op_params);
      const TfLiteTensor* input = op_context.input;
      // TODO(b/139102329): Handle the below special case in the combined
      // reference method.
      // Defer to specialized implementation for 4D Mean across axes 1 & 2.
      if (op_context.params->keep_dims && NumDimensions(input) == 4 &&
          op_params.axis_count == 2 &&
          ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
           (op_params.axis[0] == 2 && op_params.axis[1] == 1))) {
        reference_ops::Mean(op_params, input_shape, GetTensorData<float>(input),
                            GetTensorShape(op_context.output),
                            GetTensorData<float>(op_context.output));
      } else {
        TF_LITE_ENSURE(
            context,
            optimized_ops::MeanGeneral(
                GetTensorData<float>(op_context.input),
                op_context.input->dims->data, op_context.input->dims->size,
                GetTensorData<float>(op_context.output),
                op_context.output->dims->data, op_context.output->dims->size,
                GetTensorData<int>(op_context.axis), num_axis,
                op_context.params->keep_dims, GetTensorData<int>(temp_index),
                GetTensorData<int>(resolved_axis),
                GetTensorData<float>(temp_sum)));
      }
    } break;
    case kTfLiteInt32:
      TF_LITE_ENSURE(
          context,
          reference_ops::Mean(
              GetTensorData<int>(op_context.input),
              op_context.input->dims->data, op_context.input->dims->size,
              GetTensorData<int>(op_context.output),
              op_context.output->dims->data, op_context.output->dims->size,
              GetTensorData<int>(op_context.axis), num_axis,
              op_context.params->keep_dims, GetTensorData<int>(temp_index),
              GetTensorData<int>(resolved_axis),
              GetTensorData<int64_t>(temp_sum)));
      break;
    case kTfLiteInt64:
      TF_LITE_ENSURE(
          context,
          reference_ops::Mean(
              GetTensorData<int64_t>(op_context.input),
              op_context.input->dims->data, op_context.input->dims->size,
              GetTensorData<int64_t>(op_context.output),
              op_context.output->dims->data, op_context.output->dims->size,
              GetTensorData<int>(op_context.axis), num_axis,
              op_context.params->keep_dims, GetTensorData<int>(temp_index),
              GetTensorData<int>(resolved_axis),
              GetTensorData<int64_t>(temp_sum)));
      break;
    case kTfLiteInt8: {
      TF_LITE_ENSURE_OK(context, EvalMeanReferenceOps<int8_t>(
                                     context, op_context, num_axis, data,
                                     temp_index, resolved_axis, temp_sum));
    } break;
    case kTfLiteInt16: {
      TF_LITE_ENSURE_OK(context, EvalMeanReferenceOps<int16_t>(
                                     context, op_context, num_axis, data,
                                     temp_index, resolved_axis, temp_sum));
    } break;
    case kTfLiteUInt8: {
      TF_LITE_ENSURE_OK(context, EvalMeanReferenceOps<uint8_t>(
                                     context, op_context, num_axis, data,
                                     temp_index, resolved_axis, temp_sum));
    } break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <typename T>
struct EvalData {
  std::function<T(T, T)> reduce_func;
  const T* input_data;
  T output;
};

// Returns true if 'axis' holds all dims [0 ... N-1] where N is num_dims.
bool IsReduceAllDims(const TfLiteTensor* axis, int num_axis, int num_dims) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_17(mht_17_v, 837, "", "./tensorflow/lite/kernels/reduce.cc", "IsReduceAllDims");

  int dims_mask = 0;
  for (int i = 0; i < num_axis; ++i) {
    dims_mask |= 1 << (axis->data.i32[i]);
  }
  return num_dims == 0 ? dims_mask == 0 : (dims_mask == (1 << num_dims) - 1);
}

// Worker for reducing single interval. Interval is identified by index
// from [start, end).
template <typename T>
struct ReduceWorkerTask : cpu_backend_threadpool::Task {
  ReduceWorkerTask(EvalData<T>* eval_data, int start, int end)
      : eval_data(eval_data), start(start), end(end) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_18(mht_18_v, 853, "", "./tensorflow/lite/kernels/reduce.cc", "ReduceWorkerTask");
}
  void Run() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_19(mht_19_v, 857, "", "./tensorflow/lite/kernels/reduce.cc", "Run");

    auto* input_data = eval_data->input_data;
    T& output = eval_data->output;
    auto& reducer = eval_data->reduce_func;
    for (int i = start; i < end; ++i) {
      output = reducer(output, input_data[i]);
    }
  }

 private:
  EvalData<T>* eval_data;
  int start;
  int end;
};

// Apply reduce operation using the 'reducer' function on all of 'input_data'.
// and reduce all to single element.
template <typename T>
void ReduceAllDims(const T* input_data, const int* input_dims,
                   const int input_num_dims, T* output_data, T init_value,
                   T reducer(const T current, const T in),
                   TfLiteContext* context) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_20(mht_20_v, 881, "", "./tensorflow/lite/kernels/reduce.cc", "ReduceAllDims");

  EvalData<T> eval_data;
  eval_data.reduce_func = reducer;
  eval_data.input_data = input_data;
  eval_data.output = init_value;

  int num_elems = 1;
  for (int i = 0; i < input_num_dims; ++i) {
    num_elems *= input_dims[i];
  }

  // Fetch backend context and number of threads.
  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  int thread_count = cpu_backend_context->max_num_threads();
  const int kMinElementsPerThread = 1024;
  if (num_elems / thread_count < kMinElementsPerThread) thread_count = 1;

  if (thread_count == 1) {
    output_data[0] = num_elems > 0 ? input_data[0] : init_value;
    for (int i = 1; i < num_elems; ++i) {
      output_data[0] = reducer(output_data[0], input_data[i]);
    }
    return;
  }
  std::vector<ReduceWorkerTask<T>> tasks;
  std::vector<EvalData<T>> data;
  tasks.reserve(thread_count);
  data.reserve(thread_count);
  int start = 0;
  for (int i = 0; i < thread_count; ++i) {
    data.push_back(eval_data);
    int end = start + (num_elems - start) / (thread_count - i);
    tasks.emplace_back(ReduceWorkerTask<T>(&data.back(), start, end));
    start = end;
  }
  // Run all tasks on the thread pool.
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
  // Reduce all data from different workers.
  output_data[0] = data[0].output;
  for (int i = 1; i < data.size(); ++i) {
    output_data[0] = reducer(output_data[0], data[i].output);
  }
}

// The underlying logic for Reduce Sum/Prod/Max/Min/Any
template <typename T>
TfLiteStatus EvalLogic(TfLiteContext* context, TfLiteNode* node,
                       OpContext* op_context, T init_value,
                       T reducer(const T current, const T in)) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_21(mht_21_v, 934, "", "./tensorflow/lite/kernels/reduce.cc", "EvalLogic");

  int64_t num_axis = NumElements(op_context->axis);
  TfLiteTensor* temp_index;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/0, &temp_index));
  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context->output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeTempAxis(context, op_context, resolved_axis));
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, op_context));
  }

  const TfLiteTensor* input = op_context->input;
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
      input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.scale,
                      op_context->output->params.scale);
    TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                      op_context->output->params.zero_point);
  }
  int num_resolved_axis = 0;
  if (!tflite::reference_ops::ResolveAxis(
          input->dims->size, GetTensorData<int>(op_context->axis), num_axis,
          GetTensorData<int>(resolved_axis), &num_resolved_axis)) {
    return kTfLiteError;
  }
  if (IsReduceAllDims(resolved_axis, num_resolved_axis, input->dims->size)) {
    ReduceAllDims(GetTensorData<T>(input), input->dims->data, input->dims->size,
                  GetTensorData<T>(op_context->output), init_value, reducer,
                  context);
    return kTfLiteOk;
  }
  TF_LITE_ENSURE(
      context,
      reference_ops::ReduceGeneric<T>(
          GetTensorData<T>(input), input->dims->data, input->dims->size,
          GetTensorData<T>(op_context->output), op_context->output->dims->data,
          op_context->output->dims->size, GetTensorData<int>(op_context->axis),
          num_axis, op_context->params->keep_dims,
          GetTensorData<int>(temp_index), GetTensorData<int>(resolved_axis),
          init_value, reducer));
  return kTfLiteOk;
}

enum ReduceType {
  kSum,
  kProd,
  kMax,
  kMin,
  kAny,
  kAll,
};

// Eval for determined input type and reduce type.
template <typename T>
TfLiteStatus EvalType(TfLiteContext* context, TfLiteNode* node,
                      OpContext* op_context, ReduceType reduce_type) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_22(mht_22_v, 996, "", "./tensorflow/lite/kernels/reduce.cc", "EvalType");

  switch (reduce_type) {
    case kSum:
      return EvalLogic<T>(
          context, node, op_context, static_cast<T>(0),
          [](const T current, const T in) -> T { return in + current; });
      break;
    case kProd:
      return EvalLogic<T>(
          context, node, op_context, static_cast<T>(1),
          [](const T current, const T in) -> T { return in * current; });
      break;
    case kMax:
      return EvalLogic<T>(context, node, op_context,
                          std::numeric_limits<T>::lowest(),
                          [](const T current, const T in) -> T {
                            return (in > current) ? in : current;
                          });
      break;
    case kMin:
      return EvalLogic<T>(context, node, op_context,
                          std::numeric_limits<T>::max(),
                          [](const T current, const T in) -> T {
                            return (in < current) ? in : current;
                          });
      break;
    default:
      return kTfLiteError;
  }
}

// Template specialization for bool type
template <>
TfLiteStatus EvalType<bool>(TfLiteContext* context, TfLiteNode* node,
                            OpContext* op_context, ReduceType reduce_type) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_23(mht_23_v, 1033, "", "./tensorflow/lite/kernels/reduce.cc", "EvalType<bool>");

  switch (reduce_type) {
    case kAny:
      return EvalLogic<bool>(context, node, op_context, false,
                             [](const bool current, const bool in) -> bool {
                               return in || current;
                             });
    case kAll:
      return EvalLogic<bool>(context, node, op_context, true,
                             [](const bool current, const bool in) -> bool {
                               return in && current;
                             });
    default:
      return kTfLiteError;
  }
}

// The entry point that handles input types and then calls template functions to
// handle ReduceType.
template <KernelType kernel_type, ReduceType reduce_type>
TfLiteStatus EvalGeneric(TfLiteContext* context, TfLiteNode* node) {
  if (kernel_type != kReference) {
    return kTfLiteOk;
  }
  OpContext op_context(context, node);
  switch (op_context.input->type) {
    case kTfLiteFloat32:
      return EvalType<float>(context, node, &op_context, reduce_type);
      break;
    case kTfLiteInt32:
      return EvalType<int>(context, node, &op_context, reduce_type);
      break;
    case kTfLiteInt64:
      return EvalType<int64_t>(context, node, &op_context, reduce_type);
      break;
    case kTfLiteUInt8:
      return EvalType<uint8_t>(context, node, &op_context, reduce_type);
      break;
    case kTfLiteInt8:
      return EvalType<int8_t>(context, node, &op_context, reduce_type);
      break;
    case kTfLiteInt16:
      return EvalType<int16_t>(context, node, &op_context, reduce_type);
      break;
    case kTfLiteBool:
      return EvalType<bool>(context, node, &op_context, reduce_type);
      break;
    default:
      return kTfLiteError;
  }
}

TfLiteStatus EvalSum(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_24(mht_24_v, 1088, "", "./tensorflow/lite/kernels/reduce.cc", "EvalSum");

  OpContext op_context(context, node);
  ruy::profiler::ScopeLabel label("Sum");
  const auto& input = op_context.input;
  const auto& output = op_context.output;
  const bool same_scale =
      (input->params.scale == output->params.scale &&
       input->params.zero_point == output->params.zero_point);
  const bool eight_bit_quantized =
      input->type == kTfLiteUInt8 || input->type == kTfLiteInt8;
  const bool need_rescale = (eight_bit_quantized && !same_scale);
  if (need_rescale) {
    // Rescaling 8bit reduce sum.
    int num_axis = static_cast<int>(NumElements(op_context.axis));
    TfLiteTensor* temp_index;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/0, &temp_index));
    TfLiteTensor* resolved_axis;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
    TfLiteTensor* temp_sum;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, /*index=*/2, &temp_sum));
    // Resize the output tensor if the output tensor is dynamic.
    if (IsDynamicTensor(op_context.output)) {
      TF_LITE_ENSURE_OK(context,
                        ResizeTempAxis(context, &op_context, resolved_axis));
      TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
      TF_LITE_ENSURE_OK(context,
                        ResizeTempAccum(context, &op_context, temp_sum));
    }

    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE(
          context,
          reference_ops::QuantizedMeanOrSum<>(
              GetTensorData<uint8_t>(op_context.input),
              op_context.input->params.zero_point,
              op_context.input->params.scale, op_context.input->dims->data,
              op_context.input->dims->size,
              GetTensorData<uint8_t>(op_context.output),
              op_context.output->params.zero_point,
              op_context.output->params.scale, op_context.output->dims->data,
              op_context.output->dims->size,
              GetTensorData<int>(op_context.axis), num_axis,
              op_context.params->keep_dims, GetTensorData<int>(temp_index),
              GetTensorData<int>(resolved_axis), GetTensorData<int32>(temp_sum),
              /*compute_sum=*/true));
    }
    if (input->type == kTfLiteInt8) {
      TF_LITE_ENSURE(
          context,
          reference_ops::QuantizedMeanOrSum<>(
              GetTensorData<int8_t>(op_context.input),
              op_context.input->params.zero_point,
              op_context.input->params.scale, op_context.input->dims->data,
              op_context.input->dims->size,
              GetTensorData<int8_t>(op_context.output),
              op_context.output->params.zero_point,
              op_context.output->params.scale, op_context.output->dims->data,
              op_context.output->dims->size,
              GetTensorData<int>(op_context.axis), num_axis,
              op_context.params->keep_dims, GetTensorData<int>(temp_index),
              GetTensorData<int>(resolved_axis), GetTensorData<int32>(temp_sum),
              /*compute_sum=*/true));
    }
  } else {
    return EvalGeneric<kReference, kSum>(context, node);
  }

  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalQuantizedProd(TfLiteContext* context, TfLiteNode* node,
                               OpContext* op_context) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_25(mht_25_v, 1166, "", "./tensorflow/lite/kernels/reduce.cc", "EvalQuantizedProd");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const int64_t num_axis = NumElements(op_context->axis);
  TfLiteTensor* temp_index;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/0, &temp_index));
  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  TfLiteTensor* temp_prod;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_prod));

  const TfLiteTensor* input = op_context->input;
  TfLiteTensor* output = op_context->output;

  // Return early when input shape has zero dim.
  for (int i = 0; i < input->dims->size; ++i) {
    if (input->dims->data[i] == 0) return kTfLiteOk;
  }

  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeTempAxis(context, op_context, resolved_axis));
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, op_context));
    TF_LITE_ENSURE_OK(context, ResizeTempAccum(context, op_context, temp_prod));

    const int input_size = GetTensorShape(input).FlatSize();
    const int output_size = GetTensorShape(output).FlatSize();
    TF_LITE_ENSURE(context, input_size != 0);
    TF_LITE_ENSURE(context, output_size != 0);

    const int reduced_axis_size = input_size / output_size;
    const double scaling = GetQuantProdScaling(
        static_cast<double>(input->params.scale),
        static_cast<double>(output->params.scale), reduced_axis_size);
    QuantizeMultiplier(scaling, &data->multiplier, &data->shift);
  }

  TF_LITE_ENSURE(
      context,
      reference_ops::QuantizedReduceProd<T>(
          GetTensorData<T>(input), input->params.zero_point,
          GetTensorShape(input), GetTensorData<T>(output),
          output->params.zero_point, GetTensorShape(output),
          GetTensorData<int>(op_context->axis), num_axis,
          op_context->params->keep_dims, GetTensorData<int>(temp_index),
          GetTensorData<int>(resolved_axis), GetTensorData<int32>(temp_prod),
          data->multiplier, data->shift));
  return kTfLiteOk;
}

TfLiteStatus EvalProd(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_26(mht_26_v, 1223, "", "./tensorflow/lite/kernels/reduce.cc", "EvalProd");

  OpContext op_context(context, node);
  // As we need to support both quantized and non-quantized int8/int16 inputs,
  // we separate the evaluation between EvalQuantizedProd for quantized
  // int8/int16 inputs and EvalGeneric for non-quantized int8/int16 (and
  // other non-quantized types).
  if (op_context.input->quantization.type != kTfLiteNoQuantization) {
    if (op_context.input->type == kTfLiteInt8) {
      return EvalQuantizedProd<int8_t>(context, node, &op_context);
    } else if (op_context.input->type == kTfLiteInt16) {
      return EvalQuantizedProd<int16_t>(context, node, &op_context);
    } else {
      TF_LITE_KERNEL_LOG(context, "Unsupported quantized data type: %d",
                         op_context.input->type);
      return kTfLiteError;
    }
  } else {
    return EvalGeneric<reduce::kReference, reduce::kProd>(context, node);
  }
}

}  // namespace reduce

TfLiteRegistration* Register_MEAN_OPT() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_27(mht_27_v, 1249, "", "./tensorflow/lite/kernels/reduce.cc", "Register_MEAN_OPT");

  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareMeanOrSum,
                                 reduce::EvalMean<reduce::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_MEAN_REF() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_28(mht_28_v, 1259, "", "./tensorflow/lite/kernels/reduce.cc", "Register_MEAN_REF");

  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareMeanOrSum,
                                 reduce::EvalMean<reduce::kReference>};
  return &r;
}

TfLiteRegistration* Register_SUM_REF() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_29(mht_29_v, 1269, "", "./tensorflow/lite/kernels/reduce.cc", "Register_SUM_REF");

  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareMeanOrSum, reduce::EvalSum};
  return &r;
}

TfLiteRegistration* Register_REDUCE_PROD_REF() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_30(mht_30_v, 1278, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_PROD_REF");

  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareProd, reduce::EvalProd};
  return &r;
}

TfLiteRegistration* Register_REDUCE_MAX_REF() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_31(mht_31_v, 1287, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_MAX_REF");

  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareSimple,
      reduce::EvalGeneric<reduce::kReference, reduce::kMax>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_MIN_REF() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_32(mht_32_v, 1297, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_MIN_REF");

  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareSimple,
      reduce::EvalGeneric<reduce::kReference, reduce::kMin>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_ANY_REF() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_33(mht_33_v, 1307, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_ANY_REF");

  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareAllOrAny,
      reduce::EvalGeneric<reduce::kReference, reduce::kAny>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_ALL_REF() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_34(mht_34_v, 1317, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_ALL_REF");

  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareAllOrAny,
      reduce::EvalGeneric<reduce::kReference, reduce::kAll>};
  return &r;
}

TfLiteRegistration* Register_MEAN() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_35(mht_35_v, 1327, "", "./tensorflow/lite/kernels/reduce.cc", "Register_MEAN");

#ifdef USE_NEON
  return Register_MEAN_OPT();
#else
  return Register_MEAN_REF();
#endif
}

TfLiteRegistration* Register_SUM() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_36(mht_36_v, 1338, "", "./tensorflow/lite/kernels/reduce.cc", "Register_SUM");
 return Register_SUM_REF(); }
TfLiteRegistration* Register_REDUCE_PROD() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_37(mht_37_v, 1342, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_PROD");

  return Register_REDUCE_PROD_REF();
}
TfLiteRegistration* Register_REDUCE_MAX() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_38(mht_38_v, 1348, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_MAX");
 return Register_REDUCE_MAX_REF(); }
TfLiteRegistration* Register_REDUCE_MIN() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_39(mht_39_v, 1352, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_MIN");
 return Register_REDUCE_MIN_REF(); }
TfLiteRegistration* Register_REDUCE_ANY() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_40(mht_40_v, 1356, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_ANY");
 return Register_REDUCE_ANY_REF(); }
TfLiteRegistration* Register_REDUCE_ALL() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreduceDTcc mht_41(mht_41_v, 1360, "", "./tensorflow/lite/kernels/reduce.cc", "Register_REDUCE_ALL");
 return Register_REDUCE_ALL_REF(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
