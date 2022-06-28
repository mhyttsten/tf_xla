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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsvdfDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsvdfDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsvdfDTcc() {
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

// SVDF op that compresses a fully connected op via low-rank matrix
// factorization. See https://research.google.com/pubs/archive/43813.pdf for
// details.

#include "tensorflow/lite/kernels/internal/reference/svdf.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace svdf {

namespace {

struct OpData {
  int scratch_tensor_index;
  bool float_weights_time_initialized;
  int32 effective_scale_1_a;
  int effective_scale_1_b;
  int32 effective_scale_2_a;
  int effective_scale_2_b;
  bool compute_row_sums = false;
};

}  // namespace

// Input tensors.
constexpr int kInputTensor = 0;
constexpr int kWeightsFeatureTensor = 1;
constexpr int kWeightsTimeTensor = 2;
constexpr int kBiasTensor = 3;
// This is a variable tensor, and will be modified by this op.
constexpr int kStateTensor = 4;

// Output tensor.
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSsvdfDTcc mht_0(mht_0_v, 233, "", "./tensorflow/lite/kernels/svdf.cc", "Init");

  auto* op_data = new OpData();
  op_data->float_weights_time_initialized = false;
  // Note: only needs 6 scratch tensors when is_hybrid_op, only 1 otherwise.
  context->AddTensors(context, /*tensors_to_add=*/6,
                      &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsvdfDTcc mht_1(mht_1_v, 245, "", "./tensorflow/lite/kernels/svdf.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsvdfDTcc mht_2(mht_2_v, 252, "", "./tensorflow/lite/kernels/svdf.cc", "Prepare");

  const auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  int scratch_tensor_index = op_data->scratch_tensor_index;

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* weights_feature;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kWeightsFeatureTensor,
                                          &weights_feature));
  const TfLiteTensor* weights_time;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kWeightsTimeTensor, &weights_time));

  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE(context, rank != 0);
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];
  TF_LITE_ENSURE_EQ(context, input->dims->data[1],
                    weights_feature->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[0], num_filters);

  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  if (bias) {
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], num_units);
  }

  const TfLiteTensor* state;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStateTensor, &state));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Check the shape of input state tensors.
  TF_LITE_ENSURE_EQ(context, NumDimensions(state), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(state, 0), batch_size);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(state, 1),
                    memory_size * num_filters);

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(2);
  output_size_array->data[0] = batch_size;
  output_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  // The weights are of consistent type, so it suffices to check one.
  const bool is_hybrid_op = IsHybridOp(input, weights_feature);
  const bool is_full_integer = input->type == kTfLiteInt8;

  // Resize scratch.
  TfLiteIntArrayFree(node->temporaries);
  if (is_hybrid_op) {
    node->temporaries = TfLiteIntArrayCreate(6);
  } else if (is_full_integer) {
    node->temporaries = TfLiteIntArrayCreate(2);
  } else {
    node->temporaries = TfLiteIntArrayCreate(1);
  }
  node->temporaries->data[0] = scratch_tensor_index;

  TfLiteIntArray* scratch_size_array = TfLiteIntArrayCreate(2);
  scratch_size_array->data[0] = batch_size;
  scratch_size_array->data[1] = num_filters;

  TfLiteTensor* scratch_tensor;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &scratch_tensor));

  // The scratch buffer is of type int32 for full integer svdf and it's of type
  // float32 for hybrid and float case.
  if (is_full_integer) {
    scratch_tensor->type = kTfLiteInt32;
  } else {
    scratch_tensor->type = kTfLiteFloat32;
  }
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_tensor,
                                                   scratch_size_array));

  if (is_hybrid_op) {
    op_data->compute_row_sums = true;
    // Tell interpreter to allocate temporary tensors to store quantized values
    // of input tensors.
    node->temporaries->data[1] = scratch_tensor_index + 1;
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1,
                                                &input_quantized));
    input_quantized->type = weights_feature->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }

    // Tell interpreter to allocate temporary tensors to store scaling factors.
    node->temporaries->data[2] = scratch_tensor_index + 2;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/2,
                                                &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    // Used to store dequantized weights_time matrix for hybrid computation of
    // matmul(state, weights_time), which occurs in floating point.
    node->temporaries->data[3] = scratch_tensor_index + 3;
    TfLiteTensor* float_weights_time;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/3,
                                                &float_weights_time));
    float_weights_time->type = kTfLiteFloat32;
    // Persistent so that we can compute the dequantized weights only once.
    float_weights_time->allocation_type = kTfLiteArenaRwPersistent;
    if (!TfLiteIntArrayEqual(float_weights_time->dims, weights_time->dims)) {
      TfLiteIntArray* float_weights_time_size =
          TfLiteIntArrayCopy(weights_time->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, float_weights_time,
                                              float_weights_time_size));
    }

    node->temporaries->data[4] = scratch_tensor_index + 4;
    TfLiteTensor* zero_points;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/4, &zero_points));
    zero_points->type = kTfLiteFloat32;
    zero_points->allocation_type = kTfLiteArenaRw;
    int zero_points_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(zero_points->dims, 1, zero_points_dims)) {
      TfLiteIntArray* zero_points_size = TfLiteIntArrayCreate(1);
      zero_points_size->data[0] = zero_points_dims[0];
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, zero_points,
                                                       zero_points_size));
    }

    node->temporaries->data[5] = scratch_tensor_index + 5;
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, /*index=*/5, &row_sums));
    row_sums->type = kTfLiteFloat32;
    row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int row_sums_dims[1] = {num_filters};
    if (!TfLiteIntArrayEqualsArray(row_sums->dims, 1, row_sums_dims)) {
      TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(1);
      row_sums_size->data[0] = row_sums_dims[0];
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, row_sums, row_sums_size));
    }
  }
  if (is_full_integer) {
    // Allocated one extra tensor.
    TfLiteIntArray* output_temp_size_array = TfLiteIntArrayCreate(2);
    output_temp_size_array->data[0] = num_units;
    output_temp_size_array->data[1] = batch_size;
    node->temporaries->data[1] = scratch_tensor_index + 1;
    TfLiteTensor* output_temp;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/1, &output_temp));
    output_temp->type = kTfLiteInt32;
    output_temp->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_temp,
                                                     output_temp_size_array));

    // Calculate effective scales.
    TF_LITE_ENSURE(context, input->quantization.type != kTfLiteNoQuantization);
    auto* input_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    TF_LITE_ENSURE(context,
                   weights_feature->quantization.type != kTfLiteNoQuantization);
    auto* weights_feature_params = reinterpret_cast<TfLiteAffineQuantization*>(
        weights_feature->quantization.params);
    TF_LITE_ENSURE(context, state->quantization.type != kTfLiteNoQuantization);
    auto* state_params =
        reinterpret_cast<TfLiteAffineQuantization*>(state->quantization.params);
    TF_LITE_ENSURE(context,
                   weights_time->quantization.type != kTfLiteNoQuantization);
    auto* weight_time_params = reinterpret_cast<TfLiteAffineQuantization*>(
        weights_time->quantization.params);
    TF_LITE_ENSURE(context, output->quantization.type != kTfLiteNoQuantization);
    auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);
    const double effective_scale_1 = input_params->scale->data[0] *
                                     weights_feature_params->scale->data[0] /
                                     state_params->scale->data[0];
    const double effective_scale_2 = state_params->scale->data[0] *
                                     weight_time_params->scale->data[0] /
                                     output_params->scale->data[0];
    QuantizeMultiplier(effective_scale_1, &op_data->effective_scale_1_a,
                       &op_data->effective_scale_1_b);
    QuantizeMultiplier(effective_scale_2, &op_data->effective_scale_2_a,
                       &op_data->effective_scale_2_b);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsvdfDTcc mht_3(mht_3_v, 469, "", "./tensorflow/lite/kernels/svdf.cc", "Eval");

  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* weights_feature;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kWeightsFeatureTensor,
                                          &weights_feature));
  const TfLiteTensor* weights_time;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kWeightsTimeTensor, &weights_time));
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);

  TfLiteTensor* scratch;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/0, &scratch));

  TfLiteTensor* state = GetVariableInput(context, node, kStateTensor);
  TF_LITE_ENSURE(context, state != nullptr);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (weights_feature->type) {
    case kTfLiteFloat32: {
      reference_ops::EvalFloatSVDF(
          params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(weights_feature),
          GetTensorData<float>(weights_feature), GetTensorShape(weights_time),
          GetTensorData<float>(weights_time), GetTensorShape(bias),
          GetTensorData<float>(bias), GetTensorData<float>(scratch),
          GetTensorData<float>(state), GetTensorShape(output),
          GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      if (input->type == kTfLiteFloat32) {
        TfLiteTensor* input_quantized;
        TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1,
                                                    &input_quantized));
        TfLiteTensor* scaling_factors;
        TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/2,
                                                    &scaling_factors));
        TfLiteTensor* float_weights_time;
        TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/3,
                                                    &float_weights_time));
        TfLiteTensor* zero_points;
        TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/4,
                                                    &zero_points));
        TfLiteTensor* row_sums;
        TF_LITE_ENSURE_OK(
            context, GetTemporarySafe(context, node, /*index=*/5, &row_sums));
        // Dequantize weights time.
        // TODO(alanchiao): this dequantization initialization only needs to
        // happen once per model and should theoretically be placed in either
        // Init or Prepare. However, TFLite doesn't allocate float_weights_time
        // until the Eval function.
        // TODO(alanchiao): refactor logic out into dequantize function.
        if (!op_data->float_weights_time_initialized) {
          const float dequantization_scale = weights_time->params.scale;
          const int8_t* weights_time_ptr = GetTensorData<int8_t>(weights_time);
          float* float_weights_time_ptr =
              GetTensorData<float>(float_weights_time);
          for (int i = 0; i < NumElements(float_weights_time); ++i) {
            float_weights_time_ptr[i] =
                weights_time_ptr[i] * dequantization_scale;
          }
          op_data->float_weights_time_initialized = true;
        }

        int32_t* zero_points_ptr = nullptr;
        int32_t* row_sums_ptr = nullptr;
        if (params->asymmetric_quantize_inputs && row_sums != nullptr) {
          zero_points_ptr = GetTensorData<int32_t>(zero_points);
          row_sums_ptr = GetTensorData<int32_t>(row_sums);
        }

        reference_ops::EvalHybridSVDF(
            params, GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(weights_feature),
            GetTensorData<int8_t>(weights_feature),
            weights_feature->params.scale, GetTensorShape(float_weights_time),
            GetTensorData<float>(float_weights_time), GetTensorShape(bias),
            GetTensorData<float>(bias), GetTensorData<float>(scratch),
            GetTensorData<float>(scaling_factors),
            GetTensorData<int8_t>(input_quantized), GetTensorData<float>(state),
            GetTensorShape(output), GetTensorData<float>(output),
            zero_points_ptr, row_sums_ptr, &op_data->compute_row_sums);
        return kTfLiteOk;
      }
      auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
          input->quantization.params);
      auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
          output->quantization.params);
      TfLiteTensor* output_temp;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, /*index=*/1, &output_temp));

      // Currently supports only ReLU.
      // TODO(jianlijianli): support other activations.
      TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActRelu);

      reference_ops::EvalIntegerSVDF(
          params, GetTensorShape(input), GetTensorData<int8_t>(input),
          GetTensorShape(weights_feature),
          GetTensorData<int8_t>(weights_feature), GetTensorShape(weights_time),
          GetTensorData<int16_t>(weights_time), GetTensorShape(bias),
          GetTensorData<int32_t>(bias), GetTensorData<int16_t>(state),
          GetTensorShape(output), GetTensorData<int8_t>(output),
          GetTensorData<int32_t>(scratch), GetTensorData<int32_t>(output_temp),
          op_data->effective_scale_1_a, op_data->effective_scale_1_b,
          op_data->effective_scale_2_a, op_data->effective_scale_2_b,
          input_params->zero_point->data[0],
          output_params->zero_point->data[0]);
      return kTfLiteOk;
    }
    default:
      context->ReportError(context, "Type %s not currently supported.",
                           TfLiteTypeGetName(weights_feature->type));
      return kTfLiteError;
  }
}

}  // namespace svdf

TfLiteRegistration* Register_SVDF() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsvdfDTcc mht_4(mht_4_v, 599, "", "./tensorflow/lite/kernels/svdf.cc", "Register_SVDF");

  static TfLiteRegistration r = {svdf::Init, svdf::Free, svdf::Prepare,
                                 svdf::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
