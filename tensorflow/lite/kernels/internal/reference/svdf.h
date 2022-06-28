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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsvdfDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsvdfDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsvdfDTh() {
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


#include <stdint.h>

#include <algorithm>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

// SVDF op that compresses a fully connected op via low-rank matrix
// factorization. See https://research.google.com/pubs/archive/43813.pdf for
// details.

namespace tflite {
namespace reference_ops {

static inline void ApplyTimeWeightsBiasAndActivation(
    int batch_size, int memory_size, int num_filters, int num_units, int rank,
    const float* const __restrict__ weights_time_data,
    const float* const __restrict__ bias_ptr, TfLiteFusedActivation activation,
    float* const __restrict__ state_ptr, float* const __restrict__ scratch_ptr,
    float* const __restrict__ output_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsvdfDTh mht_0(mht_0_v, 211, "", "./tensorflow/lite/kernels/internal/reference/svdf.h", "ApplyTimeWeightsBiasAndActivation");

  // Compute matmul(state, weights_time).
  for (int b = 0; b < batch_size; ++b) {
    float* state_ptr_batch = state_ptr + b * memory_size * num_filters;
    float* scratch_ptr_batch = scratch_ptr + b * num_filters;
    tensor_utils::BatchVectorBatchVectorDotProduct(
        weights_time_data, state_ptr_batch, memory_size, num_filters,
        scratch_ptr_batch);
  }

  // Reduction sum.
  tensor_utils::ReductionSumVector(scratch_ptr, output_ptr,
                                   batch_size * num_units, rank);
  // Add bias if provided.
  if (bias_ptr) {
    tensor_utils::VectorBatchVectorAdd(bias_ptr, num_units, batch_size,
                                       output_ptr);
  }

  // Apply activation.
  tensor_utils::ApplyActivationToVector(output_ptr, batch_size * num_units,
                                        activation, output_ptr);
}

inline void EvalIntegerSVDF(
    const TfLiteSVDFParams* params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& weights_feature_shape,
    const int8_t* weights_feature_data, const RuntimeShape& weights_time_shape,
    const int16_t* weights_time_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, int16_t* state_data,
    const RuntimeShape& output_shape, int8_t* output_data,
    int32_t* scratch_data, int32_t* output_temp_data, int32_t scale_1_a,
    int scale_1_b, int32_t scale_2_a, int scale_2_b, int32_t input_zp,
    int32_t output_zp) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsvdfDTh mht_1(mht_1_v, 247, "", "./tensorflow/lite/kernels/internal/reference/svdf.h", "EvalIntegerSVDF");

  const int n_rank = params->rank;
  const int n_batch = input_shape.Dims(0);
  const int n_input = input_shape.Dims(1);
  const int n_filter = weights_feature_shape.Dims(0);
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_shape.Dims(1);

  // Left shift the activation_state.
  // std::copy is fine for overlapping ranges if the output is outside of the
  // input range. (This is not true for copy_n.)
  std::copy(state_data + 1, state_data + n_batch * n_memory * n_filter,
            state_data);

  // Feature matmul.
  // Note: no need to clear the latest activation, matmul is not accumulative.
  {
    const int32_t output_max = std::numeric_limits<int16_t>::max();
    const int32_t output_min = std::numeric_limits<int16_t>::min();
    int16_t* result_in_batch = state_data + (n_memory - 1);
    for (int b = 0; b < n_batch; b++) {
      const int8_t* matrix_data = weights_feature_data;
      for (int r = 0; r < n_filter; r++) {
        int32_t dot_prod = 0;
        const int8_t* vector_in_batch = input_data + b * n_input;
        for (int c = 0; c < n_input; c++) {
          dot_prod += *matrix_data++ * (*vector_in_batch++ - input_zp);
        }
        dot_prod =
            MultiplyByQuantizedMultiplier(dot_prod, scale_1_a, scale_1_b);
        dot_prod = std::min(std::max(output_min, dot_prod), output_max);
        // This assumes state is symmetrically quantized. Otherwise last bit of
        // state should be initialized to its zero point and accumulate the
        // dot_prod.
        // Equivalent as the following:
        //     result_in_batch = zero point, which happens to be zero.
        //     result_in_batch += dot_prod.
        *result_in_batch = dot_prod;
        result_in_batch += n_memory;
      }
    }
  }

  // Time.
  {
    for (int b = 0; b < n_batch; ++b) {
      const int16_t* state_data_batch = state_data + b * n_memory * n_filter;
      int32_t* scratch_data_batch = scratch_data + b * n_filter;
      tensor_utils::BatchVectorBatchVectorDotProduct(
          weights_time_data, state_data_batch, n_memory, n_filter,
          scratch_data_batch);
    }
  }

  // Reduce, add bias, rescale, activation.
  {
    // Reduce.
    tensor_utils::ReductionSumVector(scratch_data, output_temp_data,
                                     n_batch * n_unit, n_rank);
    // Add bias.
    if (bias_data) {
      tensor_utils::VectorBatchVectorAdd(bias_data, n_unit, n_batch,
                                         output_temp_data);
    }
    // Rescale.
    const int32_t output_max = std::numeric_limits<int8_t>::max();
    const int32_t output_min = std::numeric_limits<int8_t>::min();
    for (int i = 0; i < n_batch * n_unit; ++i) {
      int32_t x1 = output_temp_data[i];
      int32_t x2 = MultiplyByQuantizedMultiplier(x1, scale_2_a, scale_2_b);
      int32_t x3 = x2 + output_zp;
      int32_t x4 = std::min(std::max(output_min, x3), output_max);
      output_data[i] = static_cast<int8_t>(x4);
    }
  }
}

inline void EvalFloatSVDF(
    const TfLiteSVDFParams* params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_feature_shape,
    const float* weights_feature_data, const RuntimeShape& weights_time_shape,
    const float* weights_time_data, const RuntimeShape& bias_shape,
    const float* bias_data, float* scratch_data, float* state_data,
    const RuntimeShape& output_shape, float* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsvdfDTh mht_2(mht_2_v, 333, "", "./tensorflow/lite/kernels/internal/reference/svdf.h", "EvalFloatSVDF");

  const int rank = params->rank;
  const int batch_size = input_shape.Dims(0);
  const int input_size = input_shape.Dims(1);
  const int num_filters = weights_feature_shape.Dims(0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time_shape.Dims(1);

  // Left shift the activation_state.
  // std::copy is fine for overlapping ranges if the output is outside of the
  // input range. (This is not true for copy_n.)
  std::copy(state_data + 1, state_data + batch_size * memory_size * num_filters,
            state_data);

  // Clear scratch (the matmul is accumulative).
  std::fill_n(scratch_data, batch_size * num_filters, 0.0f);

  // Compute conv1d(inputs, weights_feature).
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      weights_feature_data, num_filters, input_size, input_data, batch_size,
      scratch_data);

  // Copy the latest activation from scratch into activation_state:
  // The last, i.e. (memory_size-1)th entry for each batch, and filter.
  for (int i = 0; i < batch_size * num_filters; ++i) {
    state_data[i * memory_size + memory_size - 1] = scratch_data[i];
  }

  ApplyTimeWeightsBiasAndActivation(
      batch_size, memory_size, num_filters, num_units, rank, weights_time_data,
      bias_data, params->activation, state_data, scratch_data, output_data);
}

inline void EvalHybridSVDF(
    const TfLiteSVDFParams* params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_feature_shape,
    const int8_t* weights_feature_data, const float weights_feature_scale,
    const RuntimeShape& weights_time_shape, const float* weights_time_data,
    const RuntimeShape& bias_shape, const float* bias_data, float* scratch,
    float* scaling_factors, int8_t* quantized_input, float* state,
    const RuntimeShape& output_shape, float* output_data, int32_t* zero_points,
    int32_t* row_sums, bool* compute_row_sums) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSsvdfDTh mht_3(mht_3_v, 377, "", "./tensorflow/lite/kernels/internal/reference/svdf.h", "EvalHybridSVDF");

  const int rank = params->rank;
  const int batch_size = input_shape.Dims(0);
  const int input_size = input_shape.Dims(1);
  const int num_filters = weights_feature_shape.Dims(0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time_shape.Dims(1);

  // Left shift the activation_state.
  // std::copy is fine for overlapping ranges if the output is outside of the
  // input range. (This is not true for copy_n.)
  std::copy(state + 1, state + batch_size * memory_size * num_filters, state);

  // Clear scratch (the matmul is accumulative).
  std::fill_n(scratch, batch_size * num_filters, 0.0f);

  if (!tensor_utils::IsZeroVector(input_data, batch_size * input_size)) {
    // Quantize input from float to int8_t.
    tensor_utils::BatchQuantizeFloats(
        input_data, batch_size, input_size, quantized_input, scaling_factors,
        zero_points, params->asymmetric_quantize_inputs);
    for (int b = 0; b < batch_size; ++b) {
      scaling_factors[b] *= weights_feature_scale;
    }

    // Compute conv1d(inputs, weights_feature).
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        weights_feature_data, num_filters, input_size, quantized_input,
        scaling_factors, batch_size, scratch,
        /*per_channel_scale=*/nullptr, zero_points,
        reinterpret_cast<int32_t*>(scratch), row_sums, compute_row_sums,
        /*context=*/nullptr);
  }
  // Copy the latest activation from scratch into activation_state:
  // The last, i.e. (memory_size-1)th entry for each batch, and filter.
  for (int i = 0; i < batch_size * num_filters; ++i) {
    state[i * memory_size + memory_size - 1] = scratch[i];
  }

  // TODO(b/174275776): can optimize hybrid case ~5% by unrolling loop in
  // applying time weights so that the inner loop multiplies eight elements at
  // a time.
  ApplyTimeWeightsBiasAndActivation(
      batch_size, memory_size, num_filters, num_units, rank, weights_time_data,
      bias_data, params->activation, state, scratch, output_data);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SVDF_H_
