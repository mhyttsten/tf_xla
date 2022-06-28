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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSkernel_utilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSkernel_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSkernel_utilsDTcc() {
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
#include "tensorflow/lite/kernels/internal/kernel_utils.h"

#include <algorithm>

#include "tensorflow/lite/kernels/internal/tensor_utils.h"

namespace tflite {
namespace kernel_utils {

void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int num_units, int batch_size,
                  int output_batch_leading_dim,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSkernel_utilsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/kernels/internal/kernel_utils.cc", "RnnBatchStep");

  RnnBatchStep(input_ptr_batch, input_weights_ptr,
               /*aux_input_ptr_batch=*/nullptr,
               /*aux_input_weights_ptr=*/nullptr, recurrent_weights_ptr,
               bias_ptr, input_size, /*aux_input_size=*/0, num_units,
               batch_size, output_batch_leading_dim, activation,
               hidden_state_ptr_batch, output_ptr_batch);
}

void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* aux_input_ptr_batch,
                  const float* aux_input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int aux_input_size, int num_units,
                  int batch_size, int output_batch_leading_dim,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSkernel_utilsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/kernels/internal/kernel_utils.cc", "RnnBatchStep");

  // Since the output batch rows may not be contiguous (output_batch_leading_dim
  // != n_output), we unroll the batched operations where this is the case.
  if (output_batch_leading_dim == num_units) {
    // Output = bias
    tensor_utils::VectorBatchVectorAssign(bias_ptr, num_units, batch_size,
                                          output_ptr_batch);

    // Output += input * input_weights
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_weights_ptr, num_units, input_size, input_ptr_batch, batch_size,
        output_ptr_batch);

    // Output += aux_input * aux_input_weights (if they are not empty).
    if (aux_input_size > 0) {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          aux_input_weights_ptr, num_units, aux_input_size, aux_input_ptr_batch,
          batch_size, output_ptr_batch);
    }

    // Output += recurrent_weights * hidden_state
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_weights_ptr, num_units, num_units, hidden_state_ptr_batch,
        batch_size, output_ptr_batch);

    // Output = activation(Output) and update hidden_state
    tensor_utils::ApplyActivationToVector(
        output_ptr_batch, num_units * batch_size, activation, output_ptr_batch);
    std::copy_n(output_ptr_batch, num_units * batch_size,
                hidden_state_ptr_batch);
  } else {
    // Output = bias
    for (int k = 0; k < batch_size; k++) {
      std::copy_n(bias_ptr, num_units,
                  output_ptr_batch + k * output_batch_leading_dim);
    }

    // Output += input * input_weights
    for (int k = 0; k < batch_size; k++) {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          input_weights_ptr, num_units, input_size,
          input_ptr_batch + k * input_size, /*n_batch=*/1,
          output_ptr_batch + k * output_batch_leading_dim);
    }

    // Output += aux_input * aux_input_weights (if they are not empty).
    if (aux_input_size > 0) {
      for (int k = 0; k < batch_size; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            aux_input_weights_ptr, num_units, aux_input_size,
            aux_input_ptr_batch + k * aux_input_size,
            /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim);
      }
    }

    // Output += recurrent_weights * hidden_state
    for (int k = 0; k < batch_size; k++) {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          recurrent_weights_ptr, num_units, num_units,
          hidden_state_ptr_batch + k * num_units,
          /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim);
    }

    // Output = activation(Output) and update hidden_state
    for (int k = 0; k < batch_size; k++) {
      tensor_utils::ApplyActivationToVector(
          output_ptr_batch + k * output_batch_leading_dim, num_units,
          activation, output_ptr_batch + k * output_batch_leading_dim);
      std::copy_n(output_ptr_batch + k * output_batch_leading_dim, num_units,
                  hidden_state_ptr_batch + k * num_units);
    }
  }
}

void RnnBatchStep(
    const float* input_ptr_batch, const int8_t* input_weights_ptr,
    float input_weights_scale, const int8_t* recurrent_weights_ptr,
    float recurrent_weights_scale, const float* bias_ptr, int input_size,
    int num_units, int batch_size, int output_batch_leading_dim,
    TfLiteFusedActivation activation, int8_t* quantized_input_ptr_batch,
    int8_t* quantized_hidden_state_ptr_batch, float* scaling_factors,
    float* hidden_state_ptr_batch, float* output_ptr_batch,
    bool asymmetric_quantize_inputs, int32_t* zero_points,
    int32_t* accum_scratch, int32_t* row_sums, bool* compute_row_sums) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSkernel_utilsDTcc mht_2(mht_2_v, 303, "", "./tensorflow/lite/kernels/internal/kernel_utils.cc", "RnnBatchStep");

  RnnBatchStep(input_ptr_batch, input_weights_ptr, input_weights_scale,
               /*aux_input_ptr_batch=*/nullptr,
               /*aux_input_weights_ptr=*/nullptr,
               /*aux_input_weights_scale=*/0.0f, recurrent_weights_ptr,
               recurrent_weights_scale, bias_ptr, input_size,
               /*aux_input_size=*/0, num_units, batch_size,
               output_batch_leading_dim, activation, quantized_input_ptr_batch,
               /*aux_quantized_input_ptr_batch=*/nullptr,
               quantized_hidden_state_ptr_batch, scaling_factors,
               hidden_state_ptr_batch, output_ptr_batch,
               asymmetric_quantize_inputs, zero_points, accum_scratch, row_sums,
               compute_row_sums);
}

void RnnBatchStep(
    const float* input_ptr_batch, const int8_t* input_weights_ptr,
    float input_weights_scale, const float* aux_input_ptr_batch,
    const int8_t* aux_input_weights_ptr, float aux_input_weights_scale,
    const int8_t* recurrent_weights_ptr, float recurrent_weights_scale,
    const float* bias_ptr, int input_size, int aux_input_size, int num_units,
    int batch_size, int output_batch_leading_dim,
    TfLiteFusedActivation activation, int8_t* quantized_input_ptr_batch,
    int8_t* aux_quantized_input_ptr_batch,
    int8_t* quantized_hidden_state_ptr_batch, float* scaling_factors,
    float* hidden_state_ptr_batch, float* output_ptr_batch,
    bool asymmetric_quantize_inputs, int32_t* zero_points,
    int32_t* accum_scratch, int32_t* row_sums, bool* compute_row_sums) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSkernel_utilsDTcc mht_3(mht_3_v, 333, "", "./tensorflow/lite/kernels/internal/kernel_utils.cc", "RnnBatchStep");

  // Since the output batch rows may not be contiguous (output_batch_leading_dim
  // != n_output), we unroll the batched operations where this is the case.

  int32_t* input_row_sums = nullptr;
  int32_t* aux_input_row_sums = nullptr;
  int32_t* recurrent_row_sums = nullptr;
  if (asymmetric_quantize_inputs) {
    input_row_sums = row_sums;
    aux_input_row_sums = row_sums;
    if (aux_input_ptr_batch) {
      aux_input_row_sums += num_units;
    }
    recurrent_row_sums = aux_input_row_sums + num_units;
    if (*compute_row_sums) {
      tensor_utils::ReductionSumVector(input_weights_ptr, input_row_sums,
                                       num_units, input_size);
      if (aux_input_ptr_batch) {
        tensor_utils::ReductionSumVector(aux_input_weights_ptr,
                                         aux_input_row_sums, num_units,
                                         aux_input_size);
      }
      tensor_utils::ReductionSumVector(
          recurrent_weights_ptr, recurrent_row_sums, num_units, num_units);
      *compute_row_sums = false;
    }
  }

  if (output_batch_leading_dim == num_units) {
    // Output = bias
    tensor_utils::VectorBatchVectorAssign(bias_ptr, num_units, batch_size,
                                          output_ptr_batch);

    // Save quantization and matmul computation for all zero input.
    if (!tensor_utils::IsZeroVector(input_ptr_batch, batch_size * input_size)) {
      // Quantize input from float to uint8 + quantization params (scaling
      // factor).
      tensor_utils::BatchQuantizeFloats(
          input_ptr_batch, batch_size, input_size, quantized_input_ptr_batch,
          scaling_factors, zero_points, asymmetric_quantize_inputs);
      for (int b = 0; b < batch_size; ++b) {
        scaling_factors[b] *= input_weights_scale;
      }
      // Output += input * input_weights
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          input_weights_ptr, num_units, input_size, quantized_input_ptr_batch,
          scaling_factors, batch_size, output_ptr_batch,
          /*per_channel_scale=*/nullptr, zero_points, accum_scratch,
          input_row_sums, compute_row_sums, /*context=*/nullptr);
    }

    if (aux_input_ptr_batch &&
        !tensor_utils::IsZeroVector(aux_input_ptr_batch,
                                    batch_size * aux_input_size)) {
      tensor_utils::BatchQuantizeFloats(
          aux_input_ptr_batch, batch_size, aux_input_size,
          aux_quantized_input_ptr_batch, scaling_factors, zero_points,
          asymmetric_quantize_inputs);
      for (int b = 0; b < batch_size; ++b) {
        scaling_factors[b] *= aux_input_weights_scale;
      }

      // Output += aux_input * aux_input_weights
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          aux_input_weights_ptr, num_units, aux_input_size,
          aux_quantized_input_ptr_batch, scaling_factors, batch_size,
          output_ptr_batch, /*per_channel_scale=*/nullptr, zero_points,
          accum_scratch, aux_input_row_sums, compute_row_sums,
          /*context=*/nullptr);
    }

    // Save quantization and matmul computation for all zero input.
    if (!tensor_utils::IsZeroVector(hidden_state_ptr_batch,
                                    batch_size * num_units)) {
      // Quantize hidden_state
      tensor_utils::BatchQuantizeFloats(
          hidden_state_ptr_batch, batch_size, num_units,
          quantized_hidden_state_ptr_batch, scaling_factors, zero_points,
          asymmetric_quantize_inputs);
      for (int b = 0; b < batch_size; ++b) {
        scaling_factors[b] *= recurrent_weights_scale;
      }

      // Output += recurrent_weights * hidden_state
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          recurrent_weights_ptr, num_units, num_units,
          quantized_hidden_state_ptr_batch, scaling_factors, batch_size,
          output_ptr_batch, /*per_channel_scale=*/nullptr, zero_points,
          accum_scratch, recurrent_row_sums, compute_row_sums,
          /*context=*/nullptr);
    }

    // Output = activation(Output) and update hidden_state
    tensor_utils::ApplyActivationToVector(
        output_ptr_batch, num_units * batch_size, activation, output_ptr_batch);
    std::copy_n(output_ptr_batch, num_units * batch_size,
                hidden_state_ptr_batch);
  } else {
    // Output = bias
    for (int k = 0; k < batch_size; k++) {
      std::copy_n(bias_ptr, num_units,
                  output_ptr_batch + k * output_batch_leading_dim);
    }

    // Save quantization and matmul computation for all zero input.
    if (!tensor_utils::IsZeroVector(input_ptr_batch, batch_size * input_size)) {
      // Quantize input from float to uint8 + quantization params (scaling
      // factor).
      tensor_utils::BatchQuantizeFloats(
          input_ptr_batch, batch_size, input_size, quantized_input_ptr_batch,
          scaling_factors, zero_points, asymmetric_quantize_inputs);
      for (int b = 0; b < batch_size; ++b) {
        scaling_factors[b] *= input_weights_scale;
      }

      // Output += input * input_weights
      for (int k = 0; k < batch_size; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            input_weights_ptr, num_units, input_size,
            quantized_input_ptr_batch + k * input_size, &scaling_factors[k],
            /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim,
            /*per_channel_scale=*/nullptr, zero_points + k, accum_scratch,
            input_row_sums, compute_row_sums, /*context=*/nullptr);
      }
    }

    if (aux_input_ptr_batch &&
        !tensor_utils::IsZeroVector(aux_input_ptr_batch,
                                    batch_size * aux_input_size)) {
      tensor_utils::BatchQuantizeFloats(
          aux_input_ptr_batch, batch_size, aux_input_size,
          aux_quantized_input_ptr_batch, scaling_factors, zero_points,
          asymmetric_quantize_inputs);
      for (int b = 0; b < batch_size; ++b) {
        scaling_factors[b] *= aux_input_weights_scale;
      }

      // Output += aux_input * aux_input_weights
      for (int k = 0; k < batch_size; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            aux_input_weights_ptr, num_units, aux_input_size,
            aux_quantized_input_ptr_batch + k * aux_input_size,
            &scaling_factors[k],
            /*n_batch=*/1, output_ptr_batch + k * output_batch_leading_dim,
            /*per_channel_scale=*/nullptr, zero_points + k, accum_scratch,
            aux_input_row_sums, compute_row_sums, /*context=*/nullptr);
      }
    }

    // Save quantization and matmul computation for all zero input.
    if (!tensor_utils::IsZeroVector(hidden_state_ptr_batch,
                                    batch_size * num_units)) {
      // Quantize hidden_state
      tensor_utils::BatchQuantizeFloats(
          hidden_state_ptr_batch, batch_size, num_units,
          quantized_hidden_state_ptr_batch, scaling_factors, zero_points,
          asymmetric_quantize_inputs);
      for (int b = 0; b < batch_size; ++b) {
        scaling_factors[b] *= recurrent_weights_scale;
      }

      // Output += recurrent_weights * hidden_state
      for (int k = 0; k < batch_size; k++) {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            recurrent_weights_ptr, num_units, num_units,
            quantized_hidden_state_ptr_batch + k * num_units,
            &scaling_factors[k], /*n_batch=*/1,
            output_ptr_batch + k * output_batch_leading_dim,
            /*per_channel_scale=*/nullptr, zero_points + k, accum_scratch,
            recurrent_row_sums, compute_row_sums, /*context=*/nullptr);
      }
    }

    // Output = activation(Output) and update hidden_state
    for (int k = 0; k < batch_size; k++) {
      tensor_utils::ApplyActivationToVector(
          output_ptr_batch + k * output_batch_leading_dim, num_units,
          activation, output_ptr_batch + k * output_batch_leading_dim);
      std::copy_n(output_ptr_batch + k * output_batch_leading_dim, num_units,
                  hidden_state_ptr_batch + k * num_units);
    }
  }
}

}  // namespace kernel_utils
}  // namespace tflite
