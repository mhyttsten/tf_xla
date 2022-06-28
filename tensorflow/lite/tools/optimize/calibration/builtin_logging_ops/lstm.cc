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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc() {
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
#include "tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/lstm_shared.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_logger.h"

namespace tflite {
namespace optimize {
namespace calibration {
namespace builtin {

namespace {

inline void CalculateLstmGateFloat(
    const float* input, const float* input_to_gate_weights,
    const float* aux_input, const float* aux_input_to_gate_weights,
    const float* output_state, const float* recurrent_to_gate_weights,
    const float* cell_state, const float* cell_to_gate_weights,
    const float* layer_norm_coefficients, const float* gate_bias,
    const int n_batch, const int n_input, const int n_aux_input,
    const int n_output, const int n_cell,
    const TfLiteFusedActivation activation, float* gate,
    const bool is_input_all_zeros, const bool is_aux_input_all_zeros,
    Logger* logger, int intermediate_tensor_index, const int subgraph_index,
    ErrorReporter* error_reporter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc mht_0(mht_0_v, 220, "", "./tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc", "CalculateLstmGateFloat");

  const bool use_peephole = (cell_to_gate_weights != nullptr);
  const bool use_layer_norm = (layer_norm_coefficients != nullptr);

  // Initialize scratch buffers with bias for regular lstm or initialize with
  // zero for layer norm lstm.
  if (use_layer_norm) {
    std::fill_n(gate, n_cell * n_batch, 0.0f);
  } else {
    tensor_utils::VectorBatchVectorAssign(gate_bias, n_cell, n_batch, gate);
  }
  // For each batch and cell: compute input_weight * input.
  // Skip if input is all zeros.
  if (!is_input_all_zeros) {
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_gate_weights, n_cell, n_input, input, n_batch, gate);
  }
  // For each batch and cell: compute aux_input_weight * aux_input.
  // Skip if auxiliary input is not available or all zeros.
  if (!is_aux_input_all_zeros) {
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(aux_input_to_gate_weights,
                                                      n_cell, n_aux_input,
                                                      aux_input, n_batch, gate);
  }
  // For each batch and cell: compute recurrent_weight * output_state.
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      recurrent_to_gate_weights, n_cell, n_output, output_state, n_batch, gate);
  // For each batch and cell: compute cell_weight .* cell_state (peephole LSTM)
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_gate_weights, n_cell, cell_state, n_batch, gate);
  }
  // Do layer normalization (if layer norm LSTM)
  if (use_layer_norm) {
    logger->LogTensorValue(subgraph_index, intermediate_tensor_index, gate,
                           n_cell * n_batch, error_reporter);

    tensor_utils::MeanStddevNormalization(gate, gate, n_cell, n_batch);
    tensor_utils::VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell,
                                                gate, n_batch, gate);
    tensor_utils::VectorBatchVectorAdd(gate_bias, n_cell, n_batch, gate);
  }
  // Apply activation
  tensor_utils::ApplyActivationToVector(gate, n_batch * n_cell, activation,
                                        gate);
}

// TODO(b/159066113): This is the exact same function as UpdateLstmCellFloat in
// kernels/lstm_eval.cc, make that public and remove this.
void UpdateLstmCellFloat(int n_batch, int n_cell, float* cell_state,
                         const float* input_gate, float* forget_gate,
                         const float* cell_gate, bool use_cifg, float clip) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc mht_1(mht_1_v, 274, "", "./tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc", "UpdateLstmCellFloat");

  tensor_utils::VectorVectorCwiseProduct(forget_gate, cell_state,
                                         n_batch * n_cell, cell_state);

  if (use_cifg) {
    // With CIFG, input_gate = 1-forget_gate. Use the forget_gate array as
    // scratch, as input_gate array is not allocated in this case. (Be careful
    // not to write to the scratch before reading the forget gate data.)
    float* scratch = forget_gate;
    tensor_utils::Sub1Vector(forget_gate, n_batch * n_cell, scratch);
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_gate, scratch, n_batch * n_cell, cell_state);
  } else {
    tensor_utils::VectorVectorCwiseProductAccumulate(
        cell_gate, input_gate, n_batch * n_cell, cell_state);
  }
  if (clip > 0.0f) {
    tensor_utils::CwiseClipping(cell_state, n_batch * n_cell, clip);
  }
}

void CalculateLstmOutputCalibration(
    int n_batch, int n_cell, int n_output, const float* cell_state,
    const float* output_gate, TfLiteFusedActivation activation,
    const float* projection_weights, const float* projection_bias,
    const float proj_clip, float* output_state, float* scratch, Logger* logger,
    int intermediate_tensor_index, const int subgraph_index,
    ErrorReporter* error_reporter) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc mht_2(mht_2_v, 304, "", "./tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc", "CalculateLstmOutputCalibration");

  tensor_utils::ApplyActivationToVector(cell_state, n_batch * n_cell,
                                        activation, scratch);
  tensor_utils::VectorVectorCwiseProduct(output_gate, scratch, n_batch * n_cell,
                                         scratch);

  logger->LogTensorValue(subgraph_index, intermediate_tensor_index, scratch,
                         n_cell * n_batch, error_reporter);

  const bool use_projection = (projection_weights != nullptr);
  const bool use_projection_bias = (projection_bias != nullptr);

  if (use_projection) {
    if (use_projection_bias) {
      tensor_utils::VectorBatchVectorAssign(projection_bias, n_output, n_batch,
                                            output_state);
    } else {
      std::fill_n(output_state, n_batch * n_output, 0.0f);
    }
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        projection_weights, n_output, n_cell, scratch, n_batch, output_state);
    if (proj_clip > 0.0f) {
      tensor_utils::CwiseClipping(output_state, n_batch * n_output, proj_clip);
    }
  } else {
    std::copy_n(scratch, n_batch * n_output, output_state);
  }
}

inline void LstmStepCalibration(
    const float* input_ptr, const float* input_to_input_weights_ptr,
    const float* input_to_forget_weights_ptr,
    const float* input_to_cell_weights_ptr,
    const float* input_to_output_weights_ptr, const float* aux_input_ptr,
    const float* aux_input_to_input_weights_ptr,
    const float* aux_input_to_forget_weights_ptr,
    const float* aux_input_to_cell_weights_ptr,
    const float* aux_input_to_output_weights_ptr,
    const float* recurrent_to_input_weights_ptr,
    const float* recurrent_to_forget_weights_ptr,
    const float* recurrent_to_cell_weights_ptr,
    const float* recurrent_to_output_weights_ptr,
    const float* cell_to_input_weights_ptr,
    const float* cell_to_forget_weights_ptr,
    const float* cell_to_output_weights_ptr,
    const float* input_layer_norm_coefficients_ptr,
    const float* forget_layer_norm_coefficients_ptr,
    const float* cell_layer_norm_coefficients_ptr,
    const float* output_layer_norm_coefficients_ptr,
    const float* input_gate_bias_ptr, const float* forget_gate_bias_ptr,
    const float* cell_gate_bias_ptr, const float* output_gate_bias_ptr,
    const float* projection_weights_ptr, const float* projection_bias_ptr,
    const TfLiteLSTMParams* params, int n_batch, int n_cell, int n_input,
    int n_aux_input, int n_output, int output_batch_leading_dim,
    float* output_state_ptr, float* cell_state_ptr, float* scratch0,
    float* scratch1, float* scratch2, float* scratch3, float* output_ptr,
    Logger* logger, const std::vector<int>& intermediate_tensor_indexes,
    const int subgraph_index, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc mht_3(mht_3_v, 364, "", "./tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc", "LstmStepCalibration");

  ruy::profiler::ScopeLabel label("LstmStepCalibration");
  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);

  // Make named scratch buffers.
  float* input_gate_scratch = scratch0;
  float* forget_gate_scratch = scratch1;
  float* cell_gate_scratch = scratch2;
  float* output_gate_scratch = scratch3;

  // Check if inputs are all zeros so we can skip some computations.
  const bool is_input_all_zeros =
      tensor_utils::IsZeroVector(input_ptr, n_batch * n_input);
  const bool is_aux_input_all_zeros =
      (aux_input_ptr == nullptr ||
       tensor_utils::IsZeroVector(aux_input_ptr, n_batch * n_aux_input));
  if (!use_cifg) {
    // Calculate the input gate. (If not CIFG.)
    CalculateLstmGateFloat(
        input_ptr, input_to_input_weights_ptr, aux_input_ptr,
        aux_input_to_input_weights_ptr, output_state_ptr,
        recurrent_to_input_weights_ptr, cell_state_ptr,
        cell_to_input_weights_ptr, input_layer_norm_coefficients_ptr,
        input_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
        /*activation=*/kTfLiteActSigmoid, input_gate_scratch,
        is_input_all_zeros, is_aux_input_all_zeros, logger,
        intermediate_tensor_indexes[0], subgraph_index, error_reporter);
  }
  // Calculate the forget gate.
  CalculateLstmGateFloat(
      input_ptr, input_to_forget_weights_ptr, aux_input_ptr,
      aux_input_to_forget_weights_ptr, output_state_ptr,
      recurrent_to_forget_weights_ptr, cell_state_ptr,
      cell_to_forget_weights_ptr, forget_layer_norm_coefficients_ptr,
      forget_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
      /*activation=*/kTfLiteActSigmoid, forget_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros, logger, intermediate_tensor_indexes[1],
      subgraph_index, error_reporter);
  // Calculate the cell update gate.
  CalculateLstmGateFloat(
      input_ptr, input_to_cell_weights_ptr, aux_input_ptr,
      aux_input_to_cell_weights_ptr, output_state_ptr,
      recurrent_to_cell_weights_ptr, /*cell_state=*/nullptr,
      /*cell_to_gate_weights=*/nullptr, cell_layer_norm_coefficients_ptr,
      cell_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
      params->activation, cell_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros, logger, intermediate_tensor_indexes[2],
      subgraph_index, error_reporter);
  // Update the cell state.
  UpdateLstmCellFloat(n_batch, n_cell, cell_state_ptr, input_gate_scratch,
                      forget_gate_scratch, cell_gate_scratch, use_cifg,
                      params->cell_clip);
  // Calculate output gate.
  CalculateLstmGateFloat(
      input_ptr, input_to_output_weights_ptr, aux_input_ptr,
      aux_input_to_output_weights_ptr, output_state_ptr,
      recurrent_to_output_weights_ptr, cell_state_ptr,
      cell_to_output_weights_ptr, output_layer_norm_coefficients_ptr,
      output_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
      /*activation=*/kTfLiteActSigmoid, output_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros, logger, intermediate_tensor_indexes[3],
      subgraph_index, error_reporter);
  // Update the output state.
  CalculateLstmOutputCalibration(
      n_batch, n_cell, n_output, cell_state_ptr, output_gate_scratch,
      params->activation, projection_weights_ptr, projection_bias_ptr,
      params->proj_clip, output_state_ptr, scratch2, logger,
      intermediate_tensor_indexes[4], subgraph_index, error_reporter);
  // Copy output state to the output. Note that the output's rows may not be
  // contiguous (output_batch_leading_dim != n_output).
  for (int b = 0; b < n_batch; b++) {
    std::copy_n(output_state_ptr + b * n_output, n_output,
                output_ptr + b * output_batch_leading_dim);
  }
}

TfLiteStatus EvalCalibration(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* cell_to_input_weights,
    const TfLiteTensor* cell_to_forget_weights,
    const TfLiteTensor* cell_to_output_weights,
    const TfLiteTensor* input_layer_norm_coefficients,
    const TfLiteTensor* forget_layer_norm_coefficients,
    const TfLiteTensor* cell_layer_norm_coefficients,
    const TfLiteTensor* output_layer_norm_coefficients,
    const TfLiteTensor* aux_input,
    const TfLiteTensor* aux_input_to_input_weights,
    const TfLiteTensor* aux_input_to_forget_weights,
    const TfLiteTensor* aux_input_to_cell_weights,
    const TfLiteTensor* aux_input_to_output_weights,
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    int output_offset, TfLiteTensor* scratch_buffer, TfLiteTensor* output_state,
    TfLiteTensor* cell_state, TfLiteTensor* output, Logger* logger,
    const std::vector<int>& intermediate_tensor_indexes,
    const int subgraph_index, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc mht_4(mht_4_v, 473, "", "./tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc", "EvalCalibration");

  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  int max_time, n_batch;
  if (input->dims->size == 3) {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  } else {
    max_time = 1;
    n_batch = input->dims->data[0];
  }
  const int n_input = input->dims->data[input->dims->size - 1];
  const int aux_input_size =
      (aux_input) ? aux_input->dims->data[aux_input->dims->size - 1] : 0;

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);

  // Index the scratch buffers pointers to the global scratch buffer.
  float* scratch_buffer_ptr = GetTensorData<float>(scratch_buffer);
  float* input_gate_scratch = nullptr;
  float* cell_gate_scratch = nullptr;
  float* forget_gate_scratch = nullptr;
  float* output_gate_scratch = nullptr;
  if (use_cifg) {
    cell_gate_scratch = scratch_buffer_ptr;
    forget_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
  } else {
    input_gate_scratch = scratch_buffer_ptr;
    cell_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 3 * n_cell * n_batch;
  }

  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  if (time_major) {
    // Loop through the sequence.
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      // If this is the forward_sequence, step forward, otherwise step
      // backwards.
      const int t_rel = forward_sequence ? t : max_time - t - 1;
      const float* input_ptr = GetTensorData<float>(input) + t_rel * input_step;
      const float* aux_input_ptr = nullptr;
      if (aux_input) {
        aux_input_ptr = GetTensorData<float>(aux_input) + t_rel * input_step;
      }
      float* output_ptr_time =
          GetTensorData<float>(output) + t_rel * output_step + output_offset;

      LstmStepCalibration(
          input_ptr, GetTensorData<float>(input_to_input_weights),
          GetTensorData<float>(input_to_forget_weights),
          GetTensorData<float>(input_to_cell_weights),
          GetTensorData<float>(input_to_output_weights), aux_input_ptr,
          GetTensorData<float>(aux_input_to_input_weights),
          GetTensorData<float>(aux_input_to_forget_weights),
          GetTensorData<float>(aux_input_to_cell_weights),
          GetTensorData<float>(aux_input_to_output_weights),
          GetTensorData<float>(recurrent_to_input_weights),
          GetTensorData<float>(recurrent_to_forget_weights),
          GetTensorData<float>(recurrent_to_cell_weights),
          GetTensorData<float>(recurrent_to_output_weights),
          GetTensorData<float>(cell_to_input_weights),
          GetTensorData<float>(cell_to_forget_weights),
          GetTensorData<float>(cell_to_output_weights),
          GetTensorData<float>(input_layer_norm_coefficients),
          GetTensorData<float>(forget_layer_norm_coefficients),
          GetTensorData<float>(cell_layer_norm_coefficients),
          GetTensorData<float>(output_layer_norm_coefficients),
          GetTensorData<float>(input_gate_bias),
          GetTensorData<float>(forget_gate_bias),
          GetTensorData<float>(cell_gate_bias),
          GetTensorData<float>(output_gate_bias),
          GetTensorData<float>(projection_weights),
          GetTensorData<float>(projection_bias), params, n_batch, n_cell,
          n_input, aux_input_size, n_output, output_batch_leading_dim,
          GetTensorData<float>(output_state), GetTensorData<float>(cell_state),
          input_gate_scratch, forget_gate_scratch, cell_gate_scratch,
          output_gate_scratch, output_ptr_time, logger,
          intermediate_tensor_indexes, subgraph_index, error_reporter);
    }
  } else {
    for (int b = 0; b < n_batch; b++) {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;
      for (int t = 0; t < max_time; t++) {
        // If this is the forward_sequence, step forward, otherwise step
        // backwards.
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const float* input_ptr =
            GetTensorData<float>(input) + time_offset * input_step;
        const float* aux_input_ptr = nullptr;
        if (aux_input) {
          aux_input_ptr =
              GetTensorData<float>(aux_input) + time_offset * input_step;
        }
        float* output_ptr = GetTensorData<float>(output) +
                            time_offset * output_step + output_offset;

        // Offset the {output,cell}_state pointers to the right batch.
        float* output_state_ptr =
            GetTensorData<float>(output_state) + b * output_batch_leading_dim;
        float* cell_state_ptr = GetTensorData<float>(cell_state) + b * n_cell;
        // Offset the scratch pointers to the right batch.
        float* input_gate_scratch_ptr =
            input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
        float* forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
        float* cell_gate_scratch_ptr = cell_gate_scratch + b * n_cell;
        float* output_gate_scratch_ptr = output_gate_scratch + b * n_cell;

        LstmStepCalibration(
            input_ptr, GetTensorData<float>(input_to_input_weights),
            GetTensorData<float>(input_to_forget_weights),
            GetTensorData<float>(input_to_cell_weights),
            GetTensorData<float>(input_to_output_weights), aux_input_ptr,
            GetTensorData<float>(aux_input_to_input_weights),
            GetTensorData<float>(aux_input_to_forget_weights),
            GetTensorData<float>(aux_input_to_cell_weights),
            GetTensorData<float>(aux_input_to_output_weights),
            GetTensorData<float>(recurrent_to_input_weights),
            GetTensorData<float>(recurrent_to_forget_weights),
            GetTensorData<float>(recurrent_to_cell_weights),
            GetTensorData<float>(recurrent_to_output_weights),
            GetTensorData<float>(cell_to_input_weights),
            GetTensorData<float>(cell_to_forget_weights),
            GetTensorData<float>(cell_to_output_weights),
            GetTensorData<float>(input_layer_norm_coefficients),
            GetTensorData<float>(forget_layer_norm_coefficients),
            GetTensorData<float>(cell_layer_norm_coefficients),
            GetTensorData<float>(output_layer_norm_coefficients),
            GetTensorData<float>(input_gate_bias),
            GetTensorData<float>(forget_gate_bias),
            GetTensorData<float>(cell_gate_bias),
            GetTensorData<float>(output_gate_bias),
            GetTensorData<float>(projection_weights),
            GetTensorData<float>(projection_bias), params, /*n_batch=*/1,
            n_cell, n_input, aux_input_size, n_output, output_batch_leading_dim,
            output_state_ptr, cell_state_ptr, input_gate_scratch_ptr,
            forget_gate_scratch_ptr, cell_gate_scratch_ptr,
            output_gate_scratch_ptr, output_ptr, logger,
            intermediate_tensor_indexes, subgraph_index, error_reporter);
      }
    }
  }
  return kTfLiteOk;
}

struct OpData {
  // Which kernel type to use. Full kernel (24 inputs) or basic kernel (5
  // inputs).
  // Please note the 20-input full kernel is deprecated and only kept
  // here for backward compatibility.
  TfLiteLSTMKernelType kernel_type;

  // If the lstm is layer norm.
  bool use_layer_norm;

  // These fields are only used by full kernel.
  int scratch_tensor_index;
};

// Resize the output, state tensors based on the sizes of the input tensors.
// Allocate a temporary scratch tensor. Also check that the sizes of the input
// tensors match each other.
TfLiteStatus lstm_eval(TfLiteContext* context, int subgraph_index,
                       TfLiteNode* node, LSTMType lstm_type, Logger* logger,
                       ErrorReporter* error_reporter) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc mht_5(mht_5_v, 651, "", "./tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc", "lstm_eval");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node,
                            ops::builtin::lstm::full::kInputTensor, &input));

  const TfLiteTensor* input_to_input_weights = GetOptionalInputTensor(
      context, node, ops::builtin::lstm::full::kInputToInputWeightsTensor);
  const TfLiteTensor* input_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node,
                   ops::builtin::lstm::full::kInputToForgetWeightsTensor,
                   &input_to_forget_weights));
  const TfLiteTensor* input_to_cell_weights;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node,
                            ops::builtin::lstm::full::kInputToCellWeightsTensor,
                            &input_to_cell_weights));
  const TfLiteTensor* input_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node,
                   ops::builtin::lstm::full::kInputToOutputWeightsTensor,
                   &input_to_output_weights));

  const TfLiteTensor* recurrent_to_input_weights = GetOptionalInputTensor(
      context, node, ops::builtin::lstm::full::kRecurrentToInputWeightsTensor);
  const TfLiteTensor* recurrent_to_forget_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node,
                   ops::builtin::lstm::full::kRecurrentToForgetWeightsTensor,
                   &recurrent_to_forget_weights));
  const TfLiteTensor* recurrent_to_cell_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node,
                   ops::builtin::lstm::full::kRecurrentToCellWeightsTensor,
                   &recurrent_to_cell_weights));
  const TfLiteTensor* recurrent_to_output_weights;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node,
                   ops::builtin::lstm::full::kRecurrentToOutputWeightsTensor,
                   &recurrent_to_output_weights));

  const TfLiteTensor* cell_to_input_weights = GetOptionalInputTensor(
      context, node, ops::builtin::lstm::full::kCellToInputWeightsTensor);
  const TfLiteTensor* cell_to_forget_weights = GetOptionalInputTensor(
      context, node, ops::builtin::lstm::full::kCellToForgetWeightsTensor);
  const TfLiteTensor* cell_to_output_weights = GetOptionalInputTensor(
      context, node, ops::builtin::lstm::full::kCellToOutputWeightsTensor);

  const TfLiteTensor* input_layer_norm_coefficients = GetOptionalInputTensor(
      context, node,
      ops::builtin::lstm::full::kInputLayerNormCoefficientsTensor);
  const TfLiteTensor* forget_layer_norm_coefficients = GetOptionalInputTensor(
      context, node,
      ops::builtin::lstm::full::kForgetLayerNormCoefficientsTensor);
  const TfLiteTensor* cell_layer_norm_coefficients = GetOptionalInputTensor(
      context, node,
      ops::builtin::lstm::full::kCellLayerNormCoefficientsTensor);
  const TfLiteTensor* output_layer_norm_coefficients = GetOptionalInputTensor(
      context, node,
      ops::builtin::lstm::full::kOutputLayerNormCoefficientsTensor);

  const TfLiteTensor* input_gate_bias = GetOptionalInputTensor(
      context, node, ops::builtin::lstm::full::kInputGateBiasTensor);
  const TfLiteTensor* forget_gate_bias;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node,
                            ops::builtin::lstm::full::kForgetGateBiasTensor,
                            &forget_gate_bias));
  const TfLiteTensor* cell_gate_bias;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, ops::builtin::lstm::full::kCellGateBiasTensor,
                   &cell_gate_bias));
  const TfLiteTensor* output_gate_bias;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node,
                            ops::builtin::lstm::full::kOutputGateBiasTensor,
                            &output_gate_bias));

  const TfLiteTensor* projection_weights = GetOptionalInputTensor(
      context, node, ops::builtin::lstm::full::kProjectionWeightsTensor);
  const TfLiteTensor* projection_bias = GetOptionalInputTensor(
      context, node, ops::builtin::lstm::full::kProjectionBiasTensor);

  // Index the scratch buffers pointers to the global scratch buffer.
  TfLiteTensor* scratch_buffer;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &scratch_buffer));

  TfLiteTensor* output_state = GetVariableInput(
      context, node, ops::builtin::lstm::full::kOutputStateTensor);
  TF_LITE_ENSURE(context, output_state != nullptr);
  TfLiteTensor* cell_state = GetVariableInput(
      context, node, ops::builtin::lstm::full::kCellStateTensor);
  TF_LITE_ENSURE(context, cell_state != nullptr);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node,
                             ops::builtin::lstm::full::kOutputTensor, &output));

  std::vector<int> intermediate_tensor_indexes(node->intermediates->size);
  // LSTM expect 5 intermediate tensors.
  TF_LITE_ENSURE_EQ(context, node->intermediates->size, 5);
  for (int i = 0; i < node->intermediates->size; ++i) {
    intermediate_tensor_indexes[i] = node->intermediates->data[i];
  }

  TfLiteLSTMParams lstm_params;
  bool time_major = true;
  switch (lstm_type) {
    case LSTMType::kLSTM: {
      lstm_params = *(static_cast<TfLiteLSTMParams*>(node->builtin_data));
      time_major = true;
      break;
    }
    case LSTMType::kUnidirectionalSequenceLSTM: {
      const auto* params = static_cast<TfLiteUnidirectionalSequenceLSTMParams*>(
          node->builtin_data);
      // Copy out the LSTM specific params so they can be passed in the
      // function.
      lstm_params.activation = params->activation;
      lstm_params.cell_clip = params->cell_clip;
      lstm_params.proj_clip = params->proj_clip;
      lstm_params.asymmetric_quantize_inputs =
          params->asymmetric_quantize_inputs;
      time_major = params->time_major;
      break;
    }
    default:
      return kTfLiteError;
  }

  switch (input_to_output_weights->type) {
    case kTfLiteFloat32: {
      return EvalCalibration(
          input, input_to_input_weights, input_to_forget_weights,
          input_to_cell_weights, input_to_output_weights,
          recurrent_to_input_weights, recurrent_to_forget_weights,
          recurrent_to_cell_weights, recurrent_to_output_weights,
          cell_to_input_weights, cell_to_forget_weights, cell_to_output_weights,
          input_layer_norm_coefficients, forget_layer_norm_coefficients,
          cell_layer_norm_coefficients, output_layer_norm_coefficients,
          /*aux_input=*/nullptr,
          /*aux_input_to_input_weights=*/nullptr,
          /*aux_input_to_forget_weights=*/nullptr,
          /*aux_input_to_cell_weights=*/nullptr,
          /*aux_input_to_output_weights=*/nullptr, input_gate_bias,
          forget_gate_bias, cell_gate_bias, output_gate_bias,
          projection_weights, projection_bias, &lstm_params,
          /*forward_sequence=*/true,
          /*time_major=*/time_major,
          /*output_offset=*/0, scratch_buffer, output_state, cell_state, output,
          logger, intermediate_tensor_indexes, subgraph_index, error_reporter);
    }
    case kTfLiteUInt8:
    case kTfLiteInt8:
    default:
      printf("Error. Only float model can be calibrated\n");
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus lstm_logging_kernel(TfLiteContext* context,
                                 const int subgraph_index, TfLiteNode* node,
                                 Logger* logger,
                                 ErrorReporter* error_reporter) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc mht_6(mht_6_v, 828, "", "./tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc", "lstm_logging_kernel");

  return lstm_eval(context, subgraph_index, node, LSTMType::kLSTM, logger,
                   error_reporter);
}

TfLiteStatus unidirectional_sequence_lstm_logging_kernel(
    TfLiteContext* context, const int subgraph_index, TfLiteNode* node,
    Logger* logger, ErrorReporter* error_reporter) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSbuiltin_logging_opsPSlstmDTcc mht_7(mht_7_v, 838, "", "./tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc", "unidirectional_sequence_lstm_logging_kernel");

  return lstm_eval(context, subgraph_index, node,
                   LSTMType::kUnidirectionalSequenceLSTM, logger,
                   error_reporter);
}

}  // namespace builtin
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
