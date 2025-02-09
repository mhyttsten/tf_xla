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
class MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/lstm_eval.h"

#include <math.h>
#include <string.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace lstm_eval {
namespace {

void ComputeRowSums(
    int32_t* input_to_input_row_sums, int32_t* input_to_forget_row_sums,
    int32_t* input_to_cell_row_sums, int32_t* input_to_output_row_sums,
    int32_t* aux_input_to_input_row_sums, int32_t* aux_input_to_forget_row_sums,
    int32_t* aux_input_to_cell_row_sums, int32_t* aux_input_to_output_row_sums,
    int32_t* recurrent_to_input_row_sums, int32_t* recurrent_to_forget_row_sums,
    int32_t* recurrent_to_cell_row_sums, int32_t* recurrent_to_output_row_sums,
    int32_t* projection_weights_row_sums, int32_t* row_sums, int n_cell,
    int n_input, int n_aux_input, int n_output,
    const int8_t* input_to_input_weights_ptr,
    const int8_t* input_to_forget_weights_ptr,
    const int8_t* input_to_cell_weights_ptr,
    const int8_t* input_to_output_weights_ptr,
    const int8_t* aux_input_to_input_weights_ptr,
    const int8_t* aux_input_to_forget_weights_ptr,
    const int8_t* aux_input_to_cell_weights_ptr,
    const int8_t* aux_input_to_output_weights_ptr,
    const int8_t* recurrent_to_input_weights_ptr,
    const int8_t* recurrent_to_forget_weights_ptr,
    const int8_t* recurrent_to_cell_weights_ptr,
    const int8_t* recurrent_to_output_weights_ptr,
    const int8_t* projection_weights_ptr, bool use_cifg,
    const float* aux_input_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_0(mht_0_v, 232, "", "./tensorflow/lite/kernels/lstm_eval.cc", "ComputeRowSums");

  // Compute the row sums for dequantization
  if (!use_cifg) {
    tensor_utils::ReductionSumVector(input_to_input_weights_ptr,
                                     input_to_input_row_sums, n_cell, n_input);
  }
  tensor_utils::ReductionSumVector(input_to_forget_weights_ptr,
                                   input_to_forget_row_sums, n_cell, n_input);
  tensor_utils::ReductionSumVector(input_to_cell_weights_ptr,
                                   input_to_cell_row_sums, n_cell, n_input);
  tensor_utils::ReductionSumVector(input_to_output_weights_ptr,
                                   input_to_output_row_sums, n_cell, n_input);

  if (aux_input_ptr) {
    if (!use_cifg) {
      tensor_utils::ReductionSumVector(aux_input_to_input_weights_ptr,
                                       aux_input_to_input_row_sums, n_cell,
                                       n_aux_input);
    }
    tensor_utils::ReductionSumVector(aux_input_to_forget_weights_ptr,
                                     aux_input_to_forget_row_sums, n_cell,
                                     n_aux_input);
    tensor_utils::ReductionSumVector(aux_input_to_cell_weights_ptr,
                                     aux_input_to_cell_row_sums, n_cell,
                                     n_aux_input);
    tensor_utils::ReductionSumVector(aux_input_to_output_weights_ptr,
                                     aux_input_to_output_row_sums, n_cell,
                                     n_aux_input);
  }
  if (!use_cifg) {
    tensor_utils::ReductionSumVector(recurrent_to_input_weights_ptr,
                                     recurrent_to_input_row_sums, n_cell,
                                     n_output);
  }
  tensor_utils::ReductionSumVector(recurrent_to_forget_weights_ptr,
                                   recurrent_to_forget_row_sums, n_cell,
                                   n_output);
  tensor_utils::ReductionSumVector(recurrent_to_cell_weights_ptr,
                                   recurrent_to_cell_row_sums, n_cell,
                                   n_output);
  tensor_utils::ReductionSumVector(recurrent_to_output_weights_ptr,
                                   recurrent_to_output_row_sums, n_cell,
                                   n_output);

  if (projection_weights_ptr != nullptr) {
    tensor_utils::ReductionSumVector(
        projection_weights_ptr, projection_weights_row_sums, n_output, n_cell);
  }
}

inline float GetTensorScale(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_1(mht_1_v, 285, "", "./tensorflow/lite/kernels/lstm_eval.cc", "GetTensorScale");

  return tensor == nullptr ? 1.0f : tensor->params.scale;
}

// LINT.IfChange
// Calculates a single LSTM gate.
//
// Implements the following formula: (* is matrix multiply)
//   gate = activate(W_input    * input + W_aux       * aux_input   +
//                   W_peephole * cell  + W_recurrent * prev_output + bias)
// with layer norm:
//   gate = activate(W_norm * normalize(...) + bias) // not adding bias inside
//
// Activation is sigmoid except for the "cell" gate (configurable, usually tanh)
//
// Parameters:
// Input vectors (to LSTM):    | Size:                | Optional?
//   input                     | n_input              |
//   aux_input                 | n_aux_input          | y (bidir LSTM)
// Input vectors (persistent states):
//   output_state              | n_output             |
//   cell_state                | n_cell               |
// 'Constant' inputs:
//   input_to_gate_weights     | n_cell * n_input     |
//   aux_input_to_gate_weights | n_cell * n_aux_input | y (bidir LSTM)
//   recurrent_to_gate_weights | n_cell * n_output    |
//   cell_to_gate_weights      | n_cell               | y (peephole)
//   gate_bias                 | n_cell               |
//   layer_norm_coefficients   | n_cell               | y (layer norm)
// Output vector:
//   gate                      | n_cell               |
// Scalar parameters:
//   n_batch                                    - batch size / number of vectors
//   n_input, n_aux_input, n_output, n_cell     - size of vectors.
//   activation                                 - activation to use.
//   is_input_all_zeros, is_aux_input_all_zeros - if input vectors are all zero.
//   use_layer_norm                             - if doing layer norm LSTM.
inline void CalculateLstmGateFloat(
    const float* input, const float* input_to_gate_weights,
    const float* aux_input, const float* aux_input_to_gate_weights,
    const float* output_state, const float* recurrent_to_gate_weights,
    const float* cell_state, const float* cell_to_gate_weights,
    const float* layer_norm_coefficients, const float* gate_bias,
    const int n_batch, const int n_input, const int n_aux_input,
    const int n_output, const int n_cell,
    const TfLiteFusedActivation activation, float* gate,
    const bool is_input_all_zeros, const bool is_aux_input_all_zeros) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_2(mht_2_v, 334, "", "./tensorflow/lite/kernels/lstm_eval.cc", "CalculateLstmGateFloat");

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
    tensor_utils::MeanStddevNormalization(gate, gate, n_cell, n_batch);
    tensor_utils::VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell,
                                                gate, n_batch, gate);
    tensor_utils::VectorBatchVectorAdd(gate_bias, n_cell, n_batch, gate);
  }
  // Apply activation
  tensor_utils::ApplyActivationToVector(gate, n_batch * n_cell, activation,
                                        gate);
}

// Updates the LSTM cell state, used by both float and hybrid LSTM versions.
//
// Implements the following formula:
//   cell_state_new = clip(forget_gate * cell_state + input_gate * cell_gate)
//
// With CIFG LSTM, input gate is replaced by (1-forget_gate).
//
// Parameters:
//  - n_batch, n_cell: sizes of vectors
//  - cell_state: input/output vector, size n_batch*n_cell
//  - input_gate: input vector, size n_batch*n_cell.
//  - forget_gate: input/scratch vector, size n_batch*n_cell, modified with CIFG
//  - cell_gate: input vector, size n_batch*n_cell.
//  - use_cifg: use 1-forget_gate instead of input_gate.
//  - clip: if > 0, clip the resulting cell state to [-clip, +clip].
void UpdateLstmCellFloat(int n_batch, int n_cell, float* cell_state,
                         const float* input_gate, float* forget_gate,
                         const float* cell_gate, bool use_cifg, float clip) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_3(mht_3_v, 398, "", "./tensorflow/lite/kernels/lstm_eval.cc", "UpdateLstmCellFloat");

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

// Calculates the output state tensor of an LSTM step.
//
// Implements the following formula:
//   output_no_projection = output_gate .* activate(cell_state)
//     (elementwise vector product)
// If no projection is used:
//   output = output_state = output_no_projection
// With projection:
//   output = output_state = clip(W*output_no_projection + bias)
//
// Output might not have a different 'stride' than n_batch, so we need to copy.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - projection_weights, projection_weights_scale, projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - scratch: scratch area, size n_batch*n_cell.
void CalculateLstmOutputFloat(int n_batch, int n_cell, int n_output,
                              const float* cell_state, const float* output_gate,
                              TfLiteFusedActivation activation,
                              const float* projection_weights,
                              const float* projection_bias,
                              const float proj_clip, float* output_state,
                              float* scratch) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_4(mht_4_v, 449, "", "./tensorflow/lite/kernels/lstm_eval.cc", "CalculateLstmOutputFloat");

  tensor_utils::ApplyActivationToVector(cell_state, n_batch * n_cell,
                                        activation, scratch);
  tensor_utils::VectorVectorCwiseProduct(output_gate, scratch, n_batch * n_cell,
                                         scratch);

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
// LINT.ThenChange(../tools/optimize/calibration/builtin_logging_ops/lstm.cc,\
//                 ../experimental/kernels/fp16/lstm_eval.cc)

// Calculates a single LSTM gate, hybrid version.
// Implements the same functionality as CalculateLstmGateFloat.
void CalculateLstmGateHybrid(
    // Input and weights
    const int8_t* input, const float* input_sf, const int32_t* input_zp,
    const int8_t* input_to_gate_weights,
    const uint8_t* input_to_gate_weights_ledger,
    const float input_to_gate_weights_scale, int32_t* input_to_gate_row_sums,
    // Aux input and weights
    const int8_t* aux_input, const float* aux_input_sf,
    const int32_t* aux_input_zp, const int8_t* aux_input_to_gate_weights,
    const float aux_input_to_gate_weights_scale,
    int32_t* aux_input_to_gate_row_sums,
    // Output state and weights
    const int8_t* output_state, const float* output_state_sf,
    const int32_t* output_state_zp, const int8_t* recurrent_to_gate_weights,
    const uint8_t* recurrent_to_gate_weights_ledger,
    const float recurrent_to_gate_weights_scale,
    int32_t* recurrent_to_gate_row_sums,
    // Cell state and weights (peephole LSTM)
    const float* cell_state, const int8_t* cell_to_gate_weights,
    const float cell_to_gate_weights_scale,
    // Layer normalization coefficients (layer norm LSTM) + gate bias
    const float* layer_norm_coefficients, const float* gate_bias,
    // Array sizes
    const int n_batch, const int n_input, const int n_aux_input,
    const int n_output, const int n_cell,
    const TfLiteFusedActivation activation,
    // Output
    float* gate,
    // Parameters for performance optimizations
    const bool is_input_all_zeros, const bool is_aux_input_all_zeros,
    const bool is_output_state_all_zeros, bool* compute_row_sums,
    CpuBackendContext* context,
    // Scratch arrays
    float* scratch0,        // size: n_batch
    float* scratch1,        // size: n_cell, only used if peephole LSTM
    int32_t* accum_scratch  // For MatrixBatchVectorMultiplyAccumulate
) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_5(mht_5_v, 518, "", "./tensorflow/lite/kernels/lstm_eval.cc", "CalculateLstmGateHybrid");

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
    if (input_to_gate_weights_ledger != nullptr) {
      std::vector<float> scales(n_batch);
      for (int i = 0; i < n_batch; i++) {
        scales[i] = input_to_gate_weights_scale * input_sf[i];
      }
      tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
          input_to_gate_weights, input_to_gate_weights_ledger, n_cell, n_input,
          input, scales.data(), n_batch, gate);

    } else {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          input_to_gate_weights, n_cell, n_input, input,
          input_to_gate_weights_scale, input_sf, n_batch, gate,
          /*per_channel_scale=*/nullptr, input_zp, accum_scratch,
          input_to_gate_row_sums, compute_row_sums, scratch0, context);
    }
  }
  // For each batch and cell: compute aux_input_weight * aux_input.
  // Skip if auxiliary input is not available or all zeros.
  if (!is_aux_input_all_zeros) {
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        aux_input_to_gate_weights, n_cell, n_aux_input, aux_input,
        aux_input_to_gate_weights_scale, aux_input_sf, n_batch, gate,
        /*per_channel_scale=*/nullptr, aux_input_zp, accum_scratch,
        aux_input_to_gate_row_sums, compute_row_sums, scratch0, context);
  }
  // For each batch and cell: compute recurrent_weight * output_state.
  // Skip if output state is all zeros.
  if (!is_output_state_all_zeros) {
    if (recurrent_to_gate_weights_ledger != nullptr) {
      std::vector<float> scales(n_batch);
      for (int i = 0; i < n_batch; i++) {
        scales[i] = recurrent_to_gate_weights_scale * input_sf[i];
      }
      tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
          recurrent_to_gate_weights, recurrent_to_gate_weights_ledger, n_cell,
          n_output, output_state, scales.data(), n_batch, gate);
    } else {
      tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          recurrent_to_gate_weights, n_cell, n_output, output_state,
          recurrent_to_gate_weights_scale, output_state_sf, n_batch, gate,
          /*per_channel_scale=*/nullptr, output_state_zp, accum_scratch,
          recurrent_to_gate_row_sums, compute_row_sums, scratch0, context);
    }
  }
  // For each batch and cell: compute cell_weight .* cell_state (peephole LSTM)
  if (use_peephole) {
    float* recovered_cell_weights = scratch1;
    tensor_utils::VectorScalarMultiply(cell_to_gate_weights, n_cell,
                                       cell_to_gate_weights_scale,
                                       recovered_cell_weights);
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        recovered_cell_weights, n_cell, cell_state, n_batch, gate);
  }
  // Do layer normalization (if layer norm LSTM)
  if (use_layer_norm) {
    tensor_utils::MeanStddevNormalization(gate, gate, n_cell, n_batch);
    tensor_utils::VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell,
                                                gate, n_batch, gate);
    tensor_utils::VectorBatchVectorAdd(gate_bias, n_cell, n_batch, gate);
  }
  // Apply activation
  tensor_utils::ApplyActivationToVector(gate, n_cell * n_batch, activation,
                                        gate);
}

// Calculates the output state tensor of an LSTM step. See Float version too.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - projection_weights, projection_weights_scale, projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - asymmetric_quantize_inputs: parameter to control quantization.
//  - projection_weights_row_sums, compute_row_sums, context: Data for optimized
//      MatrixBatchVectorMultiplyAccumulate.
//  - scratch0: scratch area of size n_batch*n_cell
//  - scratch1: scratch area of size n_batch*n_cell
//  - scratch2: scratch area of size n_batch
//  - scratch3: scratch area of size n_batch
//  - scratch4: scratch area used by MatrixBatchVectorMultiplyAccumulate
void CalculateLstmOutputHybrid(
    int n_batch, int n_cell, int n_output, const float* cell_state,
    const float* output_gate, TfLiteFusedActivation activation,
    const int8_t* projection_weights, const uint8_t* projection_weights_ledger,
    float projection_weights_scale, const float* projection_bias,
    const float proj_clip, float* output_state, bool asymmetric_quantize_inputs,
    int32_t* projection_weights_row_sums, bool* compute_row_sums,
    CpuBackendContext* context, float* scratch0, int8_t* scratch1,
    float* scratch2, int32_t* scratch3, int32_t* scratch4) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_6(mht_6_v, 627, "", "./tensorflow/lite/kernels/lstm_eval.cc", "CalculateLstmOutputHybrid");

  tensor_utils::ApplyActivationToVector(cell_state, n_batch * n_cell,
                                        activation, scratch0);
  tensor_utils::VectorVectorCwiseProduct(output_gate, scratch0,
                                         n_batch * n_cell, scratch0);

  const bool use_projection = (projection_weights != nullptr);
  const bool use_projection_bias = (projection_bias != nullptr);

  if (use_projection) {
    if (use_projection_bias) {
      tensor_utils::VectorBatchVectorAssign(projection_bias, n_output, n_batch,
                                            output_state);
    } else {
      std::fill_n(output_state, n_batch * n_output, 0.0f);
    }
    if (!tensor_utils::IsZeroVector(scratch0, n_batch * n_cell)) {
      // Save quantization and matmul computation for all zero output.
      tensor_utils::BatchQuantizeFloats(scratch0, n_batch, n_cell, scratch1,
                                        scratch2, scratch3,
                                        asymmetric_quantize_inputs);
      if (projection_weights_ledger != nullptr) {
        std::vector<float> scales(n_batch);
        for (int i = 0; i < n_batch; i++) {
          scales[i] = projection_weights_scale * scratch2[i];
        }
        tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
            projection_weights, projection_weights_ledger, n_output, n_cell,
            scratch1, scales.data(), n_batch, output_state);
      } else {
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            projection_weights, n_output, n_cell, scratch1,
            projection_weights_scale, scratch2, n_batch, output_state,
            /*per_channel_scale=*/nullptr, scratch3, scratch4,
            projection_weights_row_sums, compute_row_sums, scratch2, context);
      }
    }
    if (proj_clip > 0.0f) {
      tensor_utils::CwiseClipping(output_state, n_batch * n_output, proj_clip);
    }
  } else {
    std::copy_n(scratch0, n_batch * n_output, output_state);
  }
}

// Calculates a single LSTM gate, int8x8_16 version.
// Implements the same functionality as CalculateLstmGateFloat.
void CalculateLstmGateInteger8x8_16(
    // Input and weights
    const int8_t* input, const int8_t* input_to_gate_weights,
    const int32_t* input_to_gate_bias, const int32_t input_to_gate_scale_a,
    const int32_t input_to_gate_scale_b,
    // Output state and weights
    const int8_t* output_state, const int8_t* recurrent_to_gate_weights,
    const int32_t* recurrent_to_gate_bias,
    const int32_t recurrent_to_gate_scale_a,
    const int32_t recurrent_to_gate_scale_b,
    // Cell state and weights
    const int16_t* cell_state, const int16_t* cell_to_gate_weights,
    const int32_t cell_to_gate_scale_a, const int32_t cell_to_gate_scale_b,
    // Layer normalization parameters (layer norm LSTM)
    const int16_t* layer_norm_coefficients, const int32_t* layer_norm_bias,
    const int32_t layer_norm_input_scale_a,
    const int32_t layer_norm_input_scale_b,
    const int32_t layer_norm_variance_guard,
    // Array sizes
    const int n_batch, const int n_input, const int n_output, const int n_cell,
    const TfLiteFusedActivation activation,
    // Output
    int16_t* gate,
    // Parameters for performance optimizations
    CpuBackendContext* context,
    // Scratch arrays
    int32_t* scratch5) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_7(mht_7_v, 703, "", "./tensorflow/lite/kernels/lstm_eval.cc", "CalculateLstmGateInteger8x8_16");

  const bool use_peephole = (cell_to_gate_weights != nullptr);
  const bool use_layer_norm = (layer_norm_coefficients != nullptr);

  // Initialize scratch buffers with zeros. Note that unlike float and hybrid
  // versions, bias is only used in layer normalization.
  std::fill_n(gate, n_batch * n_cell, 0);
  // For each batch and cell: compute input_weight * input.
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      input, input_to_gate_bias, input_to_gate_weights, input_to_gate_scale_a,
      input_to_gate_scale_b, n_batch, n_input, n_cell, 0, scratch5, gate,
      context);
  // Note: no aux_input.

  // For each batch and cell: compute recurrent_weight * output_state.
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      output_state, recurrent_to_gate_bias, recurrent_to_gate_weights,
      recurrent_to_gate_scale_a, recurrent_to_gate_scale_b, n_batch, n_output,
      n_cell, 0, scratch5, gate, context);
  // For each batch and cell: compute cell_weight * cell_state (peephole LSTM)
  if (use_peephole) {
    tensor_utils::VectorBatchVectorCwiseProductAccumulate(
        cell_to_gate_weights, n_output, cell_state, n_batch,
        cell_to_gate_scale_a, cell_to_gate_scale_b, gate);
  }
  // Do layer normalization (if layer norm LSTM)
  if (use_layer_norm) {
    tensor_utils::ApplyLayerNorm(
        gate, layer_norm_coefficients, layer_norm_bias,
        layer_norm_input_scale_a, layer_norm_input_scale_b,
        layer_norm_variance_guard, n_batch, n_cell, gate);
  }
  // Apply activation
  switch (activation) {
    case kTfLiteActSigmoid:
      tensor_utils::ApplySigmoid(gate, n_batch, n_cell, gate);
      break;
    case kTfLiteActTanh:
      tensor_utils::ApplyTanh(3, gate, n_batch, n_cell, gate);
      break;
    default:
      // Only Sigmoid or Tanh is used.
      TFLITE_ASSERT_FALSE;
  }
}

// Updates the LSTM cell state, used by both integer LSTM versions.
// Also see UpdateLstmCellFloat.
//
// Parameters:
//  - n_batch, n_cell: sizes of vectors
//  - cell_state: input/output vector, size n_batch*n_cell
//  - cell_state_scale: scaling factor of cell state.
//  - input_gate: input vector, size n_batch*n_cell.
//  - forget_gate: input/scratch vector, size n_batch*n_cell, always modified.
//  - cell_gate: input vector, size n_batch*n_cell.
//  - use_cifg: use 1-forget_gate instead of input_gate.
//  - clip: if > 0, clip the resulting cell state to [-clip, +clip].
void UpdateLstmCellInteger(int n_batch, int n_cell, int16_t* cell_state,
                           int32_t cell_state_scale, const int16_t* input_gate,
                           int16_t* forget_gate, const int16_t* cell_gate,
                           bool use_cifg, int16_t clip) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_8(mht_8_v, 767, "", "./tensorflow/lite/kernels/lstm_eval.cc", "UpdateLstmCellInteger");

  // Use the forget_gate array as scratch, as input_gate array is not allocated
  // in CIFG case. (Be careful not to write to the scratch before reading the
  // forget gate data.)
  int16_t* scratch = forget_gate;

  tensor_utils::CwiseMul(forget_gate, cell_state, n_batch, n_cell, 15,
                         cell_state);
  if (use_cifg) {
    tensor_utils::Sub1Vector(forget_gate, n_batch * n_cell, scratch);
    tensor_utils::CwiseMul(scratch, cell_gate, n_batch, n_cell,
                           30 + cell_state_scale, scratch);
  } else {
    tensor_utils::CwiseMul(input_gate, cell_gate, n_batch, n_cell,
                           30 + cell_state_scale, scratch);
  }
  tensor_utils::CwiseAdd(cell_state, scratch, n_batch, n_cell, cell_state);

  if (clip > 0) {
    tensor_utils::CwiseClipping(cell_state, n_batch * n_cell, clip);
  }
}

// Calculates the output state tensor of an LSTM step. See Float and hybrid
// versions as well.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - cell_state_scale: scaling of cell_state.
//  - hidden_scale_[a|b]: effective scale of cell_state.*output_gate
//  - hidden_zp: zero_point for cell_state.*output_gate
//  - projection_weights, proj_scale_[a|b], projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - output_state_zp: zero point of output_state. (Input, calibrated value.)
//  - quantized_proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - context: data for optimized MatrixBatchVectorMultiplyAccumulate.
//  - scratch0: scratch area of size n_batch*n_cell
//  - scratch1: scratch area of size n_batch*n_cell
//  - scratch2: scratch area used by MatrixBatchVectorMultiplyAccumulate
void CalculateLstmOutputInteger8x8_16(
    int n_batch, int n_cell, int n_output, const int16_t* cell_state,
    int32_t cell_state_scale, const int16_t* output_gate,
    int32_t hidden_scale_a, int32_t hidden_scale_b, int32_t hidden_zp,
    const int8_t* projection_weights, int32_t proj_scale_a,
    int32_t proj_scale_b, const int32_t* projection_bias,
    int32_t output_state_zp, int8_t quantized_proj_clip, int8_t* output_state,
    CpuBackendContext* context, int16_t* scratch0, int8_t* scratch1,
    int32_t* scratch2) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_9(mht_9_v, 820, "", "./tensorflow/lite/kernels/lstm_eval.cc", "CalculateLstmOutputInteger8x8_16");

  // Note: unlike float/hybrid, the activation is always Tanh.
  tensor_utils::ApplyTanh(15 + cell_state_scale, cell_state, n_batch, n_cell,
                          scratch0);
  tensor_utils::CwiseMul(output_gate, scratch0, hidden_scale_a, hidden_scale_b,
                         n_batch, n_cell, hidden_zp, scratch1);

  const bool use_projection = (projection_weights != nullptr);

  if (use_projection) {
    // Note: no bias like in float/hybrid
    std::fill_n(output_state, n_batch * n_output, 0);
    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        scratch1, projection_bias, projection_weights, proj_scale_a,
        proj_scale_b, n_batch, n_cell, n_output, output_state_zp, scratch2,
        output_state, context);
    if (quantized_proj_clip > 0) {
      tensor_utils::CwiseClipping(output_state, n_batch * n_output,
                                  quantized_proj_clip);
    }
  } else {
    std::copy_n(scratch1, n_batch * n_output, output_state);
  }
}

// Calculates a single LSTM gate, int8x8_8 version.
// Implements the same functionality as CalculateLstmGateFloat.
void CalculateLstmGateInteger8x8_8(
    // Inputs and weights
    const int8_t* input, int32_t input_zp, const int8_t* input_to_gate_weight,
    const int32_t input_to_gate_scale_a, const int32_t input_to_gate_scale_b,
    const int32_t input_times_weights_scale_a,
    const int32_t input_times_weights_scale_b,
    const int32_t input_times_weights_zp,
    // Output state and weights
    const int8_t* output_state, const int32_t output_state_zp,
    const int8_t* recurrent_to_gate_weight,
    const int32_t recurrent_to_gate_scale_a,
    const int32_t recurrent_to_gate_scale_b,
    const int32_t output_state_times_weights_scale_a,
    const int32_t output_state_times_weights_scale_b,
    const int32_t output_state_times_weights_zp,
    // Layer normalization parameters (layer norm LSTM)
    const int16_t* layer_norm_gate_weight,
    const int32_t layer_norm_gate_scale_a,
    const int32_t layer_norm_gate_scale_b, const int32_t* gate_bias,
    // Array sizes
    const int n_batch, const int n_input, const int n_output, const int n_cell,
    const TfLiteFusedActivation activation,
    // Output
    int16_t* gate,
    // Scratch arrays, both sized n_batch*n_cell
    int8_t* scratch0, int8_t* scratch1) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_10(mht_10_v, 875, "", "./tensorflow/lite/kernels/lstm_eval.cc", "CalculateLstmGateInteger8x8_8");

  // Multiply input * input_weights => scratch0
  tensor_utils::MatrixBatchVectorMultiply(
      input, input_zp, input_to_gate_weight, input_to_gate_scale_a,
      input_to_gate_scale_b, n_batch, n_input, n_cell, scratch0,
      input_times_weights_zp);
  // Multiply output_state * recurrent_weights => scratch1
  tensor_utils::MatrixBatchVectorMultiply(
      output_state, output_state_zp, recurrent_to_gate_weight,
      recurrent_to_gate_scale_a, recurrent_to_gate_scale_b, n_batch, n_output,
      n_cell, scratch1, output_state_times_weights_zp);
  // Add scratch0 + scratch1 => gate
  tensor_utils::TwoGateSaturatingAdd(
      scratch0, input_times_weights_zp, scratch1, output_state_times_weights_zp,
      input_times_weights_scale_a, input_times_weights_scale_b,
      output_state_times_weights_scale_a, output_state_times_weights_scale_b,
      n_batch, n_cell, gate);
  // Apply layer normalization.
  tensor_utils::ApplyLayerNormFloat(
      gate, layer_norm_gate_weight, layer_norm_gate_scale_a,
      layer_norm_gate_scale_b, gate_bias, n_batch, n_cell, gate);
  // Apply activation.
  switch (activation) {
    case kTfLiteActSigmoid:
      tensor_utils::ApplySigmoidFloat(gate, n_batch, n_cell, gate);
      break;
    case kTfLiteActTanh:
      tensor_utils::ApplyTanhFloat(gate, n_batch, n_cell, -12, gate);
      break;
    default:
      // Only Sigmoid or Tanh is used.
      TFLITE_ASSERT_FALSE;
  }
}

// Calculates the output state tensor of an LSTM step. See Float and hybrid
// versions as well.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - projection_weights, proj_scale_[a|b], projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - output_state_zp: zero point of the output state.
//  - quantized_proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - scratch: scratch area of size n_batch*n_cell
void CalculateLstmOutputInteger8x8_8(
    int n_batch, int n_cell, int n_output, const int16_t* cell_state,
    const int16_t* output_gate, const int8_t* projection_weights,
    int32_t proj_scale_a, int32_t proj_scale_b, const int32_t* projection_bias,
    int32_t output_state_zp, int32_t quantized_proj_clip, int8_t* output_state,
    int16_t* scratch) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_11(mht_11_v, 931, "", "./tensorflow/lite/kernels/lstm_eval.cc", "CalculateLstmOutputInteger8x8_8");

  // Note: unlike float/hybrid, the activation is always Tanh.
  tensor_utils::ApplyTanhFloat(cell_state, n_batch, n_cell, -15, scratch);
  tensor_utils::CwiseMul(output_gate, scratch, n_batch, n_cell, 15 + 15 - 15,
                         scratch);
  // Note: no bias like in float/hybrid
  tensor_utils::MatrixBatchVectorMultiply(
      scratch, projection_weights, proj_scale_a, proj_scale_b, projection_bias,
      n_batch, n_cell, n_output, output_state_zp, output_state);
  if (quantized_proj_clip > 0) {
    tensor_utils::CwiseClipping(output_state, n_batch * n_output,
                                quantized_proj_clip);
  }
}

// Performs an LSTM batch inference step for input specified by input_ptr.
// The LSTM cell is specified by the pointers to its weights (*_weights_ptr) and
// biases (*_bias_ptr), and buffers (*_scratch), along with additional
// parameters:
//  - params: various LSTM params including activation, clipping, etc.,
//  - n_batch: size of batch,
//  - n_cell: number of cells (or units),
//  - n_input: the input size,
//  - n_aux_input: the auxiliary input size.
//  - n_output: the output size.
//  - output_batch_leading_dim: the leading dimension of the output buffer.
//
// Input of size 'n_batch * n_input':
//   input_ptr
// Input of size 'n_batch * n_aux_input':
//   aux_input_ptr                     - optional (can be nullptr)
//
// LSTM weights:
// Input weights of size 'n_cell * n_input':
//   input_to_input_weights            - optional
//   input_to_forget_weights
//   input_to_cell_weights
//   input_to_output_weights
// Auxiliary input weights of size 'n_cell * n_aux_input':
//   aux_input_to_input_weights        - optional
//   aux_input_to_forget_weights       - optional
//   aux_input_to_cell_weights         - optional
//   aux_input_to_output_weights       - optional
// Recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weights        - optional
//   recurrent_to_forget_weights
//   recurrent_to_cell_weights
//   recurrent_to_input_weights
// Peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights             - optional
//   cell_to_cell_weights              - optional
//   cell_to_output_weights            - optional
// Projection weights of size 'n_output * n_cell'
//   projection_weights_ptr            - optional
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr               - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   input_layer_norm_coefficients_ptr  - optional
//   forget_layer_norm_coefficients_ptr - optional
//   cell_layer_norm_coefficients_ptr   - optional
//   output_layer_norm_coefficients_ptr - optional
//
// The pointers to the cell and output state and the output are updated.
//
// The pointers input_ptr, aux_input_ptr, and output_ptr point to data aligned
// in batch_major order, and each step processes batch_size many inputs from
// input_ptr, and updates batch_size many cell and output states.
//
// The output_batch_dim is output.shape[-1], i.e. the outermost dimension of the
// output tensor, and in most cases will be equal to n_output. It is usually not
// when we want to store the LSTM output into a slice of the output tensor, e.g.
// for bidirectional LSTMs with merge_outputs. In this case, the batched
// operations cannot be used since they assume that the batched outputs are
// contiguous, and we manually loop over the batched outputs.
// LINT.IfChange
inline void LstmStepFloat(
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
    float* scratch1, float* scratch2, float* scratch3, float* output_ptr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_12(mht_12_v, 1039, "", "./tensorflow/lite/kernels/lstm_eval.cc", "LstmStepFloat");

  ruy::profiler::ScopeLabel label("LstmStepFloat");
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
        is_input_all_zeros, is_aux_input_all_zeros);
  }
  // Calculate the forget gate.
  CalculateLstmGateFloat(
      input_ptr, input_to_forget_weights_ptr, aux_input_ptr,
      aux_input_to_forget_weights_ptr, output_state_ptr,
      recurrent_to_forget_weights_ptr, cell_state_ptr,
      cell_to_forget_weights_ptr, forget_layer_norm_coefficients_ptr,
      forget_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
      /*activation=*/kTfLiteActSigmoid, forget_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros);
  // Calculate the cell update gate.
  CalculateLstmGateFloat(input_ptr, input_to_cell_weights_ptr, aux_input_ptr,
                         aux_input_to_cell_weights_ptr, output_state_ptr,
                         recurrent_to_cell_weights_ptr, /*cell_state=*/nullptr,
                         /*cell_to_gate_weights=*/nullptr,
                         cell_layer_norm_coefficients_ptr, cell_gate_bias_ptr,
                         n_batch, n_input, n_aux_input, n_output, n_cell,
                         params->activation, cell_gate_scratch,
                         is_input_all_zeros, is_aux_input_all_zeros);
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
      is_aux_input_all_zeros);
  // Update the output state.
  CalculateLstmOutputFloat(n_batch, n_cell, n_output, cell_state_ptr,
                           output_gate_scratch, params->activation,
                           projection_weights_ptr, projection_bias_ptr,
                           params->proj_clip, output_state_ptr, scratch2);
  // Copy output state to the output. Note that the output's rows may not be
  // contiguous (output_batch_leading_dim != n_output).
  for (int b = 0; b < n_batch; b++) {
    std::copy_n(output_state_ptr + b * n_output, n_output,
                output_ptr + b * output_batch_leading_dim);
  }
}
// LINT.ThenChange(../tools/optimize/calibration/builtin_logging_ops/lstm.cc,\
//                 ../experimental/kernels/fp16/lstm_eval.cc)

// Same as above but with quantized weight matrices. In detail:
// Input of size 'n_batch * n_input':
//   input_ptr
// Input of size 'n_batch * n_aux_input':
//   aux_input_ptr                     - optional (can be nullptr)
//
// LSTM weights:
// Quantized input weights of size 'n_cell * n_input':
//   input_to_input_weights            - optional
//   input_to_forget_weights
//   input_to_cell_weights
//   input_to_input_weights
// Quantized auxiliary input weights of size 'n_cell * n_aux_input':
//   aux_input_to_input_weights        - optional
//   aux_input_to_forget_weights       - optional
//   aux_input_to_cell_weights         - optional
//   aux_input_to_output_weights       - optional
// Quantized recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weights        - optional
//   recurrent_to_forget_weights
//   recurrent_to_cell_weights
//   recurrent_to_input_weights
// Quantized peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights             - optional
//   cell_to_cell_weights              - optional
//   cell_to_output_weights            - optional
// Quantized projection weights of size 'n_output * n_cell'
//   projection_weights_ptr            - optional
// Weight scales (scalars) for each of the weights above.
//   input_to_input_weights_scale      - optional
//   input_to_forget_weights_scale
//   input_to_cell_weights_scale
//   input_to_output_weights_scale
//   aux_input_to_input_weights_scale  - optional
//   aux_input_to_forget_weights_scale - optional
//   aux_input_to_cell_weights_scale   - optional
//   aux_input_to_output_weights_scale - optional
//   recurrent_to_input_weights_scale  - optional
//   recurrent_to_forget_weights_scale
//   recurrent_to_cell_weights_scale
//   recurrent_to_output_weights_scale
//   cell_to_input_weights_scale,
//   cell_to_forget_weights_scale,
//   cell_to_output_weights_scale,
//   projection_weights_scale          - optional
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr               - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   input_layer_norm_coefficients_ptr  - optional
//   forget_layer_norm_coefficients_ptr - optional
//   cell_layer_norm_coefficients_ptr   - optional
//   output_layer_norm_coefficients_ptr - optional
//
// Temporary pre-allocated storage for quantized values:
//   quantized_input_ptr (same size as input_ptr)
//   quantized_output_state_ptr (same size as output_state_ptr)
//   quantized_output_scratch (same size as cell_state_ptr)
// Temporary pre-allocated storage for recovered values:
//   recovered_cell_weights (same size as cell_to_*_weights)
//
// Outputs:
//   output_state_ptr - size 'n_batch * n_output'
//   cell_state_ptr   - size 'n_batch * n_cell'
//   output_ptr       - size 'n_batch * output_batch_leading_dim'
inline void LstmStepHybrid(
    const float* input_ptr, const int8_t* input_to_input_weights_ptr,
    const uint8_t* input_to_input_weights_ledger_ptr,
    float input_to_input_weights_scale,
    const int8_t* input_to_forget_weights_ptr,
    const uint8_t* input_to_forget_weights_ledger_ptr,
    float input_to_forget_weights_scale,
    const int8_t* input_to_cell_weights_ptr,
    const uint8_t* input_to_cell_weights_ledger_ptr,
    float input_to_cell_weights_scale,
    const int8_t* input_to_output_weights_ptr,
    const uint8_t* input_to_output_weights_ledger_ptr,
    float input_to_output_weights_scale, const float* aux_input_ptr,
    const int8_t* aux_input_to_input_weights_ptr,
    float aux_input_to_input_weights_scale,
    const int8_t* aux_input_to_forget_weights_ptr,
    float aux_input_to_forget_weights_scale,
    const int8_t* aux_input_to_cell_weights_ptr,
    float aux_input_to_cell_weights_scale,
    const int8_t* aux_input_to_output_weights_ptr,
    float aux_input_to_output_weights_scale,
    const int8_t* recurrent_to_input_weights_ptr,
    const uint8_t* recurrent_to_input_weights_ledger_ptr,
    float recurrent_to_input_weights_scale,
    const int8_t* recurrent_to_forget_weights_ptr,
    const uint8_t* recurrent_to_forget_weights_ledger_ptr,
    float recurrent_to_forget_weights_scale,
    const int8_t* recurrent_to_cell_weights_ptr,
    const uint8_t* recurrent_to_cell_weights_ledger_ptr,
    float recurrent_to_cell_weights_scale,
    const int8_t* recurrent_to_output_weights_ptr,
    const uint8_t* recurrent_to_output_weights_ledger_ptr,
    float recurrent_to_output_weights_scale,
    const int8_t* cell_to_input_weights_ptr, float cell_to_input_weights_scale,
    const int8_t* cell_to_forget_weights_ptr,
    float cell_to_forget_weights_scale,
    const int8_t* cell_to_output_weights_ptr,
    float cell_to_output_weights_scale,
    const float* input_layer_norm_coefficients_ptr,
    const float* forget_layer_norm_coefficients_ptr,
    const float* cell_layer_norm_coefficients_ptr,
    const float* output_layer_norm_coefficients_ptr,
    const float* input_gate_bias_ptr, const float* forget_gate_bias_ptr,
    const float* cell_gate_bias_ptr, const float* output_gate_bias_ptr,
    const int8_t* projection_weights_ptr,
    const uint8_t* projection_weights_ledger_ptr,
    float projection_weights_scale, const float* projection_bias_ptr,
    const TfLiteLSTMParams* params, int n_batch, int n_cell, int n_input,
    int n_aux_input, int n_output, int output_batch_leading_dim,
    float* scratch0, float* scratch1, float* scratch2, float* scratch3,
    float* input_sf, float* aux_input_sf, float* output_state_sf,
    float* scaling_factors_scratch, float* recovered_cell_weights,
    int8_t* quantized_input_ptr, int8_t* quantized_aux_input_ptr,
    int8_t* quantized_output_state_ptr, int8_t* quantized_output_scratch,
    float* output_state_ptr, float* cell_state_ptr, int32_t* accum_scratch_ptr,
    float* output_ptr, int32_t* input_zp, int32_t* aux_input_zp,
    int32_t* output_state_zp, int32_t* row_sums, int row_sums_size,
    bool* compute_row_sums, bool asymmetric_quantize_inputs,
    CpuBackendContext* context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_13(mht_13_v, 1243, "", "./tensorflow/lite/kernels/lstm_eval.cc", "LstmStepHybrid");

  ruy::profiler::ScopeLabel label("LstmStepHybrid");
  // Since we have already checked that weights are all there or none, we
  // can check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
  // Make named scratch buffers for the different gates.
  float* input_gate_scratch = scratch0;
  float* forget_gate_scratch = scratch1;
  float* cell_gate_scratch = scratch2;
  float* output_gate_scratch = scratch3;

  int32_t* input_to_input_row_sums = nullptr;
  int32_t* input_to_forget_row_sums = nullptr;
  int32_t* input_to_cell_row_sums = nullptr;
  int32_t* input_to_output_row_sums = nullptr;
  int32_t* aux_input_to_input_row_sums = nullptr;
  int32_t* aux_input_to_forget_row_sums = nullptr;
  int32_t* aux_input_to_cell_row_sums = nullptr;
  int32_t* aux_input_to_output_row_sums = nullptr;
  int32_t* recurrent_to_input_row_sums = nullptr;
  int32_t* recurrent_to_forget_row_sums = nullptr;
  int32_t* recurrent_to_cell_row_sums = nullptr;
  int32_t* recurrent_to_output_row_sums = nullptr;
  int32_t* projection_weights_row_sums = nullptr;

  if (asymmetric_quantize_inputs) {
    int num_row_sums = use_cifg ? 6 : 8;
    if (aux_input_ptr != nullptr) {
      num_row_sums += use_cifg ? 3 : 4;
    }
    if (projection_weights_ptr != nullptr) {
      num_row_sums += ceil(static_cast<float>(n_output) / n_cell);
    }
    TF_LITE_ASSERT(row_sums_size == num_row_sums);
    input_to_input_row_sums = row_sums;
    input_to_forget_row_sums =
        use_cifg ? input_to_input_row_sums : input_to_input_row_sums + n_cell;
    input_to_cell_row_sums = input_to_forget_row_sums + n_cell;
    input_to_output_row_sums = input_to_cell_row_sums + n_cell;
    if (aux_input_ptr != nullptr) {
      aux_input_to_input_row_sums = input_to_output_row_sums + n_cell;
      aux_input_to_forget_row_sums = use_cifg
                                         ? aux_input_to_input_row_sums
                                         : aux_input_to_input_row_sums + n_cell;
      aux_input_to_cell_row_sums = aux_input_to_forget_row_sums + n_cell;
      aux_input_to_output_row_sums = aux_input_to_cell_row_sums + n_cell;
    }
    recurrent_to_input_row_sums = aux_input_ptr
                                      ? aux_input_to_output_row_sums + n_cell
                                      : input_to_output_row_sums + n_cell;
    recurrent_to_forget_row_sums = use_cifg
                                       ? recurrent_to_input_row_sums
                                       : recurrent_to_input_row_sums + n_cell;
    recurrent_to_cell_row_sums = recurrent_to_forget_row_sums + n_cell;
    recurrent_to_output_row_sums = recurrent_to_cell_row_sums + n_cell;
    if (projection_weights_ptr != nullptr) {
      projection_weights_row_sums = recurrent_to_output_row_sums + n_cell;
    }
    if (*compute_row_sums) {
      ComputeRowSums(
          input_to_input_row_sums, input_to_forget_row_sums,
          input_to_cell_row_sums, input_to_output_row_sums,
          aux_input_to_input_row_sums, aux_input_to_forget_row_sums,
          aux_input_to_cell_row_sums, aux_input_to_output_row_sums,
          recurrent_to_input_row_sums, recurrent_to_forget_row_sums,
          recurrent_to_cell_row_sums, recurrent_to_output_row_sums,
          projection_weights_row_sums, row_sums, n_cell, n_input, n_aux_input,
          n_output, input_to_input_weights_ptr, input_to_forget_weights_ptr,
          input_to_cell_weights_ptr, input_to_output_weights_ptr,
          aux_input_to_input_weights_ptr, aux_input_to_forget_weights_ptr,
          aux_input_to_cell_weights_ptr, aux_input_to_output_weights_ptr,
          recurrent_to_input_weights_ptr, recurrent_to_forget_weights_ptr,
          recurrent_to_cell_weights_ptr, recurrent_to_output_weights_ptr,
          projection_weights_ptr, use_cifg, aux_input_ptr);
      *compute_row_sums = false;
    }
  }

  // Check if inputs are all zeros so we can skip some computations.
  const bool is_input_all_zeros =
      tensor_utils::IsZeroVector(input_ptr, n_batch * n_input);
  const bool is_aux_input_all_zeros =
      (aux_input_ptr == nullptr ||
       tensor_utils::IsZeroVector(aux_input_ptr, n_batch * n_aux_input));
  const bool is_output_state_all_zeros =
      tensor_utils::IsZeroVector(output_state_ptr, n_batch * n_output);
  // Quantize inputs.
  if (!is_input_all_zeros) {
    tensor_utils::BatchQuantizeFloats(input_ptr, n_batch, n_input,
                                      quantized_input_ptr, input_sf, input_zp,
                                      asymmetric_quantize_inputs);
  }
  if (!is_aux_input_all_zeros) {
    tensor_utils::BatchQuantizeFloats(aux_input_ptr, n_batch, n_aux_input,
                                      quantized_aux_input_ptr, aux_input_sf,
                                      aux_input_zp, asymmetric_quantize_inputs);
  }
  if (!is_output_state_all_zeros) {
    tensor_utils::BatchQuantizeFloats(
        output_state_ptr, n_batch, n_output, quantized_output_state_ptr,
        output_state_sf, output_state_zp, asymmetric_quantize_inputs);
  }
  if (!use_cifg) {
    // Calculate the input gate. (If not CIFG.)
    CalculateLstmGateHybrid(
        quantized_input_ptr, input_sf, input_zp, input_to_input_weights_ptr,
        input_to_input_weights_ledger_ptr, input_to_input_weights_scale,
        input_to_input_row_sums, quantized_aux_input_ptr, aux_input_sf,
        aux_input_zp, aux_input_to_input_weights_ptr,
        aux_input_to_input_weights_scale, aux_input_to_input_row_sums,
        quantized_output_state_ptr, output_state_sf, output_state_zp,
        recurrent_to_input_weights_ptr, recurrent_to_input_weights_ledger_ptr,
        recurrent_to_input_weights_scale, recurrent_to_input_row_sums,
        cell_state_ptr, cell_to_input_weights_ptr, cell_to_input_weights_scale,
        input_layer_norm_coefficients_ptr, input_gate_bias_ptr, n_batch,
        n_input, n_aux_input, n_output, n_cell, kTfLiteActSigmoid,
        input_gate_scratch, is_input_all_zeros, is_aux_input_all_zeros,
        is_output_state_all_zeros, compute_row_sums, context,
        scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr);
  }
  // Calculate the forget gate.
  CalculateLstmGateHybrid(
      quantized_input_ptr, input_sf, input_zp, input_to_forget_weights_ptr,
      input_to_forget_weights_ledger_ptr, input_to_forget_weights_scale,
      input_to_forget_row_sums, quantized_aux_input_ptr, aux_input_sf,
      aux_input_zp, aux_input_to_forget_weights_ptr,
      aux_input_to_forget_weights_scale, aux_input_to_forget_row_sums,
      quantized_output_state_ptr, output_state_sf, output_state_zp,
      recurrent_to_forget_weights_ptr, recurrent_to_forget_weights_ledger_ptr,
      recurrent_to_forget_weights_scale, recurrent_to_forget_row_sums,
      cell_state_ptr, cell_to_forget_weights_ptr, cell_to_forget_weights_scale,
      forget_layer_norm_coefficients_ptr, forget_gate_bias_ptr, n_batch,
      n_input, n_aux_input, n_output, n_cell, kTfLiteActSigmoid,
      forget_gate_scratch, is_input_all_zeros, is_aux_input_all_zeros,
      is_output_state_all_zeros, compute_row_sums, context,
      scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr);
  // Calculate the cell update gate.
  CalculateLstmGateHybrid(
      quantized_input_ptr, input_sf, input_zp, input_to_cell_weights_ptr,
      input_to_cell_weights_ledger_ptr, input_to_cell_weights_scale,
      input_to_cell_row_sums, quantized_aux_input_ptr, aux_input_sf,
      aux_input_zp, aux_input_to_cell_weights_ptr,
      aux_input_to_cell_weights_scale, aux_input_to_cell_row_sums,
      quantized_output_state_ptr, output_state_sf, output_state_zp,
      recurrent_to_cell_weights_ptr, recurrent_to_cell_weights_ledger_ptr,
      recurrent_to_cell_weights_scale, recurrent_to_cell_row_sums,
      /*cell_state=*/nullptr, /*cell_to_gate_weights=*/nullptr,
      /*cell_to_gate_weights_scale=*/0.0f, cell_layer_norm_coefficients_ptr,
      cell_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
      params->activation, cell_gate_scratch, is_input_all_zeros,
      is_aux_input_all_zeros, is_output_state_all_zeros, compute_row_sums,
      context, scaling_factors_scratch, recovered_cell_weights,
      accum_scratch_ptr);
  // Update the cell state.
  UpdateLstmCellFloat(n_batch, n_cell, cell_state_ptr, input_gate_scratch,
                      forget_gate_scratch, cell_gate_scratch, use_cifg,
                      params->cell_clip);
  // Calculate the output gate.
  CalculateLstmGateHybrid(
      quantized_input_ptr, input_sf, input_zp, input_to_output_weights_ptr,
      input_to_output_weights_ledger_ptr, input_to_output_weights_scale,
      input_to_output_row_sums, quantized_aux_input_ptr, aux_input_sf,
      aux_input_zp, aux_input_to_output_weights_ptr,
      aux_input_to_output_weights_scale, aux_input_to_output_row_sums,
      quantized_output_state_ptr, output_state_sf, output_state_zp,
      recurrent_to_output_weights_ptr, recurrent_to_output_weights_ledger_ptr,
      recurrent_to_output_weights_scale, recurrent_to_output_row_sums,
      cell_state_ptr, cell_to_output_weights_ptr, cell_to_output_weights_scale,
      output_layer_norm_coefficients_ptr, output_gate_bias_ptr, n_batch,
      n_input, n_aux_input, n_output, n_cell, kTfLiteActSigmoid,
      output_gate_scratch, is_input_all_zeros, is_aux_input_all_zeros,
      is_output_state_all_zeros, compute_row_sums, context,
      scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr);
  // Update the output state.
  CalculateLstmOutputHybrid(
      n_batch, n_cell, n_output, cell_state_ptr, output_gate_scratch,
      params->activation, projection_weights_ptr, projection_weights_ledger_ptr,
      projection_weights_scale, projection_bias_ptr, params->proj_clip,
      output_state_ptr, asymmetric_quantize_inputs, projection_weights_row_sums,
      compute_row_sums, context, scratch2, quantized_output_scratch, input_sf,
      input_zp, accum_scratch_ptr);
  // Copy output state to the output. Note that the output's rows may not be
  // contiguous (output_batch_leading_dim != n_output).
  for (int b = 0; b < n_batch; b++) {
    std::copy_n(output_state_ptr + b * n_output, n_output,
                output_ptr + b * output_batch_leading_dim);
  }
}

// Fully quantized lstm kernel for 16 bit gate matmul output.
//
// Input tensor of size n_batch * n_input:
//   input_ptr
//
// LSTM weights:
// Quantized input weights of size 'n_cell * n_input':
//   input_to_input_weight_ptr            - optional
//   input_to_forget_weight_ptr           - optional
//   input_to_cell_weight_ptr             - optional
//   input_to_output_weight_ptr           - optional
//
// Quantized recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weight_ptr        - optional
//   recurrent_to_forget_weights_ptr
//   recurrent_to_cell_weights_ptr
//   recurrent_to_input_weights_ptr
//
// Quantized peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights               - optional
//   cell_to_cell_weights                - optional
//   cell_to_output_weights              - optional
//
// Quantized projection weights of size 'n_output * n_cell'
//   projection_weight_ptr                     - optional
//
// Weight scales (scalars) for each of the weights above.
//   effective_input_to_input_scale_a    - optional
//   effective_input_to_input_scale_b    - optional
//   effective_input_to_forget_scale_a
//   effective_input_to_forget_scale_b
//   effective_input_to_cell_scale_a
//   effective_input_to_cell_scale_b
//   effective_input_to_output_scale_a
//   effective_input_to_output_scale_b
//   effective_recurrent_to_input_scale_a    - optional
//   effective_recurrent_to_input_scale_b    - optional
//   effective_recurrent_to_forget_scale_a
//   effective_recurrent_to_forget_scale_b
//   effective_recurrent_to_cell_scale_a
//   effective_recurrent_to_cell_scale_b
//   effective_recurrent_to_output_scale_a
//   effective_recurrent_to_output_scale_b
//   effective_proj_scale_a                  - optional
//   effective_proj_scale_b                  - optional
//
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr                 - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   layer_norm_input_weight_ptr    - optional
//   layer_norm_forget_weight_ptr   - optional
//   layer_norm_cell_weight_ptr     - optional
//   layer_norm_output_weight_ptr   - optional
//
// Layer norm scales of size 'n_cell'.
//   layer_norm_input_scale_a     - optional
//   layer_norm_input_scale_b     - optional
//   layer_norm_forget_scale_a    - optional
//   layer_norm_forget_scale_b    - optional
//   layer_norm_cell_scale_a      - optional
//   layer_norm_cell_scale_b      - optional
//   layer_norm_output_scale_a    - optional
//   layer_norm_output_scale_b    - optional
//
// Scalar values:
//   quantized_cell_clip: quantized clip value for cell.
//   quantized_proj_clip: quantized clip value for projection.
//   cell_state_scale: the power of two scale for cell state.
//
// Zero points:
//   output_state_zp: zero point of output state
//   hidden_zp: zero point for hidden state.
//
// Temporary pre-allocated storage for the calculation. Each is of size n_cell *
// n_batch.
//   scratch0
//   scratch1
//   scratch2
//   scratch3
//   scratch4
//   scratch5: this scratch buffer is created purely for optimizing the
//              MatrixBatchVectorMultiplyAccumulate.
//
// Outputs:
//   output_state_ptr - size 'n_batch * n_output'
//   cell_state_ptr   - size 'n_batch * n_cell'
//   output_ptr       - size 'n_batch * n_output'
// TODO(b/159947023): scratch0 is not used if (!cifg). Don't allocate then.
inline void LstmStepInteger8x8_16(
    const int8_t* input_ptr, const int8_t* input_to_input_weight_ptr,
    int32_t effective_input_to_input_scale_a,
    int32_t effective_input_to_input_scale_b,
    const int8_t* input_to_forget_weight_ptr,
    int32_t effective_input_to_forget_scale_a,
    int32_t effective_input_to_forget_scale_b,
    const int8_t* input_to_cell_weight_ptr,
    int32_t effective_input_to_cell_scale_a,
    int32_t effective_input_to_cell_scale_b,
    const int8_t* input_to_output_weight_ptr,
    int32_t effective_input_to_output_scale_a,
    int32_t effective_input_to_output_scale_b,
    const int8_t* recurrent_to_input_weight_ptr,
    int32_t effective_recurrent_to_input_scale_a,
    int32_t effective_recurrent_to_input_scale_b,
    const int8_t* recurrent_to_forget_weight_ptr,
    int32_t effective_recurrent_to_forget_scale_a,
    int32_t effective_recurrent_to_forget_scale_b,
    const int8_t* recurrent_to_cell_weight_ptr,
    int32_t effective_recurrent_to_cell_scale_a,
    int32_t effective_recurrent_to_cell_scale_b,
    const int8_t* recurrent_to_output_weight_ptr,
    int32_t effective_recurrent_to_output_scale_a,
    int32_t effective_recurrent_to_output_scale_b,
    const int16_t* cell_to_input_weight_ptr,
    int32_t effective_cell_to_input_scale_a,
    int32_t effective_cell_to_input_scale_b,
    const int16_t* cell_to_forget_weight_ptr,
    int32_t effective_cell_to_forget_scale_a,
    int32_t effective_cell_to_forget_scale_b,
    const int16_t* cell_to_output_weight_ptr,
    int32_t effective_cell_to_output_scale_a,
    int32_t effective_cell_to_output_scale_b,
    const int8_t* projection_weight_ptr, int32_t effective_proj_scale_a,
    int32_t effective_proj_scale_b, int32_t hidden_zp,
    int32_t effective_hidden_scale_a, int32_t effective_hidden_scale_b,
    const int16_t* layer_norm_input_weight_ptr,
    int32_t layer_norm_input_scale_a, int32_t layer_norm_input_scale_b,
    const int16_t* layer_norm_forget_weight_ptr,
    int32_t layer_norm_forget_scale_a, int32_t layer_norm_forget_scale_b,
    const int16_t* layer_norm_cell_weight_ptr, int32_t layer_norm_cell_scale_a,
    int32_t layer_norm_cell_scale_b,
    const int16_t* layer_norm_output_weight_ptr,
    int32_t layer_norm_output_scale_a, int32_t layer_norm_output_scale_b,
    const int32_t* input_gate_bias_ptr, const int32_t* forget_gate_bias_ptr,
    const int32_t* cell_gate_bias_ptr, const int32_t* output_gate_bias_ptr,
    int16_t quantized_cell_clip, int8_t quantized_proj_clip,
    int32_t cell_state_scale, int32_t input_variance_guard,
    int32_t forget_variance_guard, int32_t cell_variance_guard,
    int32_t output_variance_guard,
    const int32_t* input_to_forget_effective_bias,
    const int32_t* recurrent_to_forget_effective_bias,
    const int32_t* input_to_cell_effective_bias,
    const int32_t* recurrent_to_cell_effective_bias,
    const int32_t* input_to_output_effective_bias,
    const int32_t* recurrent_to_output_effective_bias,
    const int32_t* input_to_input_effective_bias,
    const int32_t* recurrent_to_input_effective_bias,
    const int32_t* projection_effective_bias, int n_batch, int n_cell,
    int n_input, int n_output, int8_t* output_state_ptr,
    int32_t output_state_zp, int16_t* cell_state_ptr, int8_t* output_ptr,
    int16_t* scratch0, int16_t* scratch1, int16_t* scratch2, int16_t* scratch3,
    int8_t* scratch4, int32_t* scratch5, CpuBackendContext* context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_14(mht_14_v, 1590, "", "./tensorflow/lite/kernels/lstm_eval.cc", "LstmStepInteger8x8_16");

  ruy::profiler::ScopeLabel label("LstmStepInteger8x8_16");
  // Make named scratch buffers for the different gates.
  int16_t* input_gate_scratch = scratch0;
  int16_t* forget_gate_scratch = scratch1;
  int16_t* cell_gate_scratch = scratch2;
  int16_t* output_gate_scratch = scratch3;

  // Since we have already checked that weights are all there or none, we
  // can check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weight_ptr == nullptr);

  // Check for nullptrs.
  TFLITE_DCHECK(input_to_forget_effective_bias);
  TFLITE_DCHECK(recurrent_to_forget_effective_bias);
  TFLITE_DCHECK(input_to_cell_effective_bias);
  TFLITE_DCHECK(recurrent_to_cell_effective_bias);
  TFLITE_DCHECK(input_to_output_effective_bias);
  TFLITE_DCHECK(recurrent_to_output_effective_bias);
  if (!use_cifg) {
    TFLITE_DCHECK(input_to_input_effective_bias);
    TFLITE_DCHECK(recurrent_to_input_effective_bias);
  }
  const bool use_projection = (projection_weight_ptr != nullptr);
  if (use_projection) {
    TFLITE_DCHECK(projection_effective_bias);
  }
  if (!use_cifg) {
    // Calculate the input gate. (If not CIFG.)
    CalculateLstmGateInteger8x8_16(
        input_ptr, input_to_input_weight_ptr, input_to_input_effective_bias,
        effective_input_to_input_scale_a, effective_input_to_input_scale_b,
        output_state_ptr, recurrent_to_input_weight_ptr,
        recurrent_to_input_effective_bias, effective_recurrent_to_input_scale_a,
        effective_recurrent_to_input_scale_b, cell_state_ptr,
        cell_to_input_weight_ptr, effective_cell_to_input_scale_a,
        effective_cell_to_input_scale_b, layer_norm_input_weight_ptr,
        input_gate_bias_ptr, layer_norm_input_scale_a, layer_norm_input_scale_b,
        input_variance_guard, n_batch, n_input, n_output, n_cell,
        kTfLiteActSigmoid, input_gate_scratch, context, scratch5);
  }
  // Calculate the forget gate.
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_forget_weight_ptr, input_to_forget_effective_bias,
      effective_input_to_forget_scale_a, effective_input_to_forget_scale_b,
      output_state_ptr, recurrent_to_forget_weight_ptr,
      recurrent_to_forget_effective_bias, effective_recurrent_to_forget_scale_a,
      effective_recurrent_to_forget_scale_b, cell_state_ptr,
      cell_to_forget_weight_ptr, effective_cell_to_forget_scale_a,
      effective_cell_to_forget_scale_b, layer_norm_forget_weight_ptr,
      forget_gate_bias_ptr, layer_norm_forget_scale_a,
      layer_norm_forget_scale_b, forget_variance_guard, n_batch, n_input,
      n_output, n_cell, kTfLiteActSigmoid, forget_gate_scratch, context,
      scratch5);
  // Calculate the cell update gate.
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_cell_weight_ptr, input_to_cell_effective_bias,
      effective_input_to_cell_scale_a, effective_input_to_cell_scale_b,
      output_state_ptr, recurrent_to_cell_weight_ptr,
      recurrent_to_cell_effective_bias, effective_recurrent_to_cell_scale_a,
      effective_recurrent_to_cell_scale_b, cell_state_ptr,
      /*cell_to_gate_weights=*/nullptr, /*cell_to_gate_scale_a=*/0,
      /*cell_to_gate_scale_b=*/0, layer_norm_cell_weight_ptr,
      cell_gate_bias_ptr, layer_norm_cell_scale_a, layer_norm_cell_scale_b,
      cell_variance_guard, n_batch, n_input, n_output, n_cell, kTfLiteActTanh,
      cell_gate_scratch, context, scratch5);
  // Update the cell state.
  UpdateLstmCellInteger(n_batch, n_cell, cell_state_ptr, cell_state_scale,
                        input_gate_scratch, forget_gate_scratch,
                        cell_gate_scratch, use_cifg, quantized_cell_clip);
  // Calculate the output gate.
  CalculateLstmGateInteger8x8_16(
      input_ptr, input_to_output_weight_ptr, input_to_output_effective_bias,
      effective_input_to_output_scale_a, effective_input_to_output_scale_b,
      output_state_ptr, recurrent_to_output_weight_ptr,
      recurrent_to_output_effective_bias, effective_recurrent_to_output_scale_a,
      effective_recurrent_to_output_scale_b, cell_state_ptr,
      cell_to_output_weight_ptr, effective_cell_to_output_scale_a,
      effective_cell_to_output_scale_b, layer_norm_output_weight_ptr,
      output_gate_bias_ptr, layer_norm_output_scale_a,
      layer_norm_output_scale_b, output_variance_guard, n_batch, n_input,
      n_output, n_cell, kTfLiteActSigmoid, output_gate_scratch, context,
      scratch5);
  // Update the output state.
  CalculateLstmOutputInteger8x8_16(
      n_batch, n_cell, n_output, cell_state_ptr, cell_state_scale,
      output_gate_scratch, effective_hidden_scale_a, effective_hidden_scale_b,
      hidden_zp, projection_weight_ptr, effective_proj_scale_a,
      effective_proj_scale_b, projection_effective_bias, output_state_zp,
      quantized_proj_clip, output_state_ptr, context, scratch0, scratch4,
      scratch5);
  // Copy output state to the output. Note that unlike float or hybrid, output
  // is always contiguous.
  std::copy_n(output_state_ptr, n_batch * n_output, output_ptr);
}

// Fully quantized lstm kernel for 8 bit gate matmul output.
//
// Input tensor of size n_batch * n_input:
//   input_ptr
//
// LSTM weights:
// Quantized input weights of size 'n_cell * n_input':
//   input_to_input_weight_ptr            - optional
//   input_to_forget_weight_ptr           - optional
//   input_to_cell_weight_ptr             - optional
//   input_to_output_weight_ptr           - optional
//
// Quantized recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weight_ptr        - optional
//   recurrent_to_forget_weights_ptr
//   recurrent_to_cell_weights_ptr
//   recurrent_to_input_weights_ptr
//
// Quantized peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights               - optional
//   cell_to_cell_weights                - optional
//   cell_to_output_weights              - optional
//
// Quantized projection weights of size 'n_output * n_cell'
//   projection_weight_ptr                     - optional
//
// Weight scales (scalars) for each of the weights above.
//   effective_input_to_input_scale_a    - optional
//   effective_input_to_input_scale_b    - optional
//   effective_input_to_forget_scale_a
//   effective_input_to_forget_scale_b
//   effective_input_to_cell_scale_a
//   effective_input_to_cell_scale_b
//   effective_input_to_output_scale_a
//   effective_input_to_output_scale_b
//   effective_recurrent_to_input_scale_a    - optional
//   effective_recurrent_to_input_scale_b    - optional
//   effective_recurrent_to_forget_scale_a
//   effective_recurrent_to_forget_scale_b
//   effective_recurrent_to_cell_scale_a
//   effective_recurrent_to_cell_scale_b
//   effective_recurrent_to_output_scale_a
//   effective_recurrent_to_output_scale_b
//   effective_proj_scale_a                  - optional
//   effective_proj_scale_b                  - optional
//
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr                 - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   layer_norm_input_weight_ptr    - optional
//   layer_norm_forget_weight_ptr   - optional
//   layer_norm_cell_weight_ptr     - optional
//   layer_norm_output_weight_ptr   - optional
//
// Layer norm scales of size 'n_cell'.
//   layer_norm_input_scale_a     - optional
//   layer_norm_input_scale_b     - optional
//   layer_norm_forget_scale_a    - optional
//   layer_norm_forget_scale_b    - optional
//   layer_norm_cell_scale_a      - optional
//   layer_norm_cell_scale_b      - optional
//   layer_norm_output_scale_a    - optional
//   layer_norm_output_scale_b    - optional
//
// Scalar values:
//   quantized_cell_clip: quantized clip value for cell.
//   quantized_proj_clip: quantized clip value for projection.
//   cell_state_scale: the power of two scale for cell state.
//
// Zero points:
//   input_zp: zero point for input tensor.
//   output_state_zp: zero point of output state.
//   hidden_zp: zero point for hidden state.
//
// Temporary pre-allocated storage for the calculation. Each is of size n_cell *
// n_batch.
//   scratch0
//   scratch1
//   scratch2
//   scratch3
//   scratch4
//   scratch5
//   scratch6
//   scratch7
//
// Outputs:
//   output_state_ptr - size 'n_batch * n_output'
//   cell_state_ptr   - size 'n_batch * n_cell'
//   output_ptr       - size 'n_batch * n_output'
//
// Can move zero point calculation into Prepare() for better perfomance.
// TODO(b/159947023): scratch5 is unused, remove.
inline void LstmStepInteger8x8_8(
    const int8_t* input_ptr, int32_t input_zp,
    const int8_t* input_to_input_weight_ptr,
    int32_t effective_input_to_input_scale_a,
    int32_t effective_input_to_input_scale_b,
    const int8_t* input_to_forget_weight_ptr,
    int32_t effective_input_to_forget_scale_a,
    int32_t effective_input_to_forget_scale_b,
    const int8_t* input_to_cell_weight_ptr,
    int32_t effective_input_to_cell_scale_a,
    int32_t effective_input_to_cell_scale_b,
    const int8_t* input_to_output_weight_ptr,
    int32_t effective_input_to_output_scale_a,
    int32_t effective_input_to_output_scale_b,
    const int8_t* recurrent_to_input_weight_ptr,
    int32_t effective_recurrent_to_input_scale_a,
    int32_t effective_recurrent_to_input_scale_b,
    const int8_t* recurrent_to_forget_weight_ptr,
    int32_t effective_recurrent_to_forget_scale_a,
    int32_t effective_recurrent_to_forget_scale_b,
    const int8_t* recurrent_to_cell_weight_ptr,
    int32_t effective_recurrent_to_cell_scale_a,
    int32_t effective_recurrent_to_cell_scale_b,
    const int8_t* recurrent_to_output_weight_ptr,
    int32_t effective_recurrent_to_output_scale_a,
    int32_t effective_recurrent_to_output_scale_b,
    const int8_t* cell_to_input_weight_ptr,
    int32_t effective_cell_to_input_scale_a,
    int32_t effective_cell_to_input_scale_b,
    const int8_t* cell_to_forget_weight_ptr,
    int32_t effective_cell_to_forget_scale_a,
    int32_t effective_cell_to_forget_scale_b,
    const int8_t* cell_to_output_weight_ptr,
    int32_t effective_cell_to_output_scale_a,
    int32_t effective_cell_to_output_scale_b,
    const int8_t* projection_weight_ptr, int32_t effective_proj_scale_a,
    int32_t effective_proj_scale_b, const int16_t* layer_norm_input_weight_ptr,
    int32_t layer_norm_input_scale_a, int32_t layer_norm_input_scale_b,
    const int16_t* layer_norm_forget_weight_ptr,
    int32_t layer_norm_forget_scale_a, int32_t layer_norm_forget_scale_b,
    const int16_t* layer_norm_cell_weight_ptr, int32_t layer_norm_cell_scale_a,
    int32_t layer_norm_cell_scale_b,
    const int16_t* layer_norm_output_weight_ptr,
    int32_t layer_norm_output_scale_a, int32_t layer_norm_output_scale_b,
    const int32_t* input_gate_bias_ptr, const int32_t* forget_gate_bias_ptr,
    const int32_t* cell_gate_bias_ptr, const int32_t* output_gate_bias_ptr,
    const int32_t* projection_bias_ptr, const TfLiteLSTMParams* params,
    const int32_t* intermediate_scale_a, const int32_t* intermediate_scale_b,
    const int32_t* intermediate_zp, int16_t quantized_cell_clip,
    int8_t quantized_proj_clip, int n_batch, int n_cell, int n_input,
    int n_output, int output_batch_leading_dim, int8_t* output_state_ptr,
    int32_t output_state_zp, int16_t* cell_state_ptr, int8_t* output_ptr,
    int8_t* scratch0, int8_t* scratch1, int16_t* scratch2, int16_t* scratch3,
    int16_t* scratch4, int16_t* scratch5, int16_t* scratch6,
    int16_t* scratch7) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_15(mht_15_v, 1839, "", "./tensorflow/lite/kernels/lstm_eval.cc", "LstmStepInteger8x8_8");

  // TODO(b/159066113): scratch5 is unused, remove.

  ruy::profiler::ScopeLabel label("LstmStepInteger8x8_8");
  // Make named scratch buffers for the different gates.
  int16_t* forget_gate_scratch = scratch2;
  int16_t* cell_gate_scratch = scratch3;
  int16_t* output_gate_scratch = scratch4;
  // no-CIFG is not supported here

  // Calculate the forget gate.
  CalculateLstmGateInteger8x8_8(
      input_ptr, input_zp, input_to_forget_weight_ptr,
      effective_input_to_forget_scale_a, effective_input_to_forget_scale_b,
      intermediate_scale_a[2], intermediate_scale_b[2], intermediate_zp[4],
      output_state_ptr, output_state_zp, recurrent_to_forget_weight_ptr,
      effective_recurrent_to_forget_scale_a,
      effective_recurrent_to_forget_scale_b, intermediate_scale_a[3],
      intermediate_scale_b[3], intermediate_zp[5], layer_norm_forget_weight_ptr,
      layer_norm_forget_scale_a, layer_norm_forget_scale_b,
      forget_gate_bias_ptr, n_batch, n_input, n_output, n_cell,
      kTfLiteActSigmoid, forget_gate_scratch, scratch0, scratch1);
  // Calculate the cell update gate.
  CalculateLstmGateInteger8x8_8(
      input_ptr, input_zp, input_to_cell_weight_ptr,
      effective_input_to_cell_scale_a, effective_input_to_cell_scale_b,
      intermediate_scale_a[4], intermediate_scale_b[4], intermediate_zp[7],
      output_state_ptr, output_state_zp, recurrent_to_cell_weight_ptr,
      effective_recurrent_to_cell_scale_a, effective_recurrent_to_cell_scale_b,
      intermediate_scale_a[5], intermediate_scale_b[5], intermediate_zp[8],
      layer_norm_cell_weight_ptr, layer_norm_cell_scale_a,
      layer_norm_cell_scale_b, cell_gate_bias_ptr, n_batch, n_input, n_output,
      n_cell, kTfLiteActTanh, cell_gate_scratch, scratch0, scratch1);
  // Update the cell state.
  UpdateLstmCellInteger(n_batch, n_cell, cell_state_ptr,
                        /*cell_state_scale=*/-15, /*input_gate=*/nullptr,
                        forget_gate_scratch, cell_gate_scratch,
                        /*use_cifg=*/true, quantized_cell_clip);
  // Calculate the output gate.
  CalculateLstmGateInteger8x8_8(
      input_ptr, input_zp, input_to_output_weight_ptr,
      effective_input_to_output_scale_a, effective_input_to_output_scale_b,
      intermediate_scale_a[6], intermediate_scale_b[6], intermediate_zp[10],
      output_state_ptr, output_state_zp, recurrent_to_output_weight_ptr,
      effective_recurrent_to_output_scale_a,
      effective_recurrent_to_output_scale_b, intermediate_scale_a[11],
      intermediate_scale_b[7], intermediate_zp[7], layer_norm_output_weight_ptr,
      layer_norm_output_scale_a, layer_norm_output_scale_b,
      output_gate_bias_ptr, n_batch, n_input, n_output, n_cell,
      kTfLiteActSigmoid, output_gate_scratch, scratch0, scratch1);
  // Update the output state.
  CalculateLstmOutputInteger8x8_8(
      n_batch, n_cell, n_output, cell_state_ptr, output_gate_scratch,
      projection_weight_ptr, effective_proj_scale_a, effective_proj_scale_b,
      projection_bias_ptr, output_state_zp, quantized_proj_clip,
      output_state_ptr, scratch2);
  // Copy output state to the output. Note that unlike float or hybrid, output
  // is always contigous.
  std::copy_n(output_state_ptr, n_batch * n_output, output_ptr);
}

}  // namespace

// LINT.IfChange
TfLiteStatus EvalFloat(
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
    TfLiteTensor* cell_state, TfLiteTensor* output) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_16(mht_16_v, 1932, "", "./tensorflow/lite/kernels/lstm_eval.cc", "EvalFloat");

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
      float* output_ptr =
          GetTensorData<float>(output) + t_rel * output_step + output_offset;

      LstmStepFloat(
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
          output_gate_scratch, output_ptr);
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

        LstmStepFloat(
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
            output_gate_scratch_ptr, output_ptr);
      }
    }
  }
  return kTfLiteOk;
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/calibration/builtin_logging_ops/lstm.cc)

TfLiteStatus EvalHybrid(
    const TfLiteTensor* input, const TfLiteTensor* input_to_input_weights,
    const TfLiteTensor* input_to_input_weights_ledger,
    const TfLiteTensor* input_to_forget_weights,
    const TfLiteTensor* input_to_forget_weights_ledger,
    const TfLiteTensor* input_to_cell_weights,
    const TfLiteTensor* input_to_cell_weights_ledger,
    const TfLiteTensor* input_to_output_weights,
    const TfLiteTensor* input_to_output_weights_ledger,
    const TfLiteTensor* recurrent_to_input_weights,
    const TfLiteTensor* recurrent_to_input_weights_ledger,
    const TfLiteTensor* recurrent_to_forget_weights,
    const TfLiteTensor* recurrent_to_forget_weights_ledger,
    const TfLiteTensor* recurrent_to_cell_weights,
    const TfLiteTensor* recurrent_to_cell_weights_ledger,
    const TfLiteTensor* recurrent_to_output_weights,
    const TfLiteTensor* recurrent_to_output_weights_ledger,
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
    const TfLiteTensor* projection_weights,
    const TfLiteTensor* projection_weights_ledger,
    const TfLiteTensor* projection_bias, const TfLiteLSTMParams* params,
    bool forward_sequence, bool time_major, int output_offset,
    TfLiteTensor* scratch_buffer, TfLiteTensor* input_sf,
    TfLiteTensor* aux_input_sf, TfLiteTensor* output_state_sf,
    TfLiteTensor* prod_scaling_factors, TfLiteTensor* recovered_cell_weights,
    TfLiteTensor* input_quantized, TfLiteTensor* aux_input_quantized,
    TfLiteTensor* output_state_quantized, TfLiteTensor* cell_state_quantized,
    TfLiteTensor* output_state, TfLiteTensor* cell_state,
    TfLiteTensor* output_scratch_buffer, TfLiteTensor* output,
    TfLiteTensor* input_zp, TfLiteTensor* aux_input_zp,
    TfLiteTensor* output_state_zp, TfLiteTensor* row_sums, int row_sums_size,
    bool* compute_row_sums, CpuBackendContext* context) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_17(mht_17_v, 2134, "", "./tensorflow/lite/kernels/lstm_eval.cc", "EvalHybrid");

  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  const int n_input = input->dims->data[input->dims->size - 1];
  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  }
  const int aux_input_size =
      (aux_input) ? aux_input->dims->data[aux_input->dims->size - 1] : 0;
  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to get the condition.
  const bool use_cifg = (input_to_input_weights == nullptr);

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

  int32_t* input_zp_ptr = nullptr;
  int32_t* aux_input_zp_ptr = nullptr;
  int32_t* output_state_zp_ptr = nullptr;
  int32_t* row_sums_ptr = nullptr;
  if (params->asymmetric_quantize_inputs) {
    input_zp_ptr = GetTensorData<int32_t>(input_zp);
    aux_input_zp_ptr = GetTensorData<int32_t>(aux_input_zp);
    output_state_zp_ptr = GetTensorData<int32_t>(output_state_zp);
    row_sums_ptr = GetTensorData<int32_t>(row_sums);
  }

  if (time_major) {
    // Feed the sequence into the LSTM step-by-step.
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
      float* output_ptr =
          GetTensorData<float>(output) + t_rel * output_step + output_offset;
      LstmStepHybrid(
          input_ptr, GetTensorData<int8_t>(input_to_input_weights),
          GetTensorData<uint8_t>(input_to_input_weights_ledger),
          GetTensorScale(input_to_input_weights),
          GetTensorData<int8_t>(input_to_forget_weights),
          GetTensorData<uint8_t>(input_to_forget_weights_ledger),
          GetTensorScale(input_to_forget_weights),
          GetTensorData<int8_t>(input_to_cell_weights),
          GetTensorData<uint8_t>(input_to_cell_weights_ledger),
          GetTensorScale(input_to_cell_weights),
          GetTensorData<int8_t>(input_to_output_weights),
          GetTensorData<uint8_t>(input_to_output_weights_ledger),
          GetTensorScale(input_to_output_weights), aux_input_ptr,
          GetTensorData<int8_t>(aux_input_to_input_weights),
          GetTensorScale(aux_input_to_input_weights),
          GetTensorData<int8_t>(aux_input_to_forget_weights),
          GetTensorScale(aux_input_to_forget_weights),
          GetTensorData<int8_t>(aux_input_to_cell_weights),
          GetTensorScale(aux_input_to_cell_weights),
          GetTensorData<int8_t>(aux_input_to_output_weights),
          GetTensorScale(aux_input_to_output_weights),
          GetTensorData<int8_t>(recurrent_to_input_weights),
          GetTensorData<uint8_t>(recurrent_to_input_weights_ledger),
          GetTensorScale(recurrent_to_input_weights),
          GetTensorData<int8_t>(recurrent_to_forget_weights),
          GetTensorData<uint8_t>(recurrent_to_forget_weights_ledger),
          GetTensorScale(recurrent_to_forget_weights),
          GetTensorData<int8_t>(recurrent_to_cell_weights),
          GetTensorData<uint8_t>(recurrent_to_cell_weights_ledger),
          GetTensorScale(recurrent_to_cell_weights),
          GetTensorData<int8_t>(recurrent_to_output_weights),
          GetTensorData<uint8_t>(recurrent_to_output_weights_ledger),
          GetTensorScale(recurrent_to_output_weights),
          GetTensorData<int8_t>(cell_to_input_weights),
          GetTensorScale(cell_to_input_weights),
          GetTensorData<int8_t>(cell_to_forget_weights),
          GetTensorScale(cell_to_forget_weights),
          GetTensorData<int8_t>(cell_to_output_weights),
          GetTensorScale(cell_to_output_weights),
          GetTensorData<float>(input_layer_norm_coefficients),
          GetTensorData<float>(forget_layer_norm_coefficients),
          GetTensorData<float>(cell_layer_norm_coefficients),
          GetTensorData<float>(output_layer_norm_coefficients),
          GetTensorData<float>(input_gate_bias),
          GetTensorData<float>(forget_gate_bias),
          GetTensorData<float>(cell_gate_bias),
          GetTensorData<float>(output_gate_bias),
          GetTensorData<int8_t>(projection_weights),
          GetTensorData<uint8_t>(projection_weights_ledger),
          GetTensorScale(projection_weights),
          GetTensorData<float>(projection_bias), params, n_batch, n_cell,
          n_input, aux_input_size, n_output, output_batch_leading_dim,
          input_gate_scratch, forget_gate_scratch, cell_gate_scratch,
          output_gate_scratch, GetTensorData<float>(input_sf),
          GetTensorData<float>(aux_input_sf),
          GetTensorData<float>(output_state_sf),
          GetTensorData<float>(prod_scaling_factors),
          GetTensorData<float>(recovered_cell_weights),
          GetTensorData<int8_t>(input_quantized),
          GetTensorData<int8_t>(aux_input_quantized),
          GetTensorData<int8_t>(output_state_quantized),
          GetTensorData<int8_t>(cell_state_quantized),
          GetTensorData<float>(output_state), GetTensorData<float>(cell_state),
          GetTensorData<int32_t>(output_scratch_buffer), output_ptr,
          input_zp_ptr, aux_input_zp_ptr, output_state_zp_ptr, row_sums_ptr,
          row_sums_size, compute_row_sums, params->asymmetric_quantize_inputs,
          context);
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

        LstmStepHybrid(
            input_ptr, GetTensorData<int8_t>(input_to_input_weights),
            GetTensorData<uint8_t>(input_to_input_weights_ledger),
            GetTensorScale(input_to_input_weights),
            GetTensorData<int8_t>(input_to_forget_weights),
            GetTensorData<uint8_t>(input_to_forget_weights_ledger),
            GetTensorScale(input_to_forget_weights),
            GetTensorData<int8_t>(input_to_cell_weights),
            GetTensorData<uint8_t>(input_to_cell_weights_ledger),
            GetTensorScale(input_to_cell_weights),
            GetTensorData<int8_t>(input_to_output_weights),
            GetTensorData<uint8_t>(input_to_output_weights_ledger),
            GetTensorScale(input_to_output_weights), aux_input_ptr,
            GetTensorData<int8_t>(aux_input_to_input_weights),
            GetTensorScale(aux_input_to_input_weights),
            GetTensorData<int8_t>(aux_input_to_forget_weights),
            GetTensorScale(aux_input_to_forget_weights),
            GetTensorData<int8_t>(aux_input_to_cell_weights),
            GetTensorScale(aux_input_to_cell_weights),
            GetTensorData<int8_t>(aux_input_to_output_weights),
            GetTensorScale(aux_input_to_output_weights),
            GetTensorData<int8_t>(recurrent_to_input_weights),
            GetTensorData<uint8_t>(recurrent_to_input_weights_ledger),
            GetTensorScale(recurrent_to_input_weights),
            GetTensorData<int8_t>(recurrent_to_forget_weights),
            GetTensorData<uint8_t>(recurrent_to_forget_weights_ledger),
            GetTensorScale(recurrent_to_forget_weights),
            GetTensorData<int8_t>(recurrent_to_cell_weights),
            GetTensorData<uint8_t>(recurrent_to_cell_weights_ledger),
            GetTensorScale(recurrent_to_cell_weights),
            GetTensorData<int8_t>(recurrent_to_output_weights),
            GetTensorData<uint8_t>(recurrent_to_output_weights_ledger),
            GetTensorScale(recurrent_to_output_weights),
            GetTensorData<int8_t>(cell_to_input_weights),
            GetTensorScale(cell_to_input_weights),
            GetTensorData<int8_t>(cell_to_forget_weights),
            GetTensorScale(cell_to_forget_weights),
            GetTensorData<int8_t>(cell_to_output_weights),
            GetTensorScale(cell_to_output_weights),
            GetTensorData<float>(input_layer_norm_coefficients),
            GetTensorData<float>(forget_layer_norm_coefficients),
            GetTensorData<float>(cell_layer_norm_coefficients),
            GetTensorData<float>(output_layer_norm_coefficients),
            GetTensorData<float>(input_gate_bias),
            GetTensorData<float>(forget_gate_bias),
            GetTensorData<float>(cell_gate_bias),
            GetTensorData<float>(output_gate_bias),
            GetTensorData<int8_t>(projection_weights),
            GetTensorData<uint8_t>(projection_weights_ledger),
            GetTensorScale(projection_weights),
            GetTensorData<float>(projection_bias), params,
            /*n_batch=*/1, n_cell, n_input, aux_input_size, n_output,
            output_batch_leading_dim, input_gate_scratch_ptr,
            forget_gate_scratch_ptr, cell_gate_scratch_ptr,
            output_gate_scratch_ptr, GetTensorData<float>(input_sf),
            GetTensorData<float>(aux_input_sf),
            GetTensorData<float>(output_state_sf),
            GetTensorData<float>(prod_scaling_factors),
            GetTensorData<float>(recovered_cell_weights),
            GetTensorData<int8_t>(input_quantized),
            GetTensorData<int8_t>(aux_input_quantized),
            GetTensorData<int8_t>(output_state_quantized),
            GetTensorData<int8_t>(cell_state_quantized), output_state_ptr,
            cell_state_ptr, GetTensorData<int32_t>(output_scratch_buffer),
            output_ptr, input_zp_ptr, aux_input_zp_ptr, output_state_zp_ptr,
            row_sums_ptr, row_sums_size, compute_row_sums,
            params->asymmetric_quantize_inputs, context);
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalInteger8x8_16(
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
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, bool forward_sequence, bool time_major,
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteTensor* output_state, TfLiteTensor* cell_state, TfLiteTensor* output,
    TfLiteTensor* scratch0, TfLiteTensor* scratch1, TfLiteTensor* scratch2,
    TfLiteTensor* scratch3, TfLiteTensor* scratch4, TfLiteTensor* scratch5,
    CpuBackendContext* context) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_18(mht_18_v, 2399, "", "./tensorflow/lite/kernels/lstm_eval.cc", "EvalInteger8x8_16");

  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  const int n_input = input->dims->data[input->dims->size - 1];
  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = (time_major) ? input->dims->data[0] : input->dims->data[1];
    n_batch = (time_major) ? input->dims->data[1] : input->dims->data[0];
  }

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  // Activation zero point
  int output_state_zp = output_state->params.zero_point;

  // Get params for time/batch/sequence.
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];

  if (time_major) {
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;
    for (int t = 0; t < max_time; t++) {
      const int t_rel = t;
      int8_t* output_ptr = GetTensorData<int8_t>(output) + t_rel * output_step;
      const int8_t* input_ptr =
          GetTensorData<int8_t>(input) + t_rel * input_step;
      LstmStepInteger8x8_16(
          input_ptr, GetTensorData<int8_t>(input_to_input_weights),
          integer_lstm_param->effective_input_to_input_scale_a,
          integer_lstm_param->effective_input_to_input_scale_b,
          GetTensorData<int8_t>(input_to_forget_weights),
          integer_lstm_param->effective_input_to_forget_scale_a,
          integer_lstm_param->effective_input_to_forget_scale_b,
          GetTensorData<int8_t>(input_to_cell_weights),
          integer_lstm_param->effective_input_to_cell_scale_a,
          integer_lstm_param->effective_input_to_cell_scale_b,
          GetTensorData<int8_t>(input_to_output_weights),
          integer_lstm_param->effective_input_to_output_scale_a,
          integer_lstm_param->effective_input_to_output_scale_b,
          GetTensorData<int8_t>(recurrent_to_input_weights),
          integer_lstm_param->effective_recurrent_to_input_scale_a,
          integer_lstm_param->effective_recurrent_to_input_scale_b,
          GetTensorData<int8_t>(recurrent_to_forget_weights),
          integer_lstm_param->effective_recurrent_to_forget_scale_a,
          integer_lstm_param->effective_recurrent_to_forget_scale_b,
          GetTensorData<int8_t>(recurrent_to_cell_weights),
          integer_lstm_param->effective_recurrent_to_cell_scale_a,
          integer_lstm_param->effective_recurrent_to_cell_scale_b,
          GetTensorData<int8_t>(recurrent_to_output_weights),
          integer_lstm_param->effective_recurrent_to_output_scale_a,
          integer_lstm_param->effective_recurrent_to_output_scale_b,
          GetTensorData<int16_t>(cell_to_input_weights),
          integer_lstm_param->effective_cell_to_input_scale_a,
          integer_lstm_param->effective_cell_to_input_scale_b,
          GetTensorData<int16_t>(cell_to_forget_weights),
          integer_lstm_param->effective_cell_to_forget_scale_a,
          integer_lstm_param->effective_cell_to_forget_scale_b,
          GetTensorData<int16_t>(cell_to_output_weights),
          integer_lstm_param->effective_cell_to_output_scale_a,
          integer_lstm_param->effective_cell_to_output_scale_b,
          GetTensorData<int8_t>(projection_weights),
          integer_lstm_param->effective_proj_scale_a,
          integer_lstm_param->effective_proj_scale_b,
          integer_lstm_param->hidden_zp,
          integer_lstm_param->effective_hidden_scale_a,
          integer_lstm_param->effective_hidden_scale_b,
          GetTensorData<int16_t>(input_layer_norm_coefficients),
          integer_lstm_param->layer_norm_input_scale_a,
          integer_lstm_param->layer_norm_input_scale_b,
          GetTensorData<int16_t>(forget_layer_norm_coefficients),
          integer_lstm_param->layer_norm_forget_scale_a,
          integer_lstm_param->layer_norm_forget_scale_b,
          GetTensorData<int16_t>(cell_layer_norm_coefficients),
          integer_lstm_param->layer_norm_cell_scale_a,
          integer_lstm_param->layer_norm_cell_scale_b,
          GetTensorData<int16_t>(output_layer_norm_coefficients),
          integer_lstm_param->layer_norm_output_scale_a,
          integer_lstm_param->layer_norm_output_scale_b,
          GetTensorData<int32_t>(input_gate_bias),
          GetTensorData<int32_t>(forget_gate_bias),
          GetTensorData<int32_t>(cell_gate_bias),
          GetTensorData<int32_t>(output_gate_bias),
          integer_lstm_param->quantized_cell_clip,
          integer_lstm_param->quantized_proj_clip,
          integer_lstm_param->cell_scale,
          integer_lstm_param->input_variance_guard,
          integer_lstm_param->forget_variance_guard,
          integer_lstm_param->cell_variance_guard,
          integer_lstm_param->output_variance_guard,
          integer_lstm_param->input_to_forget_effective_bias.get(),
          integer_lstm_param->recurrent_to_forget_effective_bias.get(),
          integer_lstm_param->input_to_cell_effective_bias.get(),
          integer_lstm_param->recurrent_to_cell_effective_bias.get(),
          integer_lstm_param->input_to_output_effective_bias.get(),
          integer_lstm_param->recurrent_to_output_effective_bias.get(),
          integer_lstm_param->input_to_input_effective_bias.get(),
          integer_lstm_param->recurrent_to_input_effective_bias.get(),
          integer_lstm_param->projection_effective_bias.get(), n_batch, n_cell,
          n_input, n_output, GetTensorData<int8_t>(output_state),
          output_state_zp, GetTensorData<int16_t>(cell_state), output_ptr,
          GetTensorData<int16_t>(scratch0), GetTensorData<int16_t>(scratch1),
          GetTensorData<int16_t>(scratch2), GetTensorData<int16_t>(scratch3),
          GetTensorData<int8_t>(scratch4), GetTensorData<int32_t>(scratch5),
          context);
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
        const int8_t* input_ptr =
            GetTensorData<int8_t>(input) + time_offset * input_step;
        int8_t* output_ptr =
            GetTensorData<int8_t>(output) + time_offset * output_step;

        // Offset the {output,cell}_state pointers to the right batch.
        int8_t* output_state_ptr =
            GetTensorData<int8_t>(output_state) + b * output_batch_leading_dim;
        int16_t* cell_state_ptr =
            GetTensorData<int16_t>(cell_state) + b * n_cell;

        LstmStepInteger8x8_16(
            input_ptr, GetTensorData<int8_t>(input_to_input_weights),
            integer_lstm_param->effective_input_to_input_scale_a,
            integer_lstm_param->effective_input_to_input_scale_b,
            GetTensorData<int8_t>(input_to_forget_weights),
            integer_lstm_param->effective_input_to_forget_scale_a,
            integer_lstm_param->effective_input_to_forget_scale_b,
            GetTensorData<int8_t>(input_to_cell_weights),
            integer_lstm_param->effective_input_to_cell_scale_a,
            integer_lstm_param->effective_input_to_cell_scale_b,
            GetTensorData<int8_t>(input_to_output_weights),
            integer_lstm_param->effective_input_to_output_scale_a,
            integer_lstm_param->effective_input_to_output_scale_b,
            GetTensorData<int8_t>(recurrent_to_input_weights),
            integer_lstm_param->effective_recurrent_to_input_scale_a,
            integer_lstm_param->effective_recurrent_to_input_scale_b,
            GetTensorData<int8_t>(recurrent_to_forget_weights),
            integer_lstm_param->effective_recurrent_to_forget_scale_a,
            integer_lstm_param->effective_recurrent_to_forget_scale_b,
            GetTensorData<int8_t>(recurrent_to_cell_weights),
            integer_lstm_param->effective_recurrent_to_cell_scale_a,
            integer_lstm_param->effective_recurrent_to_cell_scale_b,
            GetTensorData<int8_t>(recurrent_to_output_weights),
            integer_lstm_param->effective_recurrent_to_output_scale_a,
            integer_lstm_param->effective_recurrent_to_output_scale_b,
            GetTensorData<int16_t>(cell_to_input_weights),
            integer_lstm_param->effective_cell_to_input_scale_a,
            integer_lstm_param->effective_cell_to_input_scale_b,
            GetTensorData<int16_t>(cell_to_forget_weights),
            integer_lstm_param->effective_cell_to_forget_scale_a,
            integer_lstm_param->effective_cell_to_forget_scale_b,
            GetTensorData<int16_t>(cell_to_output_weights),
            integer_lstm_param->effective_cell_to_output_scale_a,
            integer_lstm_param->effective_cell_to_output_scale_b,
            GetTensorData<int8_t>(projection_weights),
            integer_lstm_param->effective_proj_scale_a,
            integer_lstm_param->effective_proj_scale_b,
            integer_lstm_param->hidden_zp,
            integer_lstm_param->effective_hidden_scale_a,
            integer_lstm_param->effective_hidden_scale_b,
            GetTensorData<int16_t>(input_layer_norm_coefficients),
            integer_lstm_param->layer_norm_input_scale_a,
            integer_lstm_param->layer_norm_input_scale_b,
            GetTensorData<int16_t>(forget_layer_norm_coefficients),
            integer_lstm_param->layer_norm_forget_scale_a,
            integer_lstm_param->layer_norm_forget_scale_b,
            GetTensorData<int16_t>(cell_layer_norm_coefficients),
            integer_lstm_param->layer_norm_cell_scale_a,
            integer_lstm_param->layer_norm_cell_scale_b,
            GetTensorData<int16_t>(output_layer_norm_coefficients),
            integer_lstm_param->layer_norm_output_scale_a,
            integer_lstm_param->layer_norm_output_scale_b,
            GetTensorData<int32_t>(input_gate_bias),
            GetTensorData<int32_t>(forget_gate_bias),
            GetTensorData<int32_t>(cell_gate_bias),
            GetTensorData<int32_t>(output_gate_bias),
            integer_lstm_param->quantized_cell_clip,
            integer_lstm_param->quantized_proj_clip,
            integer_lstm_param->cell_scale,
            integer_lstm_param->input_variance_guard,
            integer_lstm_param->forget_variance_guard,
            integer_lstm_param->cell_variance_guard,
            integer_lstm_param->output_variance_guard,
            integer_lstm_param->input_to_forget_effective_bias.get(),
            integer_lstm_param->recurrent_to_forget_effective_bias.get(),
            integer_lstm_param->input_to_cell_effective_bias.get(),
            integer_lstm_param->recurrent_to_cell_effective_bias.get(),
            integer_lstm_param->input_to_output_effective_bias.get(),
            integer_lstm_param->recurrent_to_output_effective_bias.get(),
            integer_lstm_param->input_to_input_effective_bias.get(),
            integer_lstm_param->recurrent_to_input_effective_bias.get(),
            integer_lstm_param->projection_effective_bias.get(), /*n_batch=*/1,
            n_cell, n_input, n_output, output_state_ptr, output_state_zp,
            cell_state_ptr, output_ptr, GetTensorData<int16_t>(scratch0),
            GetTensorData<int16_t>(scratch1), GetTensorData<int16_t>(scratch2),
            GetTensorData<int16_t>(scratch3), GetTensorData<int8_t>(scratch4),
            GetTensorData<int32_t>(scratch5), context);
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalInteger8x8_8(
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
    const TfLiteTensor* input_gate_bias, const TfLiteTensor* forget_gate_bias,
    const TfLiteTensor* cell_gate_bias, const TfLiteTensor* output_gate_bias,
    const TfLiteTensor* projection_weights, const TfLiteTensor* projection_bias,
    const TfLiteLSTMParams* params, TfLiteTensor* output_state,
    TfLiteTensor* cell_state, TfLiteTensor* output,
    const lstm_eval::IntegerLstmParameter* integer_lstm_param,
    TfLiteTensor* scratch0, TfLiteTensor* scratch1, TfLiteTensor* scratch2,
    TfLiteTensor* scratch3, TfLiteTensor* scratch4, TfLiteTensor* scratch5,
    TfLiteTensor* scratch6, TfLiteTensor* scratch7) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSlstm_evalDTcc mht_19(mht_19_v, 2640, "", "./tensorflow/lite/kernels/lstm_eval.cc", "EvalInteger8x8_8");

  TF_LITE_ASSERT(input->dims->size >= 2 && input->dims->size <= 3);
  const int n_input = input->dims->data[input->dims->size - 1];
  int max_time, n_batch;
  if (input->dims->size == 2) {
    max_time = 1;
    n_batch = input->dims->data[0];
  } else {
    max_time = input->dims->data[0];
    n_batch = input->dims->data[1];
  }

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->dims->data[0];
  const int n_output = recurrent_to_output_weights->dims->data[1];

  const int32_t input_zp = input->params.zero_point;
  const int32_t output_state_zp = output_state->params.zero_point;

  // Get params for time/batch/sequence.
  const int output_batch_leading_dim =
      output->dims->data[output->dims->size - 1];
  const int input_step = n_batch * n_input;
  const int output_step = n_batch * output_batch_leading_dim;

  for (int t = 0; t < max_time; t++) {
    const int t_rel = t;
    int8_t* output_ptr = GetTensorData<int8_t>(output) + t_rel * output_step;
    // Input can be int8 asymmetric or int16 symmetric.
    const int8_t* input_ptr = GetTensorData<int8_t>(input) + t_rel * input_step;
    lstm_eval::LstmStepInteger8x8_8(
        input_ptr, input_zp,

        GetTensorData<int8_t>(input_to_input_weights),
        integer_lstm_param->effective_input_to_input_scale_a,
        integer_lstm_param->effective_input_to_input_scale_b,

        GetTensorData<int8_t>(input_to_forget_weights),
        integer_lstm_param->effective_input_to_forget_scale_a,
        integer_lstm_param->effective_input_to_forget_scale_b,

        GetTensorData<int8_t>(input_to_cell_weights),
        integer_lstm_param->effective_input_to_cell_scale_a,
        integer_lstm_param->effective_input_to_cell_scale_b,

        GetTensorData<int8_t>(input_to_output_weights),
        integer_lstm_param->effective_input_to_output_scale_a,
        integer_lstm_param->effective_input_to_output_scale_b,

        GetTensorData<int8_t>(recurrent_to_input_weights),
        integer_lstm_param->effective_recurrent_to_input_scale_a,
        integer_lstm_param->effective_recurrent_to_input_scale_b,

        GetTensorData<int8_t>(recurrent_to_forget_weights),
        integer_lstm_param->effective_recurrent_to_forget_scale_a,
        integer_lstm_param->effective_recurrent_to_forget_scale_b,

        GetTensorData<int8_t>(recurrent_to_cell_weights),
        integer_lstm_param->effective_recurrent_to_cell_scale_a,
        integer_lstm_param->effective_recurrent_to_cell_scale_b,

        GetTensorData<int8_t>(recurrent_to_output_weights),
        integer_lstm_param->effective_recurrent_to_output_scale_a,
        integer_lstm_param->effective_recurrent_to_output_scale_b,

        GetTensorData<int8_t>(cell_to_input_weights),
        integer_lstm_param->effective_cell_to_input_scale_a,
        integer_lstm_param->effective_cell_to_input_scale_b,

        GetTensorData<int8_t>(cell_to_forget_weights),
        integer_lstm_param->effective_cell_to_forget_scale_a,
        integer_lstm_param->effective_cell_to_forget_scale_b,

        GetTensorData<int8_t>(cell_to_output_weights),
        integer_lstm_param->effective_cell_to_output_scale_a,
        integer_lstm_param->effective_cell_to_output_scale_b,

        GetTensorData<int8_t>(projection_weights),
        integer_lstm_param->effective_proj_scale_a,
        integer_lstm_param->effective_proj_scale_b,

        GetTensorData<int16_t>(input_layer_norm_coefficients),
        integer_lstm_param->layer_norm_input_scale_a,
        integer_lstm_param->layer_norm_input_scale_b,

        GetTensorData<int16_t>(forget_layer_norm_coefficients),
        integer_lstm_param->layer_norm_forget_scale_a,
        integer_lstm_param->layer_norm_forget_scale_b,

        GetTensorData<int16_t>(cell_layer_norm_coefficients),
        integer_lstm_param->layer_norm_cell_scale_a,
        integer_lstm_param->layer_norm_cell_scale_b,

        GetTensorData<int16_t>(output_layer_norm_coefficients),
        integer_lstm_param->layer_norm_output_scale_a,
        integer_lstm_param->layer_norm_output_scale_b,

        GetTensorData<int32_t>(input_gate_bias),
        GetTensorData<int32_t>(forget_gate_bias),
        GetTensorData<int32_t>(cell_gate_bias),
        GetTensorData<int32_t>(output_gate_bias),
        GetTensorData<int32_t>(projection_bias),

        params, integer_lstm_param->intermediate_scale_a,
        integer_lstm_param->intermediate_scale_b,
        integer_lstm_param->intermediate_zp,
        integer_lstm_param->quantized_cell_clip,
        integer_lstm_param->quantized_proj_clip, n_batch, n_cell, n_input,
        n_output, output_batch_leading_dim, GetTensorData<int8_t>(output_state),
        output_state_zp, GetTensorData<int16_t>(cell_state), output_ptr,
        GetTensorData<int8_t>(scratch0), GetTensorData<int8_t>(scratch1),
        GetTensorData<int16_t>(scratch2), GetTensorData<int16_t>(scratch3),
        GetTensorData<int16_t>(scratch4), GetTensorData<int16_t>(scratch5),
        GetTensorData<int16_t>(scratch6), GetTensorData<int16_t>(scratch7));
  }

  return kTfLiteOk;
}

}  // namespace lstm_eval
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
