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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh() {
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


#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h"

namespace tflite {
namespace tensor_utils {

void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_0(mht_0_v, 198, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MatrixBatchVectorMultiplyAccumulate");

  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                   vector, n_batch, result);
}

void MatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                         const int m_rows, const int m_cols,
                                         const int8_t* __restrict__ vectors,
                                         const float* scaling_factors,
                                         int n_batch,
                                         float* __restrict__ result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_1(mht_1_v, 211, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MatrixBatchVectorMultiplyAccumulate");

  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                   vectors, scaling_factors, n_batch, result);
}

void MatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                         const int m_rows, const int m_cols,
                                         const int8_t* __restrict__ vectors,
                                         const float* scaling_factors,
                                         int n_batch, int32_t* scratch,
                                         float* __restrict__ result,
                                         CpuBackendContext* context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_2(mht_2_v, 225, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MatrixBatchVectorMultiplyAccumulate");

  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                   vectors, scaling_factors, n_batch, scratch, result, context);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_3(mht_3_v, 238, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MatrixBatchVectorMultiplyAccumulate");

  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                   vectors, scaling_factors, n_batch, result, per_channel_scale,
                   input_offset, scratch, row_sums, compute_row_sums, context);
}

void SparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const float* __restrict__ vector, int n_batch, float* __restrict__ result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_4(mht_4_v, 250, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "SparseMatrixBatchVectorMultiplyAccumulate1x4");

  NEON_OR_PORTABLE(SparseMatrixBatchVectorMultiplyAccumulate1x4, matrix,
                   segments, indices, m_rows, m_cols, vector, n_batch, result);
}

void SparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_5(mht_5_v, 261, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "SparseMatrixBatchVectorMultiplyAccumulate");

  NEON_OR_PORTABLE(SparseMatrixBatchVectorMultiplyAccumulate, matrix, ledger,
                   m_rows, m_cols, vector, n_batch, result);
}

void SparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const int8_t* __restrict__ vector, const int32_t* __restrict__ bias_vector,
    int n_batch, const int32_t input_offset, const int32_t output_multiplier,
    const int32_t output_shift, const int32_t output_offset,
    const int32_t output_activation_min, const int32_t output_activation_max,
    int8_t* __restrict__ result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_6(mht_6_v, 276, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "SparseMatrixBatchVectorMultiplyAccumulate1x16");

  NEON_OR_PORTABLE(SparseMatrixBatchVectorMultiplyAccumulate1x16, matrix,
                   segments, indices, m_rows, m_cols, vector, bias_vector,
                   n_batch, input_offset, output_multiplier, output_shift,
                   output_offset, output_activation_min, output_activation_max,
                   result);
}

void SparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_7(mht_7_v, 290, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "SparseMatrixBatchVectorMultiplyAccumulate");

  NEON_OR_PORTABLE(SparseMatrixBatchVectorMultiplyAccumulate, matrix, ledger,
                   m_rows, m_cols, vectors, scaling_factors, n_batch, result);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_8(mht_8_v, 302, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MatrixBatchVectorMultiplyAccumulate");

  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, input, bias,
                   input_to_gate_weights, multiplier, shift, n_batch, n_input,
                   n_output, output_zp, scratch, output, context);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_9(mht_9_v, 315, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MatrixBatchVectorMultiplyAccumulate");

  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, input, bias,
                   input_to_gate_weights, multiplier, shift, n_batch, n_input,
                   n_output, output_zp, scratch, output, context);
}

void MatrixBatchVectorMultiply(const int8_t* input, int32_t input_zeropoint,
                               const int8_t* input_to_gate_weights,
                               int32_t input_to_gate_effective_scale_a,
                               int32_t input_to_gate_effective_scale_b,
                               int32_t n_batch, int32_t n_input, int32_t n_cell,
                               int8_t* gate_output, int8_t gate_output_zp) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_10(mht_10_v, 329, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MatrixBatchVectorMultiply");

  PortableMatrixBatchVectorMultiply(
      input, input_zeropoint, input_to_gate_weights,
      input_to_gate_effective_scale_a, input_to_gate_effective_scale_b, n_batch,
      n_input, n_cell, gate_output, gate_output_zp);
}

void MatrixBatchVectorMultiply(const int16_t* hidden,
                               const int8_t* hidden_to_output_weights,
                               int32_t proj_effective_scale_a,
                               int32_t proj_effective_scale_b,
                               const int32_t* gate_bias, int32_t n_batch,
                               int32_t n_hidden, int32_t n_output,
                               int32_t output_zp, int8_t* proj_output) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_11(mht_11_v, 345, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MatrixBatchVectorMultiply");

  PortableMatrixBatchVectorMultiply(hidden, hidden_to_output_weights,
                                    proj_effective_scale_a,
                                    proj_effective_scale_b, gate_bias, n_batch,
                                    n_hidden, n_output, output_zp, proj_output);
}

void MatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                    int32_t n_row, int32_t n_col,
                                    int32_t* output) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_12(mht_12_v, 357, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MatrixScalarMultiplyAccumulate");

  NEON_OR_PORTABLE(MatrixScalarMultiplyAccumulate, matrix, scalar, n_row, n_col,
                   output);
}

void ApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                    const int32_t* bias, int32_t layer_norm_scale_a,
                    int32_t layer_norm_scale_b, int32_t variance_limit,
                    int n_batch, int n_input, int16_t* output) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_13(mht_13_v, 368, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "ApplyLayerNorm");

  NEON_OR_PORTABLE(ApplyLayerNorm, input, layer_norm_weights, bias,
                   layer_norm_scale_a, layer_norm_scale_b, variance_limit,
                   n_batch, n_input, output);
}

void ApplyLayerNormFloat(const int16_t* input,
                         const int16_t* layer_norm_weights,
                         int32_t layer_norm_scale_a, int32_t layer_norm_scale_b,
                         const int32_t* bias, int n_batch, int n_input,
                         int16_t* output) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_14(mht_14_v, 381, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "ApplyLayerNormFloat");

  PortableApplyLayerNormFloat(input, layer_norm_weights, layer_norm_scale_a,
                              layer_norm_scale_b, bias, n_batch, n_input,
                              output);
}

void ApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                  int16_t* output) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_15(mht_15_v, 391, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "ApplySigmoid");

  NEON_OR_PORTABLE(ApplySigmoid, input, n_batch, n_input, output);
}

void ApplySigmoidFloat(const int16_t* input, int32_t n_batch, int32_t n_input,
                       int16_t* output) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_16(mht_16_v, 399, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "ApplySigmoidFloat");

  PortableApplySigmoidFloat(input, n_batch, n_input, output);
}

void ApplyTanh(int32_t integer_bits, const int16_t* input, int32_t n_batch,
               int32_t n_input, int16_t* output) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_17(mht_17_v, 407, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "ApplyTanh");

  NEON_OR_PORTABLE(ApplyTanh, integer_bits, input, n_batch, n_input, output);
}

void ApplyTanhFloat(const int16_t* input, int32_t n_batch, int32_t n_input,
                    int32_t integer_bits, int16_t* output) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_18(mht_18_v, 415, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "ApplyTanhFloat");

  PortableApplyTanhFloat(input, n_batch, n_input, integer_bits, output);
}

void CwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int shift, int16_t* output) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_19(mht_19_v, 423, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "CwiseMul");

  NEON_OR_PORTABLE(CwiseMul, input_1, input_2, n_batch, n_input, shift, output);
}

void CwiseMul(const int16_t* input_1, const int16_t* input_2,
              int32_t multiplier, int shift, int n_batch, int n_input,
              int32_t output_zp, int8_t* output) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_20(mht_20_v, 432, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "CwiseMul");

  NEON_OR_PORTABLE(CwiseMul, input_1, input_2, multiplier, shift, n_batch,
                   n_input, output_zp, output);
}

void CwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int16_t* output) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_21(mht_21_v, 441, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "CwiseAdd");

  NEON_OR_PORTABLE(CwiseAdd, input_1, input_2, n_batch, n_input, output);
}

void CwiseClipping(float* vector, const int v_size,
                   const float clipping_value) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_22(mht_22_v, 449, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "CwiseClipping");

  NEON_OR_PORTABLE(CwiseClipping, vector, v_size, clipping_value);
}
void CwiseClipping(int16_t* vector, const int v_size,
                   const int16_t clipping_value) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_23(mht_23_v, 456, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "CwiseClipping");

  NEON_OR_PORTABLE(CwiseClipping, vector, v_size, clipping_value);
}
void CwiseClipping(int8_t* vector, const int v_size,
                   const int8_t clipping_value) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_24(mht_24_v, 463, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "CwiseClipping");

  NEON_OR_PORTABLE(CwiseClipping, vector, v_size, clipping_value);
}

void BatchVectorBatchVectorDotProduct(const int16_t* vector1,
                                      const int16_t* vector2, int v_size,
                                      int n_batch, int32_t* result) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_25(mht_25_v, 472, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "BatchVectorBatchVectorDotProduct");

  PortableBatchVectorBatchVectorDotProduct(vector1, vector2, v_size, n_batch,
                                           result);
}

void VectorBatchVectorCwiseProductAccumulate(const int16_t* vector, int v_size,
                                             const int16_t* batch_vector,
                                             int n_batch, int32_t multiplier,
                                             int shift, int16_t* result) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_26(mht_26_v, 483, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "VectorBatchVectorCwiseProductAccumulate");

  NEON_OR_PORTABLE(VectorBatchVectorCwiseProductAccumulate, vector, v_size,
                   batch_vector, n_batch, multiplier, shift, result);
}

float VectorVectorDotProduct(const float* vector1, const float* vector2,
                             int v_size) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_27(mht_27_v, 492, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "VectorVectorDotProduct");

  return NEON_OR_PORTABLE(VectorVectorDotProduct, vector1, vector2, v_size);
}

void Sub1Vector(const float* vector, int v_size, float* result) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_28(mht_28_v, 499, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "Sub1Vector");

  NEON_OR_PORTABLE(Sub1Vector, vector, v_size, result);
}

void Sub1Vector(const int16_t* vector, int v_size, int16_t* result) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_29(mht_29_v, 506, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "Sub1Vector");

  NEON_OR_PORTABLE(Sub1Vector, vector, v_size, result);
}

// Check if all entries of a vector are zero for float.
bool IsZeroVector(const float* vector, int v_size) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_30(mht_30_v, 514, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "IsZeroVector");

  return NEON_OR_PORTABLE(IsZeroVector, vector, v_size);
}

// Check if all entries of a vector are zero for int8.
bool IsZeroVector(const int8_t* vector, int v_size) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_31(mht_31_v, 522, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "IsZeroVector");

  return NEON_OR_PORTABLE(IsZeroVector, vector, v_size);
}

void VectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                          float* result) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_32(mht_32_v, 530, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "VectorScalarMultiply");

  NEON_OR_PORTABLE(VectorScalarMultiply, vector, v_size, scale, result);
}

void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float* min_value,
                             float* max_value, float* scaling_factor) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_33(mht_33_v, 539, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "SymmetricQuantizeFloats");

  NEON_OR_PORTABLE(SymmetricQuantizeFloats, values, size, quantized_values,
                   min_value, max_value, scaling_factor);
}

void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float min_value,
                             float max_value, float* scaling_factor) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_34(mht_34_v, 549, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "SymmetricQuantizeFloats");

  NEON_OR_PORTABLE(SymmetricQuantizeFloats, values, size, quantized_values,
                   min_value, max_value, scaling_factor);
}

void AsymmetricQuantizeFloats(const float* values, const int size,
                              int8_t* quantized_values, float* scaling_factor,
                              int32_t* offset) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_35(mht_35_v, 559, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "AsymmetricQuantizeFloats");

  NEON_OR_PORTABLE(AsymmetricQuantizeFloats, values, size, quantized_values,
                   scaling_factor, offset);
}

void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_36(mht_36_v, 568, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "ReductionSumVector");

  NEON_OR_PORTABLE(ReductionSumVector, input_vector, output_vector, output_size,
                   reduction_size);
}

void ReductionSumVector(const int32_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_37(mht_37_v, 577, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "ReductionSumVector");

  PortableReductionSumVector(input_vector, output_vector, output_size,
                             reduction_size);
}

void ReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_38(mht_38_v, 586, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "ReductionSumVector");

  NEON_OR_PORTABLE(ReductionSumVector, input_vector, output_vector, output_size,
                   reduction_size);
}

void MeanStddevNormalization(const float* __restrict__ input_vector,
                             float* __restrict__ output_vector, int v_size,
                             int n_batch) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_39(mht_39_v, 596, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "MeanStddevNormalization");

  NEON_OR_PORTABLE(MeanStddevNormalization, input_vector, output_vector, v_size,
                   n_batch);
}

void TwoGateSaturatingAdd(const int8_t* input, int8_t input_zp,
                          const int8_t* recurrent, int8_t recurrent_zp,
                          int32_t input_effective_scale_a,
                          int32_t input_effective_scale_b,
                          int32_t recurrent_effective_scale_a,
                          int32_t recurrent_effective_scale_b, int32_t n_batch,
                          int32_t n_cell, int16_t* output) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSneon_tensor_utilsDTh mht_40(mht_40_v, 610, "", "./tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h", "TwoGateSaturatingAdd");

  PortableTwoGateSaturatingAdd(
      input, input_zp, recurrent, recurrent_zp, input_effective_scale_a,
      input_effective_scale_b, recurrent_effective_scale_a,
      recurrent_effective_scale_b, n_batch, n_cell, output);
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_H_
