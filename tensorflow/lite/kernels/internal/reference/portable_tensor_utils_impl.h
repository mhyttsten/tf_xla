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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_IMPL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_IMPL_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSportable_tensor_utils_implDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSportable_tensor_utils_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSportable_tensor_utils_implDTh() {
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


#include <algorithm>
#include <cstdint>

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation.
class CpuBackendContext;

namespace tensor_utils {

template <typename T>
bool PortableIsZeroVector(const T* vector, int v_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSportable_tensor_utils_implDTh mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h", "PortableIsZeroVector");

  for (int i = 0; i < v_size; ++i) {
    if (vector[i] != 0) {
      return false;
    }
  }
  return true;
}

void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float* min_value,
                                     float* max_value, float* scaling_factor);

void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float min_value,
                                     float max_value, float* scaling_factor);

void PortableAsymmetricQuantizeFloats(const float* values, const int size,
                                      int8_t* quantized_values,
                                      float* scaling_factor, int32_t* offset);

// Multiply a matrix by a batch vector, and store results in a batch-size
// vector.
void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vector, const float* scaling_factors,
    int n_batch, int32_t* scratch, float* __restrict__ result,
    CpuBackendContext* context);

void PortableSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const float* __restrict__ vector, int n_batch, float* __restrict__ result);

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result);

void PortableSparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const int8_t* __restrict__ vector, const int32_t* __restrict__ bias_vector,
    int n_batch, const int32_t input_offset, const int32_t output_multiplier,
    const int32_t output_shift, const int32_t output_offset,
    const int32_t output_activation_min, const int32_t output_activation_max,
    int8_t* __restrict__ result);

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result);

// Dot product of two vectors.
float PortableVectorVectorDotProduct(const float* vector1, const float* vector2,
                                     int v_size);

void PortableBatchVectorBatchVectorDotProduct(const int16_t* vector1,
                                              const int16_t* vector2,
                                              int v_size, int n_batch,
                                              int32_t* result);

void PortableVectorBatchVectorCwiseProductAccumulate(
    const int16_t* vector, int v_size, const int16_t* batch_vector, int n_batch,
    int32_t multiplier, int shift, int16_t* result);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context);

void PortableMatrixBatchVectorMultiply(const int8_t* input,
                                       int32_t input_zeropoint,
                                       const int8_t* input_to_gate_weights,
                                       int32_t input_to_gate_effective_scale_a,
                                       int32_t input_to_gate_effective_scale_b,
                                       int32_t n_batch, int32_t n_input,
                                       int32_t n_cell, int8_t* gate_output,
                                       int8_t gate_output_zp);

void PortableMatrixBatchVectorMultiply(
    const int16_t* hidden, const int8_t* hidden_to_output_weights,
    int32_t proj_effective_scale_a, int32_t proj_effective_scale_b,
    const int32_t* gate_bias, int32_t n_batch, int32_t n_hidden,
    int32_t n_output, int32_t output_zp, int8_t* proj_output);

void PortableMatrixScalarMultiplyAccumulate(const int8_t* matrix,
                                            int32_t scalar, int32_t n_row,
                                            int32_t n_col, int32_t* output);

void PortableApplyLayerNorm(const int16_t* input,
                            const int16_t* layer_norm_weights,
                            const int32_t* bias, int32_t layer_norm_scale_a,
                            int32_t layer_norm_scale_b, int32_t variance_limit,
                            int n_batch, int n_input, int16_t* output);

void PortableApplyLayerNormFloat(const int16_t* input,
                                 const int16_t* layer_norm_weights,
                                 int32_t layer_norm_scale_a,
                                 int32_t layer_norm_scale_b,
                                 const int32_t* bias, int n_batch, int n_input,
                                 int16_t* output);

void PortableApplySigmoid(const int16_t* input, int32_t n_batch,
                          int32_t n_input, int16_t* output);

void PortableApplySigmoidFloat(const int16_t* input, int32_t n_batch,
                               int32_t n_input, int16_t* output);

void PortableApplyTanh(int32_t integer_bits, const int16_t* input,
                       int32_t n_batch, int32_t n_input, int16_t* output);

void PortableApplyTanhFloat(const int16_t* input, int32_t n_batch,
                            int32_t n_input, int32_t integer_bits,
                            int16_t* output);

void PortableCwiseMul(const int16_t* input_1, const int16_t* input_2,
                      int n_batch, int n_input, int shift, int16_t* output);

void PortableCwiseMul(const int16_t* input_1, const int16_t* input_2,
                      int32_t multiplier, int32_t shift, int32_t n_batch,
                      int32_t n_input, int32_t output_zp, int8_t* output);

void PortableCwiseAdd(const int16_t* input_1, const int16_t* input_2,
                      int n_batch, int n_input, int16_t* output);

template <typename T>
void PortableCwiseClipping(T* vector, const int v_size,
                           const T& clipping_value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSportable_tensor_utils_implDTh mht_1(mht_1_v, 358, "", "./tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h", "PortableCwiseClipping");

  for (int i = 0; i < v_size; i++) {
    vector[i] = std::max(std::min(clipping_value, vector[i]),
                         static_cast<T>(-clipping_value));
  }
}

// Batch vector initialization with another vector.
void PortableVectorBatchVectorAssign(const float* vector, int v_size,
                                     int n_batch, float* batch_vector);

// Compute "1.0f - elements of vector" (used in CIFG).
void PortableSub1Vector(const float* vector, int v_size, float* result);

void PortableSub1Vector(const int16_t* vector, int v_size, int16_t* result);

// Multiply all elements of vector with a scalar.
void PortableVectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                                  float* result);

// Reduce-sum on a vector:
// input_vector: pointer to input vector.
// output_vector: pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
template <typename INPUT, typename OUTPUT>
void PortableReductionSumVector(const INPUT* input_vector,
                                OUTPUT* output_vector, int output_size,
                                int reduction_size) {
  for (int o = 0; o < output_size; o++) {
    OUTPUT result = 0;
    for (int r = 0; r < reduction_size; r++) {
      result += input_vector[r];
    }
    output_vector[o] = result;
    input_vector += reduction_size;
  }
}

// Layer norm for each batch.
void PortableMeanStddevNormalization(const float* __restrict__ input_vector,
                                     float* __restrict__ output_vector,
                                     int v_size, int n_batch);

// Saturate Add.
void PortableTwoGateSaturatingAdd(const int8_t* input, int8_t input_zp,
                                  const int8_t* recurrent, int8_t recurrent_zp,
                                  int32_t input_effective_scale_a,
                                  int32_t input_effective_scale_b,
                                  int32_t recurrent_effective_scale_a,
                                  int32_t recurrent_effective_scale_b,
                                  int32_t n_batch, int32_t n_cell,
                                  int16_t* output);

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_IMPL_H_
