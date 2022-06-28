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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh() {
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
#include <cmath>
#include <cstdint>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation. Use of CpuBackendContext in method
// implementations is purely optional.
class CpuBackendContext;

namespace tensor_utils {

// Same as the function above, but provide a scratch buffer for the
// int8 x int8 -> int32 and a CpuBackendContext for the accumulator
// computation.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    int32_t* __restrict__ scratch, float* __restrict__ result,
    CpuBackendContext* __restrict__ context);

// Same as the function above except that can make use of cached row sums.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context);

// Same as the function above, but provides separate scaling factor for the
// matrix and the vectors. The scaling factors are multiplied in the
// scaling_factor_scratch buffer.
inline void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float matrix_scaling_factor,
    const float* vector_scaling_factors, int n_batch,
    float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, float* scaling_factor_scratch,
    CpuBackendContext* context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh mht_0(mht_0_v, 236, "", "./tensorflow/lite/kernels/internal/tensor_utils.h", "MatrixBatchVectorMultiplyAccumulate");

  for (int b = 0; b < n_batch; ++b) {
    scaling_factor_scratch[b] =
        vector_scaling_factors[b] * matrix_scaling_factor;
  }
  MatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                      scaling_factor_scratch, n_batch, result,
                                      per_channel_scale, input_offset, scratch,
                                      row_sums, compute_row_sums, context);
}

// Multiplies a matrix by a "batched" vector (i.e. a matrix with a batch
// dimension composed by input vectors independent from each other). The result
// of the multiplication is accumulated to the passed result buffer.
// More specifically, for a matrix M of shape [n, i] and a batched-vector
// of shape [i, batch] it will first compute the product of shape [n, batch].
// This product will be accumulated to the result buffer,
// Parameters:
//     - input: batch vector of size n_batch * n_input
//     - bias:  vector of size b_input
//     - input_to_gate_weights: matrix of size n_input * n_output
//     - multiplier: scalar
//     - shift: scalar
//     - n_batch: the batch size
//     - n_input: the input size
//     - n_output: the output size
//     - output_zp: the zero point of the output.
//     - scratch: batch vector of size n_batch * n_output
//     - output: the 16 bit output
// Notes:
//     - this is used for gate matmul: for non-cifg it is for input, forget,
//       cell, output gates; for cifg, it is for forget, cell, output gates.
//     - multiplier and shift combined gives the scale.
//     - assumes input zero point is 0.
//     - scratch is created for optimization purpose only.
// TODO(b/152066492): this can be removed if some future optimization
// work makes it unnecessary.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context);

// Multiplies a matrix by a "batched" vector (i.e. a matrix with a batch
// dimension composed by input vectors independent from each other). The result
// of the multiplication is accumulated to the passed result buffer.
// More specifically, for a matrix M of shape [n, i] and a batched-vector
// of shape [i, batch] it will first compute the product of shape [n, batch].
// This product will be accumulated to the result buffer,
// Parameters:
//     - input: batch vector of size n_batch * n_input
//     - bias:  vector of size b_input
//     - input_to_gate_weights: matrix of size n_input * n_output
//     - multiplier: scalar
//     - shift: scalar
//     - n_batch: the batch size
//     - n_input: the input size
//     - n_output: the output size
//     - output_zp: the zero point of the output.
//     - scratch: batch vector of size n_batch * n_output
//     - output: the 8 bit output
// Notes:
//     - this is used for projection matmul.
//     - multiplier and shift combined gives the scale.
//     - assumes input zero point is 0.
//     - scratch is created for optimization purpose only.
// TODO(b/152066492): this can be removed if some future optimization
// work makes it unnecessary.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context);

// Apply Rectified Linear to elements of a vector.
inline void ApplyReluToVector(const float* __restrict__ vector, int v_size,
                              float* __restrict__ result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh mht_1(mht_1_v, 315, "", "./tensorflow/lite/kernels/internal/tensor_utils.h", "ApplyReluToVector");

  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(0.0f, vector[v]);
  }
}

// Apply Rectified Linear 1 (cap to [-1;1]) to elements of a vector
inline void ApplyRelu1ToVector(const float* __restrict__ vector, int v_size,
                               float* __restrict__ result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh mht_2(mht_2_v, 326, "", "./tensorflow/lite/kernels/internal/tensor_utils.h", "ApplyRelu1ToVector");

  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(-1.0f, std::min(vector[v], 1.0f));
  }
}

// Apply Rectified Linear 6 (cap to [0;6]) to elements of a vector
inline void ApplyRelu6ToVector(const float* __restrict__ vector, int v_size,
                               float* __restrict__ result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh mht_3(mht_3_v, 337, "", "./tensorflow/lite/kernels/internal/tensor_utils.h", "ApplyRelu6ToVector");

  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(0.0f, std::min(vector[v], 6.0f));
  }
}

// Apply tanh to elements of a vector
inline void ApplyTanhToVector(const float* __restrict__ vector, int v_size,
                              float* __restrict__ result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh mht_4(mht_4_v, 348, "", "./tensorflow/lite/kernels/internal/tensor_utils.h", "ApplyTanhToVector");

  using VectorMap = Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>;
  VectorMap input_map(const_cast<float* __restrict__>(vector), v_size);
  VectorMap output_map(result, v_size);
  output_map.array() = input_map.array().tanh();
}

// Apply signbit to elements of a vector
inline void ApplySignbitToVector(const float* __restrict__ vector, int v_size,
                                 float* __restrict__ result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh mht_5(mht_5_v, 360, "", "./tensorflow/lite/kernels/internal/tensor_utils.h", "ApplySignbitToVector");

  for (int v = 0; v < v_size; v++) {
    result[v] = std::signbit(vector[v]);
  }
}

// Apply sigmoid to elements of a vector.
inline void ApplySigmoidToVector(const float* __restrict__ vector, int v_size,
                                 float* __restrict__ result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh mht_6(mht_6_v, 371, "", "./tensorflow/lite/kernels/internal/tensor_utils.h", "ApplySigmoidToVector");

  using VectorMap = Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>;
  VectorMap input_map(const_cast<float* __restrict__>(vector), v_size);
  VectorMap output_map(result, v_size);
  output_map.array() = input_map.array().logistic();
}

// Apply appropriate activation function to elements of a vector.
inline void ApplyActivationToVector(const float* __restrict__ vector,
                                    int v_size,
                                    TfLiteFusedActivation activation,
                                    float* __restrict__ result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStensor_utilsDTh mht_7(mht_7_v, 385, "", "./tensorflow/lite/kernels/internal/tensor_utils.h", "ApplyActivationToVector");

  switch (activation) {
    case kTfLiteActNone:
      return;
    case kTfLiteActRelu:
      return ApplyReluToVector(vector, v_size, result);
    case kTfLiteActReluN1To1:
      return ApplyRelu1ToVector(vector, v_size, result);
    case kTfLiteActRelu6:
      return ApplyRelu6ToVector(vector, v_size, result);
    case kTfLiteActTanh:
      return ApplyTanhToVector(vector, v_size, result);
    case kTfLiteActSignBit:
      return ApplySignbitToVector(vector, v_size, result);
    case kTfLiteActSigmoid:
      return ApplySigmoidToVector(vector, v_size, result);
  }
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
