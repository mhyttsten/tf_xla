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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_BATCH_MATMUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_BATCH_MATMUL_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh() {
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


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

inline void BatchMatMul(const RuntimeShape& lhs_shape, const float* lhs_data,
                        const RuntimeShape& rhs_shape, const float* rhs_data,
                        const RuntimeShape& output_shape, float* output_data,
                        CpuBackendContext* context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh mht_0(mht_0_v, 200, "", "./tensorflow/lite/kernels/internal/optimized/batch_matmul.h", "BatchMatMul");

  using ::tflite::cpu_backend_gemm::Gemm;
  using ::tflite::cpu_backend_gemm::GemmParams;
  using ::tflite::cpu_backend_gemm::MatrixParams;
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  // Determine which dimension is the broadcast dimension.
  auto broadcast_dim = [](int lhs_dim, int rhs_dim) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh mht_1(mht_1_v, 213, "", "./tensorflow/lite/kernels/internal/optimized/batch_matmul.h", "lambda");

    if (lhs_dim == rhs_dim) return lhs_dim;
    if (lhs_dim == 1) return rhs_dim;
    TFLITE_DCHECK_EQ(rhs_dim, 1);
    return lhs_dim;
  };

  // Compute the "extent" for iterating on this dimension.
  // If we are broadcasting, then don't advance (i.e return 0).
  auto extent = [](const RuntimeShape& shape, int x) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh mht_2(mht_2_v, 225, "", "./tensorflow/lite/kernels/internal/optimized/batch_matmul.h", "lambda");

    if (shape.Dims(x) == 1) {
      return 0;
    }
    int prod = 1;
    for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
      prod *= shape.Dims(i);
    }
    return prod;
  };

  const int batch_dim0 =
      broadcast_dim(extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 =
      broadcast_dim(extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 =
      broadcast_dim(extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = extent(extended_lhs_shape, 0);
  const int lhs_ext1 = extent(extended_lhs_shape, 1);
  const int lhs_ext2 = extent(extended_lhs_shape, 2);
  const int rhs_ext0 = extent(extended_rhs_shape, 0);
  const int rhs_ext1 = extent(extended_rhs_shape, 1);
  const int rhs_ext2 = extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = lhs_rows;
  lhs_params.cols = accum_depth;

  MatrixParams<float> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = accum_depth;
  rhs_params.cols = rhs_cols;

  MatrixParams<float> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = lhs_rows;
  dst_params.cols = rhs_cols;

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const float* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const float* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const float* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const float* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const float* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const float* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        float* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                        b1 * batch_dim2 + b2) *
                                           lhs_rows * rhs_cols;
        GemmParams<float, float> gemm_params;
        cpu_backend_gemm::Gemm(lhs_params, lhs_ptr2, rhs_params, rhs_ptr2,
                               dst_params, out_ptr, gemm_params, context);
      }
    }
  }
}

inline void BatchMatMul(const RuntimeShape& lhs_shape, const int8_t* lhs_data,
                        const RuntimeShape& rhs_shape, const int8_t* rhs_data,
                        const float* scaling_factors,
                        const int32_t* input_offset, int32_t* row_sums,
                        const RuntimeShape& output_shape,
                        int32_t* accum_scratch, float* output_data,
                        bool* compute_row_sums, CpuBackendContext* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh mht_3(mht_3_v, 299, "", "./tensorflow/lite/kernels/internal/optimized/batch_matmul.h", "BatchMatMul");

  using ::tflite::cpu_backend_gemm::Gemm;
  using ::tflite::cpu_backend_gemm::GemmParams;
  using ::tflite::cpu_backend_gemm::MatrixParams;

  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  // Determine which dimension is the broadcast dimension.
  auto broadcast_dim = [](int lhs_dim, int rhs_dim) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh mht_4(mht_4_v, 313, "", "./tensorflow/lite/kernels/internal/optimized/batch_matmul.h", "lambda");

    if (lhs_dim == rhs_dim) return lhs_dim;
    if (lhs_dim == 1) return rhs_dim;
    TFLITE_DCHECK_EQ(rhs_dim, 1);
    return lhs_dim;
  };

  // Compute the "extent" for iterating on this dimension.
  // If we are broadcasting, then don't advance (i.e return 0).
  auto extent = [](const RuntimeShape& shape, int x) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh mht_5(mht_5_v, 325, "", "./tensorflow/lite/kernels/internal/optimized/batch_matmul.h", "lambda");

    if (shape.Dims(x) == 1) {
      return 0;
    }
    int prod = 1;
    for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
      prod *= shape.Dims(i);
    }
    return prod;
  };

  const int batch_dim0 =
      broadcast_dim(extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 =
      broadcast_dim(extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 =
      broadcast_dim(extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = extent(extended_lhs_shape, 0);
  const int lhs_ext1 = extent(extended_lhs_shape, 1);
  const int lhs_ext2 = extent(extended_lhs_shape, 2);
  const int rhs_ext0 = extent(extended_rhs_shape, 0);
  const int rhs_ext1 = extent(extended_rhs_shape, 1);
  const int rhs_ext2 = extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  const int ioff_ext0 = rhs_ext0 == 0 ? 0 : rhs_cols;
  const int ioff_ext1 = rhs_ext1 == 0 ? 0 : rhs_cols;
  const int ioff_ext2 = rhs_ext2 == 0 ? 0 : rhs_cols;
  const int woff_ext0 = lhs_ext0 == 0 ? 0 : lhs_rows;
  const int woff_ext1 = lhs_ext1 == 0 ? 0 : lhs_rows;
  const int woff_ext2 = lhs_ext2 == 0 ? 0 : lhs_rows;

  if (!compute_row_sums || *compute_row_sums) {
    int num_weights_matrices = 1;
    for (int i = 1; i < extended_lhs_shape.DimensionsCount() - 2; ++i) {
      num_weights_matrices *= extended_lhs_shape.Dims(i);
    }
    tensor_utils::ReductionSumVector(
        lhs_data, row_sums, num_weights_matrices * lhs_rows, accum_depth);
    if (compute_row_sums) {
      *compute_row_sums = false;
    }
  }

  MatrixParams<int8_t> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = lhs_rows;
  lhs_params.cols = accum_depth;

  MatrixParams<int8_t> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = accum_depth;
  rhs_params.cols = rhs_cols;

  MatrixParams<int32_t> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = lhs_rows;
  dst_params.cols = rhs_cols;

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const int8_t* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const int8_t* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    const int32_t* ioff_ptr0 = input_offset + (b0 * ioff_ext0);
    const float* scale_ptr0 = scaling_factors + (b0 * ioff_ext0);
    const int32_t* woff_ptr0 = row_sums + (b0 * woff_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const int8_t* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const int8_t* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      const int32_t* ioff_ptr1 = ioff_ptr0 + (b1 * ioff_ext1);
      const float* scale_ptr1 = scale_ptr0 + (b1 * ioff_ext1);
      const int32_t* woff_ptr1 = woff_ptr0 + (b1 * woff_ext1);
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const int8_t* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const int8_t* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        const int32_t* ioff_ptr2 = ioff_ptr1 + (b2 * ioff_ext2);
        const float* scale_ptr2 = scale_ptr1 + (b2 * ioff_ext2);
        const int32_t* woff_ptr2 = woff_ptr1 + (b2 * woff_ext2);
        float* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                        b1 * batch_dim2 + b2) *
                                           lhs_rows * rhs_cols;
        GemmParams<int32_t, int32_t> gemm_params;
        cpu_backend_gemm::Gemm(lhs_params, lhs_ptr2, rhs_params, rhs_ptr2,
                               dst_params, accum_scratch, gemm_params, context);
        for (int j = 0; j < rhs_cols; ++j) {
          const float batch_scaling_factor = scale_ptr2[j];
          const float batch_offset = static_cast<float>(ioff_ptr2[j]);
          int i = 0;
#ifdef USE_NEON
          const float32x4_t scaling_factor0 = vdupq_n_f32(batch_scaling_factor);
          const float32x4_t scaling_factor1 = vdupq_n_f32(batch_scaling_factor);
          const int32x4_t input_offset0 = vdupq_n_s32(-batch_offset);
          const int32x4_t input_offset1 = vdupq_n_s32(-batch_offset);
          for (; i < lhs_rows - 8; i += 8) {
            // Load the row sums;
            const int32x4_t row_sum0 = vld1q_s32(woff_ptr2 + i);
            const int32x4_t row_sum1 = vld1q_s32(woff_ptr2 + i + 4);
            // Load the accumulated values.
            int idx = lhs_rows * j + i;
            const int32x4_t scratch_val0 = vld1q_s32(accum_scratch + idx);
            const int32x4_t scratch_val1 = vld1q_s32(accum_scratch + idx + 4);
            const int32x4_t dotprod0 =
                vmlaq_s32(scratch_val0, row_sum0, input_offset0);
            const int32x4_t dotprod1 =
                vmlaq_s32(scratch_val1, row_sum1, input_offset1);
            const float32x4_t float_val0 = vcvtq_f32_s32(dotprod0);
            const float32x4_t float_val1 = vcvtq_f32_s32(dotprod1);
            const float32x4_t result0 = vmlaq_f32(vld1q_f32(out_ptr + idx),
                                                  float_val0, scaling_factor0);
            const float32x4_t result1 = vmlaq_f32(vld1q_f32(out_ptr + idx + 4),
                                                  float_val1, scaling_factor1);
            vst1q_f32(out_ptr + idx, result0);
            vst1q_f32(out_ptr + idx + 4, result1);
          }
#endif  // USE_NEON
          for (; i < lhs_rows; ++i) {
            int idx = lhs_rows * j + i;
            accum_scratch[idx] -= woff_ptr2[i] * batch_offset;
            out_ptr[idx] += batch_scaling_factor * accum_scratch[idx];
          }
        }
      }
    }
  }
}

inline void BatchMatMul(const FullyConnectedParams& params,
                        const RuntimeShape& lhs_shape, const int8_t* lhs_data,
                        const RuntimeShape& rhs_shape, const int8_t* rhs_data,
                        const RuntimeShape& output_shape, int8_t* output_data,
                        CpuBackendContext* context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh mht_6(mht_6_v, 462, "", "./tensorflow/lite/kernels/internal/optimized/batch_matmul.h", "BatchMatMul");

  using ::tflite::cpu_backend_gemm::Gemm;
  using ::tflite::cpu_backend_gemm::GemmParams;
  using ::tflite::cpu_backend_gemm::MatrixParams;

  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  // Determine which dimension is the broadcast dimension.
  auto broadcast_dim = [](int lhs_dim, int rhs_dim) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh mht_7(mht_7_v, 476, "", "./tensorflow/lite/kernels/internal/optimized/batch_matmul.h", "lambda");

    if (lhs_dim == rhs_dim) return lhs_dim;
    if (lhs_dim == 1) return rhs_dim;
    TFLITE_DCHECK_EQ(rhs_dim, 1);
    return lhs_dim;
  };

  // Compute the "extent" for iterating on this dimension.
  // If we are broadcasting, then don't advance (i.e return 0).
  auto extent = [](const RuntimeShape& shape, int x) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSbatch_matmulDTh mht_8(mht_8_v, 488, "", "./tensorflow/lite/kernels/internal/optimized/batch_matmul.h", "lambda");

    if (shape.Dims(x) == 1) {
      return 0;
    }
    int prod = 1;
    for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
      prod *= shape.Dims(i);
    }
    return prod;
  };

  const int batch_dim0 =
      broadcast_dim(extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 =
      broadcast_dim(extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 =
      broadcast_dim(extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = extent(extended_lhs_shape, 0);
  const int lhs_ext1 = extent(extended_lhs_shape, 1);
  const int lhs_ext2 = extent(extended_lhs_shape, 2);
  const int rhs_ext0 = extent(extended_rhs_shape, 0);
  const int rhs_ext1 = extent(extended_rhs_shape, 1);
  const int rhs_ext2 = extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  MatrixParams<int8_t> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = lhs_rows;
  lhs_params.cols = accum_depth;
  lhs_params.zero_point = -filter_offset;

  MatrixParams<int8_t> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = accum_depth;
  rhs_params.cols = rhs_cols;
  rhs_params.zero_point = -input_offset;

  MatrixParams<int8_t> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = lhs_rows;
  dst_params.cols = rhs_cols;
  dst_params.zero_point = output_offset;

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const int8_t* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const int8_t* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const int8_t* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const int8_t* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const int8_t* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const int8_t* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        int8_t* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                         b1 * batch_dim2 + b2) *
                                            lhs_rows * rhs_cols;

        GemmParams<int32_t, int8_t> gemm_params;
        gemm_params.clamp_min = output_activation_min;
        gemm_params.clamp_max = output_activation_max;
        gemm_params.multiplier_fixedpoint = output_multiplier;
        gemm_params.multiplier_exponent = output_shift;
        cpu_backend_gemm::Gemm(lhs_params, lhs_ptr2, rhs_params, rhs_ptr2,
                               dst_params, out_ptr, gemm_params, context);
      }
    }
  }
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_BATCH_MATMUL_H_
