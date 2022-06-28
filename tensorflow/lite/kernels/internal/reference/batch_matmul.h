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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_MATMUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_MATMUL_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbatch_matmulDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbatch_matmulDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbatch_matmulDTh() {
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

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {
namespace batch_matmul {

// Determine which dimension is the broadcast dimension.
inline int broadcast_dim(int lhs_dim, int rhs_dim) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbatch_matmulDTh mht_0(mht_0_v, 200, "", "./tensorflow/lite/kernels/internal/reference/batch_matmul.h", "broadcast_dim");

  if (lhs_dim == rhs_dim) return lhs_dim;
  if (lhs_dim == 1) return rhs_dim;
  TFLITE_DCHECK_EQ(rhs_dim, 1);
  return lhs_dim;
}

// Compute the "extent" for iterating on this dimension.
// If we are broadcasting, then don't advance (i.e return 0).
inline int extent(const RuntimeShape& shape, int x) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbatch_matmulDTh mht_1(mht_1_v, 212, "", "./tensorflow/lite/kernels/internal/reference/batch_matmul.h", "extent");

  if (shape.Dims(x) == 1) {
    return 0;
  }
  int prod = 1;
  for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
    prod *= shape.Dims(i);
  }
  return prod;
}

}  // namespace batch_matmul

template <typename Ta, typename Tb, typename Tout>
inline void BatchMatMul(const RuntimeShape& lhs_shape, const Ta* lhs_data,
                        const RuntimeShape& rhs_shape, const Tb* rhs_data,
                        const RuntimeShape& output_shape, Tout* output_data) {
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  const int batch_dim0 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = batch_matmul::extent(extended_lhs_shape, 0);
  const int lhs_ext1 = batch_matmul::extent(extended_lhs_shape, 1);
  const int lhs_ext2 = batch_matmul::extent(extended_lhs_shape, 2);
  const int rhs_ext0 = batch_matmul::extent(extended_rhs_shape, 0);
  const int rhs_ext1 = batch_matmul::extent(extended_rhs_shape, 1);
  const int rhs_ext2 = batch_matmul::extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const Ta* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const Tb* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const Ta* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const Tb* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const Ta* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const Tb* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        Tout* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                       b1 * batch_dim2 + b2) *
                                          lhs_rows * rhs_cols;
        for (int j = 0; j < rhs_cols; ++j) {
          for (int i = 0; i < lhs_rows; ++i) {
            Tout total = 0;
            for (int k = 0; k < accum_depth; ++k) {
              total += static_cast<Tout>(lhs_ptr2[accum_depth * i + k]) *
                       static_cast<Tout>(rhs_ptr2[j * accum_depth + k]);
            }
            int idx = lhs_rows * j + i;
            out_ptr[idx] = total;
          }
        }
      }
    }
  }
}

inline void BatchMatMul(const RuntimeShape& lhs_shape, const int8_t* lhs_data,
                        const RuntimeShape& rhs_shape, const int8_t* rhs_data,
                        const float* scaling_factors,
                        const int32_t* input_offset, int32_t* row_sums,
                        const RuntimeShape& output_shape, float* output_data,
                        bool* compute_row_sums) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSbatch_matmulDTh mht_2(mht_2_v, 289, "", "./tensorflow/lite/kernels/internal/reference/batch_matmul.h", "BatchMatMul");

  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  const int batch_dim0 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = batch_matmul::extent(extended_lhs_shape, 0);
  const int lhs_ext1 = batch_matmul::extent(extended_lhs_shape, 1);
  const int lhs_ext2 = batch_matmul::extent(extended_lhs_shape, 2);
  const int rhs_ext0 = batch_matmul::extent(extended_rhs_shape, 0);
  const int rhs_ext1 = batch_matmul::extent(extended_rhs_shape, 1);
  const int rhs_ext2 = batch_matmul::extent(extended_rhs_shape, 2);

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
        for (int j = 0; j < rhs_cols; ++j) {
          const float batch_scaling_factor = scale_ptr2[j];
          const float batch_offset = static_cast<float>(ioff_ptr2[j]);
          for (int i = 0; i < lhs_rows; ++i) {
            int32_t total = 0;
            for (int k = 0; k < accum_depth; ++k) {
              total +=
                  lhs_ptr2[accum_depth * i + k] * rhs_ptr2[j * accum_depth + k];
            }
            int32_t row_sum = woff_ptr2[i];
            total -= row_sum * batch_offset;
            int idx = lhs_rows * j + i;
            out_ptr[idx] += batch_scaling_factor * total;
          }
        }
      }
    }
  }
}

template <typename T, typename AccumT>
inline void BatchMatMul(const FullyConnectedParams& params,
                        const RuntimeShape& lhs_shape, const T* lhs_data,
                        const RuntimeShape& rhs_shape, const T* rhs_data,
                        const RuntimeShape& output_shape, T* output_data) {
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  const int batch_dim0 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = batch_matmul::extent(extended_lhs_shape, 0);
  const int lhs_ext1 = batch_matmul::extent(extended_lhs_shape, 1);
  const int lhs_ext2 = batch_matmul::extent(extended_lhs_shape, 2);
  const int rhs_ext0 = batch_matmul::extent(extended_rhs_shape, 0);
  const int rhs_ext1 = batch_matmul::extent(extended_rhs_shape, 1);
  const int rhs_ext2 = batch_matmul::extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const T* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const T* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const T* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const T* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const T* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const T* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        T* out_ptr = output_data +
                     ((b0 * batch_dim1 * batch_dim2) + b1 * batch_dim2 + b2) *
                         lhs_rows * rhs_cols;

        for (int j = 0; j < rhs_cols; ++j) {
          for (int i = 0; i < lhs_rows; ++i) {
            AccumT total = 0;
            for (int k = 0; k < accum_depth; ++k) {
              AccumT lhs_val = lhs_ptr2[accum_depth * i + k];
              AccumT rhs_val = rhs_ptr2[accum_depth * j + k];
              total += (lhs_val + filter_offset) * (rhs_val + input_offset);
            }
            int32_t total_scaled = MultiplyByQuantizedMultiplier(
                total, output_multiplier, output_shift);
            total_scaled += output_offset;
            total_scaled = std::max(total_scaled, output_activation_min);
            total_scaled = std::min(total_scaled, output_activation_max);
            const int idx = lhs_rows * j + i;
            out_ptr[idx] = static_cast<T>(total_scaled);
          }
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_MATMUL_H_
