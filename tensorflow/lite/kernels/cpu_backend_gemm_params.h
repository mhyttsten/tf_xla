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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_PARAMS_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_PARAMS_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_paramsDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_paramsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_paramsDTh() {
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


#include <cstdint>
#include <limits>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {

namespace cpu_backend_gemm {

// Matrix storage order: column-major or row-major.
enum class Order { kColMajor, kRowMajor };

enum class CachePolicy : std::uint8_t {
  kNeverCache,
  kCacheIfLargeSpeedup,
  kAlwaysCache,
};

inline CachePolicy DefaultCachePolicy(bool is_constant_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_paramsDTh mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/cpu_backend_gemm_params.h", "DefaultCachePolicy");

  return is_constant_data ? CachePolicy::kCacheIfLargeSpeedup
                          : CachePolicy::kNeverCache;
}

// MatrixParams encapsulates the parameters that Gemm needs about each
// matrix, besides the buffer data pointer.
// Compare to ruy::Matrix, which also encapsulates the data pointer.
// Rationale for leaving the data pointer out of here: doing so
// requires complicated const-correctness mechanics. See
// ruy::ConstCheckingPtr.
template <typename Scalar>
struct MatrixParams {
  // Storage layout order. For now we only do plain linear non-strided
  // layout. It would be easy to support a stride if needed.
  Order order = Order::kColMajor;
  // Number of rows of the matrix.
  int rows = 0;
  // Number of columns of the matrix.
  int cols = 0;
  // The zero_point, i.e. which Scalar value is to be interpreted as zero.
  // When Scalar is floating-point, this must be 0.
  Scalar zero_point = 0;
  // When the data pointed to by this matrix is constant data, so that it is
  // valid to assume that equality of pointers implies equality of data,
  // a CachePolicy may be used instead of the default kNeverCache,
  // which will enable ruy to take advantage of this constancy of the data to
  // cache the packing work, which can be a large speedup in matrix*vector
  // and other narrow shapes.
  CachePolicy cache_policy = CachePolicy::kNeverCache;
};

// Enumeration of broad categories of Gemm.
//
// The primary reason for this to exist is to allow Gemm to compile
// only uniform-quantized or only per-channel-quantized code paths.
// This is unneeded with ruy as the back-end, as this is only a runtime
// difference in ruy, but with gemmlowp these really are separate code
// paths and templatizing in a QuantizationFlavor is necessary to avoid
// compiling unused gemmlowp code. Indeed, TFLite currently uses
// uint8 with uniform quantization and int8 with per-channel quantization,
// and does not use uint8 with per-channel. We want to avoid compiling
// the gemmlowp uint8 per-channel path when gemmlowp is the back-end.
//
// It's possible to drop this in the future if gemmlowp goes away and no
// other then-relevant backend library handles quantized paths in a way that
// requires knowing this at compile-time.
enum class QuantizationFlavor {
  // Floating-point Gemm: the accumulators are not multiplied by any
  // 'multiplier'.
  kFloatingPoint,
  // Quantized Gemm using a single multiplier for all accumulators.
  kIntegerWithUniformMultiplier,
  // Quantized Gemm using a separate multipliers for accumulators of each
  // row of the destination matrix. This is what is called 'per-channel'
  // in GemmParams. Here we use the more specific 'per-row' terminology
  // to allow for the possibility of 'per-column' in the future, and to
  // allow for that to be a separate code path in some back-end such as
  // gemmlowp.
  kIntegerWithPerRowMultiplier
};

// Additional parameters that Gemm needs, beyond what falls into
// the MatrixParams that it takes. Compare to ruy::Spec.
//
// Decoupling AccumScalar from DstScalar (rather than deducing it from that)
// is useful future-proofing. Think of a float16 path using float32 accum.
//
// QuantizationFlavor is passed here even though it's technically not used
// in this class. This is so that we retain the ability in the future to
// specialize this class for quantization flavor, and this allows for
// Gemm to be templatized in quantization_flavor via the GemmParams that it
// takes, allowing for automatic template parameter deduction to take place,
// so that most call sites don't need to specify a QuantizationFlavor
// (only those that need perchannel quantization do).
template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor =
              std::is_floating_point<AccumScalar>::value
                  ? QuantizationFlavor::kFloatingPoint
                  : QuantizationFlavor::kIntegerWithUniformMultiplier>
struct GemmParams {
  // Only for non-floating-point cases. The fixed-point part (i.e. the mantissa)
  // of the multiplier by which accumulators are multiplied before being casted
  // to the destination type.
  AccumScalar multiplier_fixedpoint = 0;
  // Only for non-floating-point cases. The exponent part of the aforementioned
  // multiplier.
  int multiplier_exponent = 0;
  // Per-channel variant of multiplier_fixedpoint. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_fixedpoint.
  const AccumScalar* multiplier_fixedpoint_perchannel = nullptr;
  // Per-channel variant of multiplier_exponent. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_exponent.
  //
  // Either none or both of multiplier_exponent_perchannel and
  // multiplier_fixedpoint_perchannel must be nullptr.
  const int* multiplier_exponent_perchannel = nullptr;
  // The bias vector data, if not null.
  const AccumScalar* bias = nullptr;
  // min clamp bound of destination values.
  DstScalar clamp_min = std::is_floating_point<DstScalar>::value
                            ? -std::numeric_limits<DstScalar>::infinity()
                            : std::numeric_limits<DstScalar>::lowest();
  // max clamp bound of destination values.
  DstScalar clamp_max = std::is_floating_point<DstScalar>::value
                            ? std::numeric_limits<DstScalar>::infinity()
                            : std::numeric_limits<DstScalar>::max();
};

/* Convenience typedefs */

template <typename DstScalar>
using QuantizedGemmParams = GemmParams<std::int32_t, DstScalar>;

using FloatGemmParams = GemmParams<float, float>;

/* Validation functions */

// Note that this uses TFLITE_DCHECK from kernels/internal/compatibility.h
// and not TF_LITE_ASSERT from op_macros.h. We want this to be explicitly
// debug-build-only assertions so that there's not reason not to
// generously validate, and TF_LITE_ASSERT is actually at the moment
// a release-build assertion. See b/131587258.

// Validates self-consistency of GemmParams.
template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
void ValidateGemmParams(
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params) {
  // Guard consistency of the quantized multiplier fields.
  if (quantization_flavor == QuantizationFlavor::kFloatingPoint) {
    TFLITE_DCHECK(!params.multiplier_fixedpoint);
    TFLITE_DCHECK(!params.multiplier_exponent);
    TFLITE_DCHECK(!params.multiplier_fixedpoint_perchannel);
    TFLITE_DCHECK(!params.multiplier_exponent_perchannel);
  } else if (quantization_flavor ==
                 QuantizationFlavor::kIntegerWithUniformMultiplier &&
             !std::is_same<DstScalar, int32_t>::value) {
    TFLITE_DCHECK(params.multiplier_fixedpoint);
    // Nothing to check about multiplier_exponent
    TFLITE_DCHECK(!params.multiplier_fixedpoint_perchannel);
    TFLITE_DCHECK(!params.multiplier_exponent_perchannel);
  } else if (quantization_flavor ==
                 QuantizationFlavor::kIntegerWithPerRowMultiplier &&
             !std::is_same<DstScalar, int32_t>::value) {
    TFLITE_DCHECK(!params.multiplier_fixedpoint);
    TFLITE_DCHECK(!params.multiplier_exponent);
    TFLITE_DCHECK(params.multiplier_fixedpoint_perchannel);
    TFLITE_DCHECK(params.multiplier_exponent_perchannel);
  } else {
    // For the get raw accumulator case, we should make sure none of the
    // quantization params are set.
    TFLITE_DCHECK(!params.multiplier_fixedpoint);
    TFLITE_DCHECK(!params.multiplier_exponent);
    TFLITE_DCHECK(!params.multiplier_fixedpoint_perchannel);
    TFLITE_DCHECK(!params.multiplier_exponent_perchannel);
  }
}

namespace detail {

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct ValidateTypes {
  // This generic implementation is for quantized flavors.
  // kFloatingPoint will be a specialization below.
  static_assert(!std::is_floating_point<LhsScalar>::value, "");
  static_assert(!std::is_floating_point<RhsScalar>::value, "");
  static_assert(!std::is_floating_point<AccumScalar>::value, "");
  // No requirement on DstScalar --- we might in the future allow it
  // to be floating point even in a quantized Gemm.
};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct ValidateTypes<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                     QuantizationFlavor::kFloatingPoint> {
  static_assert(std::is_floating_point<LhsScalar>::value, "");
  static_assert(std::is_floating_point<RhsScalar>::value, "");
  static_assert(std::is_floating_point<AccumScalar>::value, "");
  static_assert(std::is_floating_point<DstScalar>::value, "");
};

}  // namespace detail

// Validates overall consistency of all the parameters taken by a Gemm call:
// the 3 MatrixParams and the GemmParams.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
void ValidateParams(
    const MatrixParams<LhsScalar>& lhs_params,
    const MatrixParams<RhsScalar>& rhs_params,
    const MatrixParams<DstScalar>& dst_params,
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params) {
  (void)detail::ValidateTypes<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                              quantization_flavor>();
  ValidateGemmParams(params);
}

// Test if the Gemm is degenerate in some way, e.g. nonsensical dimenions.
template <typename LhsScalar, typename RhsScalar, typename DstScalar>
bool IsValidGemm(const MatrixParams<LhsScalar>& lhs_params,
                 const MatrixParams<RhsScalar>& rhs_params,
                 const MatrixParams<DstScalar>& dst_params) {
  bool valid = true;
  valid &= lhs_params.rows >= 1;
  valid &= lhs_params.cols >= 1;
  valid &= rhs_params.rows >= 1;
  valid &= rhs_params.cols >= 1;
  valid &= dst_params.rows >= 1;
  valid &= dst_params.cols >= 1;
  valid &= lhs_params.cols == rhs_params.rows;
  valid &= rhs_params.cols == dst_params.cols;
  valid &= lhs_params.rows == lhs_params.rows;
  return valid;
}

}  // namespace cpu_backend_gemm

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_PARAMS_H_
