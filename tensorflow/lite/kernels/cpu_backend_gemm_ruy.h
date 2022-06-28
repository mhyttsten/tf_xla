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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RUY_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RUY_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_ruyDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_ruyDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_ruyDTh() {
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


#include "ruy/matrix.h"  // from @ruy
#include "ruy/mul_params.h"  // from @ruy
#include "ruy/ruy.h"  // from @ruy
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

inline ruy::CachePolicy ToRuyCachePolicy(CachePolicy cache_policy) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_ruyDTh mht_0(mht_0_v, 199, "", "./tensorflow/lite/kernels/cpu_backend_gemm_ruy.h", "ToRuyCachePolicy");

  switch (cache_policy) {
    case CachePolicy::kNeverCache:
      return ruy::CachePolicy::kNeverCache;
    case CachePolicy::kCacheIfLargeSpeedup:
      return ruy::CachePolicy::kCacheIfLargeSpeedup;
    case CachePolicy::kAlwaysCache:
      return ruy::CachePolicy::kAlwaysCache;
    default:
      TFLITE_DCHECK(false);
      return ruy::CachePolicy::kNeverCache;
  }
}

template <typename Scalar, typename DataPointer>
void MakeRuyMatrix(const MatrixParams<Scalar>& params, DataPointer data_ptr,
                   ruy::Matrix<Scalar>* dst, bool use_caching = false) {
  ruy::Order ruy_order = params.order == Order::kColMajor
                             ? ruy::Order::kColMajor
                             : ruy::Order::kRowMajor;
  ruy::MakeSimpleLayout(params.rows, params.cols, ruy_order,
                        dst->mutable_layout());
  // Note that ruy::Matrix::data is a ConstCheckingPtr, not a plain pointer.
  // It does care whether we assign to it a Scalar* or a const Scalar*.
  dst->set_data(data_ptr);
  dst->set_zero_point(params.zero_point);
  if (use_caching) {
    dst->set_cache_policy(ToRuyCachePolicy(params.cache_policy));
  }
}

// Floating-point case.
template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
struct MakeRuyMulParamsImpl final {
  static void Run(
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      ruy::MulParams<AccumScalar, DstScalar>* ruy_mul_params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_ruyDTh mht_1(mht_1_v, 239, "", "./tensorflow/lite/kernels/cpu_backend_gemm_ruy.h", "Run");

    static_assert(quantization_flavor == QuantizationFlavor::kFloatingPoint,
                  "");
    ruy_mul_params->set_bias(params.bias);
    ruy_mul_params->set_clamp_min(params.clamp_min);
    ruy_mul_params->set_clamp_max(params.clamp_max);
  }
};

// Integer-quantized case with destination type narrower than int32
template <typename DstScalar, QuantizationFlavor quantization_flavor>
struct MakeRuyMulParamsImpl<std::int32_t, DstScalar, quantization_flavor>
    final {
  static void Run(
      const GemmParams<std::int32_t, DstScalar, quantization_flavor>& params,
      ruy::MulParams<std::int32_t, DstScalar>* ruy_mul_params) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_ruyDTh mht_2(mht_2_v, 257, "", "./tensorflow/lite/kernels/cpu_backend_gemm_ruy.h", "Run");

    static_assert(sizeof(DstScalar) < sizeof(std::int32_t), "");
    if (quantization_flavor ==
        QuantizationFlavor::kIntegerWithUniformMultiplier) {
      ruy_mul_params->set_multiplier_fixedpoint(params.multiplier_fixedpoint);
      ruy_mul_params->set_multiplier_exponent(params.multiplier_exponent);
    }
    if (quantization_flavor ==
        QuantizationFlavor::kIntegerWithPerRowMultiplier) {
      ruy_mul_params->set_multiplier_fixedpoint_perchannel(
          params.multiplier_fixedpoint_perchannel);
      ruy_mul_params->set_multiplier_exponent_perchannel(
          params.multiplier_exponent_perchannel);
    }
    ruy_mul_params->set_bias(params.bias);
    ruy_mul_params->set_clamp_min(params.clamp_min);
    ruy_mul_params->set_clamp_max(params.clamp_max);
  }
};

// Raw-integer case with destination type int32.
template <QuantizationFlavor quantization_flavor>
struct MakeRuyMulParamsImpl<std::int32_t, std::int32_t, quantization_flavor>
    final {
  static void Run(
      const GemmParams<std::int32_t, std::int32_t, quantization_flavor>& params,
      ruy::MulParams<std::int32_t, std::int32_t>* ruy_mul_params) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_ruyDTh mht_3(mht_3_v, 286, "", "./tensorflow/lite/kernels/cpu_backend_gemm_ruy.h", "Run");

    ruy_mul_params->set_bias(params.bias);
  }
};

template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
void MakeRuyMulParams(
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
    ruy::MulParams<AccumScalar, DstScalar>* ruy_mul_params) {
  MakeRuyMulParamsImpl<AccumScalar, DstScalar, quantization_flavor>::Run(
      params, ruy_mul_params);
}

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplUsingRuy {
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      CpuBackendContext* context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_gemm_ruyDTh mht_4(mht_4_v, 311, "", "./tensorflow/lite/kernels/cpu_backend_gemm_ruy.h", "Run");

    ruy::Matrix<LhsScalar> ruy_lhs;
    ruy::Matrix<RhsScalar> ruy_rhs;
    ruy::Matrix<DstScalar> ruy_dst;
    MakeRuyMatrix(lhs_params, lhs_data, &ruy_lhs, context->use_caching());
    MakeRuyMatrix(rhs_params, rhs_data, &ruy_rhs, context->use_caching());
    MakeRuyMatrix(dst_params, dst_data, &ruy_dst);

    ruy::MulParams<AccumScalar, DstScalar> ruy_mul_params;
    MakeRuyMulParams(params, &ruy_mul_params);

    ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, context->ruy_context(),
             &ruy_dst);
  }
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RUY_H_
