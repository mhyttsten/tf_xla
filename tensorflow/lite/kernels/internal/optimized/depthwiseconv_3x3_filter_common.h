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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_3X3_FILTER_COMMON_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_3X3_FILTER_COMMON_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh() {
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


#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {
namespace depthwise_conv {

constexpr int kDepthwiseConvScratchWorkspaceSize = 10 * 10 * 64;
constexpr int kDepthwiseConvAdjustedBiasLimit = 64;
// In cases such as depth multiplication, we want to be able to load data from
// the workspace that is beyond the valid range. Macro-block sizes are adjusted
// to allow for this.
constexpr int kWorkspaceExtension = 16;

#ifdef USE_NEON

#ifndef __aarch64__
inline int8x16_t vqtbl4q_s8(int8x16x4_t a, int8x16_t b) {
  const uint8x16_t mask = vtstq_s8(b, vdupq_n_s8(8));

  // Delete bit 3 from the indices.
  const int8x16_t high_bits = vshrq_n_s8(b, 4);
  int8x16_t deleted_bit_3 = b;
  deleted_bit_3 = vsliq_n_s8(deleted_bit_3, high_bits, 3);

  int8x8x4_t repacked_data;

  // Calculate for lower indices.
  repacked_data.val[0] = vget_low_s8(a.val[0]);
  repacked_data.val[1] = vget_low_s8(a.val[1]);
  repacked_data.val[2] = vget_low_s8(a.val[2]);
  repacked_data.val[3] = vget_low_s8(a.val[3]);
  const int8x16_t output_for_lower =
      vcombine_s8(vtbl4_s8(repacked_data, vget_low_s8(deleted_bit_3)),
                  vtbl4_s8(repacked_data, vget_high_s8(deleted_bit_3)));

  // Calculate for high indices.
  repacked_data.val[0] = vget_high_s8(a.val[0]);
  repacked_data.val[1] = vget_high_s8(a.val[1]);
  repacked_data.val[2] = vget_high_s8(a.val[2]);
  repacked_data.val[3] = vget_high_s8(a.val[3]);
  const int8x16_t output_for_higher =
      vcombine_s8(vtbl4_s8(repacked_data, vget_low_s8(deleted_bit_3)),
                  vtbl4_s8(repacked_data, vget_high_s8(deleted_bit_3)));

  // Merge.
  int8x16_t output = vbslq_s8(mask, output_for_higher, output_for_lower);
  return output;
}
#endif  // !__aarch64__

// Convenience-compatibility functions.
// Compatibility: Intrinsics reflect a mixture of older and newer ARM
//     instructions. This actually results in ZIP1 / ZIP2 asm instructions, but
//     one intrinsic is provided. Also older instructions operated in place,
//     and it seems more defensive to assume that some versions of intrinsics
//     might reflect this
// Convenience: Callers in these kernels want both ZIP1 and ZIP2, and we do not
//     want the calling code to get cluttered with unpacking int8x16x2_t.
inline void vzipq_s8_in_place(int8x16_t* a, int8x16_t* b) {
  int8x16x2_t r8x16;
  r8x16 = vzipq_s8(*a, *b);
  *a = r8x16.val[0];
  *b = r8x16.val[1];
}

inline void vzipq_s8x2_in_place(int8x16_t* a, int8x16_t* b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_0(mht_0_v, 255, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "vzipq_s8x2_in_place");

  int16x8x2_t r16x8;
  r16x8 = vzipq_s16(vreinterpretq_s16_s8(*a), vreinterpretq_s16_s8(*b));
  *a = vreinterpretq_s8_s16(r16x8.val[0]);
  *b = vreinterpretq_s8_s16(r16x8.val[1]);
}

// Similar rationale to the zip-in_place functions, but callers only actually
// need the TRN1 asm instruction result.
inline void vtrn1_s8x2_in_place(int8x16_t* a, int8x16_t* b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_1(mht_1_v, 267, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "vtrn1_s8x2_in_place");

  int16x8x2_t r16x8;
  r16x8 = vtrnq_s16(vreinterpretq_s16_s8(*a), vreinterpretq_s16_s8(*b));
  *a = vreinterpretq_s8_s16(r16x8.val[0]);
}

// Similar rationale to the zip-in_place functions, but callers only actually
// need the ZIP1 or ZIP2 asm instruction results.
inline int8x16_t vzip1q_s8(int8x16_t a, int8x16_t b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_2(mht_2_v, 278, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "vzip1q_s8");

  return vzipq_s8(a, b).val[0];
}
inline int8x16_t vzip2q_s8(int8x16_t a, int8x16_t b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_3(mht_3_v, 284, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "vzip2q_s8");

  return vzipq_s8(a, b).val[1];
}

inline void biregister_rotate_8(int8x16_t* left, int8x16_t* right) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_4(mht_4_v, 291, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "biregister_rotate_8");

  *left = vreinterpretq_s8_u32(vshrq_n_u32(vreinterpretq_u32_s8(*left), 8));
  *left = vreinterpretq_s8_u32(vsliq_n_u32(vreinterpretq_u32_s8(*left),
                                           vreinterpretq_u32_s8(*right), 24));
  *right = vreinterpretq_s8_u32(vshrq_n_u32(vreinterpretq_u32_s8(*right), 8));
}

#ifndef __aarch64__
inline int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
  int32x4x2_t deinterleaved = vuzpq_s32(a, b);
  return vqaddq_s32(deinterleaved.val[0], deinterleaved.val[1]);
}
#endif  // !__aarch64__

#ifdef __ARM_FEATURE_DOTPROD
// The vdotq_lane_s32 takes int8x8t for the rhs parameter, whereas the actual
// instruction selects from between 4 32-bit (4x8-bit packed) sub-registers, an
// unusual interpretation of "lane".
inline int32x4_t vdotq_four_lane_s32(int32x4_t acc, int8x16_t lhs,
                                     int8x16_t rhs, const int lane) {
  switch (lane) {
    case 0:
      return vdotq_lane_s32(acc, lhs, vreinterpret_s32_s8(vget_low_s8(rhs)), 0);
    case 1:
      return vdotq_lane_s32(acc, lhs, vreinterpret_s32_s8(vget_low_s8(rhs)), 1);
    case 2:
      return vdotq_lane_s32(acc, lhs, vreinterpret_s32_s8(vget_high_s8(rhs)),
                            0);
    case 3:
    default:
      return vdotq_lane_s32(acc, lhs, vreinterpret_s32_s8(vget_high_s8(rhs)),
                            1);
  }
}

#else

inline int32x4_t vdotq_s32(int32x4_t acc, int8x16_t lhs, int8x16_t rhs) {
  int32x4_t sum0 = vpaddlq_s16(vmull_s8(vget_low_s8(lhs), vget_low_s8(rhs)));
  int32x4_t sum1 = vpaddlq_s16(vmull_s8(vget_high_s8(lhs), vget_high_s8(rhs)));
  int32x4_t sum = vpaddq_s32(sum0, sum1);
  return vaddq_s32(acc, sum);
}

inline int32x4_t vdotq_four_lane_s32(int32x4_t acc, int8x16_t lhs,
                                     int8x16_t rhs, int lane) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_5(mht_5_v, 339, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "vdotq_four_lane_s32");

  int8x8_t lane_rhs;
  if (lane == 0) {
    lane_rhs = vreinterpret_s8_s32(
        vdup_lane_s32(vreinterpret_s32_s8(vget_low_s8(rhs)), 0));
  } else if (lane == 1) {
    lane_rhs = vreinterpret_s8_s32(
        vdup_lane_s32(vreinterpret_s32_s8(vget_low_s8(rhs)), 1));
  } else if (lane == 2) {
    lane_rhs = vreinterpret_s8_s32(
        vdup_lane_s32(vreinterpret_s32_s8(vget_high_s8(rhs)), 0));
  } else {
    lane_rhs = vreinterpret_s8_s32(
        vdup_lane_s32(vreinterpret_s32_s8(vget_high_s8(rhs)), 1));
  }
  int32x4_t sum0 = vpaddlq_s16(vmull_s8(vget_low_s8(lhs), lane_rhs));
  int32x4_t sum1 = vpaddlq_s16(vmull_s8(vget_high_s8(lhs), lane_rhs));
  int32x4_t sum = vpaddq_s32(sum0, sum1);
  return vaddq_s32(acc, sum);
}

#endif  // !__ARM_FEATURE_DOTPROD
#endif  // ARM NEON

//  This structure is typically used for reducing the magnitude of outputs, and
//  the historical name reflects that.
template <DepthwiseConvOutputRounding output_rounding>
struct DivideByPOT {};

template <>
struct DivideByPOT<DepthwiseConvOutputRounding::kAwayFromZero> {
  template <typename IntegerType>
  static inline IntegerType Run(IntegerType x, int exponent) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_6(mht_6_v, 374, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "Run");

    return RoundingDivideByPOT(x, exponent);
  }
  // Mult versions use the exponents directly, rather than negated.
  template <typename IntegerType>
  static inline IntegerType RunMult(IntegerType x, int exponent) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_7(mht_7_v, 382, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "RunMult");

    return RoundingDivideByPOT(x, -exponent);
  }
};

#ifdef USE_NEON
template <>
struct DivideByPOT<DepthwiseConvOutputRounding::kUpward> {
  template <typename IntegerType>
  static inline IntegerType Run(IntegerType x, int exponent) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_8(mht_8_v, 394, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "Run");

    return vqrshlq_s32(x, vdupq_n_s32(static_cast<int32>(-exponent)));
  }
  template <typename IntegerType>
  static inline IntegerType RunMult(IntegerType x, IntegerType exponent) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_9(mht_9_v, 401, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "RunMult");

    return vqrshlq_s32(x, exponent);
  }
  template <typename IntegerType>
  static inline IntegerType RunMult(IntegerType x, int exponent) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_10(mht_10_v, 408, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "RunMult");

    return vqrshlq_s32(x, vdupq_n_s32(static_cast<int32>(exponent)));
  }
};
#endif  // ARM NEON

// See CategorizeDotProductKernel for definitive taxonomy.
enum class DotProduct3x3KernelType {
  kNone = 0,  // Parameter combination is not supported for dot product kernels.
  kPlain,
  kWithDepthMultiplicationStride1,
  kWithDepthMultiplicationStride2,
  kStride2,
};

enum class QuantizationType {
  kNonPerChannelUint8 = 0,
  kPerChannelInt8 = 1,
};

template <QuantizationType quantization_type>
struct QuantizationTypeImpl {};

template <>
struct QuantizationTypeImpl<QuantizationType::kNonPerChannelUint8> {
  typedef uint8 ExternalType;

  static constexpr int kIntSymmetricZeroPoint = 128;
  static constexpr uint8 kUint8SignBit = 0x80;
};

template <>
struct QuantizationTypeImpl<QuantizationType::kPerChannelInt8> {
  typedef int8 ExternalType;

  static constexpr int kIntSymmetricZeroPoint = 0;
  static constexpr uint8 kUint8SignBit = 0x0;
};

template <
    QuantizationType quantization_type = QuantizationType::kNonPerChannelUint8>
inline DotProduct3x3KernelType CategorizeDotProductKernel(
    const RuntimeShape& input_shape, const RuntimeShape& filter_shape,
    const RuntimeShape& output_shape, const DepthwiseParams& params,
    const int32* output_shift_ptr = nullptr) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_11(mht_11_v, 455, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "CategorizeDotProductKernel");

  constexpr int kSymmetricZeroPoint =
      QuantizationTypeImpl<quantization_type>::kIntSymmetricZeroPoint;
  const int padding =
      std::max(params.padding_values.width, params.padding_values.height);
  const int stride = params.stride_width;
  const int32 input_depth = input_shape.Dims(3);
  const int32 depth_multiplier = params.depth_multiplier;
  const int32 filter_height = filter_shape.Dims(1);
  const int32 filter_width = filter_shape.Dims(2);

  bool supported = stride == params.stride_height && stride <= 2 &&
                   padding <= 1 && filter_width == 3 && filter_height == 3 &&
                   params.dilation_width_factor == 1 &&
                   params.dilation_height_factor == 1 &&
                   (((input_depth % 8) == 0 && depth_multiplier == 1) ||
                    (input_depth == 1 && depth_multiplier > 1));

  if (!supported) {
    return DotProduct3x3KernelType::kNone;
  }

  if (params.weights_offset != -kSymmetricZeroPoint) {
    return DotProduct3x3KernelType::kNone;
  }

  if (quantization_type == QuantizationType::kPerChannelInt8) {
    if (output_shift_ptr == nullptr) {
      return DotProduct3x3KernelType::kNone;
    }
  } else if (params.output_shift > 0) {
    return DotProduct3x3KernelType::kNone;
  }

  if (params.depth_multiplier == 1) {
    if (stride == 1) {
      return DotProduct3x3KernelType::kPlain;
    } else if (stride == 2) {
      return DotProduct3x3KernelType::kStride2;
    } else {
      return DotProduct3x3KernelType::kNone;
    }
  } else {
    if (stride == 1) {
      return DotProduct3x3KernelType::kWithDepthMultiplicationStride1;
    } else if (stride == 2) {
      return DotProduct3x3KernelType::kWithDepthMultiplicationStride2;
    } else {
      return DotProduct3x3KernelType::kNone;
    }
  }
}

// Encapsulates constant parameters used in DepthwiseConv.
// 64-bit is used for types that will be added to 64-bit addresses in asm.
struct DepthwiseConvParams {
  int64_t input_depth;
  int64_t input_row_size;
  int64_t output_depth;
  int64_t output_row_size;
  int64_t filter_row_size;
  int32 input_offset;
  int32 output_offset;
  int32 filter_offset;
  int32 output_multiplier;
  int32 output_activation_min;
  int32 output_activation_max;
  int32 output_right_shift;
  int32 input_width;
  int32 input_height;
  int32 stride_width;
  int32 stride_height;
  int32 output_width;
  int32 output_height;
  float float_output_activation_min;
  float float_output_activation_max;
};

// Encapsulates constant parameters used in DepthwiseConv using dot-product ops.
// 64-bit is used for types that will be added to 64-bit addresses in asm.
//
// This structure is specifically designed for use in asm.
struct DepthwiseConvDotProdParams {
  int64_t input_depth;
  int64_t output_depth;
  int32 stride;
  int32 bias_increment;
  //
  int32 input_offset;
  int32 output_offset;
  int32 output_multiplier;
  int32 output_shift;
  int32 quantized_activation_min;
  int32 quantized_activation_max;
  //
  int32 padding_left;
  int32 padding_right;
  int32 padding_top;
  int32 padding_bottom;
  //
  int32 depth_micro_repeats;
  //
  int32 width_macro_count;
  int32 input_width_overall_micro_repeats;
  int32 input_width_micro_repeats;
  int32 residual_width;
  int32 output_width_overall_micro_repeats;
  int32 output_width_micro_repeats;
  int32 output_residual_width;
  int32 workspace_width_micro_repeats;
  //
  int32 height_macro_count;
  int32 inbound_block_height;
  int32 outbound_block_height;
  int32 input_height_stride;
  int32 output_height_stride;
  int32 workspace_height_stride;
  //
  int32 four_over_stride;
  //
  const int32* output_multiplier_per_channel;
  const int32* output_shift_per_channel;
};

template <DepthwiseConvOutputRounding output_rounding, int32 kDepth,
          int32 kStrideWidth, int32 kStrideHeight>
struct DepthwiseConvWindow {};

template <DepthwiseConvOutputRounding output_rounding, int32 kDepth,
          int32 kStrideWidth, int32 kStrideHeight>
struct DepthwiseConvWindowPerChannel {};

enum class EdgeType { kCorner, kHorizontal, kVertical, kCenter };

template <DepthwiseConvOutputRounding output_rounding, EdgeType kEdgeType,
          int kPadWidth, int kPadHeight>
struct DepthwiseConvPartial {};

template <DepthwiseConvOutputRounding output_rounding, EdgeType kEdgeType,
          int kPadWidth, int kPadHeight>
struct DepthwiseConvPartialPerChannel {};

// Copies a subset of the input designated by |input_ptr| into |output_ptr|
// with the specified output dimensions. Supports output depths of 64 only as
// this is the cache line size.
template <typename T>
inline void ShuffleInput(const T* input_ptr, int64_t input_depth,
                         int32 input_width, int32 input_height,
                         int64_t output_depth, int32 output_width,
                         int32 output_height, T* output_ptr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_12(mht_12_v, 607, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "ShuffleInput");

  const int64_t input_row_size = input_depth * input_width;
  for (int32 y = 0; y < output_height; y++) {
    const T* ptr = input_ptr;
    for (int32 x = 0; x < output_width; x++) {
      memcpy(output_ptr, ptr, output_depth);
      output_ptr += output_depth;
      ptr += input_depth;
    }
    input_ptr += input_row_size;
  }
}

// Calculates the input size depending on stride and output.
inline int32 get_shuffle_input_size(int32 stride, int32 output) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_13(mht_13_v, 624, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "get_shuffle_input_size");

  return stride * (output - 1) + 3;
}

// Indicates the input and output dimensions used when shuffling input
// activations.
struct ShuffleParams {
  int32 output_width;
  int32 output_height;
  int32 input_width;
  int32 input_height;

  ShuffleParams() = default;
  ShuffleParams(int32 output_width, int32 output_height, int32 stride_width,
                int32 stride_height)
      : output_width(output_width),
        output_height(output_height),
        input_width(get_shuffle_input_size(stride_width, output_width)),
        input_height(get_shuffle_input_size(stride_height, output_height)) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_14(mht_14_v, 645, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "ShuffleParams");
}
};

template <
    QuantizationType quantization_type = QuantizationType::kNonPerChannelUint8>
inline bool Fast3x3FilterKernelSupported(
    const RuntimeShape& input_shape, const RuntimeShape& filter_shape,
    int32 stride_width, int32 stride_height, int32 dilation_width_factor,
    int32 dilation_height_factor, int32 pad_width, int32 pad_height,
    int32 depth_multiplier, const RuntimeShape& output_shape,
    int32 output_shift, const int32* output_shift_ptr = nullptr) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_3x3_filter_commonDTh mht_15(mht_15_v, 658, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h", "Fast3x3FilterKernelSupported");

  const int32 input_height = input_shape.Dims(1);
  const int32 input_width = input_shape.Dims(2);
  const int32 input_depth = input_shape.Dims(3);
  const int32 filter_height = filter_shape.Dims(1);
  const int32 filter_width = filter_shape.Dims(2);
  const int32 output_height = output_shape.Dims(1);
  const int32 output_width = output_shape.Dims(2);

  bool supported =
      filter_width == 3 && filter_height == 3 && depth_multiplier == 1 &&
      (stride_width == 1 || stride_width == 2) &&
      (stride_height == 1 || stride_height == 2) &&
      (stride_width == stride_height) && (pad_width == 0 || pad_width == 1) &&
      (pad_height == 0 || pad_height == 1) && (pad_width == pad_height) &&
      (input_depth % 8) == 0 && (output_shift <= 0) &&
      dilation_width_factor == 1 && dilation_height_factor == 1;

  if (!supported) {
    return false;
  }

  // Handle case where padding is zero but padding type is not kValid.
  // This would require special boundary case handling that is not supported.

  const int32 out_x = output_width - 1;
  const int32 out_y = output_height - 1;

  const int32 in_x_origin = (out_x * stride_width) - pad_width;
  const int32 in_y_origin = (out_y * stride_height) - pad_height;

  const int32 in_x_end = in_x_origin + filter_width;
  const int32 in_y_end = in_y_origin + filter_height;

  // Supported only if filter on the right and bottom boundary lies completely
  // within the input if padding is zero.
  if (pad_width == 0 && pad_height == 0) {
    return in_x_end <= input_width && in_y_end <= input_height;
  }

  // Else if padding is 1, supported if bottom right filter lies +1 past input
  // width and height.
  supported = in_x_end <= (input_width + 1) && in_y_end <= (input_height + 1);

  if (!supported) {
    return false;
  }

  // Shapes with width 1 and height > 1, and vice versa are not supported yet.
  if (input_width == 1) {
    supported = (input_width == input_height);
  } else if (input_height == 1) {
    supported = (input_width == input_height);
  }
  return supported;
}

// Permute filter data, and adjust bias data to account for symmetric input
// offset. Details are provided in the implementation of the
// kUseCModel3x3DotProduct version.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation,
          QuantizationType quantization_type>
struct ProcessPerDepth {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

// Copy a macro block of data from the input buffer into the workspace,
// permuting data within each micro block.
//
// (a) Copy a macro block of data, padding as required along the width and
//     height.
// (b) Transpose the data within each micro block.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation,
          QuantizationType quantization_type,
          DepthwiseConvDepthMultiplication depth_multiplication,
          int32 max_padding>
struct PackMacroBlock {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

// Apply filter to macro block of input data and store results. Details are
// provided in the implementation of the kUseCModel3x3DotProduct version.
//
// Parameters for repeats and residual sizes are in terms of outputs.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation,
          QuantizationType quantization_type,
          DepthwiseConvDepthMultiplication depth_multiplication, int32 stride>
struct KernelMacroBlock {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

#if defined(__aarch64__)
// Experiments suggest that a modest performance improvement is seen, at least
// on 855 chipset big cores, with cache hints.
template <typename T>
inline void PreloadInputBlock(
    const T* input_block_data,
    const DepthwiseConvDotProdParams* function_params) {
  // Preload.
  const int input_width_micro_repeats =
      function_params->input_width_micro_repeats;
  const int block_height = function_params->inbound_block_height;
  const int residual_width = function_params->residual_width;
  const int input_height_stride = function_params->input_height_stride;
  const int input_depth = function_params->input_depth;

  const int total_width = 4 * input_width_micro_repeats + residual_width;
  const T* row_ptr = input_block_data;
  for (int k_height = 0; k_height < block_height; ++k_height) {
    const T* ptr = row_ptr;
    for (int j = 0; j < total_width; ++j) {
      // Input data is loaded once.
      optimized_ops_preload_l1_keep(ptr);
      ptr += input_depth;
    }
    row_ptr += input_height_stride;
  }
}
#endif  // __aarch64__

}  // namespace depthwise_conv
}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_3X3_FILTER_COMMON_H_
