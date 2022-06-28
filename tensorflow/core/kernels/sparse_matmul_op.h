/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_MATMUL_OP_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_MATMUL_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh() {
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


#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/types.h"

#if defined(PLATFORM_WINDOWS)
#include "tensorflow/core/platform/windows/intrinsics_port.h"
#endif

namespace Eigen {
namespace internal {

// Return the float representation of the bfloat16 value
// in the lower 16-bits of input
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pexpand_bf16_l(const Packet& from) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pexpand_bf16_l");

  tensorflow::uint32 tmp;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  tmp = (reinterpret_cast<const tensorflow::uint32&>(from)) & 0xffff0000;
#else
  tmp = (reinterpret_cast<const tensorflow::uint32&>(from) << 16) & 0xffff0000;
#endif
  return reinterpret_cast<const float&>(tmp);
}

// Return the float representation of the bfloat16 value
// in the upper 16-bits of input
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pexpand_bf16_u(const Packet& from) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_1(mht_1_v, 218, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pexpand_bf16_u");

  tensorflow::uint32 tmp;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  tmp = (reinterpret_cast<const tensorflow::uint32&>(from) << 16) & 0xffff0000;
#else
  tmp = (reinterpret_cast<const tensorflow::uint32&>(from)) & 0xffff0000;
#endif
  return reinterpret_cast<const float&>(tmp);
}

// Specialization non-scalar version on non-sse.
// Enable vectorization on z13 and higher
#if defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX) || \
    defined(EIGEN_VECTORIZE_NEON) || defined(EIGEN_VECTORIZE_ZVECTOR)
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet4f pexpand_bf16_l(const Packet4f& from) {
  float r[4];
  tensorflow::uint32 p[4];
  pstoreu(r, from);
  tensorflow::uint32* ir = reinterpret_cast<tensorflow::uint32*>(r);
  p[0] = (ir[0] << 16) & 0xffff0000;
  p[1] = ir[0] & 0xffff0000;
  p[2] = (ir[1] << 16) & 0xffff0000;
  p[3] = ir[1] & 0xffff0000;
  return ploadu<Packet4f>(reinterpret_cast<float*>(p));
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet4f pexpand_bf16_u(const Packet4f& from) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_2(mht_2_v, 249, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pexpand_bf16_u");

  float r[4];
  tensorflow::uint32 p[4];
  pstoreu(r, from);
  tensorflow::uint32* ir = reinterpret_cast<tensorflow::uint32*>(r);
  p[0] = (ir[2] << 16) & 0xffff0000;
  p[1] = ir[2] & 0xffff0000;
  p[2] = (ir[3] << 16) & 0xffff0000;
  p[3] = ir[3] & 0xffff0000;
  return ploadu<Packet4f>(reinterpret_cast<float*>(p));
}
#endif

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pinterleave4x64(const Packet& from) {
  return from;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pbroadcast_first(const Packet& a) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_3(mht_3_v, 271, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_first");

  return a;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pbroadcast_second(const Packet& a) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_4(mht_4_v, 279, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_second");

  assert(false && "Not applicable to Scalar Values");
  return a;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pbroadcast_third(const Packet& a) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_5(mht_5_v, 288, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_third");

  assert(false && "Not applicable to Scalar Values");
  return a;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pbroadcast_fourth(const Packet& a) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_6(mht_6_v, 297, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_fourth");

  assert(false && "Not applicable to Scalar Values");
  return a;
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pload4bf16(
    const typename unpacket_traits<Packet>::type* from) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_7(mht_7_v, 307, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pload4bf16");

  assert(false && "Not applicable to Scalar Values");
  return Packet();
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pload2bf16(
    const typename unpacket_traits<Packet>::type* from) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_8(mht_8_v, 317, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pload2bf16");

  assert(false && "Not applicable to Scalar Values");
  return Packet();
}

// Specialization for pload4bf16 and pload2bf16 for non-sse.
// Enable vectorization on z13 and higher.
#if defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX) || \
    defined(EIGEN_VECTORIZE_NEON) || defined(EIGEN_VECTORIZE_ZVECTOR)
template <>
EIGEN_STRONG_INLINE Packet4f pload4bf16<Packet4f>(const float* from) {
  tensorflow::uint32 p[4];
  const tensorflow::uint32* ir =
      reinterpret_cast<const tensorflow::uint32*>(from);
  p[0] = (ir[0] << 16) & 0xffff0000;
  p[1] = ir[0] & 0xffff0000;
  p[2] = (ir[1] << 16) & 0xffff0000;
  p[3] = ir[1] & 0xffff0000;
  return ploadu<Packet4f>(reinterpret_cast<float*>(p));
}

template <>
EIGEN_STRONG_INLINE Packet4f pload2bf16<Packet4f>(const float* from) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_9(mht_9_v, 342, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pload2bf16<Packet4f>");

  tensorflow::uint32 p[4];
  const tensorflow::uint32* ir =
      reinterpret_cast<const tensorflow::uint32*>(from);
  p[0] = (ir[0] << 16) & 0xffff0000;
  p[1] = ir[0] & 0xffff0000;
  p[2] = (ir[0] << 16) & 0xffff0000;
  p[3] = ir[0] & 0xffff0000;
  return ploadu<Packet4f>(reinterpret_cast<float*>(p));
}
#endif

#if defined(EIGEN_VECTORIZE_NEON)
// Return a packet with the first value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_first<Packet4f>(const Packet4f& a) {
  return pset1<Packet4f>(pfirst(a));
}
template <>
EIGEN_STRONG_INLINE Packet2f pbroadcast_first<Packet2f>(const Packet2f& a) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_10(mht_10_v, 364, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_first<Packet2f>");

  return pset1<Packet2f>(pfirst(a));
}

// Return a packet with the second value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_second<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_11(mht_11_v, 373, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_second<Packet4f>");

  return pset1<Packet4f>(vgetq_lane_f32(a, 1));
}
template <>
EIGEN_STRONG_INLINE Packet2f pbroadcast_second<Packet2f>(const Packet2f& a) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_12(mht_12_v, 380, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_second<Packet2f>");

  return pset1<Packet2f>(vget_lane_f32(a, 1));
}

// Return a packet with the third value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_third<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_13(mht_13_v, 389, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_third<Packet4f>");

  return pset1<Packet4f>(vgetq_lane_f32(a, 2));
}

// Return a packet with the fourth value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_fourth<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_14(mht_14_v, 398, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_fourth<Packet4f>");

  return pset1<Packet4f>(vgetq_lane_f32(a, 3));
}
#endif

#if defined(EIGEN_VECTORIZE_ALTIVEC) || defined(EIGEN_VECTORIZE_VSX)
// Return a packet with the first value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_first<Packet4f>(const Packet4f& a) {
  return vec_splat(a, 0);
}

// Return a packet with the second value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_second<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_15(mht_15_v, 415, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_second<Packet4f>");

  return vec_splat(a, 1);
}

// Return a packet with the third value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_third<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_16(mht_16_v, 424, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_third<Packet4f>");

  return vec_splat(a, 2);
}

// Return a packet with the fourth value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_fourth<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_17(mht_17_v, 433, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_fourth<Packet4f>");

  return vec_splat(a, 3);
}
#endif

#ifdef EIGEN_VECTORIZE_SSE2
// For PacketSize of 4 floats the Packet is not modified
template <>
EIGEN_STRONG_INLINE Packet4f pinterleave4x64<Packet4f>(const Packet4f& from) {
  return from;
}

// Return a Packet with 4 floats loaded from 4 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet4f pload4bf16<Packet4f>(const float* from) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_18(mht_18_v, 450, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pload4bf16<Packet4f>");

  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castpd_si128(_mm_load_pd1((const double*)from));
  return _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp));
}

// Return a Packet with 2 floats loaded from 2 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet4f pload2bf16<Packet4f>(const float* from) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_19(mht_19_v, 461, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pload2bf16<Packet4f>");

  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(_mm_load_ps1(from));
  return _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp));
}

// Return a Packet with 4 floats expanded from 4 bfloat16 values
// in the lower half of the 128-bit lane
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet4f pexpand_bf16_l(const Packet4f& from) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_20(mht_20_v, 473, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pexpand_bf16_l");

  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(from);
  return _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp));
}

// Return a Packet with 4 floats expanded from 4 bfloat16 values
// in the upper half of the 128-bit lane
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet4f pexpand_bf16_u(const Packet4f& from) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_21(mht_21_v, 485, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pexpand_bf16_u");

  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(from);
  return _mm_castsi128_ps(_mm_unpackhi_epi16(zero, tmp));
}

// Return a packet with the first value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_first<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_22(mht_22_v, 496, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_first<Packet4f>");

  return _mm_set1_ps(pfirst<Packet4f>(a));
}

// Return a packet with the second value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_second<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_23(mht_23_v, 505, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_second<Packet4f>");

  return _mm_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(a, a, 1)));
}

// Return a packet with the third value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_third<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_24(mht_24_v, 514, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_third<Packet4f>");

  return _mm_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(a, a, 2)));
}

// Return a packet with the fourth value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet4f pbroadcast_fourth<Packet4f>(const Packet4f& a) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_25(mht_25_v, 523, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_fourth<Packet4f>");

  return _mm_set1_ps(_mm_cvtss_f32(_mm_shuffle_ps(a, a, 3)));
}

#endif

#ifdef EIGEN_VECTORIZE_AVX512
template <>
EIGEN_STRONG_INLINE Packet16f
pbroadcast_first<Packet16f>(const Packet16f& a_in) {
  Packet4f a = _mm512_castps512_ps128(a_in);
  return _mm512_broadcastss_ps(a);
}
template <>
EIGEN_STRONG_INLINE Packet16f
pbroadcast_second<Packet16f>(const Packet16f& a_in) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_26(mht_26_v, 541, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_second<Packet16f>");

  Packet4f a = _mm512_castps512_ps128(a_in);
  return _mm512_broadcastss_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
}
template <>
EIGEN_STRONG_INLINE Packet16f
pbroadcast_third<Packet16f>(const Packet16f& a_in) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_27(mht_27_v, 550, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_third<Packet16f>");

  Packet4f a = _mm512_castps512_ps128(a_in);
  return _mm512_broadcastss_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 2, 2)));
}
template <>
EIGEN_STRONG_INLINE Packet16f
pbroadcast_fourth<Packet16f>(const Packet16f& a_in) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_28(mht_28_v, 559, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_fourth<Packet16f>");

  Packet4f a = _mm512_castps512_ps128(a_in);
  return _mm512_broadcastss_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3)));
}
template <>
EIGEN_STRONG_INLINE Packet8d pbroadcast_first<Packet8d>(const Packet8d& a_in) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_29(mht_29_v, 567, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_first<Packet8d>");

  Packet2d a = _mm512_castpd512_pd128(a_in);
  return _mm512_broadcastsd_pd(a);
}
template <>
EIGEN_STRONG_INLINE Packet8d pbroadcast_second<Packet8d>(const Packet8d& a_in) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_30(mht_30_v, 575, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_second<Packet8d>");

  Packet2d a = _mm_permute_pd(_mm512_castpd512_pd128(a_in), 3);
  return _mm512_broadcastsd_pd(a);
}
template <>
EIGEN_STRONG_INLINE Packet8d pbroadcast_third<Packet8d>(const Packet8d& a_in) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_31(mht_31_v, 583, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_third<Packet8d>");

  Packet2d a = _mm256_extractf128_pd(_mm512_castpd512_pd256(a_in), 1);
  return _mm512_broadcastsd_pd(a);
}
template <>
EIGEN_STRONG_INLINE Packet8d pbroadcast_fourth<Packet8d>(const Packet8d& a_in) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_32(mht_32_v, 591, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_fourth<Packet8d>");

  Packet2d a =
      _mm_permute_pd(_mm256_extractf128_pd(_mm512_castpd512_pd256(a_in), 1), 3);
  return _mm512_broadcastsd_pd(a);
}
template <>
EIGEN_STRONG_INLINE Packet16i
pbroadcast_first<Packet16i>(const Packet16i& a_in) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_33(mht_33_v, 601, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_first<Packet16i>");

  Packet4i a = _mm512_castsi512_si128(a_in);
  return _mm512_broadcastd_epi32(a);
}
template <>
EIGEN_STRONG_INLINE Packet16i
pbroadcast_second<Packet16i>(const Packet16i& a_in) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_34(mht_34_v, 610, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_second<Packet16i>");

  Packet4i a = _mm512_castsi512_si128(a_in);
  return _mm512_broadcastd_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(1, 1, 1, 1)));
}
template <>
EIGEN_STRONG_INLINE Packet16i
pbroadcast_third<Packet16i>(const Packet16i& a_in) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_35(mht_35_v, 619, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_third<Packet16i>");

  Packet4i a = _mm512_castsi512_si128(a_in);
  return _mm512_broadcastd_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(2, 2, 2, 2)));
}
template <>
EIGEN_STRONG_INLINE Packet16i
pbroadcast_fourth<Packet16i>(const Packet16i& a_in) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_36(mht_36_v, 628, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_fourth<Packet16i>");

  Packet4i a = _mm512_castsi512_si128(a_in);
  return _mm512_broadcastd_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 3, 3)));
}
#endif

#ifdef EIGEN_VECTORIZE_AVX
// For a Packet of Size 8 floats(256-bits), swap the 2nd and 3rd quadwords
template <>
EIGEN_STRONG_INLINE Packet8f pinterleave4x64<Packet8f>(const Packet8f& from) {
#ifdef EIGEN_VECTORIZE_AVX2
  return _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(from),
                                                      _MM_SHUFFLE(3, 1, 2, 0)));
#else
  auto tmp1 = _mm256_extract_epi32(_mm256_castps_si256(from), 2);
  auto tmp2 = _mm256_extract_epi32(_mm256_castps_si256(from), 3);
  auto tmp3 = _mm256_extract_epi32(_mm256_castps_si256(from), 4);
  auto tmp4 = _mm256_extract_epi32(_mm256_castps_si256(from), 5);
  auto tmp5 = _mm256_insert_epi32(_mm256_castps_si256(from), tmp1, 4);
  tmp5 = _mm256_insert_epi32(tmp5, tmp2, 5);
  tmp5 = _mm256_insert_epi32(tmp5, tmp3, 2);
  tmp5 = _mm256_insert_epi32(tmp5, tmp4, 3);
  return _mm256_castsi256_ps(tmp5);
#endif
}
// Return a Packet with 4 floats loaded from 4 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet8f pload4bf16<Packet8f>(const float* from) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_37(mht_37_v, 658, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pload4bf16<Packet8f>");

  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castpd_si128(_mm_load_pd1((const double*)from));
  return _mm256_castps128_ps256(
      _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp)));
}
// Return a Packet with 2 floats loaded from 2 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet8f pload2bf16<Packet8f>(const float* from) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_38(mht_38_v, 669, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pload2bf16<Packet8f>");

  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(_mm_load_ps1(from));
  return _mm256_castps128_ps256(
      _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp)));
}

#ifdef EIGEN_VECTORIZE_AVX512
// Return a Packet with 4 floats loaded from 4 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet16f pload4bf16<Packet16f>(const float* from) {
  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castpd_si128(_mm_load_pd1((const double*)from));
  return _mm512_castps128_ps512(
      _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp)));
}
// Return a Packet with 2 floats loaded from 2 bfloat16 values
template <>
EIGEN_STRONG_INLINE Packet16f pload2bf16<Packet16f>(const float* from) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_39(mht_39_v, 690, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pload2bf16<Packet16f>");

  __m128i zero = _mm_setzero_si128();
  __m128i tmp = _mm_castps_si128(_mm_load_ps1(from));
  return _mm512_castps128_ps512(
      _mm_castsi128_ps(_mm_unpacklo_epi16(zero, tmp)));
}
#endif

// For each 128-bit lane convert 4 bfloat to 4 float values from the lower half
// of the 128-bit lane
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet8f pexpand_bf16_l(const Packet8f& from) {
#ifdef EIGEN_VECTORIZE_AVX2
  __m256i zero = _mm256_setzero_si256();
  __m256i tmp = _mm256_castps_si256(from);
  return _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero, tmp));
#else
  __m128i zero = _mm_setzero_si128();
  __m128i low = _mm_castps_si128(_mm256_extractf128_ps(from, 0));
  __m128i res_l = _mm_unpacklo_epi16(zero, low);
  __m128i high = _mm_castps_si128(_mm256_extractf128_ps(from, 1));
  __m128i res_h = _mm_unpacklo_epi16(zero, high);
  __m256 res = _mm256_castps128_ps256(_mm_castsi128_ps(res_l));
  res = _mm256_insertf128_ps(res, _mm_castsi128_ps(res_h), 1);
  return res;
#endif
}

// For each 128-bit lane convert 4 bfloat to 4 float values from the upper half
// of the 128-bit lane
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet8f pexpand_bf16_u(const Packet8f& from) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_40(mht_40_v, 724, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pexpand_bf16_u");

#ifdef EIGEN_VECTORIZE_AVX2
  __m256i zero = _mm256_setzero_si256();
  __m256i tmp = _mm256_castps_si256(from);
  return _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero, tmp));
#else
  __m128i zero = _mm_setzero_si128();
  __m128i low = _mm_castps_si128(_mm256_extractf128_ps(from, 0));
  __m128i res_l = _mm_unpackhi_epi16(zero, low);
  __m128i high = _mm_castps_si128(_mm256_extractf128_ps(from, 1));
  __m128i res_h = _mm_unpackhi_epi16(zero, high);
  __m256 res = _mm256_castps128_ps256(_mm_castsi128_ps(res_l));
  res = _mm256_insertf128_ps(res, _mm_castsi128_ps(res_h), 1);
  return res;
#endif
}

// Return a packet with the first value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet8f pbroadcast_first<Packet8f>(const Packet8f& a) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_41(mht_41_v, 746, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_first<Packet8f>");

  return _mm256_set1_ps(pfirst<Packet8f>(a));
}

// Return a packet with the second value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet8f pbroadcast_second<Packet8f>(const Packet8f& a) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_42(mht_42_v, 755, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_second<Packet8f>");

  return _mm256_set1_ps(
      _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_permute_ps(a, 1))));
}

// Return a packet with the third value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet8f pbroadcast_third<Packet8f>(const Packet8f& a) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_43(mht_43_v, 765, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_third<Packet8f>");

  return _mm256_set1_ps(
      _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_permute_ps(a, 2))));
}

// Return a packet with the fourth value of the input Packet replicated
template <>
EIGEN_STRONG_INLINE Packet8f pbroadcast_fourth<Packet8f>(const Packet8f& a) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_44(mht_44_v, 775, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pbroadcast_fourth<Packet8f>");

  return _mm256_set1_ps(
      _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_permute_ps(a, 3))));
}

#endif

#ifdef EIGEN_VECTORIZE_AVX512

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet16f pexpand_bf16_l(const Packet16f& from) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm512_castsi512_si256(_mm512_castps_si512(from))),
      16));
}

template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet16f pexpand_bf16_u(const Packet16f& from) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_matmul_opDTh mht_45(mht_45_v, 795, "", "./tensorflow/core/kernels/sparse_matmul_op.h", "pexpand_bf16_u");

  Packet16i tmp = _mm512_castps_si512(from);
  Packet16i tmp2 = _mm512_alignr_epi32(tmp, tmp, 8);
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm512_castsi512_si256(tmp2)), 16));
}

#endif
}  // namespace internal
}  // namespace Eigen
#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_MATMUL_OP_H_
