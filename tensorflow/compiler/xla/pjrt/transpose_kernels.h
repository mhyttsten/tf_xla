/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_KERNELS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_KERNELS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh() {
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

#include "third_party/eigen3/Eigen/Core"

namespace xla {

// Generic transpose kernel.
//
// All of the kernels that follow in this file are optimized versions of this
// generic kernel, specialized to particular block sizes and data types.
//
// The transpose kernel requires its input to be contiguous in one of the two
// dimensions being transposed, and the output to be contiguous in the other
// dimension.
//
// lda, ldb are strides in bytes.
template <typename T, int bs>
struct TransposeMicroKernel {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/pjrt/transpose_kernels.h", "Apply");

    for (int i = 0; i < bs; ++i) {
      for (int j = 0; j < bs; ++j) {
        *reinterpret_cast<T*>(b + i * ldb + j * sizeof(T)) =
            *reinterpret_cast<T const*>(a + j * lda + i * sizeof(T));
      }
    }
  }
};

// TODO(phawkins): it would be nice to remove the use of Eigen here, and instead
// allow for runtime dispatch of, say, AVX or AVX2 kernels where they are
// supported. On the other hand, using Eigen makes for easier cross-platform
// portability.
#ifdef EIGEN_VECTORIZE_AVX

template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh mht_1(mht_1_v, 229, "", "./tensorflow/compiler/xla/pjrt/transpose_kernels.h", "Apply");

    __m128i x = _mm_set_epi32(*reinterpret_cast<const uint32_t*>(a + lda * 0),
                              *reinterpret_cast<const uint32_t*>(a + lda * 1),
                              *reinterpret_cast<const uint32_t*>(a + lda * 2),
                              *reinterpret_cast<const uint32_t*>(a + lda * 3));
    __m128i mask =
        _mm_setr_epi8(12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3);
    x = _mm_shuffle_epi8(x, mask);
    *reinterpret_cast<uint32_t*>(b + ldb * 0) = _mm_extract_epi32(x, 0);
    *reinterpret_cast<uint32_t*>(b + ldb * 1) = _mm_extract_epi32(x, 1);
    *reinterpret_cast<uint32_t*>(b + ldb * 2) = _mm_extract_epi32(x, 2);
    *reinterpret_cast<uint32_t*>(b + ldb * 3) = _mm_extract_epi32(x, 3);
  }
};

// TODO(phawkins): add an 8x8 byte transpose kernel.

// TODO(phawkins): Eigen doesn't have a SSE/AVX byte Packet16c type. Add one
// and call it here rather than using AVX intrinsics.
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/16> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh mht_2(mht_2_v, 254, "", "./tensorflow/compiler/xla/pjrt/transpose_kernels.h", "Apply");

    std::array<__m128i, 16> packet;
    for (int i = 0; i < 16; ++i) {
      packet[i] =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + lda * i));
    }

    // If we number the elements in the input thus:
    // kernel.packet[ 0] = {00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 0a, 0b, 0c,
    //                      0d, 0e, 0f}
    // kernel.packet[ 1] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1a, 1b, 1c,
    //                      1d, 1e, 1f}
    // ...
    // kernel.packet[15] = {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, fa, fb, fc,
    //                      fd, fe, ff},
    //
    // the desired output is:
    // kernel.packet[ 0] = {00, 10, 20, 30, 40, 50, 60, 70, 80, 90, a0, b0, c0,
    //                      d0, e0, f0}
    // kernel.packet[ 1] = {01, 11, 21, 31, 41, 51, 61, 71, 81, 91, a1, b1, c1,
    //                      d1, e1, f1}
    // ...
    // kernel.packet[15] = {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, af, bf, cf,
    //                      df, ef, ff},
    // 00 10 01 11 02 12 03 13 04 14 05 15 06 16 07 17
    __m128i t0 = _mm_unpacklo_epi8(packet[0], packet[1]);
    // 08 18 09 19 0a 1a 0b 1b 0c 1c 0d 1d 0e 1e 0f 1f
    __m128i t1 = _mm_unpackhi_epi8(packet[0], packet[1]);
    // 20 30 21 31 22 32 ...                     27 37
    __m128i t2 = _mm_unpacklo_epi8(packet[2], packet[3]);
    // 28 38 29 39 2a 3a ...                     2f 3f
    __m128i t3 = _mm_unpackhi_epi8(packet[2], packet[3]);
    // 40 50 41 51 42 52                         47 57
    __m128i t4 = _mm_unpacklo_epi8(packet[4], packet[5]);
    // 48 58 49 59 4a 5a
    __m128i t5 = _mm_unpackhi_epi8(packet[4], packet[5]);
    __m128i t6 = _mm_unpacklo_epi8(packet[6], packet[7]);
    __m128i t7 = _mm_unpackhi_epi8(packet[6], packet[7]);
    __m128i t8 = _mm_unpacklo_epi8(packet[8], packet[9]);
    __m128i t9 = _mm_unpackhi_epi8(packet[8], packet[9]);
    __m128i ta = _mm_unpacklo_epi8(packet[10], packet[11]);
    __m128i tb = _mm_unpackhi_epi8(packet[10], packet[11]);
    __m128i tc = _mm_unpacklo_epi8(packet[12], packet[13]);
    __m128i td = _mm_unpackhi_epi8(packet[12], packet[13]);
    __m128i te = _mm_unpacklo_epi8(packet[14], packet[15]);
    __m128i tf = _mm_unpackhi_epi8(packet[14], packet[15]);

    // 00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
    __m128i s0 = _mm_unpacklo_epi16(t0, t2);
    __m128i s1 = _mm_unpackhi_epi16(t0, t2);  // 04 14 24 34
    __m128i s2 = _mm_unpacklo_epi16(t1, t3);  // 08 18 28 38 ...
    __m128i s3 = _mm_unpackhi_epi16(t1, t3);  // 0c 1c 2c 3c ...
    // 40 50 60 70 41 51 61 71 42 52 62 72 43 53 63 73
    __m128i s4 = _mm_unpacklo_epi16(t4, t6);
    __m128i s5 = _mm_unpackhi_epi16(t4, t6);  // 44 54 64 74 ...
    __m128i s6 = _mm_unpacklo_epi16(t5, t7);
    __m128i s7 = _mm_unpackhi_epi16(t5, t7);
    __m128i s8 = _mm_unpacklo_epi16(t8, ta);
    __m128i s9 = _mm_unpackhi_epi16(t8, ta);
    __m128i sa = _mm_unpacklo_epi16(t9, tb);
    __m128i sb = _mm_unpackhi_epi16(t9, tb);
    __m128i sc = _mm_unpacklo_epi16(tc, te);
    __m128i sd = _mm_unpackhi_epi16(tc, te);
    __m128i se = _mm_unpacklo_epi16(td, tf);
    __m128i sf = _mm_unpackhi_epi16(td, tf);

    // 00 10 20 30 40 50 60 70 01 11 21 31 41 51 61 71
    __m128i u0 = _mm_unpacklo_epi32(s0, s4);
    // 02 12 22 32 42 52 62 72 03 13 23 33 43 53 63 73
    __m128i u1 = _mm_unpackhi_epi32(s0, s4);
    __m128i u2 = _mm_unpacklo_epi32(s1, s5);
    __m128i u3 = _mm_unpackhi_epi32(s1, s5);
    __m128i u4 = _mm_unpacklo_epi32(s2, s6);
    __m128i u5 = _mm_unpackhi_epi32(s2, s6);
    __m128i u6 = _mm_unpacklo_epi32(s3, s7);
    __m128i u7 = _mm_unpackhi_epi32(s3, s7);
    __m128i u8 = _mm_unpacklo_epi32(s8, sc);
    __m128i u9 = _mm_unpackhi_epi32(s8, sc);
    __m128i ua = _mm_unpacklo_epi32(s9, sd);
    __m128i ub = _mm_unpackhi_epi32(s9, sd);
    __m128i uc = _mm_unpacklo_epi32(sa, se);
    __m128i ud = _mm_unpackhi_epi32(sa, se);
    __m128i ue = _mm_unpacklo_epi32(sb, sf);
    __m128i uf = _mm_unpackhi_epi32(sb, sf);

    packet[0] = _mm_unpacklo_epi64(u0, u8);
    packet[1] = _mm_unpackhi_epi64(u0, u8);
    packet[2] = _mm_unpacklo_epi64(u1, u9);
    packet[3] = _mm_unpackhi_epi64(u1, u9);
    packet[4] = _mm_unpacklo_epi64(u2, ua);
    packet[5] = _mm_unpackhi_epi64(u2, ua);
    packet[6] = _mm_unpacklo_epi64(u3, ub);
    packet[7] = _mm_unpackhi_epi64(u3, ub);
    packet[8] = _mm_unpacklo_epi64(u4, uc);
    packet[9] = _mm_unpackhi_epi64(u4, uc);
    packet[10] = _mm_unpacklo_epi64(u5, ud);
    packet[11] = _mm_unpackhi_epi64(u5, ud);
    packet[12] = _mm_unpacklo_epi64(u6, ue);
    packet[13] = _mm_unpackhi_epi64(u6, ue);
    packet[14] = _mm_unpacklo_epi64(u7, uf);
    packet[15] = _mm_unpackhi_epi64(u7, uf);
    for (int i = 0; i < 16; ++i) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(b + ldb * i), packet[i]);
    }
  }
};

// TODO(phawkins): add an 4x4 uint16_t transpose kernel.

template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/8> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh mht_3(mht_3_v, 369, "", "./tensorflow/compiler/xla/pjrt/transpose_kernels.h", "Apply");

    using Eigen::internal::Packet8h;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 8;
    PacketBlock<Packet8h, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet8h>(
          reinterpret_cast<const Eigen::half*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<Eigen::half>(
          reinterpret_cast<Eigen::half*>(b + ldb * i), block.packet[i]);
    }
  }
};

template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh mht_4(mht_4_v, 392, "", "./tensorflow/compiler/xla/pjrt/transpose_kernels.h", "Apply");

    using Eigen::internal::Packet4f;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 4;
    PacketBlock<Packet4f, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet4f>(
          reinterpret_cast<const float*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<float>(reinterpret_cast<float*>(b + ldb * i),
                                      block.packet[i]);
    }
  }
};

template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/8> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh mht_5(mht_5_v, 415, "", "./tensorflow/compiler/xla/pjrt/transpose_kernels.h", "Apply");

    using Eigen::internal::Packet8f;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 8;
    PacketBlock<Packet8f, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet8f>(
          reinterpret_cast<const float*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<float>(reinterpret_cast<float*>(b + ldb * i),
                                      block.packet[i]);
    }
  }
};

template <>
struct TransposeMicroKernel<uint64_t, /*bs=*/2> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh mht_6(mht_6_v, 438, "", "./tensorflow/compiler/xla/pjrt/transpose_kernels.h", "Apply");

    using Eigen::internal::Packet2d;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 2;
    PacketBlock<Packet2d, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet2d>(
          reinterpret_cast<const double*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<double>(reinterpret_cast<double*>(b + ldb * i),
                                       block.packet[i]);
    }
  }
};

template <>
struct TransposeMicroKernel<uint64_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStranspose_kernelsDTh mht_7(mht_7_v, 461, "", "./tensorflow/compiler/xla/pjrt/transpose_kernels.h", "Apply");

    using Eigen::internal::Packet4d;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 4;
    PacketBlock<Packet4d, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet4d>(
          reinterpret_cast<const double*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<double>(reinterpret_cast<double*>(b + ldb * i),
                                       block.packet[i]);
    }
  }
};

#endif  // EIGEN_VECTORIZE_AVX

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_KERNELS_H_
